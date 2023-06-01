# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SpeechT5 model."""

import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "SpeechT5Config"


SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/speecht5_asr",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_vc",
    # See all SpeechT5 models at https://huggingface.co/models?filter=speecht5
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_spectrograms_right(input_values: torch.Tensor, reduction_factor: int = 1):
    """
    Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
    """
    # thin out frames for reduction factor
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1 :: reduction_factor]

    shifted_input_values = input_values.new_zeros(input_values.shape)
    shifted_input_values[:, 1:] = input_values[:, :-1].clone()

    # replace possible -100 values in labels by zeros
    shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)

    return shifted_input_values


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# https://github.com/pytorch/pytorch/issues/25661#issuecomment-845419189
def _make_causal_mask(
    bsz: int, tgt_len: int, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0,
    min16: float = torch.finfo(torch.float16).min,
    min32: float = torch.finfo(torch.float32).min,
    min64: float = torch.finfo(torch.float64).min,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    if dtype == torch.float16:
        mask = torch.full((tgt_len, tgt_len), torch.tensor(min16, device=device), device=device)
    elif dtype == torch.float32:
        mask = torch.full((tgt_len, tgt_len), torch.tensor(min32, device=device), device=device)
    else:
        mask = torch.full((tgt_len, tgt_len), torch.tensor(min64, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None,
    min16: float = torch.finfo(torch.float16).min,
    min32: float = torch.finfo(torch.float32).min,
    min64: float = torch.finfo(torch.float64).min,
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    if dtype == torch.float16:
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), min16)
    elif dtype == torch.float32:
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), min32)
    else:
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), min64)


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = 512 if layer_id > 0 else 1 # conv_dim
        self.out_conv_dim = 512 # conv_dim

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=[10, 3, 3, 3, 3, 2, 2][layer_id], # conv_kernel
            stride=[5, 2, 2, 2, 2, 2, 2][layer_id], # conv_stride
            bias=False, # conv_bias
        )
        self.activation = ACT2FN['gelu'] # feat_extract_activation

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = 512 if layer_id > 0 else 1 # conv_dim
        self.out_conv_dim = 512 # conv_dim

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=[10, 3, 3, 3, 3, 2, 2][layer_id], # conv_kernel
            stride=[5, 2, 2, 2, 2, 2, 2][layer_id], # conv_stride
            bias=False, # conv_bias
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN['gelu'] # feat_extract_activation

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SpeechT5
class SpeechT5GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = 512 if layer_id > 0 else 1 # conv_dim
        self.out_conv_dim = 512 # conv_dim

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=[10, 3, 3, 3, 3, 2, 2][layer_id], # conv_kernel
            stride=[5, 2, 2, 2, 2, 2, 2][layer_id], # conv_stride
            bias=False, # conv_bias
        )
        self.activation = ACT2FN['gelu'] # feat_extract_activation

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->SpeechT5
class SpeechT5SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SpeechT5
class SpeechT5PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            768, # hidden_size
            768, # hidden_size
            kernel_size=128, # num_conv_pos_embeddings
            padding=64, # num_conv_pos_embeddings // 2
            groups=16, # num_conv_pos_embedding_groups
        )

#         if True:
        import deepspeed

        with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
        deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
#         else:
#             self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = SpeechT5SamePadLayer(128) # num_conv_pos_embeddings
        self.activation = ACT2FN['gelu'] # feat_extract_activation

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SpeechT5ScaledPositionalEncoding(nn.Module):
    """
    Scaled positional encoding, see §3.2 in https://arxiv.org/abs/1809.08895
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        emb = emb + self.alpha * self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb


class SpeechT5RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_length=1000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.pe_k = torch.nn.Embedding(2 * max_length, dim)

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]

        pos_seq[pos_seq < -self.max_length] = -self.max_length
        pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = pos_seq + self.max_length

        return self.pe_k(pos_seq)


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->SpeechT5
class SpeechT5SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->SpeechT5
class SpeechT5FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        if True: # feat_extract_norm == "group"
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
                SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(6) # num_feat_extract_layers - 1
            ]
        else:
            conv_layers = [
                SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(7) # num_feat_extract_layers
            ]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->SpeechT5
class SpeechT5FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512, eps=1e-05) # conv_dim, layer_norm_eps
        self.projection = nn.Linear(512, 768) # conv_dim, hidden_size
        self.dropout = nn.Dropout(0.0) # feat_proj_dropout

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class SpeechT5SpeechDecoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    80 if i == 0 else 256, # num_mel_bins, speech_decoder_prenet_units
                    256, # speech_decoder_prenet_units
                )
                for i in range(2) # speech_decoder_prenet_layers
            ]
        )

        self.final_layer = nn.Linear(256, 768) # speech_decoder_prenet_units, hidden_size

        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            0.1, # positional_dropout
            768, # hidden_size
            1876, # 1876
        )

        self.speaker_embeds_layer = nn.Linear(1280, 768) # speaker_embedding_dim + hidden_size, hidden_size

    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ):
        # Dropout is always applied, even when evaluating. See §2.2 in https://arxiv.org/abs/1712.05884.

        inputs_embeds = input_values
        for layer in self.layers:
            inputs_embeds = nn.functional.relu(layer(inputs_embeds))
            inputs_embeds = nn.functional.dropout(
                inputs_embeds, 0.5, training=True # speech_decoder_prenet_dropout
            )

        inputs_embeds = self.final_layer(inputs_embeds)
        inputs_embeds = self.encode_positions(inputs_embeds)

        if speaker_embeddings is not None:
            speaker_embeddings = nn.functional.normalize(speaker_embeddings)
            speaker_embeddings = speaker_embeddings.unsqueeze(1)
            speaker_embeddings = speaker_embeddings.expand(-1, inputs_embeds.size(1), -1)
            inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
            inputs_embeds = nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

        return inputs_embeds


class SpeechT5BatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        if layer_id == 0:
            in_conv_dim = 80 # num_mel_bins
        else:
            in_conv_dim = 256 # speech_decoder_postnet_units

        if layer_id == 4: # speech_decoder_postnet_layers - 1
            out_conv_dim = 80 # num_mel_bins
        else:
            out_conv_dim = 256 # speech_decoder_postnet_units

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=5, # speech_decoder_postnet_kernel,
            stride=1,
            padding=2, # (speech_decoder_postnet_kernel - 1) // 2
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        if layer_id < 4: # speech_decoder_postnet_layers - 1
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.dropout = nn.Dropout(0.5) # speech_decoder_postnet_dropout

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SpeechT5SpeechDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feat_out = nn.Linear(768, 160) # hidden_size, num_mel_bins * reduction_factor
        self.prob_out = nn.Linear(768, 2) # hidden_size, reduction_factor

        self.layers = nn.ModuleList(
            [SpeechT5BatchNormConvLayer(config, i) for i in range(5)] # speech_decoder_postnet_layers
        )

    def forward(self, hidden_states: torch.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, 80) # num_mel_bins
        outputs_after_postnet = self.postnet(outputs_before_postnet)
        logits = self.prob_out(hidden_states).view(hidden_states.size(0), -1)
        return outputs_before_postnet, outputs_after_postnet, logits

    def postnet(self, hidden_states: torch.Tensor):
        layer_output = hidden_states.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        return hidden_states + layer_output.transpose(1, 2)


class SpeechT5TextEncoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(81, 768, 1) # vocab_size, hidden_size, pad_token_id
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            0.1, # positional_dropout
            768, # hidden_size
            600, # max_text_positions
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: torch.LongTensor):
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.encode_positions(inputs_embeds)
        return inputs_embeds


class SpeechT5TextDecoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(0.1) # positional_dropout
        self.embed_scale = 1.0 # sqrt(hidden_size) if scale_embedding

        self.embed_tokens = nn.Embedding(81, 768, 1) # vocab_size, hidden_size, pad_token_id

        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
            602, # max_text_positions + pad_token_id + 1
            768, # hidden_size
            1, # pad_token_id
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]] = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        positions = self.embed_positions(input_ids, past_key_values_length)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embeds += positions
        inputs_embeds = self.dropout(inputs_embeds)

        return inputs_embeds, attention_mask


class SpeechT5TextDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(768, 81, bias=False) # hidden_size, vocab_size

    def forward(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class SpeechT5Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # relative attention bias
        if position_bias is not None:
            reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
            rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
            rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
                bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
            )
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class SpeechT5CrossAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # relative attention bias
        if position_bias is not None:
            reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
            rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
            rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
                bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
            )
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class SpeechT5FeedForward(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(0.1) # activation_dropout

        self.intermediate_dense = nn.Linear(768, intermediate_size) # hidden_size
        self.intermediate_act_fn = ACT2FN['gelu'] # hidden_act

        self.output_dense = nn.Linear(intermediate_size, 768) # hidden_size
        self.output_dropout = nn.Dropout(0.1) # hidden_dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SpeechT5EncoderLayer(nn.Module):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.attention = SpeechT5Attention(
            embed_dim=768, # hidden_size
            num_heads=12, # encoder_attention_heads
            dropout=0.1, # attention_dropout
            is_decoder=False,
        )
        self.dropout = nn.Dropout(0.1) # hidden_dropout
        self.layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps
        self.feed_forward = SpeechT5FeedForward(config, 3072)# encoder_ffn_dim
        self.final_layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            position_bias (`torch.FloatTensor`):
                relative position embeddings of size `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            position_bias=position_bias,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states,)


class SpeechT5DecoderLayer(nn.Module):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.self_attn = SpeechT5Attention(
            embed_dim=768, # hidden_size
            num_heads=12, # decoder_attention_heads
            dropout=0.1, # attention_dropout
            is_decoder=True,
        )
        self.dropout = nn.Dropout(0.1) # hidden_dropout
        self.self_attn_layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps

        self.encoder_attn = SpeechT5CrossAttention(
            768, # hidden_size
            12, # decoder_attention_heads
            dropout=0.1, # attention_dropout
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps

        self.feed_forward = SpeechT5FeedForward(config, 3072) # decoder_ffn_dim
        self.final_layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        cross_attn_weights: Optional[torch.Tensor] = None
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        assert present_key_value is not None and cross_attn_present_key_value is not None
        return_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, return_key_value)


class SpeechT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpeechT5Config
    base_model_prefix = "speecht5"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SpeechT5PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, SpeechT5FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02) # initializer_range
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02) # initializer_range
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (SpeechT5Encoder, SpeechT5Decoder, SpeechT5FeatureEncoder)):
            module.gradient_checkpointing = value


class SpeechT5Encoder(SpeechT5PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`SpeechT5EncoderLayer`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layer_norm = nn.LayerNorm(768, eps=1e-05) # hidden_size, layer_norm_eps
        self.dropout = nn.Dropout(0.1) # hidden_dropout

        self.layers = nn.ModuleList([SpeechT5EncoderLayer(config) for _ in range(12)]) # encoder_layers

        self.embed_positions = SpeechT5RelativePositionalEncoding(
            64, 160 # hidden_size // encoder_attention_heads, encoder_max_relative_position
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Features extracted from the speech or text input by the encoder prenet.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        """
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        position_bias = self.embed_positions(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            # under deepspeed zero3 all gpus must run in sync
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class SpeechT5EncoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5TextEncoderPrenet to convert the input_ids to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5TextEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.prenet.set_input_embeddings(value)

    def forward(
        self,
        input_values: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.prenet(input_values)

        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        return outputs


class SpeechT5Decoder(SpeechT5PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SpeechT5DecoderLayer`]
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layers = nn.ModuleList([SpeechT5DecoderLayer(config) for _ in range(6)]) # decoder_layers

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.LongTensor],
        bsz: int, tgt_len: int,
        inputs_embeds: torch.FloatTensor,
        past_key_values_length: int
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask: Optional[torch.Tensor] = None
        if tgt_len > 1:
            combined_attention_mask = _make_causal_mask(
                bsz, tgt_len,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=tgt_len).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]] = None,
    ) -> Tuple[torch.Tensor, Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]]:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Features extracted from the speech or text input by the decoder prenet.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
        """
        bsz, tgt_len, _ = hidden_states.size()

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, bsz, tgt_len, hidden_states, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, hidden_states.dtype, tgt_len=tgt_len)

        # decoder layers
        next_decoder_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

            next_decoder_cache.append(layer_outputs[1])

        return (hidden_states, (next_decoder_cache[0], next_decoder_cache[1], next_decoder_cache[2], next_decoder_cache[3], next_decoder_cache[4], next_decoder_cache[5]))


class SpeechT5DecoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5SpeechDecoderPrenet to convert log-mel filterbanks to hidden
    features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.prenet = SpeechT5SpeechDecoderPrenet(config)
        self.wrapped_decoder = SpeechT5Decoder(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]] = None,
    ) -> Tuple[torch.Tensor, Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]]:
        decoder_hidden_states = self.prenet(input_values, speaker_embeddings)

        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
        )

        return outputs


class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):
    """
    Guided attention loss from the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention](https://arxiv.org/abs/1710.08969), adapted for multi-head attention.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.sigma = 0.4 # guided_attention_loss_sigma
        self.scale = 10.0 # guided_attention_loss_scale

    def forward(
        self, attentions: torch.FloatTensor, input_masks: torch.BoolTensor, output_masks: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute the attention loss.

        Args:
            attentions (`torch.FloatTensor` of shape `(batch_size, layers * heads, output_sequence_length, input_sequence_length)`):
                Batch of multi-head attention weights
            input_masks (`torch.BoolTensor` of shape `(batch_size, input_sequence_length)`):
                Input attention mask as booleans.
            output_masks (`torch.BoolTensor` of shape `(batch_size, output_sequence_length)`):
                Target attention mask as booleans.

        Returns:
            `torch.Tensor` with the loss value
        """
        guided_attn_masks = self._make_guided_attention_masks(input_masks, output_masks, attentions.device)
        masks = output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)
        masks = masks.to(attentions.device).unsqueeze(1)

        losses = guided_attn_masks * attentions
        loss = torch.mean(losses.masked_select(masks))
        return self.scale * loss

    def _make_guided_attention_masks(self, input_masks, output_masks, device):
        input_lengths = input_masks.sum(-1)
        output_lengths = output_masks.sum(-1)

        guided_attn_masks = torch.zeros((len(input_masks), output_masks.shape[1], input_masks.shape[1]), device=device)

        for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma, device)

        return guided_attn_masks.unsqueeze(1)

    @staticmethod
    def _make_guided_attention_mask(input_length, output_length, sigma, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(input_length, device=device),
            torch.arange(output_length, device=device),
            indexing="xy",
        )
        grid_x = grid_x.float() / output_length
        grid_y = grid_y.float() / input_length
        return 1.0 - torch.exp(-((grid_y - grid_x) ** 2) / (2 * (sigma**2)))


class SpeechT5SpectrogramLoss(nn.Module):
    """
    Loss computation used by SpeechT5ForTextToSpeech.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.use_guided_attention_loss = True # use_guided_attention_loss
        self.guided_attention_loss_num_heads = 2 # guided_attention_loss_num_heads
        self.reduction_factor = 2 # reduction_factor

        self.l1_criterion = L1Loss()
        self.bce_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))

        if self.use_guided_attention_loss:
            self.attn_criterion = SpeechT5GuidedMultiheadAttentionLoss(config)

    def forward(
        self,
        attention_mask: torch.LongTensor,
        outputs_before_postnet: torch.FloatTensor,
        outputs_after_postnet: torch.FloatTensor,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        cross_attentions: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        padding_mask = labels != -100.0

        # mask out the padded portions
        labels = labels.masked_select(padding_mask)
        outputs_before_postnet = outputs_before_postnet.masked_select(padding_mask)
        outputs_after_postnet = outputs_after_postnet.masked_select(padding_mask)

        # spectrogram loss
        l1_loss = self.l1_criterion(outputs_after_postnet, labels) + self.l1_criterion(outputs_before_postnet, labels)

        # construct stop labels from the padding mask
        masks = padding_mask[:, :, 0]
        stop_labels = torch.cat([~masks * 1.0, torch.ones(masks.size(0), 1).to(masks.device)], dim=1)
        stop_labels = stop_labels[:, 1:].masked_select(masks)
        logits = logits.masked_select(masks)

        # stop token loss
        bce_loss = self.bce_criterion(logits, stop_labels)

        # combined loss
        loss = l1_loss + bce_loss

        # guided attention loss
        if self.use_guided_attention_loss:
            attn = torch.cat([x[:, : self.guided_attention_loss_num_heads] for x in cross_attentions], dim=1)
            input_masks = attention_mask == 1
            output_masks = padding_mask[:, :, 0]
            if self.reduction_factor > 1:
                output_masks = output_masks[:, self.reduction_factor - 1 :: self.reduction_factor]
            attn_loss = self.attn_criterion(attn, input_masks, output_masks)
            loss += attn_loss

        return loss


SPEECHT5_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`SpeechT5EncoderWithTextPrenet`]):
            The Transformer encoder module that applies the appropiate text encoder prenet.
        decoder ([`SpeechT5DecoderWithSpeechPrenet`]):
            The Transformer decoder module that applies the appropiate speech decoder prenet.
"""


SPEECHT5_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SPEECHT5_INPUTS_DOCSTRING = r"""
    Args:
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.

            </Tip>

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read [`SpeechT5Decoder._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

        head_mask (`torch.FloatTensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_values` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_values` of shape `(batch_size, sequence_length)`. decoder_inputs_embeds (`torch.FloatTensor`
            of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
            `decoder_input_values` you can choose to directly pass an embedded representation. If `past_key_values` is
            used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is
            useful if you want more control over how to convert `decoder_input_values` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.",
    SPEECHT5_BASE_START_DOCSTRING,
)
class SpeechT5Model(SpeechT5PreTrainedModel):
    def __init__(
        self,
        config: SpeechT5Config,
        encoder: SpeechT5EncoderWithTextPrenet,
        decoder: SpeechT5DecoderWithSpeechPrenet,
    ):
        super().__init__(config)
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self):
        return None


@add_start_docstrings(
    """SpeechT5 Model with a text encoder and a speech decoder.""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForTextToSpeech(SpeechT5PreTrainedModel):
    _keys_to_ignore_on_load_missing = []
    _keys_to_ignore_on_save = []

    main_input_name = "input_ids"

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        text_encoder = SpeechT5EncoderWithTextPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, text_encoder, speech_decoder)

        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def forward(
        self,
        input_ids: torch.LongTensor,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
    ) -> torch.FloatTensor:
        r"""
        Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
        speech waveform using a vocoder.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. The `batch_size` should be 1 currently.

                Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
                [`~PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.

        Returns:
            `tuple(torch.FloatTensor)`:
            - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
              `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
        """
        encoder_attention_mask = torch.ones_like(input_ids)

        encoder_last_hidden_state = self.speecht5.encoder(
            input_values=input_ids,
            attention_mask=encoder_attention_mask,
        )

        maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / 2) # reduction_factor
        minlen = int(encoder_last_hidden_state.size(1) * minlenratio / 2) # reduction_factor

        # Start the output sequence with a mel spectrum that is all zeros.
        output_sequence = encoder_last_hidden_state.new_zeros(1, 1, 80) # num_mel_bins

        spectrogram = []
        past_key_values: Optional[Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ]] = None

        for idx in range(maxlen + 1):
            # Run the decoder prenet on the entire output sequence.
            decoder_hidden_states = self.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

            # Run the decoder layers on the last element of the prenet output.
            last_decoder_output, past_key_values = self.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                attention_mask=None,
                encoder_hidden_states=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )

            last_decoder_output = last_decoder_output[0, -1]

            # Predict the new mel spectrum for this step in the sequence.
            spectrum = self.speech_decoder_postnet.feat_out(last_decoder_output)
            spectrum = spectrum.view(2, 80) # reduction_factor, num_mel_bins
            spectrogram.append(spectrum)

            # Extend the output sequence with the new mel spectrum.
            output_sequence = torch.cat((output_sequence, spectrum[-1].view(1, 1, 80)), dim=1) # num_mel_bins

            # Predict the probability that this is the stop token.
            prob = torch.sigmoid(self.speech_decoder_postnet.prob_out(last_decoder_output))

            # Finished when stop token or maximum length is reached.
            if idx >= minlen and int(sum(prob >= threshold)) > 0:
                break

        spectrogram = torch.cat(spectrogram, dim=0).unsqueeze(0)
        spectrogram = self.speech_decoder_postnet.postnet(spectrogram)
        spectrogram = spectrogram.squeeze(0)
        return spectrogram


HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5HifiGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
class SpeechT5HifiGan(PreTrainedModel):
    config_class = SpeechT5HifiGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: SpeechT5HifiGanConfig):
        super().__init__(config)
        self.num_kernels = 3 # len(resblock_kernel_sizes)
        self.num_upsamples = 4 # len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            80, # model_in_dim
            512, # upsample_initial_channel
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip([4, 4, 4, 4], [8, 8, 8, 8])): # upsample_rates, upsample_kernel_sizes
            self.upsampler.append(
                nn.ConvTranspose1d(
                    512 // (2**i), # upsample_initial_channel
                    512 // (2 ** (i + 1)), # upsample_initial_channel
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = 512 // (2 ** (i + 1)) # upsample_initial_channel
            for kernel_size, dilation in zip([3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]]): # resblock_kernel_sizes, resblock_dilation_sizes
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, 0.1)) # leaky_relu_slope

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        self.register_buffer("mean", torch.zeros(80)) # model_in_dim
        self.register_buffer("scale", torch.ones(80)) # model_in_dim

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.01) # initializer_range
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)

    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if True: # normalize_before
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)

        hidden_states = spectrogram.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)
        
        hidden_states = nn.functional.leaky_relu(hidden_states, 0.1) # leaky_relu_slope
        hidden_states = self.upsampler[0](hidden_states)
        res_state = self.resblocks[0](hidden_states) + self.resblocks[1](hidden_states) + self.resblocks[2](hidden_states)
        hidden_states = res_state / self.num_kernels
        
        hidden_states = nn.functional.leaky_relu(hidden_states, 0.1) # leaky_relu_slope
        hidden_states = self.upsampler[1](hidden_states)
        res_state = self.resblocks[3](hidden_states) + self.resblocks[4](hidden_states) + self.resblocks[5](hidden_states)
        hidden_states = res_state / self.num_kernels
        
        hidden_states = nn.functional.leaky_relu(hidden_states, 0.1) # leaky_relu_slope
        hidden_states = self.upsampler[2](hidden_states)
        res_state = self.resblocks[6](hidden_states) + self.resblocks[7](hidden_states) + self.resblocks[8](hidden_states)
        hidden_states = res_state / self.num_kernels
        
        hidden_states = nn.functional.leaky_relu(hidden_states, 0.1) # leaky_relu_slope
        hidden_states = self.upsampler[3](hidden_states)
        res_state = self.resblocks[9](hidden_states) + self.resblocks[10](hidden_states) + self.resblocks[11](hidden_states)
        hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform
