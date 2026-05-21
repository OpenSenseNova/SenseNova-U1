#!/usr/bin/env python
# --------------------------------------------------------
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from Flash-Attention (Tri Dao, BSD-3-Clause).
# RotaryEmbedding follows RoFormer (Su et al., 2021) and the GPT-NeoX
# implementation; XPos extensions follow Sun et al., 2022.
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause portions retain their license.
# Upstream references:
#   https://github.com/Dao-AILab/flash-attention
#   https://github.com/ZhuiyiTechnology/roformer
#   https://github.com/EleutherAI/gpt-neox
# --------------------------------------------------------
# -*- encoding: utf-8 -*-
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.ops.rotary_emb import apply_rotary_emb
from sensenovalm.utils.parallel import is_using_isp


class Embedding1D(nn.Module):
    """
    1D Embedding.

    Args:
        num_embeddings (int): The size of vocab.
        embedding_dim (int): The dimention of model.
        padding_idx (int): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                            therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                            i.e. it remains as a fixed "pad". None by default.
        dtype (Optional[torch.dtype]): Data type None by default.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        vocab_parallel: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs
        self.vocab_parallel = vocab_parallel

        parallel_size = gpc.weight_parallel_size if is_using_isp() else gpc.tensor_parallel_size

        if vocab_parallel:
            assert num_embeddings % parallel_size == 0, f"{num_embeddings} is not divisible by {parallel_size}"

            self.num_embeddings_per_partition = num_embeddings // parallel_size
            self.embed_dim_per_partition = embedding_dim
            self.vocab_start_index = gpc.get_local_rank(ParallelMode.TENSOR) * self.num_embeddings_per_partition
            self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        else:
            assert embedding_dim % parallel_size == 0, f"{embedding_dim} is not divisible by {parallel_size}"

            self.num_embeddings_per_partition = num_embeddings
            self.embed_dim_per_partition = embedding_dim // parallel_size
            self.vocab_start_index = 0
            self.vocab_end_index = self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim_per_partition), dtype=dtype)
        )

    def forward(self, input_: Tensor) -> Tensor:
        if self.vocab_parallel and not is_using_isp():
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        output = F.embedding(masked_input, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        if self.vocab_parallel and not is_using_isp():
            output[input_mask, :] = 0.0

        return output


class RotaryEmbedding(torch.nn.Module):     # NOTE:
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None, rope_scaling=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim = dim
        self.base = base

        assert rope_scaling == None
        # NOTE:
        print("################ legacy init rope ############")
        dim = dim * 2
        if not rope_scaling:
            self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        else:
            raise NotImplementedError
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            factor = rope_scaling["factor"]  # `8` in the original implementation
            low_freq_factor = rope_scaling["low_freq_factor"]  # `1` in the original implementation
            high_freq_factor = rope_scaling["high_freq_factor"]  # `4` in the original implementation
            old_context_len = rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        # NOTE:
        self.inv_freq = self.inv_freq[::2]

        self.scale_base = scale_base
        # NOTE:
        self.scale = (
            #(torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[::2] + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        assert (scale_base == 0) and (self.scale == None)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(
        self, x: torch.Tensor, indexes: Union[int, torch.Tensor] = 0, max_seqlen: Optional[int] = None
    ):
        """x: (batch, seqlen, nheads, headdim)"""
        assert max_seqlen is not None
        if max_seqlen is not None:   
            seqlen = max_seqlen
        elif isinstance(indexes, int):
            seqlen = indexes + x.shape[1]
        else:
            # Note that this statement may cause synchronization between CPU and GPU,
            # so it's best to precompute and pass in max_seqlen ahead of time
            seqlen = indexes.max().item()

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                raise NotImplementedError
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _get_slice(self, tensor: torch.Tensor, offsets: Union[int, torch.Tensor] = 0):
        if isinstance(offsets, int):
            return tensor[offsets:]
        else:
            return tensor[offsets]

    def _convert_padding(
        self, x: torch.Tensor, empties: torch.Tensor, convert_type: str = "left2right", in_place: bool = False
    ):
        # TODO: impl in_place = True.
        assert not in_place, "in_place = True is NYI."
        assert convert_type in ("left2right", "right2left"), f"Unknown convert type {convert_type}"

        ret = x.clone()

        for i in range(len(empties)):
            if empties[i] == 0:
                continue

            if convert_type == "left2right":
                ret[i][: -empties[i]] = x[i][empties[i] :]
                ret[i][-empties[i] :] = x[i][: empties[i]]
            else:  # right2left
                ret[i][empties[i] :] = x[i][: -empties[i]]
                ret[i][: empties[i]] = x[i][-empties[i] :]

        return ret

    def forward(
        self,
        x: torch.Tensor,
        offsets: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        cache_type: str = "query",
        interleaved: bool = False,
        in_place: bool = False,
        left_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Applies rotary position embeddings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            offsets (Union[int, torch.Tensor], optional): The sequence offsets for the input. Defaults to 0.
            max_seqlen (Optional[int], optional): The maximum sequence length for caching. Defaults to None.
            cache_type (str, optional): Specifies whether the cache is for 'query' or 'key'. Defaults to "query".
            interleaved (bool, optional): Whether the input tensor is interleaved. Defaults to False.
            in_place (bool, optional): Whether the operation should be done in-place. Defaults to False.
            left_padding_mask (Optional[torch.Tensor], optional): A mask for left padding. Defaults to None.

        Returns:
            torch.Tensor: The tensor with applied rotary position embeddings.
        """
        assert cache_type in ("query", "key"), f"Unknown cache type {cache_type}"
        assert isinstance(offsets, (torch.Tensor)), f"Invalid offsets type {type(offsets)}"
        assert interleaved is False

        if left_padding_mask is not None:   
            raise NotImplementedError    
            empties = left_padding_mask[..., -1].sum(dim=-1)
            x = self._convert_padding(x, empties, convert_type="left2right", in_place=in_place)

        self._update_cos_sin_cache(x, offsets, max_seqlen)      # NOTE:

        cos_cached = self._cos_k_cached if cache_type == "key" and self.scale is not None else self._cos_cached
        sin_cached = self._sin_k_cached if cache_type == "key" and self.scale is not None else self._sin_cached
        ret = apply_rotary_emb(
            x, self._get_slice(cos_cached, offsets), self._get_slice(sin_cached, offsets), interleaved, in_place
        )

        if left_padding_mask is not None:
            raise NotImplementedError
            ret = self._convert_padding(ret, empties, convert_type="right2left", in_place=in_place)

        return ret


class LinearRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/200009566639b5a83604e522a41df3a9 \
            5b6056ed/src/transformers/models/llama/modeling_llama.py#L159C1-L176C1
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item()
        else:
            seqlen = indexes + x.shape[1]

        t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/eb8489971ac1415f67b0abdd1584fde8 \
            b659ced9/src/transformers/models/llama/modeling_llama.py#L147
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update(self, seqlen, x):
        self._seq_len_cached = seqlen
        if seqlen > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
        else:
            inv_freq = self.inv_freq

        t = torch.arange(seqlen, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item()
        else:
            seqlen = indexes + x.shape[1]  # eval_forward
        if seqlen <= self.max_position_embeddings:
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                self._seq_len_cached > self.max_position_embeddings
                or seqlen > self._seq_len_cached
                or self._cos_cached.device != x.device
                or self._cos_cached.dtype != x.dtype
            ):
                self._update(seqlen, x)
        else:
            self._update(seqlen, x)


def new_rotary_embedding(
    dim: int,
    base=10000,
    scale_base=0,
    device=None,
    max_position_embeddings=2048,
    scaling_factor=1.0,
    rotary_type: str = "native",
    rope_scaling: Optional[dict] = None,
) -> RotaryEmbedding:
    assert rotary_type in ("native", "linear_scale", "dynamic_ntk"), f"Unknown rotary type {rotary_type}"

    if rotary_type == "linear_scale":
        raise NotImplementedError('LinearRotaryEmbedding')
        return LinearRotaryEmbedding(dim, base, scale_base, device, max_position_embeddings, scaling_factor)
    elif rotary_type == "dynamic_ntk":
        raise NotImplementedError('DynamicNTKScalingRotaryEmbedding')
        return DynamicNTKScalingRotaryEmbedding(dim, base, scale_base, device, max_position_embeddings, scaling_factor)
    else:  # native
        return RotaryEmbedding(dim, base, scale_base, device, rope_scaling)
