#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
from einops import rearrange

from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_expert") and param.is_expert:
        return True
    return False


def is_gate_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_gate") and param.is_gate:
        return True
    return False


def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


def Gelu(w1_o, w2_o):
    return F.gelu(w1_o) * w2_o


# may reduce numerical precision
# Silu = torch.jit.script(Silu)
Gelu = torch.jit.script(Gelu)


def update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            # For FT, k_cache has shape (b, h, headdim / packsize, s, packsize)
            # where packsize = 4 if fp32, 8 if fp16 or bf16.
            # v_cache has shape (b, h, s, headdim)
            k_cache, v_cache = inference_params.key_value_memory_dict[layer_idx]
            kv_cache = None
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0])
    assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2])
    # Copy key and values.
    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        kv = kv_cache[batch_start:batch_end, :sequence_end, ...]
        return kv
    else:
        assert inference_params.sequence_len_offset == 0
        # FT kernel requires different layouts for the k_cache and v_cache.
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        packsize = 4 if kv.dtype == torch.float32 else 8
        if kv_cache is not None:
            kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
            k_cache = rearrange(
                kv_cache[:, :, 0],
                "b s h (d packsize) -> b h d s packsize",
                packsize=packsize,
            ).contiguous()
            v_cache = rearrange(kv_cache[:, :, 1], "b s h d -> b h s d").contiguous()
            inference_params.key_value_memory_dict[layer_idx] = (k_cache, v_cache)
        else:
            k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
                kv[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize
            )
            v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(kv[:, :, 1], "b s h d -> b h s d")
        return kv
