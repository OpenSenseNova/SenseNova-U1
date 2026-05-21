# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
Some hepler functions for ops package.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def unpack_qkv_before_attn(cur_input: torch.Tensor, cu_seqlens: torch.Tensor, padding_v: int = 0):
    """
    qkv: the shape is (1, packed_length, three, head_num, head_dim)
    kv: the shape is (1, packed_length, two, head_num, head_dim)
    q/k/v: the shape is (1, packed_length, head_num, head_dim)

    Return:
    output: the shape is (micro_bsz, seq_len, three, head_num, head_dim) for qkv
                        (micro_bsz, seq_len, two, head_num, head_dim) for kv
                        (micro_bsz, seq_len, head_num, head_dim) for q/k/v
    """
    assert cur_input.shape[0] == 1
    cur_input = cur_input.squeeze(0)

    sequences = []
    for i in range(len(cu_seqlens) - 1):
        sequences.append(cur_input[cu_seqlens[i] : cu_seqlens[i + 1]])

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_v)

    return padded_sequences


def pack_output_after_attn(cur_input: torch.Tensor, cu_seqlens: torch.Tensor, packed_length: int, padding_v: int = 0):
    """
    cur_input: the shape is (micro_bsz, seq_len, head_num, head_dim)

    Return:
    output: the shape is (1, packed_length, head_num, head_dim)
    """
    output_shape = list(cur_input.shape)
    output_shape[0] = 1
    output_shape[1] = packed_length

    output = torch.full(output_shape, fill_value=padding_v, device=cur_input.device, dtype=cur_input.dtype)
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        output[0, cu_seqlens[i] : cu_seqlens[i + 1]] = cur_input[i, 0:length]

    return output
