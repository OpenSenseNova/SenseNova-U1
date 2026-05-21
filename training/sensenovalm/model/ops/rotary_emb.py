# --------------------------------------------------------
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from Flash-Attention (Tri Dao, BSD-3-Clause).
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause portions retain their license.
# Upstream reference: https://github.com/Dao-AILab/flash-attention
# --------------------------------------------------------
"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the roatry embedding operators.
"""

from typing import Callable, Tuple

import torch
from einops import rearrange
from torch import Tensor

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.core.context import global_context as gpc

try:
    from rotary_emb import apply_rotary as _flash_apply_rotary_func

    flash_rotary_impl = True
except (ModuleNotFoundError, ImportError):
    flash_rotary_impl = False

try:
    from deeplink_ext.internevo_ops import ApplyRotaryEmb as DeeplinkApplyRotaryEmb

    deeplink_rotary_impl = True
except (ModuleNotFoundError, ImportError):
    deeplink_rotary_impl = False


try:
    from torch_npu import npu_rotary_mul

    torchnpu_rotary_impl = True
except (ModuleNotFoundError, ImportError):
    torchnpu_rotary_impl = False

sensenovalm_accelerator = get_accelerator()


def _rope_to_float32_wrapper(input_idxs: Tuple, rope_func: Callable, *args, **kwargs):
    try:
        use_fp32_rope = gpc.config.model.get("use_fp32_rope", True)
    except AttributeError:
        use_fp32_rope = True

    if use_fp32_rope:
        inputs = [args[idx] for idx in input_idxs]
        input_dtype = inputs[0].dtype
        other_args = [args[idx] for idx in range(len(inputs), len(args))]

        for idx in input_idxs:
            inputs[idx] = inputs[idx].to(torch.float32)

        res = rope_func(*inputs, *other_args, **kwargs)
        if res is not None:
            return res.to(input_dtype)
    else:
        return rope_func(*args, **kwargs)


def _torch_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    # TODO: improve perfermance.
    assert x1.device == x2.device == cos.device == sin.device, "All inputs must be on the same device"
    assert x1.dtype == x2.dtype == cos.dtype == sin.dtype, "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"


    if conj:
        out1.copy_(x1 * cos + x2 * sin)
        out2.copy_(-x1 * sin + x2 * cos)
    else:
        out1.copy_(x1 * cos - x2 * sin)
        out2.copy_(x1 * sin + x2 * cos)


def _apply_npu_rotary_mul(x: Tensor, cos: Tensor, sin: Tensor):
    """
    Implement RotaryEmbedding rotation position encoding. Support FakeTensor mode.
    Ref: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/
            apiref/fmkadptapi/ptaoplist_000451.html
    Args:
        x (Tensor): q or k, shape is [B, S, N, D].
        cos (Tensor): cos, shape is [1, S, 1, D].
        sin (Tensor): sin, shape is [1, S, 1, D].
    """
    return npu_rotary_mul(x, cos, sin)


def _apply_torch_npu_rotary_mul(x: Tensor, cos: Tensor, sin: Tensor):
    """Torch implementation of 'npu_rotary_mul', baseline for unit testing.

    Args:
        x (Tensor): q or k, shape is [B, S, N, D].
        cos (Tensor): cos, shape is [1, S, 1, D].
        sin (Tensor): sin, shape is [1, S, 1, D].
    """

    # NOTE: This could probably be moved to Triton.
    def rotate_half(_x):
        x1, x2 = _x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # Handle a possible sequence length mismatch in between q and k.
    cos = cos[:, : x.shape[1], :, :]
    sin = sin[:, : x.shape[1], :, :]
    re = (x * cos) + (rotate_half(x) * sin)

    del rotate_half
    return re


def _select_apply_rotary_func_npu(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, use_fused_rope: bool = False):
    if use_fused_rope:
        return _rope_to_float32_wrapper((0, 1, 2), _apply_npu_rotary_mul, x, cos, sin)
    else:
        return _rope_to_float32_wrapper((0, 1, 2), _apply_torch_npu_rotary_mul, x, cos, sin)


def rotary_emb_in_rotate_half_style(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved=False,
    use_fused_rope=False,
):
    """The rotary_emb implemented in the rotate_half style is different from the flash_attn's rotary_emb
    in that cos and sin require [max_position_embeddings, dim/2] -> [1, max_position_embeddings, 1, dim].

    Args:
        x (Tensor): x, If x is qkv, shape is [B, S, 3, N, D]; If x is q or k, shape is [B, S, N, D].
        cos (Tensor): cos, shape is [S, D//2].
        sin (Tensor): sin, shape is [S, D//2].
    """
    assert False, "This function has some bugs. You should not arrive here."
    # reformat cos/sin shape.
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :]

    if len(x.shape) == 5:
        q, k, _ = x.unbind(dim=2)
        q, k = q.squeeze(dim=2), k.squeeze(dim=2)
        if interleaved:
            q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
            k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)

        q = _select_apply_rotary_func_npu(q, cos, sin, use_fused_rope)
        k = _select_apply_rotary_func_npu(k, cos, sin, use_fused_rope)

        if interleaved:
            x[:, :, 0, ..., : x.shape[-1] // 2].copy_(q[..., ::2])
            x[:, :, 0, ..., x.shape[-1] // 2 :].copy_(q[..., 1::2])

            x[:, :, 1, ..., : x.shape[-1] // 2].copy_(k[..., ::2])
            x[:, :, 1, ..., x.shape[-1] // 2 :].copy_(k[..., 1::2])
        else:
            x[:, :, 0, ...].copy_(q)
            x[:, :, 1, ...].copy_(k)
    else:
        if interleaved:
            x = torch.cat([x[..., ::2], x[..., 1::2]], dim=-1)
        x = _select_apply_rotary_func_npu(x, cos, sin, use_fused_rope)
        if interleaved:
            out = torch.empty_like(x)
            out[..., ::2].copy_(x[..., : x.shape[-1] // 2])
            out[..., 1::2].copy_(x[..., x.shape[-1] // 2 :])
            x = out
    return x


def _select_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
    use_fused_rope: bool = True,
) -> None:
    if use_fused_rope and flash_rotary_impl:
        _flash_apply_rotary_func(x1, x2, cos, sin, out1, out2, conj)
    else:
        _rope_to_float32_wrapper((0, 1, 2, 3), _torch_apply_rotary_func, x1, x2, cos, sin, out1, out2, conj)


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L35
class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        in_place: bool = False,
        use_fused_rope: bool = True,
    ):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        *_, seqlen, _, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert rotary_dim <= head_dim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)

        x_ro = x[..., :rotary_dim]
        x1, x2 = (x_ro[..., ::2], x_ro[..., 1::2]) if interleaved else x_ro.chunk(2, dim=-1)

        if in_place:
            out, o1, o2 = x, x1, x2
        else:
            out = torch.empty_like(x)
            out_ro = out[..., :rotary_dim]
            o1, o2 = (out_ro[..., ::2], out_ro[..., 1::2]) if interleaved else out_ro.chunk(2, dim=-1)

        _select_apply_rotary_func(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
            use_fused_rope,
        )

        if rotary_dim < head_dim and not in_place:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place
        ctx.use_fused_rope = use_fused_rope

        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        do_ro = do[..., :rotary_dim]
        do1, do2 = (do_ro[..., ::2], do_ro[..., 1::2]) if ctx.interleaved else do_ro.chunk(2, dim=-1)

        if ctx.in_place:
            dx, dx1, dx2 = do, do1, do2
        else:
            dx = torch.empty_like(do)
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (dx_ro[..., ::2], dx_ro[..., 1::2]) if ctx.interleaved else dx_ro.chunk(2, dim=-1)

        _select_apply_rotary_func(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
            ctx.use_fused_rope,
        )

        if rotary_dim < head_dim and not ctx.in_place:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])

        return dx, None, None, None, None


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False, in_place: bool = False
):
    # TODO: Support deeplink in a more unified manner
    if sensenovalm_accelerator.get_accelerator_backend() in [AcceleratorType.DIPU, AcceleratorType.DITORCH]:
        return DeeplinkApplyRotaryEmb.apply(x, cos, sin, interleaved, in_place)
    if sensenovalm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        return ApplyRotaryEmb.apply(x, cos, sin, interleaved, in_place)
    else:
        return ApplyRotaryEmb.apply(x, cos, sin, interleaved, in_place)     # NOTE:
