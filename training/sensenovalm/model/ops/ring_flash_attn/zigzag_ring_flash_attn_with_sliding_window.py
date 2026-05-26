# --------------------------------------------------------
# Derived from Ring-Flash-Attention (zhuzilin, MIT).
# Copyright (c) zhuzilin.
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original MIT portions retain their license.
# Upstream reference:
#   https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/zigzag_ring_flash_attn.py
# --------------------------------------------------------

import torch
import torch.distributed
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from sensenovalm.core.context.parallel_context import global_context as gpc

from .utils import RingComm, update_out_and_lse

fa_output_mapping = {}


def create_buffer(tensor):
    buffer_shape = list(tensor.shape)
    return torch.empty(buffer_shape, dtype=tensor.dtype, device=tensor.device)


def zigzag_double_ring_flash_attn_forward(
    ring_pg,
    p2p_pg,
    local_p2p_pg,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):
    assert causal is True, "zigzag ring is meaningless for causal=False"
    ring_comm = RingComm(ring_pg)
    p2p_comm = RingComm(p2p_pg)
    local_p2p_comm = RingComm(local_p2p_pg)

    block_seq_len = q.shape[1] // 2

    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    def _first_window_forward(q, k, v):
        out = None
        lse = None

        for step in range(local_p2p_comm.world_size):

            if step + 1 != local_p2p_comm.world_size:
                next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                local_p2p_comm.commit()

            if step == 0:
                block_out, block_lse = forward(
                    q,
                    k,
                    v,
                    causal=True,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= local_p2p_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

            if step + 1 != local_p2p_comm.world_size:
                local_p2p_comm.wait()
                k = next_k
                v = next_v

        return out, lse

    def _other_window_forward(out, lse, q, k, v, window_num_idx):

        if window_num_idx > p2p_comm.rank:

            for step in range(local_p2p_comm.world_size):

                if step + 1 != local_p2p_comm.world_size:
                    next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                    next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                    local_p2p_comm.commit()

                q1 = q[:, block_seq_len:]
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

                if step + 1 != local_p2p_comm.world_size:
                    local_p2p_comm.wait()
                    k = next_k
                    v = next_v
        else:
            for step in range(local_p2p_comm.world_size):

                if step + 1 != local_p2p_comm.world_size:
                    next_k: torch.Tensor = local_p2p_comm.send_recv(k)
                    next_v: torch.Tensor = local_p2p_comm.send_recv(v)
                    local_p2p_comm.commit()

                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

                if step + 1 != local_p2p_comm.world_size:
                    local_p2p_comm.wait()
                    k = next_k
                    v = next_v
        return out, lse

    window_size = gpc.config.parallel.sequence_2D.get("window_size", 1)
    window_num = ring_comm.world_size // window_size

    local_k = k
    local_v = v

    for j in range(window_num):

        if j > 0:
            p2p_comm.wait()
            local_k = next_k
            local_v = next_v

        if j + 1 != window_num:
            next_k: torch.Tensor = p2p_comm.send_recv(local_k.contiguous())
            next_v: torch.Tensor = p2p_comm.send_recv(local_v.contiguous())
            p2p_comm.commit()

        if j == 0:
            out, lse = _first_window_forward(q, local_k, local_v)
        else:
            out, lse = _other_window_forward(out, lse, q, local_k, local_v, window_num_idx=j)

    lse = lse.squeeze(dim=-1).transpose(1, 2)
    out = out.to(q.dtype)

    return out, lse


def zigzag_double_ring_flash_attn_backward(
    context_pg,
    inter_window_pg,
    intra_window_pg,
    dkv_inter_window_pg,
    dkv_intra_window_pg,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),  # pylint: disable=W0613
    alibi_slopes=None,  # pylint: disable=W0613
    deterministic=False,  # pylint: disable=W0613
):

    assert causal is True, "zigzag ring is meaningless for causal=False"

    context_comm = RingComm(context_pg)
    dkv_comm = RingComm(dkv_inter_window_pg)
    kv_comm = RingComm(inter_window_pg)
    local_kv_comm = RingComm(intra_window_pg)
    local_dkv_comm = RingComm(dkv_intra_window_pg)

    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = create_buffer(q)
    dk_buffer = create_buffer(k)
    dv_buffer = create_buffer(v)

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        softmax_lse = softmax_lse.contiguous()
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
        )

    def _first_window_backward(dout, q, k, v, out, softmax_lse):

        dk_comm_buffer, dv_comm_buffer = None, None
        dq, dk, dv = None, None, None

        for step in range(local_kv_comm.world_size):
            if step + 1 != local_kv_comm.world_size:
                next_k = local_kv_comm.send_recv(k)
                next_v = local_kv_comm.send_recv(v)
                local_kv_comm.commit()

            if step == 0:
                backward(dout, q, k, v, out, softmax_lse, causal=True)
                dq = dq_buffer.to(torch.float32)
                dk = dk_buffer.to(torch.float32)
                dv = dv_buffer.to(torch.float32)
            else:
                if step <= local_kv_comm.rank:
                    k0 = k[:, :block_seq_len]
                    v0 = v[:, :block_seq_len]
                    backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                    dq += dq_buffer
                else:
                    dout1 = dout.chunk(2, dim=1)[1]
                    q1 = q.chunk(2, dim=1)[1]
                    out1 = out.chunk(2, dim=1)[1]
                    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
                    backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                    # always use the first half in dq_buffer.
                    dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]  # pylint: disable=E1137

                local_dkv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv

                if step <= local_kv_comm.rank:
                    dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                    dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
                else:
                    dk += dk_buffer
                    dv += dv_buffer

            if step + 1 != local_kv_comm.world_size:
                local_kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
            next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
            local_dkv_comm.commit()

        local_dkv_comm.wait()

        return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

    def _other_window_backward(dout, q, k, v, dq, dk, dv, out, softmax_lse, window_num_idx, inter_window_dkv_comm):

        dk_comm_buffer, dv_comm_buffer = None, None

        if window_num_idx > kv_comm.rank:

            for step in range(local_kv_comm.world_size):

                if step + 1 != local_kv_comm.world_size:
                    next_k = local_kv_comm.send_recv(k)
                    next_v = local_kv_comm.send_recv(v)
                    local_kv_comm.commit()

                dout1 = dout.chunk(2, dim=1)[1]
                q1 = q.chunk(2, dim=1)[1]
                out1 = out.chunk(2, dim=1)[1]
                softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

                if step > 0:
                    local_dkv_comm.wait()
                    dk_comm_buffer, dv_comm_buffer = dk, dv
                    dk, dv = next_dk, next_dv

                if step == 0:
                    inter_window_dkv_comm.wait()

                dk += dk_buffer
                dv += dv_buffer

                if step + 1 != local_kv_comm.world_size:
                    local_kv_comm.wait()
                    k = next_k
                    v = next_v

                next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
                next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
                local_dkv_comm.commit()

            local_dkv_comm.wait()
        else:

            for step in range(local_kv_comm.world_size):

                if step + 1 != local_kv_comm.world_size:
                    next_k = local_kv_comm.send_recv(k)
                    next_v = local_kv_comm.send_recv(v)
                    local_kv_comm.commit()

                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer

                if step > 0:
                    local_dkv_comm.wait()
                    dk_comm_buffer, dv_comm_buffer = dk, dv
                    dk, dv = next_dk, next_dv

                if step == 0:
                    inter_window_dkv_comm.wait()

                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]

                if step + 1 != local_kv_comm.world_size:
                    local_kv_comm.wait()
                    k = next_k
                    v = next_v

                next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
                next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
                local_dkv_comm.commit()

            local_dkv_comm.wait()

        return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

    window_size = gpc.config.parallel.sequence_2D.get("window_size", 1)
    window_num = context_comm.world_size // window_size

    local_k = k
    local_v = v

    for j in range(window_num):

        if j > 0:
            kv_comm.wait()
            local_k = next_k
            local_v = next_v

        if j + 1 != window_num:
            next_k: torch.Tensor = kv_comm.send_recv(local_k.contiguous())
            next_v: torch.Tensor = kv_comm.send_recv(local_v.contiguous())
            kv_comm.commit()

        if j > 0:
            dk = next_dk
            dv = next_dv

        if j == 0:
            dq, dk, dv = _first_window_backward(
                dout,
                q,
                local_k,
                local_v,
                out,
                softmax_lse,
            )
        else:
            dq, dk, dv = _other_window_backward(
                dout,
                q,
                local_k,
                local_v,
                dq,
                dk,
                dv,
                out,
                softmax_lse,
                window_num_idx=j,
                inter_window_dkv_comm=dkv_comm,
            )

        next_dk: torch.Tensor = dkv_comm.send_recv(dk.contiguous())
        next_dv: torch.Tensor = dkv_comm.send_recv(dv.contiguous())
        dkv_comm.commit()

    dkv_comm.wait()

    dq = dq.to(q.dtype)
    next_dk = next_dk.to(q.dtype)
    next_dv = next_dv.to(q.dtype)

    return dq, next_dk, next_dv


class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    """ZigZagRingFlashAttnFunc"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        context_group,
        inter_window_group=None,
        intra_window_group=None,
        dkv_inter_window_group=None,
        dkv_intra_window_group=None,
        layer_idx=0,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()

        if gpc.is_forward is False and gpc.config.selective_checkpoint:
            assert layer_idx in fa_output_mapping
            out, softmax_lse = fa_output_mapping.pop(layer_idx)
        else:
            out, softmax_lse = zigzag_double_ring_flash_attn_forward(
                context_group,
                inter_window_group,
                intra_window_group,
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
            )

        # store attn forward output to avoid re-computation of attn when activation checkpoint is enabled
        if gpc.is_forward and gpc.config.selective_checkpoint:
            fa_output_mapping[layer_idx] = (out, softmax_lse)

        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.context_group = context_group
        ctx.inter_window_group = inter_window_group
        ctx.intra_window_group = intra_window_group
        ctx.dkv_inter_window_group = dkv_inter_window_group
        ctx.dkv_intra_window_group = dkv_intra_window_group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613

        q, k, v, out, softmax_lse = ctx.saved_tensors

        dq, dk, dv = zigzag_double_ring_flash_attn_backward(
            ctx.context_group,
            ctx.inter_window_group,
            ctx.intra_window_group,
            ctx.dkv_inter_window_group,
            ctx.dkv_intra_window_group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_kvpacked_func_with_sliding_window(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    context_group=None,
    inter_window_group=None,
    intra_window_group=None,
    dkv_inter_window_group=None,
    dkv_intra_window_group=None,
    layer_idx=0,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        context_group,
        inter_window_group,
        intra_window_group,
        dkv_inter_window_group,
        dkv_intra_window_group,
        layer_idx,
    )


def zigzag_ring_flash_attn_qkvpacked_func_with_sliding_window(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    context_group=None,
    inter_window_group=None,
    intra_window_group=None,
    dkv_inter_window_group=None,
    dkv_intra_window_group=None,
    layer_idx=0,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        context_group,
        inter_window_group,
        intra_window_group,
        dkv_inter_window_group,
        dkv_intra_window_group,
        layer_idx,
    )


def zigzag_ring_flash_attn_qkvsplited_func_with_sliding_window(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    context_group=None,
    inter_window_group=None,
    intra_window_group=None,
    dkv_inter_window_group=None,
    dkv_intra_window_group=None,
    layer_idx=0,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        context_group,
        inter_window_group,
        intra_window_group,
        dkv_inter_window_group,
        dkv_intra_window_group,
        layer_idx,
    )
