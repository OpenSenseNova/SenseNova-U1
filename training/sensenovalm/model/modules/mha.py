#!/usr/bin/env python
# --------------------------------------------------------
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from Flash-Attention (Tri Dao, BSD-3-Clause).
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause portions retain their license.
# Upstream reference: https://github.com/Dao-AILab/flash-attention
# --------------------------------------------------------
# -*- encoding: utf-8 -*-
import inspect
import math
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.comm.isp import _SeqAllToAll
from sensenovalm.model.modules.embedding import new_rotary_embedding
from sensenovalm.model.modules.linear import new_linear
from sensenovalm.model.modules.norm import new_layer_norm
from sensenovalm.model.modules.utils import update_kv_cache
from sensenovalm.model.ops.attention import CrossAttention, SelfAttention
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.parallel import is_using_isp

logger = get_logger(__file__)

# NOTE:
try:
    from torch.nn.attention.flex_attention import flex_attention
    flex_attention = torch.compile(flex_attention)
except ImportError:
    print("To enable flexattention, please install torch>=2.5.0")


def safe_norm(norm, x, dtype):
    y = x.to(dtype)
    if y.numel() == 0:
        return y + 0.0 * norm.weight.sum()
    return norm(y)


def pad_sequence(tensor, pad_size):
    _, H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((_, H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=2)
# NOTE:


def _convert_cu_seqlens_for_qksplited(kwargs: Dict):
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    max_seqlen = kwargs.pop("max_seqlen", None)

    if cu_seqlens is not None:
        kwargs["cu_seqlens_q"] = cu_seqlens
        kwargs["cu_seqlens_k"] = cu_seqlens

    if max_seqlen is not None:
        kwargs["max_seqlen_q"] = max_seqlen
        kwargs["max_seqlen_k"] = max_seqlen

    return kwargs


def split_fused_wqkv_weight(wqkv, *args, **kwargs):  # pylint: disable=W0613
    q_dim = kwargs["q_dim"]
    kv_dim = kwargs["kv_dim"]
    split_size = [q_dim, kv_dim, kv_dim]
    assert (q_dim + 2 * kv_dim) % wqkv.size(0) == 0
    divisor = (q_dim + 2 * kv_dim) // wqkv.size(0)
    wq, wk, wv = torch.split(wqkv, [x // divisor for x in split_size], dim=0)
    return wq, wk, wv


def _qkv_pre_load_convert(module: "GQA", state_dict, prefix: str, *args, **kwargs) -> None:  # pylint: disable=W0613
    wq_name, wk_name, wv_name, fused_name = (
        f"{prefix}wq.weight",
        f"{prefix}wk.weight",
        f"{prefix}wv.weight",
        f"{prefix}wqkv.weight",
    )
    if module.enable_qkv_fusion and fused_name not in state_dict:
        wq, wk, wv = state_dict.pop(wq_name), state_dict.pop(wk_name), state_dict.pop(wv_name)
        state_dict[fused_name] = torch.cat([wq, wk, wv], dim=0)
    if not module.enable_qkv_fusion and (
        wq_name not in state_dict or wk_name not in state_dict or wv_name not in state_dict
    ):
        state_dict[wq_name], state_dict[wk_name], state_dict[wv_name] = split_fused_wqkv_weight(
            state_dict.pop(fused_name), *args, **kwargs
        )


def _qkv_save_convert(module: "GQA", state_dict, prefix: str, *args, **kwargs) -> Dict:  # pylint: disable=W0613
    wq_name, wk_name, wv_name, fused_name = (
        f"{prefix}wq.weight",
        f"{prefix}wk.weight",
        f"{prefix}wv.weight",
        f"{prefix}wqkv.weight",
    )
    if module.enable_qkv_fusion:
        state_dict[wq_name], state_dict[wk_name], state_dict[wv_name] = split_fused_wqkv_weight(
            state_dict.pop(fused_name), *args, **kwargs
        )
    return state_dict


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        bias (bool): Whether the bias is needed for linears. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        qk_interleaved (Optional[bool]): whether the odd and even columns of wq and wk is interleaved. True by default.
        enable_qkv_fusion (bool): whether wq, wk and wv lienar is fused. True by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
        out_bias: bool = True,
    ) -> None:
        raise NotImplementedError
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.kv_dim = self.head_dim * num_heads  # num_kv_heads equals to num_heads in MHA
        self.enable_qkv_fusion = enable_qkv_fusion

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=1.0,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        if self.enable_qkv_fusion:
            # bias=True is according to https://spaces.ac.cn/archives/9577
            self.wqkv = new_linear("wqkv", embed_dim, 3 * embed_dim, bias, **factory_kwargs)
        else:
            self.wq = new_linear("wq", embed_dim, embed_dim, bias, **factory_kwargs)
            self.wk = new_linear("wk", embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = new_linear("wv", embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        # output projection always have the bias (for now) (except for baichuan2 model)
        self.out_proj = new_linear("out_proj", embed_dim, embed_dim, bias=out_bias, **factory_kwargs)

    def register_checkpoint_compatibility_hooks(
        self, pre_load_hook: Optional[Callable] = None, pre_save_hook: Optional[Callable] = None
    ):
        # Here we explicitly expose the checkpoint compatibility interface of the module,
        # hoping that model developers will make good use of it when adapting.
        # Is this interface already meeting all reasonable requirements?
        self._register_load_state_dict_pre_hook(pre_load_hook, with_module=True)
        self._register_state_dict_hook(pre_save_hook)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)

            q = qkv[:, :, 0].squeeze(2)
            k = qkv[:, :, 1].squeeze(2)
            v = qkv[:, :, 2].squeeze(2)
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        # rotary embedding
        indexes = kwargs.pop("indexes", 0)
        max_seqlen = kwargs.get("max_seqlen", None)
        q = self.rotary_emb(q, offsets=indexes, cache_type="query", interleaved=self.interleaved, max_seqlen=max_seqlen)
        k = self.rotary_emb(k, offsets=indexes, cache_type="key", interleaved=self.interleaved, max_seqlen=max_seqlen)

        # self attention
        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)
        if gpc.config.data.use_packed_dataset is False or gpc.is_evaluating:
            kwargs.pop("max_seqlen_q", None)
            kwargs.pop("max_seqlen_k", None)
        context = self.inner_attn(q, k, v, **kwargs)

        # wo
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))

    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        batch_size = x.shape[0]

        # wqkv, output: q, kv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)

            q = qkv[:, :, 0].squeeze(2)
            kv = qkv[:, :, 1:]
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)
            kv = torch.stack([k, v], dim=2)

        # rotary embedding, output: q, kv
        # q shape: [bsz, nheads, head_dim]
        # kv shape: [bsz, seqlen, 2, nheads, head_dim]
        if self.use_dynamic_ntk_rope:
            # update kv cache fisrt when enable dynamic ntk rope.
            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            if sequence_len_offset != 0:
                if sequence_len_offset > self.max_position_embeddings:
                    logger.warning(
                        "Notice your prompt's length is longer than model's max_position_embeddings: "
                        f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                    )

                if self.rotary_emb_dim > 0:
                    q = self.rotary_emb(
                        q, offsets=sequence_len_offset, cache_type="query", interleaved=self.interleaved
                    )
                    k = kv[:, :, 0].squeeze(2)
                    self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, in_place=True
                    )  # in-place is important
            else:
                if self.rotary_emb_dim > 0:
                    q = self.rotary_emb(q, offsets=0, cache_type="query", interleaved=self.interleaved)
                    k = kv[:, :, 0].squeeze(2)
                    self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, in_place=True
                    )  # in-place is important
        else:
            assert self.rotary_emb_dim > 0, "You should use rotary_emb."

            k, v = kv[:, :, 0].squeeze(2), kv[:, :, 1].squeeze(2)

            if attention_mask is None:
                q = self.rotary_emb(q, offsets=sequence_len_offset, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=sequence_len_offset, cache_type="key", interleaved=self.interleaved)
            else:
                if sequence_len_offset == 0:
                    q = self.rotary_emb(
                        q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                    )
                    k = self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                    )
                else:
                    if sequence_len_offset > self.max_position_embeddings:
                        logger.warning(
                            "Notice your prompt's length is longer than model's max_position_embeddings: "
                            f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                        )

                    empties = attention_mask[..., -1].sum(dim=-1)
                    indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                    indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                    # TODO To fit flash_attn apis, we rearrange q&k to pack them here and
                    # calculate rope for this batch input. Waiting to be optimized
                    q = rearrange(q, "b s h d -> s b h d", d=self.head_dim)  # pack input
                    k = rearrange(k, "b s h d -> s b h d", d=self.head_dim)
                    q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                    k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)
                    q = rearrange(q, "s b h d -> b s h d", d=self.head_dim)  # unpack
                    k = rearrange(k, "s b h d -> b s h d", d=self.head_dim)

            kv = torch.stack([k, v], dim=2)
            # update kv cache after rotary embedding when disable dynamic ntk rope.
            kv = update_kv_cache(kv, inference_params, self.layer_idx)

        # self-attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(batch_size, -1)

                output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, batch_size, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(batch_size, -1, 1, 1), output)
            else:
                attn_mask = attention_mask[:, -1, :].view(batch_size, 1, 1, -1)

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))


class GQA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of attention heads for key and value.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        bias (bool): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                     output projection. False by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        qk_interleaved (Optional[bool]): whether the odd and even columns of wq and wk is interleaved. True by default.
        enable_qkv_fusion (bool): whether wq, wk and wv lienar is fused. True by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 2048,
        head_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
        rope_scaling_factor: float = 1.0,
        rope_scaling: Optional[dict] = None,
        wo_no_bias: bool = True,
    ) -> None:
        raise NotImplementedError
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if head_dim:
            self.head_dim = head_dim
            q_dim = head_dim * num_heads
        else:
            self.head_dim = self.embed_dim // num_heads
            q_dim = embed_dim
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.enable_qkv_fusion = enable_qkv_fusion

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=rope_scaling_factor,
                rope_scaling=rope_scaling,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        if enable_qkv_fusion:
            self.wqkv = new_linear("wqkv", embed_dim, q_dim + 2 * self.kv_dim, bias, **factory_kwargs)
            self._register_load_state_dict_pre_hook(
                partial(_qkv_pre_load_convert, q_dim=q_dim, kv_dim=self.kv_dim), with_module=True
            )
            self._register_state_dict_hook(partial(_qkv_save_convert, q_dim=q_dim, kv_dim=self.kv_dim))
        else:
            self.wq = new_linear("wq", embed_dim, q_dim, bias, **factory_kwargs)
            self.wk = new_linear("wk", embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = new_linear("wv", embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        )
        self.inner_cross_attn = CrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        )

        self.wo = new_linear("wo", q_dim, embed_dim, not wo_no_bias if bias else False, **factory_kwargs)

    def register_checkpoint_compatibility_hooks(
        self, pre_load_hook: Optional[Callable] = None, pre_save_hook: Optional[Callable] = None
    ):
        # Here we explicitly expose the checkpoint compatibility interface of the module,
        # hoping that model developers will make good use of it when adapting.
        # Is this interface already meeting all reasonable requirements?
        self._register_load_state_dict_pre_hook(pre_load_hook, with_module=True)
        self._register_state_dict_hook(pre_save_hook)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])
            q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)
            max_seqlen_q = kwargs.get("max_seqlen_q", None)
            max_seqlen_k = kwargs.get("max_seqlen_k", None)

            q = self.rotary_emb(
                q, offsets=indexes, max_seqlen=max_seqlen_q, cache_type="query", interleaved=self.interleaved
            )
            k = self.rotary_emb(
                k, offsets=indexes, max_seqlen=max_seqlen_k, cache_type="key", interleaved=self.interleaved
            )

        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        if gpc.config.data.use_packed_dataset is False or gpc.is_evaluating:
            kwargs.pop("max_seqlen_q", None)
            kwargs.pop("max_seqlen_k", None)

        # self attention
        context = self.inner_attn(q, kv, **kwargs)

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))

    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        window_size = inference_params.window_size

        batch_size = x.shape[0]

        # wqkv, output: q, k, v
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])
            q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        # rotary embedding, output: q, kv
        assert self.rotary_emb_dim > 0
        if attention_mask is None:
            raise NotImplementedError(
                "You should make sure you are aware that you are changing the method of generating."
                "According to your generation function instead of inference/seq_generator_module.py, "
                "You may implement here for normal running."
            )
        else:
            if inference_params.sequence_len_offset == 0:
                q = self.rotary_emb(
                    q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
                k = self.rotary_emb(
                    k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
            else:
                empties = attention_mask[..., -1].sum(dim=-1)
                indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                # TODO To fit flash_attn apis, we rearrange q&k to pack them here and
                # calculate rope for this batch input. Waiting to be optimized
                q = rearrange(q, "b s h d -> s b h d", d=self.head_dim)  # pack input
                k = rearrange(k, "b s h d -> s b h d", d=self.head_dim)
                q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)
                q = rearrange(q, "s b h d -> b s h d", d=self.head_dim)  # unpack
                k = rearrange(k, "s b h d -> b s h d", d=self.head_dim)

        kv = torch.stack([k, v], dim=2)

        if window_size is None or window_size > sequence_len_offset:
            kv = update_kv_cache(kv, inference_params, self.layer_idx)
        else:  # window_size <= sequence_len_offset
            assert kv.size(1) == 1, "update kv length more than 1"

            inference_params.key_value_memory_dict[self.layer_idx][
                :, inference_params.keep_first : inference_params.window_size - 1, ...
            ] = inference_params.key_value_memory_dict[self.layer_idx][
                :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
            ].clone()
            inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
            inference_params.sequence_len_offset = inference_params.window_size - 1

            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            inference_params.sequence_len_offset = inference_params.real_sequence_len_offset

        # When using FP16, there is a high probability of NAN in the KV.
        # Since NAN cannot be removed by multiplying with and 0, it needs
        # to be removed manually here.
        kv = torch.where(torch.isnan(kv), 0, kv)

        # attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(batch_size, -1)

                output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, batch_size, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(batch_size, -1, 1, 1), output)

            else:
                attn_mask = attention_mask[:, -1, :].view(batch_size, 1, 1, -1)
                if window_size is not None and window_size <= sequence_len_offset:
                    attn_mask = torch.concat(
                        [
                            attn_mask[..., : inference_params.keep_first],
                            attn_mask[..., -(window_size - inference_params.keep_first) :],
                        ],
                        dim=-1,
                    )

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                expansion = q.size(2) // k.size(2)
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))


try:
    from flash_attn import flash_attn_func

    # flash_attn >= v2.3.0
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except (ModuleNotFoundError, ImportError):
    _flash_supports_window_size = False


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SWA(nn.Module):
    """
    sliding window attention

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        sequence_process_group (torch.distributed.ProcessGroup): The process group for attention calculation.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        qkv_bias: bool = True,
        o_bias: bool = False,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_type: str = "normal",
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_sliding_window: bool = False,
        sliding_window: int = None,
        use_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        layer_norm_epsilon: float = 1e-06,
        tp_mode: str = "mtp",
        qk_interleaved: Optional[bool] = True,
        use_logn_attn: bool = False,  # Qwen1
    ) -> None:
        assert embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"
        assert (not use_sliding_window) or (
            sliding_window is not None
        ), "Must set `sliding windows` size when `use_sliding_window` is True."
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if head_dim is None:
            head_dim = self.embed_dim // num_heads
        self.head_dim = head_dim
        self.q_dim = self.head_dim * self.num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.causal = causal
        self.layer_idx = layer_idx
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.tp_mode = tp_mode
        self.rope_type = rope_type
        self.use_logn_attn = use_logn_attn
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb_t = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=rope_scaling_factor,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )
            self.rotary_emb_hw = new_rotary_embedding(
                self.rotary_emb_dim // 2,
                base=10000,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=rope_scaling_factor,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        # notice here should change bias=True
        self.wq = new_linear(
            "wq",
            embed_dim,
            self.q_dim,  
            qkv_bias,
            **factory_kwargs,
        )
        self.wq_hw = new_linear(
            "wq",
            embed_dim,
            self.q_dim,  
            qkv_bias,
            **factory_kwargs,
        )

        self.wk = new_linear(
            "wk",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )
        self.wk_hw = new_linear(
            "wk",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.wv = new_linear(
            "wv",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = new_layer_norm(
                norm_type, self.head_dim, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm = new_layer_norm(
                norm_type, self.head_dim, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

            self.q_norm_hw = new_layer_norm(
                norm_type, self.head_dim, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm_hw = new_layer_norm(
                norm_type, self.head_dim, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

        # NOTE:
        # self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        # self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        self.inner_cross_attn_causal = causal
        self.inner_cross_attn_softmax_scale = softmax_scale
        self.inner_cross_attn_dropout = dropout

        self.wo = new_linear(
            "wo",
            self.q_dim,
            embed_dim,
            o_bias,
            **factory_kwargs,
        )

    def forward(self, x, position_embeddings, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, position_embeddings=position_embeddings, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, position_embeddings: Tuple[torch.Tensor, torch.Tensor], **kwargs):

        q_t = rearrange(self.wq(x), "b t (h d) -> b t h d", d=self.head_dim)
        q_hw = rearrange(self.wq_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_t = rearrange(self.wk(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_hw = rearrange(self.wk_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(self.wv(x), "b t (h d) -> b t h d", d=self.head_dim)

        kv_seq_len = v.size(1)
        use_window_circumstance = (
            _flash_supports_window_size
            and self.use_sliding_window
            and self.sliding_window
            and kv_seq_len > self.sliding_window
        )

        # NOTE:
        padlen = kwargs.pop("padlen", 0)
        flex_mask = kwargs.pop("flex_mask", None)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        if self.use_qk_norm:
            def apply_norm(q, k, q_norm_fn, k_norm_fn):
                return q_norm_fn(q), k_norm_fn(k)

            q_t, k_t = apply_norm(q_t, k_t, self.q_norm, self.k_norm)
            q_hw, k_hw = apply_norm(q_hw, k_hw, self.q_norm_hw, self.k_norm_hw)
        
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)
            max_seqlen_q = kwargs.get("max_seqlen_q", None)
            max_seqlen_k = kwargs.get("max_seqlen_k", None)

            def apply_rotary_func(emb_fn, q, k, offsets, interleaved):
                max_seqlen = offsets.max() + 1
                q = emb_fn(q, offsets=offsets, max_seqlen=max_seqlen, cache_type="query", interleaved=interleaved)
                k = emb_fn(k, offsets=offsets, max_seqlen=max_seqlen, cache_type="key", interleaved=interleaved)
                return q, k

            q_t, k_t = apply_rotary_func(self.rotary_emb_t, q_t, k_t, indexes[:, 0], self.interleaved)
            q_h, k_h = apply_rotary_func(self.rotary_emb_hw, q_h, k_h, indexes[:, 1], self.interleaved)
            q_w, k_w = apply_rotary_func(self.rotary_emb_hw, q_w, k_w, indexes[:, 2], self.interleaved)
        else:
            kwargs.pop("indexes", 0)
        
        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)

        if use_window_circumstance:
            kwargs["window_size"] = (self.sliding_window, 0)

        if gpc.config.data.use_packed_dataset is False or gpc.is_evaluating:
            kwargs.pop("max_seqlen_q", None)
            kwargs.pop("max_seqlen_k", None)

        # self attention
        # kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
        # context = self.inner_attn(q, kv, **kwargs)
            
        # NOTE:
        q = q.permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ISP all-to-all: scatter heads, gather sequence before flex_attention
        _is_isp = is_using_isp()
        _isp_tp_size = gpc.get_world_size(ParallelMode.TENSOR) if _is_isp else 1
        if _is_isp and _isp_tp_size > 1:
            _spg = gpc.get_group(ParallelMode.TENSOR)
            q = _SeqAllToAll.apply(_spg, 1, 2, q)   # [B, H/tp, S_full, D]
            k = _SeqAllToAll.apply(_spg, 1, 2, k)
            v = _SeqAllToAll.apply(_spg, 1, 2, v)

        if padlen > 0: 
            q = pad_sequence(q, padlen)
            k = pad_sequence(k, padlen)
            v = pad_sequence(v, padlen)

        context = flex_attention(
            q, k, v, 
            enable_gqa=True,
            block_mask=flex_mask,
            scale = 1.0 / math.sqrt(q.size(-1) // 2)
        )

        if padlen > 0:
            end_index = context.shape[2] - padlen
            context = context[:, :, :end_index, :]

        # ISP reverse all-to-all: scatter sequence, gather heads after flex_attention
        if _is_isp and _isp_tp_size > 1:
            context = _SeqAllToAll.apply(_spg, 2, 1, context)  # [B, H, S_local, D]

        context = context.permute(0, 2, 1, 3)

        return self.wo(rearrange(context, "b s h d -> b s (h d)"))
    
    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params=None, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        window_size = inference_params.window_size

        bsz = x.shape[0]

        q_t = rearrange(self.wq(x), "b t (h d) -> b t h d", d=self.head_dim)
        q_hw = rearrange(self.wq_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_t = rearrange(self.wk(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_hw = rearrange(self.wk_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(self.wv(x), "b t (h d) -> b t h d", d=self.head_dim)

        kv_seq_len = v.size(1)
        use_window_circumstance = (
            _flash_supports_window_size
            and self.use_sliding_window
            and self.sliding_window
            and kv_seq_len > self.sliding_window
        )

        # NOTE:
        padlen = kwargs.pop("padlen", 0)
        flex_mask = kwargs.pop("flex_mask", None)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        if self.use_qk_norm:
            def apply_norm(q, k, q_norm_fn, k_norm_fn):
                return q_norm_fn(q), k_norm_fn(k)

            q_t, k_t = apply_norm(q_t, k_t, self.q_norm, self.k_norm)
            q_hw, k_hw = apply_norm(q_hw, k_hw, self.q_norm_hw, self.k_norm_hw)
        
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        assert self.rotary_emb_dim > 0
        if attention_mask is None:
            raise NotImplementedError(
                "You should make sure you are aware that you are changing the method of generating."
                "According to your generation function instead of inference/seq_generator_module.py, "
                "You may implement here for normal running."
            )
        else:
            if inference_params.sequence_len_offset == 0:

                indexes = kwargs.pop("indexes", 0)
                max_seqlen_q = kwargs.get("max_seqlen_q", None)
                max_seqlen_k = kwargs.get("max_seqlen_k", None)

                def apply_rotary_func(emb_fn, q, k, offsets, interleaved):
                    max_seqlen = offsets.max() + 1
                    q = emb_fn(q, offsets=offsets, max_seqlen=max_seqlen, cache_type="query", interleaved=interleaved)
                    k = emb_fn(k, offsets=offsets, max_seqlen=max_seqlen, cache_type="key", interleaved=interleaved)
                    return q, k

                q_t, k_t = apply_rotary_func(self.rotary_emb_t, q_t, k_t, indexes[:, 0], self.interleaved)
                q_h, k_h = apply_rotary_func(self.rotary_emb_hw, q_h, k_h, indexes[:, 1], self.interleaved)
                q_w, k_w = apply_rotary_func(self.rotary_emb_hw, q_w, k_w, indexes[:, 2], self.interleaved)
        

                q = self.rotary_emb(
                    q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
                k = self.rotary_emb(
                    k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
            else:
                empties = attention_mask[..., -1].sum(dim=-1)
                indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)

        kv = torch.stack([k, v], dim=2)

        if window_size is None or window_size > sequence_len_offset:
            kv = update_kv_cache(kv, inference_params, self.layer_idx)
        else:  # window_size <= sequence_len_offset
            assert kv.size(1) == 1, "update kv length more than 1"

            inference_params.key_value_memory_dict[self.layer_idx][
                :, inference_params.keep_first : inference_params.window_size - 1, ...
            ] = inference_params.key_value_memory_dict[self.layer_idx][
                :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
            ].clone()
            inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
            inference_params.sequence_len_offset = inference_params.window_size - 1

            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            inference_params.sequence_len_offset = inference_params.real_sequence_len_offset

        # When using FP16, there is a high probability of NAN in the KV.
        # Since NAN cannot be removed by multiplying with and 0, it needs
        # to be removed manually here.
        kv = torch.where(torch.isnan(kv), 0, kv)

        # attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)

                if use_window_circumstance:
                    output = self.inner_attn(
                        *self._convert_unpacked_qkv_to_packed(q, kv, bsz, attn_mask4flsh),
                        window_size=(self.sliding_window, 0),
                    )
                else:
                    output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, bsz, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

            else:
                attn_mask = attention_mask[:, -1, :].view(bsz, 1, 1, -1)
                if window_size is not None and window_size <= sequence_len_offset:
                    attn_mask = torch.concat(
                        [
                            attn_mask[..., : inference_params.keep_first],
                            attn_mask[..., -(window_size - inference_params.keep_first) :],
                        ],
                        dim=-1,
                    )

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                expansion = q.size(2) // k.size(2)
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))


class SWA_MoT(nn.Module):
    """
    sliding window attention for MOT model

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        sequence_process_group (torch.distributed.ProcessGroup): The process group for attention calculation.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        tp_mode (str): The string value of tensor parallel mode, should be in ["mtp", "msp", "fsp", "isp"],
                       "mtp" by default.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        qkv_bias: bool = True,
        o_bias: bool = False,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_type: str = "normal",
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_sliding_window: bool = False,
        sliding_window: int = None,
        use_qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        layer_norm_epsilon: float = 1e-06,
        tp_mode: str = "mtp",
        qk_interleaved: Optional[bool] = True,
        use_logn_attn: bool = False,  # Qwen1
    ) -> None:
        assert embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"
        assert (not use_sliding_window) or (
            sliding_window is not None
        ), "Must set `sliding windows` size when `use_sliding_window` is True."
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if head_dim is None:
            head_dim = self.embed_dim // num_heads
        self.head_dim = head_dim
        self.q_dim = self.head_dim * self.num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.causal = causal
        self.layer_idx = layer_idx
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.tp_mode = tp_mode
        self.rope_type = rope_type
        self.use_logn_attn = use_logn_attn
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb_t = new_rotary_embedding(
                self.rotary_emb_dim // 2,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=rope_scaling_factor,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )
            self.rotary_emb_hw = new_rotary_embedding(
                self.rotary_emb_dim // 4,
                base=10000,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=rope_scaling_factor,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        # notice here should change bias=True
        self.wq = new_linear(
            "wq",
            embed_dim,
            self.q_dim,  
            qkv_bias,
            **factory_kwargs,
        )

        self.wk = new_linear(
            "wk",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.wv = new_linear(
            "wv",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.wq_mot_gen = new_linear(
            "wq",
            embed_dim,
            self.q_dim,  
            qkv_bias,
            **factory_kwargs,
        )

        self.wk_mot_gen = new_linear(
            "wk",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.wv_mot_gen = new_linear(
            "wv",
            embed_dim,
            self.kv_dim,
            qkv_bias,
            **factory_kwargs,
        )

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

            self.q_norm_hw = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm_hw = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

            self.q_norm_mot_gen = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm_mot_gen = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

            self.q_norm_hw_mot_gen = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # unlike olmo, only on the head dim!
            self.k_norm_hw_mot_gen = new_layer_norm(
                norm_type, self.head_dim // 2, eps=layer_norm_epsilon
            )  # thus post q_norm does not need reshape

        # NOTE:
        # self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        # self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        self.inner_cross_attn_causal = causal
        self.inner_cross_attn_softmax_scale = softmax_scale
        self.inner_cross_attn_dropout = dropout

        self.wo = new_linear(
            "wo",
            self.q_dim,
            embed_dim,
            o_bias,
            **factory_kwargs,
        )

        self.wo_mot_gen = new_linear(
            "wo",
            self.q_dim,
            embed_dim,
            o_bias,
            **factory_kwargs,
        )

    def forward(self, x, position_embeddings, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, position_embeddings=position_embeddings, image_gen_indicators=image_gen_indicators, exist_non_image_gen_tokens=exist_non_image_gen_tokens, exist_image_gen_tokens=exist_image_gen_tokens, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, position_embeddings: Tuple[torch.Tensor, torch.Tensor], image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, **kwargs):        
        num_image_gen_tokens = image_gen_indicators.sum().item()

        q = torch.cat([self.wq_mot_gen(x[:, :num_image_gen_tokens]), self.wq(x[:, num_image_gen_tokens:])], 1)
        k = torch.cat([self.wk_mot_gen(x[:, :num_image_gen_tokens]), self.wk(x[:, num_image_gen_tokens:])], 1)
        v = torch.cat([self.wv_mot_gen(x[:, :num_image_gen_tokens]), self.wv(x[:, num_image_gen_tokens:])], 1)

        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_dim)

        kv_seq_len = v.size(0)
        use_window_circumstance = (
            _flash_supports_window_size
            and self.use_sliding_window
            and self.sliding_window
            and kv_seq_len > self.sliding_window
        )

        # NOTE:
        padlen = kwargs.pop("padlen", 0)
        flex_mask = kwargs.pop("flex_mask", 0)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)
        if self.use_qk_norm:

            def apply_norm(q, k, q_norm, q_norm_mot_gen, k_norm, k_norm_mot_gen):
                q_ = q.new_zeros(q.shape)
                k_ = k.new_zeros(k.shape)

                q_ = torch.cat([safe_norm(q_norm_mot_gen, q[:, :num_image_gen_tokens], q_norm_mot_gen.weight.dtype), safe_norm(q_norm, q[:, num_image_gen_tokens:], q_norm.weight.dtype)], 1)
                k_ = torch.cat([safe_norm(k_norm_mot_gen, k[:, :num_image_gen_tokens], k_norm_mot_gen.weight.dtype), safe_norm(k_norm, k[:, num_image_gen_tokens:], k_norm.weight.dtype)], 1)

                return q_, k_

            q_t, k_t = apply_norm(q_t, k_t, self.q_norm, self.q_norm_mot_gen, self.k_norm, self.k_norm_mot_gen)
            q_hw, k_hw = apply_norm(q_hw, k_hw, self.q_norm_hw, self.q_norm_hw_mot_gen, self.k_norm_hw, self.k_norm_hw_mot_gen)
        
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)
            max_seqlen_q = kwargs.get("max_seqlen_q", None)
            max_seqlen_k = kwargs.get("max_seqlen_k", None)

            def apply_rotary_func(emb_fn, q, k, offsets, interleaved):
                max_seqlen = offsets.max() + 1
                q = emb_fn(q, offsets=offsets, max_seqlen=max_seqlen, cache_type="query", interleaved=interleaved)
                k = emb_fn(k, offsets=offsets, max_seqlen=max_seqlen, cache_type="key", interleaved=interleaved)
                return q, k

            q_t, k_t = apply_rotary_func(self.rotary_emb_t, q_t, k_t, indexes[:, 0], self.interleaved)
            q_h, k_h = apply_rotary_func(self.rotary_emb_hw, q_h, k_h, indexes[:, 1], self.interleaved)
            q_w, k_w = apply_rotary_func(self.rotary_emb_hw, q_w, k_w, indexes[:, 2], self.interleaved)
        else:
            kwargs.pop("indexes", 0)
        
        ## original

        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)


        if use_window_circumstance:
            kwargs["window_size"] = (self.sliding_window, 0)

        if gpc.config.data.use_packed_dataset is False or gpc.is_evaluating:
            kwargs.pop("max_seqlen_q", None)
            kwargs.pop("max_seqlen_k", None)

        # self attention
        # kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
        # context = self.inner_attn(q, kv, **kwargs)
            
        # NOTE:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ISP all-to-all: scatter heads, gather sequence before flex_attention
        _is_isp = is_using_isp()
        _isp_tp_size = gpc.get_world_size(ParallelMode.TENSOR) if _is_isp else 1
        if _is_isp and _isp_tp_size > 1:
            _spg = gpc.get_group(ParallelMode.TENSOR)
            q = _SeqAllToAll.apply(_spg, 1, 2, q)   # [B, H/tp, S_full, D]
            k = _SeqAllToAll.apply(_spg, 1, 2, k)
            v = _SeqAllToAll.apply(_spg, 1, 2, v)

        if padlen > 0:
            q = pad_sequence(q, padlen)
            k = pad_sequence(k, padlen)
            v = pad_sequence(v, padlen)

        context = flex_attention(
            q, k, v,
            enable_gqa=True,
            block_mask=flex_mask,
            scale = 1.0 / math.sqrt(q.size(-1))
        )

        if padlen > 0:
            end_index = context.shape[2] - padlen
            context = context[:, :, :end_index, :]

        # ISP reverse all-to-all: scatter sequence, gather heads after flex_attention
        if _is_isp and _isp_tp_size > 1:
            context = _SeqAllToAll.apply(_spg, 2, 1, context)  # [B, H, S_local, D]

        context = context.permute(0, 2, 1, 3)
        context = rearrange(context, "b s h d -> b s (h d)")

        output = torch.cat([self.wo_mot_gen(context[:, :num_image_gen_tokens]), self.wo(context[:, num_image_gen_tokens:])], 1)

        return output


    #     q_hw = x.new_zeros((x.shape[0], x.shape[1], self.num_heads * self.head_dim))
    #     if exist_non_image_gen_tokens:
    #         q_hw[~image_gen_indicators] = self.wq_hw(x[~image_gen_indicators])
    #     if exist_image_gen_tokens:
    #         q_hw[image_gen_indicators] = self.wq_hw_mot_gen(x[image_gen_indicators])

    #     k_t = x.new_zeros((x.shape[0], x.shape[1], self.num_kv_heads * self.head_dim))
    #     if exist_non_image_gen_tokens:
    #         k_t[~image_gen_indicators] = self.wk(x[~image_gen_indicators])
    #     if exist_image_gen_tokens:
    #         k_t[image_gen_indicators] = self.wk_mot_gen(x[image_gen_indicators])

    #     k_hw = x.new_zeros((x.shape[0], x.shape[1], self.num_kv_heads * self.head_dim))
    #     if exist_non_image_gen_tokens:
    #         k_hw[~image_gen_indicators] = self.wk_hw(x[~image_gen_indicators])
    #     if exist_image_gen_tokens:
    #         k_hw[image_gen_indicators] = self.wk_hw_mot_gen(x[image_gen_indicators])

    #     v = x.new_zeros((x.shape[0], x.shape[1], self.num_kv_heads * self.head_dim))
    #     if exist_non_image_gen_tokens:
    #         v[~image_gen_indicators] = self.wv(x[~image_gen_indicators])
    #     if exist_image_gen_tokens:
    #         v[image_gen_indicators] = self.wv_mot_gen(x[image_gen_indicators])

    #     q_t = rearrange(q_t, "b t (h d) -> b t h d", d=self.head_dim)
    #     q_hw = rearrange(q_hw, "b t (h d) -> b t h d", d=self.head_dim)
    #     k_t = rearrange(k_t, "b t (h d) -> b t h d", d=self.head_dim)
    #     k_hw = rearrange(k_hw, "b t (h d) -> b t h d", d=self.head_dim)
    #     v = rearrange(v, "b t (h d) -> b t h d", d=self.head_dim)

    #     kv_seq_len = v.size(0)
    #     use_window_circumstance = (
    #         _flash_supports_window_size
    #         and self.use_sliding_window
    #         and self.sliding_window
    #         and kv_seq_len > self.sliding_window
    #     )

    #     # NOTE:
    #     padlen = kwargs.pop("padlen", 0)
    #     flex_mask = kwargs.pop("flex_mask", 0)


    #     q_h, q_w = q_hw.chunk(2, dim=-1)
    #     k_h, k_w = k_hw.chunk(2, dim=-1)

    #     if self.use_qk_norm:


    #         q_t, k_t = apply_norm(q_t, k_t, self.q_norm, self.q_norm_mot_gen, self.k_norm, self.k_norm_mot_gen)
    #         q_h, k_h = apply_norm(q_h, k_h, self.q_norm_h, self.q_norm_h_mot_gen, self.k_norm_h, self.k_norm_h_mot_gen)
    #         q_w, k_w = apply_norm(q_w, k_w, self.q_norm_w, self.q_norm_w_mot_gen, self.k_norm_w, self.k_norm_w_mot_gen)

    #     # rotary embedding
    #     if self.rotary_emb_dim > 0:
    #         indexes = kwargs.pop("indexes", 0)
    #         max_seqlen_q = kwargs.get("max_seqlen_q", None)
    #         max_seqlen_k = kwargs.get("max_seqlen_k", None)


    #         q_t, k_t = apply_rotary_func(self.rotary_emb_t, q_t, k_t, indexes[:, 0], self.interleaved)
    #         q_h, k_h = apply_rotary_func(self.rotary_emb_hw, q_h, k_h, indexes[:, 1], self.interleaved)
    #         q_w, k_w = apply_rotary_func(self.rotary_emb_hw, q_w, k_w, indexes[:, 2], self.interleaved)
    #     else:
    #         kwargs.pop("indexes", 0)
        
    #     q = torch.cat([q_t, q_h, q_w], dim=-1)
    #     k = torch.cat([k_t, k_h, k_w], dim=-1)



    #     # self attention
    #     # kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    #     # context = self.inner_attn(q, kv, **kwargs)
            
    #     # NOTE:
    #     q = q.permute(0, 2, 1, 3)
    #     k = k.permute(0, 2, 1, 3)
    #     v = v.permute(0, 2, 1, 3)



        
    #     context = context.permute(0, 2, 1, 3)
    #     context = rearrange(context, "b s h d -> b s (h d)")

    #     output = context.new_zeros((context.shape[0], context.shape[1], self.embed_dim))
    #     if exist_non_image_gen_tokens:
    #         output[~image_gen_indicators] = self.wo(context[~image_gen_indicators])
    #     if exist_image_gen_tokens:
    #         output[image_gen_indicators] = self.wo_mot_gen(context[image_gen_indicators])

    
    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params=None, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        window_size = inference_params.window_size

        bsz = x.shape[0]

        q_t = rearrange(self.wq(x), "b t (h d) -> b t h d", d=self.head_dim)
        q_hw = rearrange(self.wq_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_t = rearrange(self.wk(x), "b t (h d) -> b t h d", d=self.head_dim)
        k_hw = rearrange(self.wk_hw(x), "b t (h d) -> b t h d", d=self.head_dim)
        v = rearrange(self.wv(x), "b t (h d) -> b t h d", d=self.head_dim)

        kv_seq_len = v.size(0)
        use_window_circumstance = (
            _flash_supports_window_size
            and self.use_sliding_window
            and self.sliding_window
            and kv_seq_len > self.sliding_window
        )

        # NOTE:
        padlen = kwargs.pop("padlen", 0)
        flex_mask = kwargs.pop("flex_mask", 0)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        if self.use_qk_norm:

            def apply_norm(q, k, q_norm, q_norm_mot_gen, k_norm, k_norm_mot_gen):
                q_ = q.new_zeros(q.shape)
                k_ = k.new_zeros(k.shape)

                q_ = torch.cat([safe_norm(q_norm_mot_gen, q[:, :num_image_gen_tokens], q_norm_mot_gen.weight.dtype), safe_norm(q_norm, q[:, num_image_gen_tokens:], q_norm.weight.dtype)], 1)
                k_ = torch.cat([safe_norm(k_norm_mot_gen, k[:, :num_image_gen_tokens], k_norm_mot_gen.weight.dtype), safe_norm(k_norm, k[:, num_image_gen_tokens:], k_norm.weight.dtype)], 1)

                return q_, k_

            q_t, k_t = apply_norm(q_t, k_t, self.q_norm, self.q_norm_mot_gen, self.k_norm, self.k_norm_mot_gen)
            q_hw, k_hw = apply_norm(q_hw, k_hw, self.q_norm_hw, self.q_norm_hw_mot_gen, self.k_norm_hw, self.k_norm_hw_mot_gen)
        
        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)
        
        assert self.rotary_emb_dim > 0
        if attention_mask is None:
            raise NotImplementedError(
                "You should make sure you are aware that you are changing the method of generating."
                "According to your generation function instead of inference/seq_generator_module.py, "
                "You may implement here for normal running."
            )
        else:
            if inference_params.sequence_len_offset == 0:

                indexes = kwargs.pop("indexes", 0)
                max_seqlen_q = kwargs.get("max_seqlen_q", None)
                max_seqlen_k = kwargs.get("max_seqlen_k", None)

                def apply_rotary_func(emb_fn, q, k, offsets, interleaved):
                    max_seqlen = offsets.max() + 1
                    q = emb_fn(q, offsets=offsets, max_seqlen=max_seqlen, cache_type="query", interleaved=interleaved)
                    k = emb_fn(k, offsets=offsets, max_seqlen=max_seqlen, cache_type="key", interleaved=interleaved)
                    return q, k

                q_t, k_t = apply_rotary_func(self.rotary_emb_t, q_t, k_t, indexes[:, 0], self.interleaved)
                q_h, k_h = apply_rotary_func(self.rotary_emb_hw, q_h, k_h, indexes[:, 1], self.interleaved)
                q_w, k_w = apply_rotary_func(self.rotary_emb_hw, q_w, k_w, indexes[:, 2], self.interleaved)
        

                q = self.rotary_emb(
                    q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
                k = self.rotary_emb(
                    k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
            else:
                empties = attention_mask[..., -1].sum(dim=-1)
                indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)

        kv = torch.stack([k, v], dim=2)

        if window_size is None or window_size > sequence_len_offset:
            kv = update_kv_cache(kv, inference_params, self.layer_idx)
        else:  # window_size <= sequence_len_offset
            assert kv.size(1) == 1, "update kv length more than 1"

            inference_params.key_value_memory_dict[self.layer_idx][
                :, inference_params.keep_first : inference_params.window_size - 1, ...
            ] = inference_params.key_value_memory_dict[self.layer_idx][
                :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
            ].clone()
            inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
            inference_params.sequence_len_offset = inference_params.window_size - 1

            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            inference_params.sequence_len_offset = inference_params.real_sequence_len_offset

        # When using FP16, there is a high probability of NAN in the KV.
        # Since NAN cannot be removed by multiplying with and 0, it needs
        # to be removed manually here.
        kv = torch.where(torch.isnan(kv), 0, kv)

        # attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)

                if use_window_circumstance:
                    output = self.inner_attn(
                        *self._convert_unpacked_qkv_to_packed(q, kv, bsz, attn_mask4flsh),
                        window_size=(self.sliding_window, 0),
                    )
                else:
                    output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, bsz, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

            else:
                attn_mask = attention_mask[:, -1, :].view(bsz, 1, 1, -1)
                if window_size is not None and window_size <= sequence_len_offset:
                    attn_mask = torch.concat(
                        [
                            attn_mask[..., : inference_params.keep_first],
                            attn_mask[..., -(window_size - inference_params.keep_first) :],
                        ],
                        dim=-1,
                    )

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                expansion = q.size(2) // k.size(2)
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))
