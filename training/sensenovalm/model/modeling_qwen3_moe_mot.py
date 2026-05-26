# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context.parallel_context import global_context as gpc
from sensenovalm.initialize.initialize_tensor import (
    normal_,
    scaled_init_method_normal,
    scaled_init_method_uniform,
    uniform_,
)
from sensenovalm.model.base_model import BaseModel
from sensenovalm.model.modules.embedding import Embedding1D
from sensenovalm.model.modules.linear import new_linear
from sensenovalm.model.modules.mha import SWA_MoT
from sensenovalm.model.modules.mlp import new_feed_forward
from sensenovalm.model.modules.norm import new_layer_norm
from sensenovalm.model.moe import SenseNovaVLMoE
from sensenovalm.model.utils import (
    convert_attn_args_to_kwargs,
    convert_attn_kwargs_to_args,
)
from sensenovalm.solver.activation_checkpoint import activation_checkpoint
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.parallel import is_using_isp
from sensenovalm.model.mtp import MTP
from sensenovalm.model.moe.utils import SenseNovaVLMoEOutput


sensenovalm_accelerator = get_accelerator()
logger = get_logger(__file__)

def safe_norm(norm, x, dtype):
    y = x.to(dtype)
    if y.numel() == 0:
        return y + 0.0 * norm.weight.sum()
    return norm(y)


class Qwen3MoeMoTDecoder(nn.Module):
    """
    1D Packed Flash Qwen Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        multiple_of (int): The value to make SwiGLU hidden layer size multiple of large power of 2.
    """

    def __init__(
        self,
        hidden_size,
        head_dim: int = None,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        qkv_bias=True,
        o_bias=False,
        mlp_bias=False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_type: str = "normal",
        rope_base: int = 1000000.0,
        rope_scaling_factor: float = 1.0,
        use_sliding_window: bool = False,
        sliding_window: int = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        scale_attn_weights: bool = False,  # Qwen1
        use_logn_attn: bool = False,  # Qwen1
        moe_kwargs: dict = None,
        moe_layer_kwargs: dict = None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        if scale_attn_weights:
            softmax_scale = None
        else:
            softmax_scale = 1 / math.sqrt(head_dim)
        
        self.attention = SWA_MoT(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_attention_heads,
            head_dim=head_dim,
            dropout=attn_drop_rate,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=softmax_scale,
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            use_qk_norm=True,
            rope_type=rope_type,
            rope_base=rope_base,
            rope_scaling_factor=rope_scaling_factor,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            use_logn_attn=use_logn_attn,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.attention_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.ffn_norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.attention_norm_mot_gen = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.ffn_norm_mot_gen = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        if (
            not moe_kwargs
            or moe_kwargs.get("num_experts", 1) <= 1
            or layer_idx < moe_kwargs.get("first_k_dense_replace", 0)
            or layer_idx % moe_kwargs.get("moe_layer_freq", 1) != 0
        ):  # dense, not MoE
            self.use_moe = False
            self.feed_forward = new_feed_forward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                bias=mlp_bias,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type="swiglu" if use_swiglu else "gelu",
            )
            self.feed_forward_mot_gen = new_feed_forward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                bias=mlp_bias,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type="swiglu" if use_swiglu else "gelu",
            )
        else:
            self.use_moe = True
            self.moe_kwargs = moe_kwargs
            # replace mlp by MoE module. The expert in MoE is a FeedForward module.
            # mlp_cls = get_mlp_cls(self.tp_mode)
            self.feed_forward = SenseNovaVLMoE(
                hidden_size,
                moe_kwargs.moe_intermediate_size,
                out_features=hidden_size,
                moe_type=moe_kwargs.moe_type,
                num_experts=moe_kwargs.num_experts,
                top_k=moe_kwargs.top_k,
                num_shared_experts=moe_kwargs.num_shared_experts,
                moe_layer_kwargs=moe_layer_kwargs,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
            )
            self.feed_forward_mot_gen = SenseNovaVLMoE(
                hidden_size,
                moe_kwargs.moe_intermediate_size,
                out_features=hidden_size,
                moe_type=moe_kwargs.moe_type,
                num_experts=moe_kwargs.gen_num_experts,
                top_k=moe_kwargs.top_k,
                num_shared_experts=moe_kwargs.num_shared_experts,
                moe_layer_kwargs=moe_layer_kwargs,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
            )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        if init_type == "normal":
            self.init_func = normal_
            self.scaled_init_func = scaled_init_method_normal
        else:
            self.init_func = uniform_
            self.scaled_init_func = scaled_init_method_uniform

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wq" in name or "wk" in name or "wv" in name:
                    self.init_func(std=self.attn_wqkv_init_std)(param.data)
                elif self.use_scaled_init:  # wo
                    self.scaled_init_func(sigma=self.attn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                else:
                    self.init_func(std=self.attn_other_init_std)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

            for name, param in self.feed_forward_mot_gen.named_parameters():
                if self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

    def forward(self, hidden_states, residual=None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, image_gen_indicators=None, exist_non_image_gen_tokens=None, exist_image_gen_tokens=None, **kwargs):
        if self.checkpoint and self.training:
            args = convert_attn_kwargs_to_args(kwargs)
            return activation_checkpoint(self._forward, False, hidden_states, residual, position_embeddings, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, *args)
        else:
            return self._forward(hidden_states, residual, position_embeddings, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, **kwargs)

    def _forward(self, hidden_states, residual, position_embeddings, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, *args, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        final_hidden_states = None
        num_image_gen_tokens = image_gen_indicators.sum().item()
        if self.prenorm:

            def _dropout_and_norm_attn(_residual, _hidden_states):
                _dropped = self.dropout1(_hidden_states)
                _residual = (_dropped + _residual) if _residual is not None else _dropped     

                _hidden_states = torch.cat([
                    safe_norm(self.attention_norm_mot_gen, _residual[:, :num_image_gen_tokens], self.attention_norm_mot_gen.weight.dtype),
                    safe_norm(self.attention_norm,         _residual[:, num_image_gen_tokens:], self.attention_norm.weight.dtype),
                ], dim=1)

                return _residual, _hidden_states

            if self.dropout_selective_checkpoint:
                residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, residual, hidden_states)
            else:
                residual, hidden_states = _dropout_and_norm_attn(residual, hidden_states)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            mixer_kwargs = convert_attn_args_to_kwargs(args, kwargs)
            hidden_states = self.attention(hidden_states, position_embeddings, image_gen_indicators, exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens, **mixer_kwargs)

            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:
                    
                    def _dropout_and_norm_ffn(_residual, _hidden_states): 
                        _dropped = self.dropout2(_hidden_states)
                        _residual = (_dropped + _residual) if _residual is not None else _dropped

                        _hidden_states = torch.cat([
                            safe_norm(self.ffn_norm_mot_gen, _residual[:, :num_image_gen_tokens], self.ffn_norm_mot_gen.weight.dtype),
                            safe_norm(self.ffn_norm,         _residual[:, num_image_gen_tokens:], self.ffn_norm.weight.dtype),
                        ], dim=1)

                        return _residual, _hidden_states

                    if self.dropout_selective_checkpoint:
                        residual, hidden_states = activation_checkpoint(
                            _dropout_and_norm_ffn, False, residual, hidden_states
                        )
                    else:
                        residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)

                if not self.use_moe:  # dense mlp output
                    moe_outputs = None
                    gate_logits_gen, gate_logits_und = None, None
                    hidden_states = torch.cat([self.feed_forward_mot_gen(hidden_states[:, :num_image_gen_tokens]), self.feed_forward(hidden_states[:, num_image_gen_tokens:])], 1)

                else:  # MoE output
                    hidden_states_gen,gate_logits_gen, moe_gen = self.feed_forward_mot_gen(hidden_states[:, :num_image_gen_tokens])
                    
                    
                    hidden_states_und,gate_logits_und, moe_und = self.feed_forward(hidden_states[:, num_image_gen_tokens:])
                    hidden_states = torch.cat([hidden_states_gen,hidden_states_und],1)

                    # Keep branch logits separate; compute load-balance loss in model.forward
                    # where input_ids/cu_seqlens are reliably available.
                    gate_logits = (gate_logits_gen, gate_logits_und)
                    moe_outputs = SenseNovaVLMoEOutput(
                        gate_logits=gate_logits,
                        moe_loss=(moe_gen.moe_loss + moe_und.moe_loss) if (moe_gen.moe_loss is not None and moe_und.moe_loss is not None) else (moe_gen.moe_loss or moe_und.moe_loss),
                        z_loss=(moe_gen.z_loss + moe_und.z_loss) if (moe_gen.z_loss is not None and moe_und.z_loss is not None) else (moe_gen.z_loss or moe_und.z_loss),
                        routed_coef_loss=(moe_gen.routed_coef_loss + moe_und.routed_coef_loss) if (moe_gen.routed_coef_loss is not None and moe_und.routed_coef_loss is not None) else (moe_gen.routed_coef_loss or moe_und.routed_coef_loss),
                        routed_coef=None if (moe_gen.routed_coef is None and moe_und.routed_coef is None) else (
                            (moe_gen.routed_coef * num_image_gen_tokens + moe_und.routed_coef * hidden_states_und.shape[1]) /
                            (num_image_gen_tokens + hidden_states_und.shape[1])
                            if (moe_gen.routed_coef is not None and moe_und.routed_coef is not None) else (moe_gen.routed_coef or moe_und.routed_coef)
                        ),
                        gates_max=None,
                        drop_ratio=None
                    )
            final_hidden_states = hidden_states + residual

            return final_hidden_states, gate_logits_gen, gate_logits_und,  moe_outputs
        else:
            raise NotImplementedError("Post-norm is not supported yet.")


class Qwen3MoeMoT(BaseModel):
    """
    1D Packed Flash Qwen.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        embedding_init_std (float): std used to init embedding weight. 0.02 by default,
        attn_wqkv_init_std (float): std used to init attn_wqkv weight. 0.02 by default,
        attn_other_init_std (float): std used to init attn_other weight. 0.02 by default,
        ffn_uplayer_init_std (float): std used to init w1, w2 weight in ffn when using glu
            otherwise init fc1 weight in ffn. 0.02 by default,
        ffn_other_init_std (float): std used to init ffn_other weight. 0.02 by default,
        out_head_init_std (float): std used to init output lmhead weight. 0.02 by default,
        init_type (str): Initialization type. Use uniform or normal. "normal" by default,
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        multiple_of (int): The value to make SwiGLU hidden layer size multiple of large power of 2.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        head_dim: int = None,
        vocab_size: int = 50304,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 1.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        qkv_bias=True,
        o_bias=False,
        mlp_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        rope_type: str = "normal",
        rope_base: int = 10000,
        rope_scaling_factor: float = 1.0,
        use_sliding_window: bool = False,
        max_window_layers: int = 0,
        sliding_window: int = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        scale_attn_weights: bool = False,  # Qwen1
        use_logn_attn: bool = False,  # Qwen1
        moe_kwargs: dict = None,
        moe_layer_kwargs: dict = None,
        num_mtp_layers: int = 0,
    ):
        super().__init__()

        self.use_moe = True if moe_kwargs is not None and moe_kwargs.get("num_experts", 1) > 1 else False
        self.moe_kwargs = moe_kwargs

        self.embed_grad_scale = embed_grad_scale

        checkpoint_layer_num = int((num_layers + num_mtp_layers) * checkpoint)

        if first:
            self.tok_embeddings = Embedding1D(num_embeddings=vocab_size, embedding_dim=hidden_size)
            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)

        self.layers = nn.ModuleList(
            [
                Qwen3MoeMoTDecoder(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_kv_attention_heads=num_kv_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    qkv_bias=qkv_bias,
                    o_bias=o_bias,
                    mlp_bias=mlp_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    qk_interleaved=qk_interleaved,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_type=rope_type,
                    rope_base=rope_base,
                    rope_scaling_factor=rope_scaling_factor,
                    use_sliding_window=use_sliding_window and lid >= max_window_layers,
                    sliding_window=sliding_window,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                    max_position_embeddings=max_position_embeddings,
                    scale_attn_weights=scale_attn_weights,
                    use_logn_attn=use_logn_attn,
                    moe_kwargs=moe_kwargs,
                    moe_layer_kwargs=moe_layer_kwargs,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
                self.norm_mot_gen = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

            self.output = new_linear(
                name="output",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
            )

            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)
            
            if num_mtp_layers > 0:
                self.num_mtp_layers = num_mtp_layers
                mtp_blocks = [
                    (
                        Qwen3MoeMoTDecoder(
                            hidden_size=hidden_size,
                            head_dim=head_dim,
                            num_attention_heads=num_attention_heads,
                            num_kv_attention_heads=num_kv_attention_heads,
                            mlp_ratio=mlp_ratio,
                            attn_drop_rate=attn_drop_rate,
                            drop_rate=drop_rate,
                            dtype=dtype,
                            layer_norm_epsilon=layer_norm_epsilon,
                            checkpoint=False,
                            layer_idx=lid + num_layers + start_layer_idx,  # This parameter is used for caching during generation
                            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                            residual_in_fp32=residual_in_fp32,
                            device=device,
                            apply_post_layer_norm=apply_post_layer_norm,
                            fused_dropout_add_ln=False,
                            qkv_bias=qkv_bias,
                            o_bias=o_bias,
                            mlp_bias=mlp_bias,
                            norm_type=norm_type,
                            dropout_selective_checkpoint=dropout_selective_checkpoint,
                            use_scaled_init=use_scaled_init,
                            use_swiglu=use_swiglu,
                            qk_interleaved=qk_interleaved,
                            attn_wqkv_init_std=attn_wqkv_init_std,
                            attn_other_init_std=attn_other_init_std,
                            ffn_uplayer_init_std=ffn_uplayer_init_std,
                            ffn_other_init_std=ffn_other_init_std,
                            init_type=init_type,
                            rope_type=rope_type,
                            rope_base=rope_base,
                            rope_scaling_factor=rope_scaling_factor,
                            use_sliding_window=use_sliding_window and (lid + num_layers >= max_window_layers),
                            sliding_window=sliding_window,
                            mlp_layer_fusion=mlp_layer_fusion,
                            multiple_of=multiple_of,
                            max_position_embeddings=max_position_embeddings,
                            scale_attn_weights=scale_attn_weights,
                            use_logn_attn=use_logn_attn,
                            moe_kwargs=moe_kwargs,
                            moe_layer_kwargs=moe_layer_kwargs,
                        ),
                        (lid + num_layers) < checkpoint_layer_num,
                    )
                    for lid in range(num_mtp_layers)
                ]
                self.mtp = MTP(hidden_size, mtp_blocks, layer_norm_epsilon)

        self.parallel_output = parallel_output

    def forward(self, hidden_states=None, input_ids=None, image_gen_indicators=None, **kwargs):
        exist_non_image_gen_tokens = (~image_gen_indicators).any()
        exist_image_gen_tokens = image_gen_indicators.any()
        num_image_gen_tokens = image_gen_indicators.sum().item()

        # attention_mask: compute attention on the places where the value is 1
        # old condition may fail when use shared embedding
        if gpc.is_pipeline_first_stage() and hasattr(self, "tok_embeddings") and hidden_states is None:
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        moe_outputs = []
        gate_logits_gen = []
        gate_logits_und = []
        for layer_i, block in enumerate(self.layers):
            hidden_states, gen_gate, und_gate, moe_output = block(
                hidden_states,
                residual=None,
                position_embeddings=None,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                **kwargs,
            )
            moe_outputs.append(moe_output)
            gate_logits_gen.append(gen_gate)
            gate_logits_und.append(und_gate)

        if hasattr(self, "norm"):
            hidden_states = torch.cat([
                safe_norm(self.norm_mot_gen, hidden_states[:, :num_image_gen_tokens], self.norm_mot_gen.weight.dtype),
                safe_norm(self.norm,         hidden_states[:, num_image_gen_tokens:], self.norm.weight.dtype),
            ], dim=1)
        if hasattr(self, "output"):
            main_output = self.output(hidden_states)
        else:
            main_output = hidden_states

        # load balance loss
        if self.use_moe:
            gen_image_inds = kwargs.get("gen_image_inds", None)
            und_image_inds = kwargs.get("und_image_inds", kwargs.get("image_inds", None))
            l_aux_und = self.load_balancing_loss_func(
                gate_logits_und,
                use_attn_mask=True,
                cu_seqlens=kwargs.get("und_cu_seqlens", kwargs.get("cu_seqlens", None)),
                input_ids=kwargs.get("und_input_ids", input_ids),
                num_experts=self.moe_kwargs.get("num_experts"),
                image_inds=und_image_inds,
            )
            l_aux_gen = self.load_balancing_loss_func(
                gate_logits_gen,
                use_attn_mask=True,
                cu_seqlens=kwargs.get("gen_cu_seqlens", kwargs.get("cu_seqlens", None)),
                input_ids=kwargs.get("gen_input_ids", input_ids),
                num_experts=self.moe_kwargs.get("gen_num_experts", self.moe_kwargs.get("num_experts")),
                image_inds=gen_image_inds,
            )
            
            moe_balance_gen_ratio = float(getattr(gpc.config, "moe_balance_gen_ratio", 1.0))
            moe_balance_und_ratio = float(getattr(gpc.config, "moe_balance_und_ratio", 1.0))
            if l_aux_gen is not None and l_aux_und is not None:
                gen_tokens = sum(g.shape[0] for g in gate_logits_gen) ## seq*60
                und_tokens = sum(g.shape[0] for g in gate_logits_und)
                l_aux = l_aux_gen * moe_balance_gen_ratio + l_aux_und * moe_balance_und_ratio
            elif l_aux_gen is not None:
                gen_tokens = sum(g.shape[0] for g in gate_logits_gen)
                und_tokens = sum(g.shape[0] for g in gate_logits_und)
                l_aux = l_aux_gen * moe_balance_gen_ratio
            elif l_aux_und is not None:
                gen_tokens = sum(g.shape[0] for g in gate_logits_gen)
                und_tokens = sum(g.shape[0] for g in gate_logits_und)
                l_aux = l_aux_und * moe_balance_und_ratio
            else:
                gen_tokens = sum(g.shape[0] for g in gate_logits_gen)
                und_tokens = sum(g.shape[0] for g in gate_logits_und)
                l_aux = None
                
            total_tokens = max(gen_tokens + und_tokens, 1)
            for moe_output in moe_outputs:
                if moe_output is not None:
                    moe_output.gate_logits = None
            # use layer0 moe output to handle total moe loss
            if len(moe_outputs) > 0 and l_aux is not None:
                moe_outputs[0].moe_loss = l_aux

            log_every = 50
            batch_count = int(getattr(gpc.config, "batch_count", 0))
            if gpc.is_rank_for_log() and batch_count % log_every == 0:
                logger.info(
                    "[moe-loss] step=%s gen_layers=%s und_layers=%s gen_tokens=%s und_tokens=%s gen_ratio=%s und_ratio=%s l_aux_gen=%s l_aux_und=%s l_aux=%s",
                    batch_count,
                    len(gate_logits_gen),
                    len(gate_logits_und),
                    gen_tokens,
                    und_tokens,
                    moe_balance_gen_ratio,
                    moe_balance_und_ratio,
                    None if l_aux_gen is None else float(l_aux_gen.detach().float().item()),
                    None if l_aux_und is None else float(l_aux_und.detach().float().item()),
                    None if l_aux is None else float(l_aux.detach().float().item()),
                )

        if hasattr(self, "num_mtp_layers") and self.num_mtp_layers > 0:
            mtp_outputs, mtp_moe_outputs = self.mtp(
                hidden_states, input_ids, self.tok_embeddings, self.norm, self.output
            )
        else:
            mtp_outputs = None
            mtp_moe_outputs = []

        if len(mtp_moe_outputs) > 0:
            moe_outputs.extend(mtp_moe_outputs)

        # hidden_states, hidden_states_before_output, moe_outputs
        return main_output, hidden_states, mtp_outputs, moe_outputs
    

    def load_balancing_loss_func(
            self, 
            gate_logits: List[torch.Tensor], 
            use_attn_mask: bool = False,
            cu_seqlens = None, 
            input_ids = None,
            num_experts: Optional[int] = None, 
            image_inds=None, 
            is_global=True):
        top_k = self.moe_kwargs.get("top_k")
        num_experts = int(self.moe_kwargs.get("num_experts") if num_experts is None else num_experts)
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        expert_mask = torch.mean(expert_mask.float(), dim=2) ##[n_layer,seq_length,experts]
        # if torch.distributed.get_rank() == 0:
        #     print(selected_experts.shape)
        #     print(num_experts)
        if not use_attn_mask:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask, dim=1)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=1)
        elif cu_seqlens is None or input_ids is None:
            logger.warning("cu_seqlens and input_ids must be provided when use_attn_mask is True.")
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask, dim=1)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=1)
        else:
            # seqlens = gpc.config.data.seq_len // gpc.get_world_size(ParallelMode.TENSOR) ## --> needs to be equal to current seq lens
            # In ISP mode, gate_logits are already LOCAL (ISP-split), so don't divide by tp_size
            _is_isp = is_using_isp()
            _tp_size = gpc.get_world_size(ParallelMode.TENSOR)
            if _is_isp and _tp_size > 1:
                seqlens = expert_mask.shape[1]
            else:
                seqlens = expert_mask.shape[1] // _tp_size
            ## 16384/8192
            micro_bsz = gpc.config.data.micro_bsz
            num_hidden_layers = len(gate_logits)

            # get attention mask
            # In ISP mode, input_ids and cu_seqlens are already LOCAL (split gen/und),
            # so the attention mask is computed directly on local data.
            attention_mask = torch.ones((1, micro_bsz * seqlens), dtype=torch.int8, device=compute_device) ## []
            for i in range(len(cu_seqlens) - 1, 0, -1):
                if torch.all(input_ids[0][cu_seqlens[i - 1] : cu_seqlens[i]] == 0):
                    attention_mask[0][cu_seqlens[i - 1] : cu_seqlens[i]] = 0

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, micro_bsz, seqlens, 1))
                .reshape(num_hidden_layers, -1, 1)
                .to(compute_device)
            )

            if image_inds is not None:
                assert is_global
                image_attention_mask = image_inds[:, :, None].float()
                text_attention_mask = (~image_inds)[:, :, None].float()

                image_tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask * image_attention_mask, dim=1)
                image_tokens_sum = (expert_attention_mask * image_attention_mask).sum(dim=1)
                text_tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask * text_attention_mask, dim=1)
                text_tokens_sum = (expert_attention_mask * text_attention_mask).sum(dim=1)

                router_per_expert_attention_mask = expert_attention_mask

                image_router_prob_sum_expert = torch.sum(routing_weights * router_per_expert_attention_mask * image_attention_mask, dim=1)
                text_router_prob_sum_expert = torch.sum(routing_weights * router_per_expert_attention_mask * text_attention_mask, dim=1)

                # Fuse 6 all_reduce into 1
                _ar_buf = torch.cat([
                    image_tokens_sum_expert.reshape(-1),
                    image_tokens_sum.reshape(-1),
                    text_tokens_sum_expert.reshape(-1),
                    text_tokens_sum.reshape(-1),
                    image_router_prob_sum_expert.reshape(-1),
                    text_router_prob_sum_expert.reshape(-1),
                ])
                torch.distributed.all_reduce(_ar_buf, op=torch.distributed.ReduceOp.SUM)
                _off = 0
                _ne = image_tokens_sum_expert.numel()
                _ns = image_tokens_sum.numel()
                image_tokens_sum_expert = _ar_buf[_off:_off + _ne].reshape(image_tokens_sum_expert.shape); _off += _ne
                image_tokens_sum = _ar_buf[_off:_off + _ns].reshape(image_tokens_sum.shape); _off += _ns
                text_tokens_sum_expert = _ar_buf[_off:_off + _ne].reshape(text_tokens_sum_expert.shape); _off += _ne
                text_tokens_sum = _ar_buf[_off:_off + _ns].reshape(text_tokens_sum.shape); _off += _ns
                image_router_prob_sum_expert = _ar_buf[_off:_off + _ne].reshape(image_router_prob_sum_expert.shape); _off += _ne
                text_router_prob_sum_expert = _ar_buf[_off:_off + _ne].reshape(text_router_prob_sum_expert.shape)

                image_tokens_per_expert = image_tokens_sum_expert / (image_tokens_sum + torch.finfo(torch.bfloat16).smallest_normal)
                text_tokens_per_expert = text_tokens_sum_expert / (text_tokens_sum + torch.finfo(torch.bfloat16).smallest_normal)
                image_router_prob_sum = image_tokens_sum
                image_router_prob_per_expert = image_router_prob_sum_expert / (image_router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal)
                text_router_prob_sum = text_tokens_sum
                text_router_prob_per_expert = text_router_prob_sum_expert / (text_router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal)

                # --- per-layer balance loss (for diagnostics) ---
                img_per_layer = (image_tokens_per_expert * image_router_prob_per_expert).sum(dim=-1) * num_experts  # [n_layers]
                txt_per_layer = (text_tokens_per_expert * text_router_prob_per_expert).sum(dim=-1) * num_experts    # [n_layers]
                l_aux_image = img_per_layer.sum()
                l_aux_text = txt_per_layer.sum()
                # Guard: when no image tokens globally, image balance loss is meaningless
                if image_tokens_sum.sum() == 0:
                    l_aux_image = l_aux_image * 0.0
                if text_tokens_sum.sum() == 0:
                    l_aux_text = l_aux_text * 0.0

                if torch.distributed.get_rank() == 0:
                    _n_layers = image_tokens_sum.shape[0]
                    # local token counts (before all_reduce) for this rank
                    _local_img = (expert_attention_mask * image_attention_mask).sum().item()
                    _local_txt = (expert_attention_mask * text_attention_mask).sum().item()
                    _attn_mask_sum = attention_mask.sum().item()
                    _attn_mask_total = attention_mask.numel()
                    # per-layer normalization sanity check (should be 1.0 if correct)
                    _img_tpe_sum_per_layer = image_tokens_per_expert.sum(dim=-1)  # [n_layers]
                    _img_rpe_sum_per_layer = image_router_prob_per_expert.sum(dim=-1)
                    _txt_tpe_sum_per_layer = text_tokens_per_expert.sum(dim=-1)
                    _txt_rpe_sum_per_layer = text_router_prob_per_expert.sum(dim=-1)
                    # per-layer image_tokens_sum (should be same across layers)
                    _img_ts_per_layer = image_tokens_sum.squeeze(-1)  # [n_layers]
                    _txt_ts_per_layer = text_tokens_sum.squeeze(-1)
                    print(
                        f"[und_balance] "
                        f"l_aux_image={l_aux_image.item():.2f} l_aux_text={l_aux_text.item():.2f} "
                        f"n_layers={_n_layers} num_experts={num_experts} "
                        f"theoretical_max={_n_layers * num_experts} "
                        f"| global_img_tokens={image_tokens_sum.sum().item():.0f} "
                        f"global_txt_tokens={text_tokens_sum.sum().item():.0f} "
                        f"| local_img_tokens={_local_img:.0f} local_txt_tokens={_local_txt:.0f} "
                        f"attn_mask={_attn_mask_sum:.0f}/{_attn_mask_total} "
                        f"| img_per_layer: mean={img_per_layer.mean().item():.2f} "
                        f"max={img_per_layer.max().item():.2f}(layer{img_per_layer.argmax().item()}) "
                        f"min={img_per_layer.min().item():.2f} "
                        f"| txt_per_layer: mean={txt_per_layer.mean().item():.2f} "
                        f"max={txt_per_layer.max().item():.2f}(layer{txt_per_layer.argmax().item()}) "
                        f"min={txt_per_layer.min().item():.2f} "
                        f"| norm_check img_tpe_sum: min={_img_tpe_sum_per_layer.min().item():.6f} "
                        f"max={_img_tpe_sum_per_layer.max().item():.6f} "
                        f"img_rpe_sum: min={_img_rpe_sum_per_layer.min().item():.6f} "
                        f"max={_img_rpe_sum_per_layer.max().item():.6f} "
                        f"txt_tpe_sum: min={_txt_tpe_sum_per_layer.min().item():.6f} "
                        f"max={_txt_tpe_sum_per_layer.max().item():.6f} "
                        f"| img_tokens_per_layer: min={_img_ts_per_layer.min().item():.0f} "
                        f"max={_img_ts_per_layer.max().item():.0f} "
                        f"std={_img_ts_per_layer.float().std().item():.1f} "
                        f"txt_tokens_per_layer: min={_txt_ts_per_layer.min().item():.0f} "
                        f"max={_txt_ts_per_layer.max().item():.0f} "
                        f"std={_txt_ts_per_layer.float().std().item():.1f}"
                    )
                moe_balance_image_ratio = gpc.config.moe_balance_image_ratio
                moe_balance_text_ratio = gpc.config.moe_balance_text_ratio
                l_aux_image = l_aux_image * moe_balance_image_ratio
                l_aux_text = l_aux_text * moe_balance_text_ratio
                l_aux = l_aux_image + l_aux_text
            
            else:
                # Compute the percentage of tokens routed to each experts
                if is_global:
                    tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask, dim=1)
                    tokens_sum = expert_attention_mask.sum(dim=1)

                router_per_expert_attention_mask = expert_attention_mask

                # Compute the average probability of routing to these experts
                if is_global:
                    router_prob_sum_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=1)

                    # Fuse 3 all_reduce into 1
                    _ar_buf = torch.cat([
                        tokens_sum_expert.reshape(-1),
                        tokens_sum.reshape(-1),
                        router_prob_sum_expert.reshape(-1),
                    ])
                    torch.distributed.all_reduce(_ar_buf, op=torch.distributed.ReduceOp.SUM)
                    _off = 0
                    _ne = tokens_sum_expert.numel()
                    _ns = tokens_sum.numel()
                    tokens_sum_expert = _ar_buf[_off:_off + _ne].reshape(tokens_sum_expert.shape); _off += _ne
                    tokens_sum = _ar_buf[_off:_off + _ns].reshape(tokens_sum.shape); _off += _ns
                    router_prob_sum_expert = _ar_buf[_off:_off + _ne].reshape(router_prob_sum_expert.shape)

                    tokens_per_expert = tokens_sum_expert / (tokens_sum + torch.finfo(torch.bfloat16).smallest_normal)
                    router_prob_sum = tokens_sum
                    router_prob_per_expert = router_prob_sum_expert / (router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal)
                else:
                    tokens_per_expert = torch.sum(expert_mask * expert_attention_mask, dim=1) / (torch.sum(expert_attention_mask, dim=1) + torch.finfo(torch.bfloat16).smallest_normal)
                    router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=1) / (torch.sum(router_per_expert_attention_mask, dim=1) + torch.finfo(torch.bfloat16).smallest_normal)

                
                l_aux_per_layer = (tokens_per_expert * router_prob_per_expert).sum(dim=-1) * num_experts   # [n_layer]
                l_aux = l_aux_per_layer.sum()  # 你原来的口径（对 layer 求和）

                if torch.distributed.get_rank() == 0:
                    print(
                        f"[moe_balance] l_aux_sum={l_aux.item():.4f} "
                        f"per_layer mean={l_aux_per_layer.mean().item():.4f} "
                        f"min={l_aux_per_layer.min().item():.4f} "
                        f"max={l_aux_per_layer.max().item():.4f}"
                    )
                    # 如果想看最“偏”的层是哪几层
                    topv, topi = torch.topk(l_aux_per_layer, k=min(5, l_aux_per_layer.numel()))
                    print("top layers:", [(int(i), float(v)) for i, v in zip(topi, topv)])

        return l_aux
    



    #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    #     expert_mask = torch.mean(expert_mask.float(), dim=2)
        


    #     if not use_attn_mask:
    #         tokens_per_expert = torch.mean(expert_mask, dim=1)
    #         router_prob_per_expert = torch.mean(routing_weights, dim=1)
    #     elif cu_seqlens is None or input_ids is None:
    #         logger.warning("cu_seqlens and input_ids must be provided when use_attn_mask is True.")
    #         tokens_per_expert = torch.mean(expert_mask, dim=1)
    #         router_prob_per_expert = torch.mean(routing_weights, dim=1)
    #     else:
    #         num_hidden_layers = len(gate_logits)
    #         flat_input_ids = input_ids.view(-1)
    #         num_tokens = expert_mask.shape[1]


    #         attention_mask = torch.ones((1, num_tokens), dtype=torch.int8, device=compute_device)
    #         for i in range(len(cu_seqlens) - 1, 0, -1):
    #             start = int(cu_seqlens[i - 1])
    #             end = int(cu_seqlens[i])
    #             if start >= num_tokens:
    #                 continue
    #             end = min(end, num_tokens)
    #             if start < end and torch.all(flat_input_ids[start:end] == 0):
    #                 attention_mask[0][start:end] = 0





    #             image_tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask * image_attention_mask, dim=1)
    #             _all_reduce_if_needed(image_tokens_sum_expert)
    #             image_tokens_sum = (expert_attention_mask * image_attention_mask).sum(dim=1)
    #             _all_reduce_if_needed(image_tokens_sum)
    #             image_tokens_per_expert = image_tokens_sum_expert / (
    #                 image_tokens_sum + torch.finfo(torch.bfloat16).smallest_normal
    #             )

    #             text_tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask * text_attention_mask, dim=1)
    #             _all_reduce_if_needed(text_tokens_sum_expert)
    #             text_tokens_sum = (expert_attention_mask * text_attention_mask).sum(dim=1)
    #             _all_reduce_if_needed(text_tokens_sum)
    #             text_tokens_per_expert = text_tokens_sum_expert / (
    #                 text_tokens_sum + torch.finfo(torch.bfloat16).smallest_normal
    #             )


    #             image_router_prob_sum_expert = torch.sum(
    #                 routing_weights * router_per_expert_attention_mask * image_attention_mask, dim=1
    #             )
    #             _all_reduce_if_needed(image_router_prob_sum_expert)
    #             image_router_prob_sum = image_tokens_sum
    #             image_router_prob_per_expert = image_router_prob_sum_expert / (
    #                 image_router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal
    #             )

    #             text_router_prob_sum_expert = torch.sum(
    #                 routing_weights * router_per_expert_attention_mask * text_attention_mask, dim=1
    #             )
    #             _all_reduce_if_needed(text_router_prob_sum_expert)
    #             text_router_prob_sum = text_tokens_sum
    #             text_router_prob_per_expert = text_router_prob_sum_expert / (
    #                 text_router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal
    #             )

    #             l_aux_image = torch.sum(image_tokens_per_expert * image_router_prob_per_expert) * num_experts
    #             l_aux_text = torch.sum(text_tokens_per_expert * text_router_prob_per_expert) * num_experts
    #             if torch.distributed.get_rank() == 0:
    #                 print(f"rank {torch.distributed.get_rank()}, balance image: {l_aux_image.item()}, text: {l_aux_text.item()}")
    #             moe_balance_image_ratio = gpc.config.moe_balance_image_ratio
    #             moe_balance_text_ratio = gpc.config.moe_balance_text_ratio
    #             l_aux_image = l_aux_image * moe_balance_image_ratio
    #             l_aux_text = l_aux_text * moe_balance_text_ratio
    #             l_aux = l_aux_image + l_aux_text
    #         else:
    #             if is_global:
    #                 tokens_sum_expert = torch.sum(expert_mask * expert_attention_mask, dim=1)
    #                 _all_reduce_if_needed(tokens_sum_expert)
    #                 tokens_sum = expert_attention_mask.sum(dim=1)
    #                 _all_reduce_if_needed(tokens_sum)
    #                 tokens_per_expert = tokens_sum_expert / (tokens_sum + torch.finfo(torch.bfloat16).smallest_normal)
    #             else:
    #                 tokens_per_expert = torch.sum(expert_mask * expert_attention_mask, dim=1) / (
    #                     torch.sum(expert_attention_mask, dim=1) + torch.finfo(torch.bfloat16).smallest_normal
    #                 )

    #             router_per_expert_attention_mask = expert_attention_mask
    #             if is_global:
    #                 router_prob_sum_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=1)
    #                 _all_reduce_if_needed(router_prob_sum_expert)
    #                 router_prob_sum = tokens_sum
    #                 router_prob_per_expert = router_prob_sum_expert / (
    #                     router_prob_sum + torch.finfo(torch.bfloat16).smallest_normal
    #                 )
    #             else:
    #                 router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=1) / (
    #                     torch.sum(router_per_expert_attention_mask, dim=1) + torch.finfo(torch.bfloat16).smallest_normal
    #                 )
    #             l_aux = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts

    #     if l_aux is None:
    #         l_aux = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts
    #     log_every = 50
    #     batch_count = int(getattr(gpc.config, "batch_count", 0))
    #     if gpc.is_rank_for_log() and batch_count % log_every == 0:
    #         logger.info(
    #             "[moe-load-balance] step=%s layers=%s tokens=%s use_attn_mask=%s image_split=%s l_aux=%s",
    #             batch_count,
    #             len(gate_logits),
    #             expert_mask.shape[1],
    #             use_attn_mask,
    #             image_inds is not None,
    #             None if l_aux is None else float(l_aux.detach().float().item()),
    #         )
    #     if l_aux is None:
    #         return None


    @staticmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError
