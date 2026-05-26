# --------------------------------------------------------
# SenseNovaVL — derived from InternVL (OpenGVLab, MIT).
# Copyright (c) 2023 OpenGVLab. Licensed under MIT.
# Copyright (c) SenseNovaLM contributors. Modifications licensed under Apache-2.0.
# --------------------------------------------------------
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from sensenovalm.core.context import IS_REPLICA_ZERO_PARALLEL, ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.modules.linear import new_linear
from sensenovalm.model.moe.moe import SenseNovaVLMoE
from sensenovalm.model.moe.utils import SenseNovaVLMoEOutput
from sensenovalm.utils.common import get_current_device
from sensenovavl.model.sensenovavl_moe_chat.configuration_neo_vit import NEOVisionConfig
from sensenovavl.model.sensenovavl_moe_chat.flash_attention import FlashAttention
from sensenovavl.model.sensenovavl_moe_chat.utils import (
    gather_forward_split_backward,
    get_split_size,
    split_forward_gather_backward,
    uneven_all2all_gather_split,
)
from sensenovavl.model.modules.vit_mlp import new_feed_forward
from sensenovavl.model.utils import set_parallel_attribute
from sensenovavl.model.vit_moe import new_moe_layer
from sensenovavl.utils.utils import precompute_rope_freqs_sincos, build_abs_positions_from_grid_hw, apply_2d_rotary_pos_emb

logger = logging.get_logger(__name__)


class InternRMSNorm(nn.Module):
    """RMSNorm Module."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm as InternRMSNorm  # noqa: F811

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm")
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to InternRMSNorm")
    pass

norm_cls_mapping = {"rms_norm": InternRMSNorm, "layer_norm": nn.LayerNorm}


@dataclass
class BaseModelOutputWithMoELoss(BaseModelOutput):
    moe_outputs: List[SenseNovaVLMoEOutput] = None


@dataclass
class BaseModelOutputWithPoolWithMOELoss(BaseModelOutputWithPooling):
    moe_outputs: SenseNovaVLMoEOutput = None
    sequence_length: int = None
    cu_seqlens: torch.Tensor = None


class InternVisionEmbeddings(nn.Module):
    """
    Embedding Module for Vision.
    """

    def __init__(self, config: NEOVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size[0]
        self.downsample_factor = int(1 / config.downsample_ratio[0])
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.llm_embed_dim, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        self.gelu = nn.GELU()

        self.add_pos_embedding = config.add_pos_embedding
        if self.add_pos_embedding:
            # 2D RoPE for native resolution
            max_pixels = gpc.config.data.get('max_pixels', 4194304)
            ROPE_MAX_COORD_DEFAULT = max_pixels / self.patch_size ** 2
            ROPE_BASE_DEFAULT = 10000.0

            self.rope_dim_part = self.embed_dim // 2
            cos_x, sin_x = precompute_rope_freqs_sincos(
                self.rope_dim_part, ROPE_MAX_COORD_DEFAULT, base=ROPE_BASE_DEFAULT, device=None
            )
            cos_y, sin_y = precompute_rope_freqs_sincos(
                self.rope_dim_part, ROPE_MAX_COORD_DEFAULT, base=ROPE_BASE_DEFAULT, device=None
            )
            
            self.register_buffer("cos_cached_x", cos_x, persistent=False)
            self.register_buffer("sin_cached_x", sin_x, persistent=False)
            self.register_buffer("cos_cached_y", cos_y, persistent=False)
            self.register_buffer("sin_cached_y", sin_y, persistent=False)

        # set parallel attributes
        set_parallel_attribute(self.patch_embedding, IS_REPLICA_ZERO_PARALLEL)
        set_parallel_attribute(self.dense_embedding, IS_REPLICA_ZERO_PARALLEL)

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        """
        Apply 2D Rotary Position Embedding to the patch embeddings.
        """
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=get_current_device())
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32), # RoPE calculations are often more stable in float32
            self.cos_cached_x, self.sin_cached_x,
            self.cos_cached_y, self.sin_cached_y,
            abs_pos_x,
            abs_pos_y
        ).to(self.patch_embedding.weight.dtype)
        return embeddings
        
    def forward(self, pixel_values: torch.FloatTensor, grid_hw=None) -> torch.Tensor:
        assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"

        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)  # [N_total, 768] -> [N_total, 3, 16, 16]
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)
        current_device = get_current_device()
        if self.add_pos_embedding:
            self.cos_cached_x = self.cos_cached_x.to(current_device)
            self.sin_cached_x = self.sin_cached_x.to(current_device)
            self.cos_cached_y = self.cos_cached_y.to(current_device)
            self.sin_cached_y = self.sin_cached_y.to(current_device)
            patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw) # [28072, 1024]
        assert (grid_hw[:,0] * grid_hw[:,1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)  # (N_total // downsample_factor**2, C)

        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor**2)

        return embeddings


class InternVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: NEOVisionConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn
        assert self.use_flash_attn is True

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.tp_mode = gpc.config.parallel.tensor.mode
        self.qkv = new_linear(
            "wqkv",
            self.embed_dim,
            3 * self.embed_dim,
            bias=config.qkv_bias,
            multiple_of=3 * self.head_dim if self.tp_mode != "isp" else 1,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
        )

        self.tp_size = gpc.get_world_size(ParallelMode.TENSOR)

        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            set_parallel_attribute(self.q_norm, IS_REPLICA_ZERO_PARALLEL)
            set_parallel_attribute(self.k_norm, IS_REPLICA_ZERO_PARALLEL)

        self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)

        self.proj = new_linear(
            "wo",
            self.embed_dim,
            self.embed_dim,
            bias=config.proj_bias,
            multiple_of=self.head_dim if self.tp_mode != "isp" else 1,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
        )

        # It contains the number head for each tp rank
        # It is prepared for uneven split
        if self.tp_mode == "mtp" and self.tp_size > 1:
            self.div_head = self.num_heads // self.tp_size
            self.mod_head = self.num_heads % self.tp_size
            self.head_split_size = get_split_size(self.tp_size, self.div_head, self.mod_head)

        self.enable_vit_sp = gpc.config.parallel.tensor.enable_vit

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False, sequence_length=None, cu_seqlens=None):
        qkv = self.qkv(x)

        # reshape qkv to [batch, sequence_length, 3, num_heads, head_dim](default) or [sequence_length, 3, num_heads, head_dim](only for native resolution)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if self.qk_normalization:
            q, k, v = qkv.unbind(-3)
            if self.tp_size > 1 and self.tp_mode == "mtp":
                q = gather_forward_split_backward(
                    q.contiguous(),
                    ParallelMode.TENSOR,
                    dim=-2,
                    div=self.div_head,
                    mod=self.mod_head,
                    split_size=self.head_split_size,
                )
                k = gather_forward_split_backward(
                    k.contiguous(),
                    ParallelMode.TENSOR,
                    dim=-2,
                    div=self.div_head,
                    mod=self.mod_head,
                    split_size=self.head_split_size,
                )

            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)

            if self.tp_size > 1 and self.tp_mode == "mtp":
                q = split_forward_gather_backward(
                    q,
                    ParallelMode.TENSOR,
                    dim=-2,
                    div=self.div_head,
                    mod=self.mod_head,
                    split_size=self.head_split_size,
                )
                k = split_forward_gather_backward(
                    k,
                    ParallelMode.TENSOR,
                    dim=-2,
                    div=self.div_head,
                    mod=self.mod_head,
                    split_size=self.head_split_size,
                )

            qkv = torch.stack([q, k, v], dim=-3)

        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            qkv = uneven_all2all_gather_split(
                qkv, sequence_length, self.num_heads, gather_dim=-4, split_dim=-2, tp_size=self.tp_size
            )

        assert qkv.dim() == 4, "qkv must be 4D [S, 3, H, D] for native resolution."
        assert cu_seqlens is not None, "cu_seqlens must be provided for native resolution."

        # max sequence length for flash-attention applied to a single concatenated sequence
        max_s = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        context, _ = self.inner_attn(qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False, cu_seqlens=cu_seqlens, max_s=max_s)

        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            # isp: gather_dim = head_dim, split_dim=sequence_dim
            # for native resolution, context shape: [S, H, D]
            # head_dim = -2, sequence_dim = -3 for [B, S, H, D] and [S, H, D]
            context = uneven_all2all_gather_split(
                context, self.num_heads, sequence_length, gather_dim=-2, split_dim=-3, tp_size=self.tp_size, is_clone=True
            )

        outs = self.proj(rearrange(context, "... h d -> ... (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor, sequence_length=None, cu_seqlens=None) -> torch.Tensor:
        return self._flash_attn(hidden_states, sequence_length=sequence_length, cu_seqlens=cu_seqlens)


class InternVisionEncoderLayer(nn.Module):
    """Encoder Layer for Intern Vision Model."""

    def __init__(
        self,
        config: NEOVisionConfig,
        drop_path_rate: float,
        gradient_checkpointing: float = 0.0,
        layer_id: int = -1,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_per_layer_num = config.checkpoint_per_layer_num
        self.use_flash_attn = config.use_flash_attn
        self.use_moe = config.use_moe
        self.layer_id = layer_id
        self.tp_size = gpc.get_world_size(ParallelMode.TENSOR)

        self.attn = InternVisionAttention(config)

        if config.use_moe:
            self.mlp = SenseNovaVLMoE(
                in_features=self.embed_dim,
                hidden_features=config.moe_cfg.moe_intermediate_size,
                out_features=self.embed_dim,
                num_experts=config.moe_cfg.num_experts,
                top_k=config.moe_cfg.num_routed_experts,
                num_shared_experts=config.moe_cfg.num_shared_experts,
                shared_expert_intermediate_size=config.moe_cfg.shared_expert_intermediate_size,
                moe_layer_kwargs=config.moe_layer_kwargs,
                device=get_current_device(),
                dtype=gpc.config.model.dtype,
                activation_type=config.hidden_act,
                moe_as_stack=config.moe_cfg.get("moe_as_stack", False),
                moe_output_scale=config.moe_cfg.get("moe_output_scale", 1.0),
                use_coefficient=config.moe_cfg.get("use_coefficient", False),
                coefficient_type=config.moe_cfg.get("coefficient_type", "softmax"),
                ls_init_value=config.moe_cfg.get("ls_init_value", 1e-5),
                routed_coefficient_bias=config.moe_cfg.get("routed_coefficient_bias", 0.02),
                init_parameters=True,
                moe_type=config.moe_cfg.moe_type,
                moe_cls=new_moe_layer,
                residual_mlp_cls=new_feed_forward,
            )
        else:
            self.mlp = new_feed_forward(
                config.hidden_size, config.intermediate_size, config.hidden_size, activation_type=config.hidden_act
            )
        norm_cls = norm_cls_mapping[config.norm_type]
        self.norm1 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # set parallel attribute
        set_parallel_attribute(self.norm1, IS_REPLICA_ZERO_PARALLEL)
        set_parallel_attribute(self.norm2, IS_REPLICA_ZERO_PARALLEL)
        set_parallel_attribute(self.ls1, IS_REPLICA_ZERO_PARALLEL)
        set_parallel_attribute(self.ls2, IS_REPLICA_ZERO_PARALLEL)

    def forward(self, hidden_states, image_flags=None, sequence_length=None, cu_seqlens=None):
        if self.gradient_checkpointing and self.training and self.checkpoint_per_layer_num == 1:
            mlp_hidden_states = torch.utils.checkpoint.checkpoint(
                self._forward, hidden_states, image_flags, sequence_length, cu_seqlens, use_reentrant=True
            )
        else:
            mlp_hidden_states = self._forward(hidden_states, image_flags, sequence_length, cu_seqlens)

        if isinstance(mlp_hidden_states, tuple):
            mlp_hidden_states, moe_loss, z_loss, moe_coef_loss, routed_coef, gates_max, drop_ratio = mlp_hidden_states

            return mlp_hidden_states, moe_loss, z_loss, moe_coef_loss, routed_coef, gates_max, drop_ratio
        else:
            return mlp_hidden_states

    def _forward(
        self,
        hidden_states: torch.Tensor,
        image_flags=None,
        sequence_length=None,
        cu_seqlens=None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer
                of shape `(batch, seq_len, embed_dim)`
                or shape `(seq_len, embed_dim)` for native resolution
        """

        # attention
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states), sequence_length=sequence_length, cu_seqlens=cu_seqlens) * self.ls1
        ) 
        # moe
        if not self.use_moe:  # dense mlp output
            norm2_output = self.norm2(hidden_states)
            mlp_hidden_states = self.mlp(norm2_output)
            hidden_states = hidden_states + self.drop_path2(mlp_hidden_states * self.ls2)
            return hidden_states
        else:  # MoE output
            moe_tp_size = gpc.get_world_size(ParallelMode.EXPERT_TENSOR)
            seq_len = hidden_states.shape[-2]
            div_head = seq_len // moe_tp_size
            mod_head = seq_len % moe_tp_size
            split_size = []
            for i in range(moe_tp_size):
                if i < mod_head:
                    split_size.append(div_head + bool(mod_head))
                else:
                    split_size.append(div_head)
            if moe_tp_size > 1:
                mlp_hidden_states = split_forward_gather_backward(
                    hidden_states,
                    ParallelMode.EXPERT_TENSOR,
                    dim=-2,
                    div=div_head,
                    mod=mod_head,
                    split_size=split_size,
                )
            else:
                mlp_hidden_states = hidden_states
            # image_flags is accepted upstream (dummy-image path in modeling_sensenovavl_chat_mot.py)
            # but unused under native_resolution — all tokens go through the MoE
            used_token = None
            mlp_hidden_states, moe_outputs = self.mlp(self.norm2(mlp_hidden_states), used_token)

            if self.tp_size > 1:
                mlp_hidden_states = gather_forward_split_backward(
                    mlp_hidden_states.contiguous(),
                    ParallelMode.TENSOR,
                    dim=1,
                    div=div_head,
                    mod=mod_head,
                    split_size=split_size,
                )

            hidden_states = hidden_states + self.drop_path2(mlp_hidden_states * self.ls2)

            return hidden_states, moe_outputs


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: NEOVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # stochastic depth decay rule
        checkpoint_layer_num = int(config.num_hidden_layers * config.gradient_checkpointing)
        self.num_layers = config.num_hidden_layers
        self.checkpoint_per_layer_num = self.config.checkpoint_per_layer_num
        assert self.num_layers % self.checkpoint_per_layer_num == 0
        if config.num_hidden_layers > 0:
            self.layers = nn.ModuleList(
                [
                    InternVisionEncoderLayer(
                        config,
                        drop_path_rate=dpr[lid],
                        gradient_checkpointing=lid < checkpoint_layer_num,
                        layer_id=lid,
                    )
                    for lid in range(config.num_hidden_layers)
                ]
            )
            self.is_dummy = False
        else:
            self.is_dummy = True

        self.moe_coef = getattr(gpc.config.loss, "expert_balance_coef", 0.0)
        self.moe_z_coef = getattr(gpc.config.loss, "moe_z_loss_coef", 0.0)
        self.expert_coef_balance_coef = getattr(gpc.config.loss, "expert_coef_balance_coef", 0.0)

        # moe monitor
        self.moe_monitor_cfg = gpc.config.moe_monitor

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        inputs_embeds,
        image_flags=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_length: int = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Forward method with activation checkpointing."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        moe_outputs = []

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)

                    if isinstance(x_, tuple):
                        x_, cur_moe_output = x_
                        moe_outputs.append(cur_moe_output)

                return x_

            return custom_forward

        # Uniformly divide the total number of Transformer layers and checkpoint
        # the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        layer_id = 0
        while layer_id < self.num_layers:
            hidden_states = torch.utils.checkpoint.checkpoint(
                custom(layer_id, layer_id + self.checkpoint_per_layer_num),
                hidden_states,
                image_flags,
                sequence_length,
                cu_seqlens=cu_seqlens,
                use_reentrant=True,
            )

            layer_id += self.checkpoint_per_layer_num

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, moe_outputs]
                if v is not None and (isinstance(v, list) and len(v) > 0)
            )

        return BaseModelOutputWithMoELoss(  # pylint: disable=E1123
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            moe_outputs=moe_outputs,
        )

    def forward(
        self,
        inputs_embeds,
        image_flags=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_length: int = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)` or shape `(sequence_length, hidden_size)` for native resolution):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        if self.checkpoint_per_layer_num > 1:
            return self._checkpointed_forward(
                inputs_embeds, image_flags, output_hidden_states, return_dict, sequence_length, cu_seqlens=cu_seqlens
            )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        moe_outputs = []
        if not self.is_dummy:
            for _, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                layer_outputs = encoder_layer(hidden_states, image_flags=image_flags, sequence_length=sequence_length)
                if isinstance(layer_outputs, tuple):
                    layer_outputs, cur_moe_output = layer_outputs
                    moe_outputs.append(cur_moe_output)

                hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, moe_outputs]
                if v is not None and (isinstance(v, list) and len(v) > 0)
            )

        return BaseModelOutputWithMoELoss(  # pylint: disable=E1123
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            moe_outputs=moe_outputs,
        )


class NEOVisionModel(PreTrainedModel):
    """Vision Model."""

    main_input_name = "pixel_values"
    config_class = NEOVisionConfig
    _no_split_modules = ["InternVisionEncoderLayer"]

    def __init__(self, config: NEOVisionConfig):

        super().__init__(config)
        self.config = config
        self.tp_mode = gpc.config.parallel.tensor.mode
        self.tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)
        self.enable_vit_sp = gpc.config.parallel.tensor.enable_vit
        self.use_flash_attn = gpc.config.model.use_flash_attn

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")
        
        cu_seqlens = None

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
            assert self.use_flash_attn is True, "native resolution only supports flash attention."
            hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)
            cu_seqlens = (grid_hw[:, 0] * grid_hw[:, 1]).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        seq_len = hidden_states.shape[-2]
        if self.tp_mode == "isp" and self.enable_vit_sp and self.tp_size > 1:
            div_seq = seq_len // self.tp_size
            mod_seq = seq_len % self.tp_size
            sequence_split_size = get_split_size(self.tp_size, div_seq, mod_seq)
            hidden_states = split_forward_gather_backward(
                hidden_states, ParallelMode.TENSOR, dim=1, div=div_seq, mod=mod_seq, split_size=sequence_split_size
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            image_flags=image_flags,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sequence_length=seq_len,
            cu_seqlens=cu_seqlens,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = None

        if not return_dict:
            return (last_hidden_state, pooled_output, cu_seqlens) + encoder_outputs[1:]

        return BaseModelOutputWithPoolWithMOELoss(  # pylint: disable=E1123
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            moe_outputs=encoder_outputs.moe_outputs,
            sequence_length=seq_len,
            cu_seqlens=cu_seqlens
        )
