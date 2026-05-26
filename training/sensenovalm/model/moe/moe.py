# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0). MoE composition style
# follows DeepSpeed (Microsoft, Apache-2.0).
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import _calculate_correct_fan, calculate_gain

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.naive_amp import set_fp32_attr_to_module
from sensenovalm.model.modules.mlp import new_feed_forward
from sensenovalm.model.moe.dropless_layer_qwen import QwenDroplessMoELayer
from sensenovalm.model.moe.gshard_layer import GShardMoELayer
from sensenovalm.utils.logger import get_logger

from .utils import SenseNovaVLMoEOutput, LayerScale

# global llm logger
logger = get_logger(__file__)


def new_moe_layer(moe_type: str, **kwargs):
    if moe_type == "GShard":
        return GShardMoELayer(**kwargs)
    elif moe_type == "QwenDropless":
        return QwenDroplessMoELayer(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {moe_type}")


class MoEBase(torch.nn.Module):
    """Initialize an MoE base layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        activation_type: str = "swiglu",
        shared_expert_intermediate_size: int = None,
        moe_type: str = None,
        moe_cls=None,
        residual_mlp_cls=None,
    ):

        super().__init__()

        if moe_layer_kwargs is None:
            moe_layer_kwargs = dict()

        ep_group = gpc.get_group(ParallelMode.EXPERT)
        ep_size = gpc.get_world_size(ParallelMode.EXPERT)

        if moe_type is None:
            moe_type = gpc.config.model.moe_kwargs.moe_type
        if moe_cls is None:
            moe_cls = new_moe_layer
        if residual_mlp_cls is None:
            residual_mlp_cls = new_feed_forward

        self.moe_layer = moe_cls(
            moe_type=moe_type,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            ep_group=ep_group,
            ep_size=ep_size,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
            **moe_layer_kwargs,
        )
        set_fp32_attr_to_module(self.moe_layer.gate)

        # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
        self.num_shared_experts = num_shared_experts
        if self.num_shared_experts > 0:
            if shared_expert_intermediate_size is None:
                shared_expert_intermediate_size = int(hidden_features * num_shared_experts)
            self.residual_mlp = residual_mlp_cls(
                in_features=in_features,
                hidden_features=shared_expert_intermediate_size,
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
            )


class MoE(MoEBase):
    """Initialize an MoE layer.
    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_type: str = None,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        activation_type: str = "swiglu",
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            moe_layer_kwargs=moe_layer_kwargs,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
            moe_type=moe_type,
        )
        if self.num_shared_experts > 0:
            self.coefficient = torch.nn.Linear(in_features, 2)

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.num_shared_experts > 0:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts


class Qwen2MoE(MoEBase):
    """Initialize an Qwen2MoE layer.
    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        activation_type: str = "swiglu",
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            moe_layer_kwargs=moe_layer_kwargs,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
        )
        if self.num_shared_experts > 0:
            self.coefficient = torch.nn.Linear(in_features, 1, bias=False)

    def forward(self, hidden_states, used_token=None):
        """Qwen2MoE forward
        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens
        Returns:
            A tuple including output, gate loss, and expert count.
            * output (Tensor): output of the model
            * l_aux (Tensor): gate loss value
            * exp_counts (int): expert count
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.num_shared_experts > 0:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            output_mlp = F.sigmoid(coef) * output_mlp
            output = output + output_mlp
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts


class SenseNovaVLMoE(MoEBase):
    """Initialize an SenseNovaVL MoE layer.
    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        num_shared_experts (int): The number of shard experts, alwarys placed in every device.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: int = None,
        moe_layer_kwargs=None,
        device=None,
        dtype=None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 1,
        activation_type: str = "swiglu",
        use_coefficient: bool = False,
        moe_as_stack: bool = False,
        moe_output_scale: float = 1.0,
        coefficient_type: str = "softmax",
        ls_init_value: float = 1e-5,
        routed_coefficient_bias: float = 0.02,
        init_parameters: bool = True,
        coef_loss_after_mean: bool = False,
        coef_linear_bias: bool = True,
        moe_type: str = None,
        moe_cls=None,
        residual_mlp_cls=None,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_layer_kwargs=moe_layer_kwargs,
            device=device,
            dtype=dtype,
            mlp_layer_fusion=mlp_layer_fusion,
            multiple_of=multiple_of,
            activation_type=activation_type,
            moe_type=moe_type,
            moe_cls=moe_cls,
            residual_mlp_cls=residual_mlp_cls,
        )
        self.use_coefficient = use_coefficient
        self.moe_as_stack = moe_as_stack
        self.moe_output_scale = moe_output_scale
        self.coefficient_type = coefficient_type
        self.coef_loss_after_mean = coef_loss_after_mean
        self.routed_coefficient_bias = routed_coefficient_bias
        if self.num_shared_experts > 0:
            if self.use_coefficient:
                if self.coefficient_type == "softmax":
                    self.coefficient = torch.nn.Linear(in_features, 2, bias=coef_linear_bias)
                elif self.coefficient_type == "sigmoid":
                    self.coefficient = torch.nn.Linear(in_features, 1, bias=coef_linear_bias)
                elif self.coefficient_type == "layerscale":
                    self.coefficient = LayerScale(in_features, init_values=ls_init_value, inplace=False)

            if self.moe_as_stack:
                assert self.coefficient_type == "layerscale" and self.use_coefficient

        # moe monitor
        moe_monitor_cfg = gpc.config.moe_monitor
        self.gates_max_enable = moe_monitor_cfg.get("gates_max", False)
        self.drop_ratio_enable = moe_monitor_cfg.get("drop_ratio", False)

        if init_parameters:
            self.reset_parameters()

    def reset_parameters(self):
        self._init_wg(self.moe_layer.gate.wg.weight)
        # with torch.no_grad():
        #     w = self.moe_layer.gate.wg.weight
        #     w.zero_()
        #     w.add_(1e-4 * torch.randn_like(w))
        # routed experts may do not have initialization
        for expert in self.moe_layer.experts.wrapped_experts:
            expert.apply(self._init_linear)

        if self.num_shared_experts > 0 and self.use_coefficient:
            if self.coefficient_type in ("softmax", "sigmoid"):
                self._init_linear(self.coefficient)
                with torch.no_grad():
                    ratio = self.routed_coefficient_bias
                    bias_init = float(-np.log((1 - ratio) / ratio))
                    if self.coefficient.bias is not None:
                        if self.coefficient_type == "softmax":
                            self.coefficient.bias[0].data.fill_(bias_init / 2.0)
                            self.coefficient.bias[1].data.fill_(-bias_init / 2.0)
                        elif self.coefficient_type == "sigmoid":
                            bias_init = float(-np.log((1 - ratio) / ratio))
                            self.coefficient.bias.data.fill_(bias_init)

    # kaiming_uniform_
    def _init_wg(self, tensor, a: float = math.sqrt(5), mode: str = "fan_in", nonlinearity: str = "leaky_relu"):
        if 0 in tensor.shape:
            logger.warning("Initializing zero-element tensors is a no-op")
            return tensor
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def _init_bias(self, tensor, a, b):
        with torch.no_grad():
            return tensor.uniform_(a, b)

    def _init_linear(self, m):
        if isinstance(m, torch.nn.Linear):
            self._init_wg(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self._init_bias(m.bias, -bound, bound)

    def forward(self, hidden_states, valid_index=None, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        if self.moe_as_stack:
            hidden_states = self.residual_mlp(hidden_states)

        gates_max, drop_ratio = None, None
        l_aux, z_loss, gate_logits = None, None, None
        outputs = self.moe_layer(hidden_states, valid_index, used_token)
        if len(outputs) == 3:
            output, l_aux_or_gate, z_loss = outputs
            if l_aux_or_gate.ndim == 0:
                l_aux = l_aux_or_gate
            else:
                gate_logits = l_aux_or_gate
        elif len(outputs) == 4:
            if self.gates_max_enable:
                output, l_aux, z_loss, gates_max = outputs
            elif self.drop_ratio_enable:
                output, l_aux, z_loss, drop_ratio = outputs
            else:
                raise RuntimeError("unknown moe layer output")
        elif len(outputs) == 5:
            output, l_aux, z_loss, gates_max, drop_ratio = outputs

        if self.moe_output_scale != 1.0:
            output = output * self.moe_output_scale

        # Residual MoE
        routed_coef_loss = None
        routed_coef = 1.0
        if self.num_shared_experts > 0:
            if self.moe_as_stack:
                coef, coef_mean = self.coefficient(output)
                output = coef + hidden_states
                routed_coef = coef_mean.item()
            else:
                output_mlp = self.residual_mlp(hidden_states)
                if isinstance(output_mlp, tuple):
                    output_mlp = output_mlp[0]  # Ignore the bias term for now
                if self.use_coefficient:
                    if self.coefficient_type == "softmax":
                        coef = self.coefficient(hidden_states)
                        coef = torch.nn.functional.softmax(coef, dim=-1)
                        output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
                        routed_coef = coef[:, :valid_index]
                        if self.coef_loss_after_mean:
                            routed_coef_loss = (
                                routed_coef[..., 0:1].mean()
                                * (routed_coef[..., 0:1] > routed_coef[..., 1:]).float().mean()
                                + routed_coef[..., 1:].mean()
                                * (1 - (routed_coef[..., 0:1] > routed_coef[..., 1:]).float().mean())
                            ) * 2
                        else:
                            routed_coef_loss = (
                                routed_coef[..., 0:1].mean() ** 2 / 2.0 + routed_coef[..., 1:].mean() ** 2 / 2.0
                            ).sqrt() * 2.0  # ideally --> 1
                        routed_coef = routed_coef[..., 0].mean().item()
                    elif self.coefficient_type == "sigmoid":
                        coef = self.coefficient(hidden_states)
                        coef = torch.nn.functional.sigmoid(coef)
                        output = output + output_mlp * coef
                        if self.coef_loss_after_mean:
                            coef_ = coef.mean()
                            coef__ = (coef > 0.5).float().mean()
                            routed_coef_loss = ((coef_ * coef__) + (1 - coef_) * (1 - coef__)) * 2
                            routed_coef = coef_.detach().item()
                        else:
                            coef_ = coef.mean()
                            routed_coef = coef_.detach().item()
                            routed_coef_loss = (coef * coef + (1 - coef) * (1 - coef)).mean() * 2
                    elif self.coefficient_type == "layerscale":
                        coef, coef_mean = self.coefficient(output)
                        output = coef + output_mlp
                        routed_coef = coef_mean.item()
                else:
                    # internevo deepseek
                    output = output + output_mlp
                    routed_coef = 1.0

        return output, gate_logits,SenseNovaVLMoEOutput(
            moe_loss=l_aux,
            gate_logits=gate_logits,  # to cpu?
            z_loss=z_loss,
            routed_coef_loss=routed_coef_loss,
            routed_coef=routed_coef,
            gates_max=gates_max,
            drop_ratio=drop_ratio,
        )
