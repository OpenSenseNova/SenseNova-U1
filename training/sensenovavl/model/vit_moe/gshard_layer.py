# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Originally adapted from DeepSpeed (Microsoft, Apache-2.0); routing logic
# influenced by the GShard paper (Lepikhin et al., 2020).
"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.3.x
    from tutel import moe as tutel_moe

    TUTEL_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.moe.base_layer import BaseMoELayer
from sensenovalm.model.moe.gshard_layer import (
    GatingTokenRearrangeInfo,
    _capacity,
    _top_idx,
    einsum,
    gumbel_rsample,
    multiplicative_jitter,
)
from sensenovalm.model.moe.utils import all_to_all
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.megatron_timers import megatron_timer as timer
from sensenovavl.model.modules.vit_mlp import new_feed_forward

# global llm logger
logger = get_logger(__file__)

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}


def apply_jitter(x, epsilon, generator):  # pylint: disable=W0613
    if epsilon == 0:
        return x
    # Create a uniform distribution for jittering
    # uniform = torch.distributions.uniform.Uniform(low=1.0 - epsilon, high=1.0 + epsilon)
    uniform = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) * 2 * epsilon + 1.0 - epsilon
    # Apply jitter by multiplying with a sampled value from the uniform distribution
    jittered_x = x * uniform
    return jittered_x


def top1gating(
    logits: Tensor,
    capacity_factor: float,
    min_capacity: int,
    used_token: Tensor = None,
    noisy_gate_policy: Optional[str] = None,
    drop_tokens: bool = True,
    use_rts: bool = False,
    laux_allreduce="local",
    is_training=True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == "RSample":
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == "RSample" else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to("cpu")

    # Compute l_aux
    l_aux = _compute_laux(gates, mask1, laux_allreduce, num_experts, 1, is_training)

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.DATA))
        capacity = new_capacity

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=logits.device), high=torch.tensor(1.0, device=logits.device)
            ).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[0] >= min_capacity, (
        "No. of tokens (batch-size) should be greater than min_capacity."
        "Either set min_capacity to 0 or increase your batch size."
    )

    top_idx = _top_idx(mask1_rand, capacity)  # token index

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    # Compute locations in capacity buffer

    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.type_as(logits)
    gates = gates * mask1_float

    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask


def top2gating(
    logits: Tensor,
    capacity_factor: float,
    min_capacity: int,
    used_token: Tensor = None,
    laux_allreduce="local",
    is_training=True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), torch.finfo(logits.dtype).min)
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    # mask only used tokens
    if used_token is not None:
        mask2 = einsum("s,se->se", used_token, mask2)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    # exp_counts = torch.sum(mask1, dim=0).detach().to("cpu")

    # Compute l_aux
    l_aux = _compute_laux(gates, mask1, laux_allreduce, num_experts, 1, is_training)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.type_as(logits)
    mask2_float = mask2.type_as(logits)
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).type_as(logits)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask


def fused_topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    used_token: Tensor = None,
    noisy_gate_policy: Optional[str] = None,
    use_rts: bool = True,
    laux_allreduce: str = "local",
    enable_token_rearrange_opt: bool = False,
    use_tutel: bool = False,
    is_training: bool = True,
    gates_max_enable: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    if gates_max_enable:
        with torch.no_grad():
            gates_max = torch.max(gates, dim=-1)[0].mean()
    else:
        gates_max = None

    num_experts = gates.size(1)

    capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))

    # Create a mask by top-k experts
    if noisy_gate_policy not in ("RSample", "RSample_before"):
        indices_s = torch.topk(gates, k, dim=1).indices.t()
        masks = F.one_hot(indices_s.reshape(-1), num_classes=num_experts)
        # reshape (s,e) to (k,s,e)
        exp_usage = masks.reshape(-1, gates.shape[0], num_experts).sum(dim=0)
    else:
        masks = []
        for i in range(k):
            if i == 0:
                if noisy_gate_policy == "RSample_before":
                    scores = logits + gumbel_rsample(logits.shape, device=logits.device)
                else:
                    scores = logits
            else:
                if i == 1:
                    # 从第二个开始 对于非首选专家，添加噪声并
                    if noisy_gate_policy == "RSample":
                        scores = logits + gumbel_rsample(logits.shape, device=logits.device)

                # for prev_mask in masks:
                # 遮盖之前选中的专家
                scores = scores.masked_fill(masks[-1].bool(), float("-inf"))
            # 选择当前轮次得分最高的专家
            indices = torch.argmax(scores, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts)
            # mask only used tokens
            if used_token is not None:
                mask = einsum("s,se->se", used_token, mask)

            masks.append(mask)

        exp_usage = masks[0].clone()
        for mask in masks[1:]:
            exp_usage.add_(mask)

        masks = torch.cat(masks, dim=0)

    # gating decisions
    # exp_counts = torch.sum(exp_usage, dim=0).detach().to("cpu")
    # Compute l_aux
    if used_token is not None:
        l_aux = _compute_laux(gates[used_token], exp_usage[used_token], laux_allreduce, num_experts, k, is_training)
    else:
        l_aux = _compute_laux(gates, exp_usage, laux_allreduce, num_experts, k, is_training)

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=logits.device), high=torch.tensor(1.0, device=logits.device)
            ).rsample
            exp_selection_uniform_map[logits.device] = uniform

        masks_rand = masks * uniform(masks.shape)
        top_idx = _top_idx(masks_rand, capacity)  # token index
        new_masks = masks * torch.zeros_like(masks).scatter_(0, top_idx, 1)
        masks = new_masks

    # Compute locations in capacity buffer
    if use_tutel and TUTEL_INSTALLED:
        locations = tutel_moe.fast_cumsum_sub_one(masks)
    else:
        locations = torch.cumsum(masks, dim=0) - 1

    # reshape (s,e) to (k,s,e)
    masks = masks.reshape(-1, gates.shape[0], num_experts)
    locations = locations.reshape(-1, gates.shape[0], num_experts)

    # Remove locations outside capacity from mask
    masks *= torch.lt(locations, capacity)

    # Store the capacity location for each token
    locations_s = torch.sum(locations * masks, dim=2)

    # Normalize gate probabilities
    mask_float = masks.type_as(logits)
    gate_s, indices_s = torch.max(gates * mask_float, dim=2)
    denom_s = torch.sum(gate_s, dim=0)
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gate_s /= denom_s

    if enable_token_rearrange_opt:
        token_rearranged_ec_idx = indices_s.int() * capacity + locations_s.int()
        # shape：[S, E]->[C, E]->[E, C]->[E*C]
        token_sel_exp_int_mask = masks * torch.arange(k, 0, -1, device=masks.device).reshape(k, 1, 1)
        expert_sel_top_c_token_idx = torch.topk(
            torch.sum(token_sel_exp_int_mask, dim=0), k=capacity, dim=0, sorted=True
        )[1]
        expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(num_experts * capacity)
        token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
        token_exp_weights = gate_s.reshape(-1)

        topk_gating_token_infos = GatingTokenRearrangeInfo(
            token_rearranged_ec_idx=token_rearranged_ec_idx,
            token_exp_weights=token_exp_weights,
            expert_select_token_idx=expert_select_token_idx,
        )
        return l_aux, topk_gating_token_infos, gates_max
    else:
        # Calculate combine_weights and dispatch_mask
        gate_all = einsum("ks,kse->kse", gate_s, mask_float)
        locations_sc = F.one_hot(locations_s, num_classes=capacity).type_as(logits)
        combine_sec = einsum("kse,ksc->ksec", gate_all, locations_sc)
        combine_weights = torch.sum(combine_sec, dim=0)
        dispatch_mask = combine_weights.bool()

        return l_aux, combine_weights, dispatch_mask, gates_max


def _compute_laux(gates, exp_usage, laux_allreduce, num_experts, k, is_training):  # pylint: disable=W0613
    """
    计算辅助损失以促进专家的均衡使用。

    Args:
        gates (torch.Tensor): 归一化后的门控概率，形状为 [batch_size, num_experts]
        masks (List[torch.Tensor]): 每个Top-K选择对应的mask列表
        num_samples (int): 用于辅助损失计算的样本数量
        laux_allreduce (str): 辅助损失的计算方式，'local' 或 'all_nodes'
        num_experts (int): 专家总数
        is_training (bool): 是否处于训练模式

    Returns:
        torch.Tensor: 计算得到的辅助损失
    """

    me = torch.mean(gates, dim=0)
    ce = torch.mean(exp_usage.float(), dim=0) / k
    if laux_allreduce == "all_nodes" and not gpc.is_forward_done and is_training:
        dist.all_reduce(me, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
        dist.all_reduce(ce, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))
        if gpc.config.parallel.tensor.mode == "isp" and gpc.get_world_size(ParallelMode.EXPERT_WEIGHT) > 1:
            dist.all_reduce(me, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT_WEIGHT))
            dist.all_reduce(ce, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT_WEIGHT))
        elif gpc.config.parallel.tensor.mode != "isp" and gpc.get_world_size(ParallelMode.EXPERT_TENSOR) > 1:
            dist.all_reduce(me, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT_TENSOR))
            dist.all_reduce(ce, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT_TENSOR))
        if gpc.get_world_size(ParallelMode.EXPERT) > 1:
            dist.all_reduce(me, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT))
            dist.all_reduce(ce, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.EXPERT))

    l_aux = torch.mean(me * ce) * num_experts * num_experts

    return l_aux


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk: int = 1,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: Optional[str] = None,
        jitter_eps: float = 1e-2,
        drop_tokens: bool = True,
        use_rts: bool = False,
        use_fused_gating: bool = False,
        enable_token_rearrange_opt: bool = False,
        use_tutel: bool = False,
        laux_allreduce: str = "local",
    ) -> None:
        super().__init__()
        if topk > 1 and use_rts and gpc.is_rank_for_log():
            logger.warning(f"top-{topk}gating will use random token drop")
        # alway use fp32
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = topk
        self.jitter_eps = jitter_eps
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.use_fused_gating = use_fused_gating
        self.enable_token_rearrange_opt = enable_token_rearrange_opt
        self.use_tutel = use_tutel
        self.laux_allreduce = laux_allreduce

        # moe monitor
        moe_monitor_cfg = gpc.config.moe_monitor
        self.monitor_internval = moe_monitor_cfg.get("interval_steps", 10)
        self.gates_max_enable = moe_monitor_cfg.get("gates_max", False)
        self.drop_ratio_enable = moe_monitor_cfg.get("drop_ratio", False)

    def forward(
        self, inputs: torch.Tensor, used_token: torch.Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if self.wall_clock_breakdown:
            timer("TopKGate").start()

        # input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(inputs, epsilon=self.jitter_eps, device=inputs.device)
        logits = self.wg(inputs)

        if self.training:
            z_loss = torch.logsumexp(logits, dim=1).square().mean()
        else:
            z_loss = None

        gates_max = None
        if self.use_fused_gating or self.k > 2:
            gate_output = fused_topkgating(
                logits,
                self.k,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.use_rts,
                self.laux_allreduce,
                self.enable_token_rearrange_opt,
                self.use_tutel,
                self.training,
                gates_max_enable=(self.gates_max_enable and gpc.config.batch_count % self.monitor_internval == 0),
            )
            if self.enable_token_rearrange_opt:
                l_aux, topk_gating_token_infos, gates_max = gate_output
            else:
                l_aux, combine_weights, dispatch_mask, gates_max = gate_output  # pylint: disable=W0632
        # deepspeed-style code
        elif self.k == 1:
            gate_output = top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
                self.use_rts,
                self.laux_allreduce,
                self.training,
            )
            l_aux, combine_weights, dispatch_mask = gate_output
        elif self.k == 2:
            gate_output = top2gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                self.laux_allreduce,
                self.training,
            )
            l_aux, combine_weights, dispatch_mask = gate_output
        else:
            assert False, "Unsupported gating policy"

        if self.wall_clock_breakdown:
            timer("TopKGate").stop()
            self.gate_time = timer("TopKGate").elapsed(reset=False)

        if (
            not self.enable_token_rearrange_opt
            and self.drop_ratio_enable
            and gpc.config.batch_count % self.monitor_internval == 0
        ):
            if used_token is not None:
                drop_ratio = 1.0 - dispatch_mask.sum().detach() / self.k / used_token.sum().detach()
            else:
                drop_ratio = 1.0 - dispatch_mask.sum().detach() / self.k / dispatch_mask.size(0)
        else:
            drop_ratio = None

        if self.enable_token_rearrange_opt and (self.use_fused_gating or self.k > 2):
            return l_aux, topk_gating_token_infos, z_loss, gates_max, drop_ratio
        return l_aux, combine_weights, dispatch_mask, z_loss, gates_max, drop_ratio


class InternVitGShardMoELayer(BaseMoELayer):
    """MoELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MoELayer(gate, expert)
        output = moe(inputs)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 4,
        noisy_gate_policy: str = None,
        moe_jitter_eps: float = 0.0,
        drop_tokens: bool = True,
        use_rts: bool = False,
        use_fused_gating: bool = True,
        enable_token_rearrange_opt: bool = False,
        use_grouped_mlp: bool = False,
        use_tutel: bool = False,
        laux_allreduce: str = "local",
    ) -> None:
        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample", "RSample_before"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )
        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"

        if use_grouped_mlp:
            experts = new_feed_forward(
                in_features,
                hidden_features,
                out_features,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                activation_type=activation_type,
                is_expert=True,
                use_grouped_mlp=True,
                num_groups=num_experts // ep_size,
                backend="bmm",
            )
        else:
            experts = torch.nn.ModuleList(
                [
                    new_feed_forward(
                        in_features,
                        hidden_features,
                        out_features,
                        bias=False,
                        device=device,
                        dtype=dtype,
                        mlp_layer_fusion=mlp_layer_fusion,
                        multiple_of=multiple_of,
                        activation_type=activation_type,
                        is_expert=True,
                    )
                    for _ in range(num_experts // ep_size)
                ]
            )

        gate = TopKGate(
            in_features,
            num_experts,
            top_k,
            capacity_factor,
            eval_capacity_factor,
            min_capacity,
            noisy_gate_policy,
            moe_jitter_eps,
            drop_tokens,
            use_rts,
            use_fused_gating,
            enable_token_rearrange_opt,
            use_tutel,
            laux_allreduce,
        )

        super().__init__(gate, experts, ep_group, ep_size, num_experts // ep_size)

        self.use_grouped_mlp = use_grouped_mlp

        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.wall_clock_breakdown = False
        self.enable_token_rearrange_opt = enable_token_rearrange_opt
        self.num_experts = num_experts
        self.topk = top_k

        if moe_jitter_eps > 0:
            self.add_multiplicative_jitter(moe_jitter_eps)

    def add_multiplicative_jitter(self, jitter_epsilon):
        expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)
        generator = torch.Generator(device=get_current_device())
        generator.manual_seed(expp_rank)  # 保证相同的expert rank加的jitter是一致的

        for expert in self.experts.wrapped_experts:
            for name, param in expert.named_parameters():
                if "weight" in name:
                    param.data = apply_jitter(param.data, jitter_epsilon, generator)
                if "bias" in name:
                    param.data = apply_jitter(param.data, jitter_epsilon, generator)

    def forward(self, *inputs: Tensor) -> Tensor:
        if self.wall_clock_breakdown:
            timer("moe").start()

        hidden_states = inputs[0]
        used_token = inputs[2]
        # Implement Algorithm 2 from GShard paper.
        d_model = hidden_states.shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = hidden_states.reshape(-1, d_model)

        if not self.enable_token_rearrange_opt:
            l_aux, combine_weights, dispatch_mask, z_loss, gates_max, drop_ratio = self.gate(
                reshaped_inputs, used_token
            )
            dispatched_inputs = einsum(
                "sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_inputs
            )  # TODO: heavy memory usage due to long sequence length
        else:
            l_aux, token_rearrange_infos, z_loss, gates_max, drop_ratio = self.gate(reshaped_inputs, used_token)
            org_dtype = reshaped_inputs.dtype
            if org_dtype == torch.bfloat16:  # avoid precision missing
                rearranged_input = torch.index_select(
                    reshaped_inputs.to(torch.float32), dim=0, index=token_rearrange_infos.expert_select_token_idx
                ).to(org_dtype)
            else:
                rearranged_input = torch.index_select(
                    reshaped_inputs, dim=0, index=token_rearrange_infos.expert_select_token_idx
                )
            capacity = token_rearrange_infos.expert_select_token_idx.size(0) // self.num_experts
            dispatched_inputs = rearranged_input.reshape(self.num_experts, capacity, d_model).contiguous()
        if self.wall_clock_breakdown:
            timer("falltoall").start()

        if gpc.get_world_size(ParallelMode.EXPERT) > 1:
            dispatched_inputs, _ = all_to_all(dispatched_inputs, group=self.ep_group)

        if self.wall_clock_breakdown:
            timer("falltoall").stop()
            self.time_falltoall = timer("falltoall").elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_inputs = dispatched_inputs.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        if self.use_grouped_mlp:
            # (g,e,c,m) -> (e, g*c, m)
            dispatched_inputs = (
                dispatched_inputs.transpose(0, 1).reshape(self.num_local_experts, -1, d_model).contiguous()
            )

        expert_output = self.experts(dispatched_inputs, split_dim=1)

        if self.use_grouped_mlp:
            # (e, g*c, m) -> (e, g, c, m) -> (g, e, c, m)
            expert_output = (
                expert_output.reshape(self.num_local_experts, self.ep_size, -1, d_model).transpose(0, 1).contiguous()
            )

        if self.wall_clock_breakdown:
            timer("salltoall").start()

        if gpc.get_world_size(ParallelMode.EXPERT) > 1:
            expert_output, _ = all_to_all(expert_output, group=self.ep_group)

        if self.wall_clock_breakdown:
            timer("salltoall").stop()
            self.time_salltoall = timer("salltoall").elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if not self.enable_token_rearrange_opt:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)
        else:
            E, C, M = expert_output.shape
            org_dtype = expert_output.dtype
            if org_dtype == torch.bfloat16:
                valid_expert_out = torch.index_select(
                    expert_output.view(E * C, M).to(torch.float32),
                    dim=0,
                    index=token_rearrange_infos.token_rearranged_ec_idx,
                ).to(org_dtype)
            else:
                valid_expert_out = torch.index_select(
                    expert_output.view(E * C, M), dim=0, index=token_rearrange_infos.token_rearranged_ec_idx
                )
            combined_output = valid_expert_out * token_rearrange_infos.token_exp_weights.unsqueeze(1).type_as(
                hidden_states
            )
            if self.topk > 1:
                combined_output = combined_output.reshape(self.topk, -1, M)
                combined_output = torch.sum(combined_output, dim=0)

        out = combined_output.reshape(hidden_states.shape)

        if self.wall_clock_breakdown:
            timer("moe").stop()
            self.time_moe = timer("moe").elapsed(reset=False)

        outputs = (out, l_aux, z_loss)
        if gates_max is not None:
            outputs += (gates_max,)
        if drop_ratio is not None:
            outputs += (drop_ratio,)
        return outputs
