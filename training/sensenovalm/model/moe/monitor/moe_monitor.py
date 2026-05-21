# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.moe.utils import gather_along_first_dim_expert_parallel
from sensenovalm.utils.logger import get_logger

from .base_monitor import BaseMonitor

sensenovalm_accelerator = get_accelerator()
logger = get_logger(__file__)


class MoEMonitor(BaseMonitor):
    """
    ExpertMonitor class for monitoring MoE experts.
    """

    def __init__(self, ep_size: int, num_experts: int, topk: int, moe_monitor_cfg: Dict):
        """
        Initializes the ExpertMonitor with parameters for expert profiling and monitoring.

        :param ep_size: Size of the expert parallel dimension.
        :param num_experts: Number of experts in the model.
        :param topk: Top-k routing parameter.
        :param moe_monitor_cfg: Configuration dictionary for monitoring experts.
        """
        super().__init__()
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.topk = topk
        self.tokens_above_avg_monitor = moe_monitor_cfg.get("tokens_above_avg", False)
        self.expert_activation_monitor = moe_monitor_cfg.get("expert_activation", False)
        self.get_logit_before_gate = moe_monitor_cfg.get("logit_before_gate", False)
        self.interval_steps = moe_monitor_cfg.get("interval_steps", 100)

        self.tokens_above_avg_max = None
        self.tokens_above_avg_min = None
        self.expert_activation = None

        if gpc.is_rank_for_log():
            logger.info(f"ExpertMonitor initialized with ep_size={ep_size}, num_experts={num_experts}")

    def expert_profiling(self, tokens_per_expert_before_capacity, num_tokens_per_expert=None) -> None:
        batch_count = gpc.config.get("batch_count")
        if self.tokens_above_avg_monitor and batch_count % self.interval_steps == 0:
            with torch.no_grad():
                if tokens_per_expert_before_capacity is not None:
                    if self.ep_size > 1:
                        num_global_tokens_per_expert_before_capacity = (
                            gather_along_first_dim_expert_parallel(tokens_per_expert_before_capacity)
                            .reshape(self.ep_size, self.num_experts)
                            .sum(axis=0)
                        )
                    else:
                        num_global_tokens_per_expert_before_capacity = tokens_per_expert_before_capacity
                elif num_tokens_per_expert is not None:
                    num_global_tokens_per_expert_before_capacity = num_tokens_per_expert
                else:
                    raise ValueError(
                        "Either tokens_per_expert_before_capacity or num_tokens_per_expert must be provided"
                    )
                tokens_avg_global = num_global_tokens_per_expert_before_capacity.float().mean()
                tokens_above_avg = num_global_tokens_per_expert_before_capacity - tokens_avg_global

                self.tokens_above_avg_max = tokens_above_avg.clone()
                self.tokens_above_avg_min = tokens_above_avg.clone()
            if gpc.is_initialized(ParallelMode.EXPERT_WEIGHT):
                dist.all_reduce(
                    self.tokens_above_avg_max,
                    group=gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                    op=dist.ReduceOp.MAX,
                    async_op=False,
                )
                dist.all_reduce(
                    self.tokens_above_avg_min,
                    group=gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                    op=dist.ReduceOp.MIN,
                    async_op=False,
                )

            self.handles.append(
                dist.all_reduce(
                    self.tokens_above_avg_max,
                    group=gpc.get_group(ParallelMode.EXPERT_DATA),
                    op=dist.ReduceOp.MAX,
                    async_op=True,
                )
            )
            self.handles.append(
                dist.all_reduce(
                    self.tokens_above_avg_min,
                    group=gpc.get_group(ParallelMode.EXPERT_DATA),
                    op=dist.ReduceOp.MIN,
                    async_op=True,
                )
            )

        if self.expert_activation_monitor and batch_count % self.interval_steps == 0:
            with torch.no_grad():
                self.expert_activation = (num_tokens_per_expert > 0).int()
            if gpc.is_initialized(ParallelMode.EXPERT_WEIGHT):
                dist.all_reduce(
                    self.expert_activation,
                    group=gpc.get_group(ParallelMode.EXPERT_WEIGHT),
                    op=dist.ReduceOp.SUM,
                    async_op=False,
                )
            self.handles.append(
                dist.all_reduce(
                    self.expert_activation,
                    group=gpc.get_group(ParallelMode.EXPERT_DATA),
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
            )

    def expert_profiling_wait(self):
        batch_count = gpc.config.get("batch_count")
        if batch_count % self.interval_steps == 0:
            if self.tokens_above_avg_monitor:
                if self.handles:
                    self.clear_handles()
                gpc.metric["tokens_above_avg_max"].append(self.tokens_above_avg_max)
                gpc.metric["tokens_above_avg_min"].append(self.tokens_above_avg_min)
            if self.expert_activation_monitor:
                if self.handles:
                    self.clear_handles()
                gpc.metric["expert_activation"].append(self.expert_activation)

    def pre_gate_profiling(self, inputs: Tensor):
        if self.get_logit_before_gate and gpc.config.get("batch_count") % self.interval_steps == 0:
            with torch.no_grad():
                gpc.metric["logit_before_gate_max"].append(inputs.max())
                gpc.metric["logit_before_gate_min"].append(inputs.min())
                gpc.metric["logit_before_gate_mean"].append(inputs.mean())
