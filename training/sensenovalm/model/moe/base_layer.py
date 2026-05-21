# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0). MoE composition style
# follows DeepSpeed (Microsoft, Apache-2.0).
from typing import TYPE_CHECKING, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.moe.experts import Experts
from sensenovalm.utils.common import get_current_device

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


class BaseMoELayer(Base):
    """
    Base MoE Layer.
    """

    def __init__(
        self, gate: Module, experts: Union[Module, ModuleList], ep_group, ep_size: int, num_local_experts: int
    ) -> None:
        super().__init__()
        # for elastic expert paralle, experts may have multiple groups
        expert_group_name = f"moe_ep_size_{ep_size}"
        if expert_group_name not in gpc.expert_parallel_group_names:
            gpc.expert_parallel_group_names.append(expert_group_name)
        self.gate = gate
        self.experts = Experts(experts, num_local_experts)
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.l_aux = torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
        self.exp_counts = None

        for _, param in self.gate.named_parameters():
            param.is_gate = True

        for expert in self.experts.wrapped_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for _, param in expert.named_parameters():
                param.is_expert = True
                param.group_name = expert_group_name
