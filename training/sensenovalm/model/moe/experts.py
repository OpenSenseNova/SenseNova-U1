# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Originally adapted from DeepSpeed (Microsoft, Apache-2.0).
"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""

from typing import Union, cast

import torch
from torch.nn import Module, ModuleList


class Experts(torch.nn.Module):
    """
    Local Experts.
    """

    def __init__(self, experts: Union[Module, ModuleList], num_local_experts=1):
        """
        Encapsulating the moe experts.
        Args:
            experts: moe experts, can be Module or ModuleList
            num_local_experts: the number of experts in each device
            expert_group_name: the expert group name, can vary across different layers
        """
        super().__init__()

        if isinstance(experts, ModuleList):
            self.wrapped_experts = cast(ModuleList, experts)
        else:
            self.wrapped_experts = ModuleList([experts])
        self.num_local_experts = num_local_experts

    def forward(self, inputs, split_size_or_sections=None, split_dim=0, **kwargs):
        """
        Args:
            inputs: tokens to be processed in expert's forward pass
            split_size_or_sections: if not given, the input tokens are split into the same size, otherwise
                                    split_size_or_sections determines the number of tokens allocated to each
                                    expert, should be list(int)
            kwargs: args used for expert's forward pass other than input tokens
        """

        if len(self.wrapped_experts) == 1:
            return self.wrapped_experts[0](inputs, **kwargs)

        # The following code is designed for multiple experts.
        #   1. split tokens among split_dim
        #   2. do for-loop for experts's computation
        if split_size_or_sections is None:
            # chunk can be faster than split
            chunks = inputs.chunk(len(self.wrapped_experts), dim=split_dim)
        else:
            if isinstance(split_size_or_sections, torch.Tensor):
                split_size_or_sections = split_size_or_sections.tolist()
            chunks = inputs.split(split_size_or_sections, dim=split_dim)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.wrapped_experts):
            out = expert(chunk, **kwargs)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]
        expert_output = torch.cat(expert_outputs, dim=split_dim)
        return expert_output
