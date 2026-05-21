# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
layer norm modules
"""

import inspect
from typing import List, Union

import torch
from torch import nn

from sensenovalm.model.ops.norm import RMSNorm

Shape = Union[int, List[int], torch.Size]


def new_layer_norm(norm_type: str, normalized_shape: Shape, eps: float = 1e-5, add_unit_offset=False):
    if norm_type == "rmsnorm":
        rmsnorm_params = inspect.signature(RMSNorm).parameters
        if "add_unit_offset" in rmsnorm_params:
            return RMSNorm(normalized_shape, eps, add_unit_offset)
        else:
            return RMSNorm(normalized_shape, eps)
    else:  # default: layernorm
        return nn.LayerNorm(normalized_shape, eps)
