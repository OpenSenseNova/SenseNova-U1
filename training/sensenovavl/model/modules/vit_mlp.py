#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# -*- encoding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.modules.linear import new_linear
from sensenovalm.utils.common import get_current_device


class VisionGroupedFeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.
    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        mlp_layer_fusion (Optional[Bool]):  Some linears without bias in FFN can be fused to reduce the comm cost of SP.
        activation_type (str): the activation function used for feed forward, "swiglu" by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        bias: bool = True,
        multiple_of: int = 256,
        activation_type: str = "gelu",
        num_groups: int = 1,
        backend: str = "bmm",
        is_expert: bool = False,
    ):
        super().__init__()

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.act = ACT2FN[activation_type]
        self.fc1 = new_linear(
            "grouped_w1",
            in_features,
            hidden_features,
            bias,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
            num_groups=num_groups,
            backend=backend,
            is_expert=is_expert,
        )
        self.fc2 = new_linear(
            "grouped_w2",
            hidden_features,
            out_features,
            bias,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
            num_groups=num_groups,
            backend=backend,
            is_expert=is_expert,
        )

    def forward(self, x, batch_sizes=None):
        hidden_states = self.fc1(x, batch_sizes)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states, batch_sizes)
        return hidden_states


class VisionFeedForward(nn.Module):
    """
    Vision FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of: int = 256,
        activation_type: str = "gelu",
        is_expert: bool = False,
    ):
        super().__init__()
        self.tp_mode = gpc.config.parallel.tensor.mode
        self.act = ACT2FN[activation_type]
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.fc1 = new_linear(
            "w1",
            in_features,
            hidden_features,
            bias=bias,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
            is_expert=is_expert,
        )
        self.fc2 = new_linear(
            "w2",
            hidden_features,
            out_features,
            bias=bias,
            device=get_current_device(),
            dtype=gpc.config.model.dtype,
            is_expert=is_expert,
        )

        self.is_expert = is_expert
        self.flag = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)

        # NOTICE
        if (
            not self.is_expert
            and self.tp_mode == "isp"
            and hidden_states.shape[1] == 512
            and gpc.config.parallel.tensor.size > 1
        ):
            self.flag = True
            hidden_states = F.pad(hidden_states, (0, 0, 0, 1), "constant", 0)
        hidden_states = self.fc2(hidden_states)

        if (
            self.tp_mode == "isp"
            and hidden_states.shape[1] == 513
            and self.flag is True
            and gpc.config.parallel.tensor.size > 1
        ):
            hidden_states = hidden_states[:, :512, :]
        return hidden_states


def new_feed_forward(
    in_features: int,
    hidden_features: int,
    out_features: int,
    bias: bool = True,
    multiple_of: int = 256,
    activation_type: str = "swiglu",
    is_expert: bool = False,
    use_grouped_mlp: bool = False,
    **kwargs,
):
    if use_grouped_mlp:
        num_groups = kwargs.pop("num_groups", 1)
        backend = kwargs.pop("backend", "bmm")
        return VisionGroupedFeedForward(
            in_features,
            hidden_features,
            out_features,
            bias,
            multiple_of,
            activation_type,
            num_groups=num_groups,
            backend=backend,
            is_expert=is_expert,
        )
    return VisionFeedForward(
        in_features,
        hidden_features,
        out_features,
        bias,
        multiple_of,
        activation_type,
        is_expert=is_expert,
    )
