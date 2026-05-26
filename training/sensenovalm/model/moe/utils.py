# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# AllToAll autograd Function based on https://github.com/pytorch/pytorch/pull/40762
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device


class AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: torch.distributed.ProcessGroup = None,
        async_op=False,
    ) -> Tensor:  # type: ignore

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return inputs, None

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        )
        handle = torch.distributed.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        # if async_op=False, handle will be None
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, _) -> Tuple[None, Tensor]:
        if ctx.needs_input_grad[0]:
            # Bypass the function if we are using only 1 GPU.
            world_size = torch.distributed.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            torch.distributed.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group, async_op)


class moe_gather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(input_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class moe_scatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_
        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def _gather_along_first_dim_moe(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = gpc.get_group(ParallelMode.EXPERT)
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = gpc.get_group(ParallelMode.EXPERT)
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_):  # pylint: disable=W0613
        """Symbolic function for tracing."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):  # pylint: disable=W0613
        """Forward function."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        """Backward function."""
        return _reduce_scatter_along_first_dim_moe(grad_output), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):  # pylint: disable=W0613
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):  # pylint: disable=W0613
        """Forward function."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        """Backward function."""
        return _gather_along_first_dim_moe(grad_output), None


def gather_from_parallel_region_to_moe(input_):
    """Wrapper for autograd function"""
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)


def reduce_scatter_to_parallel_region_from_moe(input_):
    """Wrapper for autograd function"""
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)


class LayerScale(torch.nn.Module):
    """
    Layer scale module
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(self.gamma), self.gamma.mean()
        else:
            return x * self.gamma, self.gamma.mean()


def gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = gpc.get_group(ParallelMode.EXPERT)
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())

    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


@dataclass
class SenseNovaVLMoEOutput:
    moe_loss: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    routed_coef_loss: torch.Tensor | None = None
    routed_coef: float | None = None
    gates_max: torch.Tensor | None = None
    drop_ratio: torch.Tensor | None = None
