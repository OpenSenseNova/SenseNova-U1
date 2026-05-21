#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from sensenovalm.core.context import global_context as gpc


class AsyncCommHandle(ABC):
    """A interface for asynchronous communication handles."""

    @abstractmethod
    def wait(self) -> None:
        """wait asynchronous communication to complete."""


class DummyAsyncCommHandle(AsyncCommHandle):
    """A fake communication handle used to maintain consistency in code writing"""

    def wait(self) -> None:
        pass


DUMMY_HANDLE_CONST = DummyAsyncCommHandle()


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup, reduce_dim: int = 0) -> Tensor:
        ctx.process_group = process_group
        ctx.reduce_dim = reduce_dim
        output, _ = reduce_scatter_raw(input_, process_group, reduce_dim=reduce_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        gather_dim = ctx.reduce_dim
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group, gather_dim=gather_dim)
        return grad_input, None, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        _ = ctx  # avoid lint warning W0613
        return grad_output, None


# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply


def _split(input_, parallel_mode, dim=-1, split_size=None):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    if split_size is None:
        assert dim_size % world_size == 0, (
            f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
            f"cannot split tensor evenly"
        )
        # It means the tensor will be splitted evenly.
        split_size = dim_size // world_size
    else:
        assert isinstance(split_size, list)
        assert sum(split_size) == dim_size, "The sum of split size should be equal to the dimension size."

    tensor_list = torch.split(input_, split_size, dim=dim)
    rank = gpc.get_local_rank(parallel_mode)
    output = tensor_list[rank].contiguous()
    output = output.detach().clone()

    return output


def _gather(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # all gather
    rank = gpc.get_local_rank(parallel_mode)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    # NOTE: Do NOT alias tensor_list[rank] = input_ here.
    # dist.all_gather writes to all output slots, including tensor_list[rank].
    # If it aliases input_, this is an in-place modification that breaks
    # autograd version tracking for any tensor saved for backward.
    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    dist.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _reduce(input_, parallel_mode):
    # skip if only one rank involved
    if gpc.get_world_size(parallel_mode) == 1:
        return input_

    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    dist.all_reduce(input_, group=group)

    return input_


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _gather(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _gather(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def gather_forward_split_backward(input_, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim)


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _split(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _split(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


def split_forward_gather_backward(input_, parallel_mode, dim):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim)


class _ReduceForward(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def symbolic(input_):
        return _reduce(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode):  # pylint: disable=W0613
        return _reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        return grad_output, None


def reduce_forward(input_, parallel_mode):
    return _ReduceForward.apply(input_, parallel_mode)


def all_gather_raw(
    input_: Tensor,
    process_group: ProcessGroup,
    async_op: bool = False,
    gather_dim: int = 0,
):
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return input_, None

    if gather_dim == 0 or (gather_dim == 1 and input_.shape[0] == 1):
        shape = list(input_.shape)
        shape[gather_dim] = shape[gather_dim] * world_size
        output = torch.empty(shape, dtype=input_.dtype, device=input_.device)

        handle = dist.all_gather_into_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    else:
        assert async_op is False, "gather_dim!=0 and async_op=True is not supported"
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        handle = dist.all_gather(gather_list, input_, group=process_group, async_op=async_op)
        output = torch.cat(gather_list, dim=gather_dim).contiguous()
    return output, handle


def reduce_scatter_raw(
    input_: Tensor,
    process_group: ProcessGroup,
    op=dist.ReduceOp.SUM,
    async_op: bool = False,
    reduce_dim: int = 0,
):
    world_size = dist.get_world_size(process_group)
    assert input_.shape[reduce_dim] % world_size == 0

    if world_size <= 1:
        return input_, None

    shape_list = list(input_.shape)
    shape_list[reduce_dim] = shape_list[reduce_dim] // world_size

    output = torch.empty(
        shape_list,
        dtype=input_.dtype,
        device=input_.device,
    ).contiguous()

    if reduce_dim == 0 or (reduce_dim == 1 and input_.shape[0] == 1):
        handle = dist.reduce_scatter_tensor(output, input_.contiguous(), op=op, group=process_group, async_op=async_op)
    else:
        input_list = [input_i.contiguous() for input_i in input_.chunk(world_size, dim=reduce_dim)]
        handle = dist.reduce_scatter(output, input_list, op=op, group=process_group, async_op=async_op)
    return output, handle


def apply_to_tensors_only(function, value):
    """
    Apply `function` to every Tensor in `value`.

    Args:
        functional: The function class to apply.
        value (Any): Target object to apply `function` to.

    Returns:
        Any: Output of `function`.
    """
    if isinstance(value, (tuple, list)):
        touched_outputs = []
        for elem in value:
            touched_output = apply_to_tensors_only(function, elem)
            touched_outputs.append(touched_output)

        return value.__class__(touched_outputs)
    elif isinstance(value, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in value.keys():
            value[key] = apply_to_tensors_only(function, value[key])
        return value

    elif isinstance(value, torch.Tensor):
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_output = function(value)

        return touched_output
    else:
        return value


class _ExpandKVPackedFunction(torch.autograd.Function):
    """
    Copy the KV head repeat times to support sequence parallel.

    Args:
        kv: input kv.
        repeat_times: the repeat number of each head.
        num_head_dim: the dimension of head number.
    """

    @staticmethod
    def forward(ctx, kv, repeat_times, num_head_dim):

        kv_shape = kv.shape
        num_heads_kv = kv_shape[num_head_dim]

        ctx.num_head_dim = num_head_dim
        ctx.num_heads_kv = num_heads_kv

        # here we construct a repeat index to indicate which dim should copy
        repeat_index = [1] * kv.ndim
        repeat_index[num_head_dim] = repeat_times

        # split the kv into head num splits
        kv_splits = torch.chunk(kv, chunks=num_heads_kv, dim=num_head_dim)
        kv_repeats = []
        # for each split, we copy it to repeat_times copys.
        for kv_split in kv_splits:
            kv_split_repeat = kv_split.repeat(repeat_index)
            kv_repeats.append(kv_split_repeat)

        # check the copy head whether is the same
        # res = torch.cat(kv_repeats, dim=num_head_dim)


        # last we concat these repeats on the num_head_dim dimension.
        return torch.cat(kv_repeats, dim=num_head_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        For backward, we sum the copy head inside a query group.
        """

        num_head_dim = ctx.num_head_dim
        num_heads_kv = ctx.num_heads_kv

        # we split the grad into query groups splits.
        grad_output_splits = torch.chunk(grad_output, chunks=num_heads_kv, dim=num_head_dim)
        grad_output_sums = []
        # for each split, we sum the head
        for grad_output_split in grad_output_splits:
            grad_output_sum = grad_output_split.sum(dim=num_head_dim, keepdim=True)
            grad_output_sums.append(grad_output_sum)
        # then we concat the split sums on the num_head_dim dimension.
        return torch.cat(grad_output_sums, dim=num_head_dim), None, None


expandKVPacked = _ExpandKVPackedFunction.apply
