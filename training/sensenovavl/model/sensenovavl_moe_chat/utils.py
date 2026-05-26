# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
import torch
import torch.nn.functional as F

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.comm.utils import _gather, _split


def get_split_size(num_chunks: int, div: int, mod: int):
    """
    The torch.split support split tensor unevenly.
    However, we should pass a 'split size' to it,
    This function is to get the 'split size'.
    For example:
    tensor with shape (2, 3), which could be splitted into two parts in dim=1
    the 'split size' should be pass to torch.split is [2, 1]

    Args:
    num_chunks: the number of chunks that should be splitted into
    div: dim_shape / num_chunks = div
    mod: dim_shape % num_chunks = mod

    """
    split_size = []
    for i in range(num_chunks):
        if i < mod:
            split_size.append(div + bool(mod))
        else:
            split_size.append(div)

    return split_size


def before_gather(input_, parallel_mode, mod, padding):
    """
    input_ shape: (b, s, 13, d) and (b, s, 12, d)
    before conduct gather communication,
    (b, s, 12, d) should be padding to (b, s, 13, d)
    """
    if gpc.get_local_rank(parallel_mode) >= mod:
        input_ = F.pad(input_, padding, "constant", 0)
    return input_


def after_gather(input_, parallel_mode, mod, dim):
    """
    input_ shape: (b, s, 26, d)
    after gather, we should delete the padding head
    so that input_ shape becomes (b, s, 25, d)
    """
    tp_size = gpc.get_world_size(parallel_mode)
    input_list = list(torch.chunk(input_, tp_size, dim))
    indices = [slice(None)] * input_.ndim
    indices[dim] = slice(0, -1)
    for i in range(len(input_list)):
        if i >= mod:
            input_list[i] = input_list[i][tuple(indices)].clone()
    input_ = torch.cat(input_list, dim=dim)
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
    def forward(ctx, input_, parallel_mode, dim, div, mod, split_size):  # pylint: disable=W0613
        ctx.mode = parallel_mode
        ctx.dim = dim
        ctx.mod_head = mod
        ctx.split_size = split_size

        # if the head could be even split
        if mod == 0:
            return _gather(input_.contiguous(), parallel_mode, dim)

        # for those ranks who have less heads, we should to padding it before communicate.
        # after communicating, we should delete the padding head.
        # padding is used in F.pad,
        # the use function could be seen in https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        padding = tuple([0] * (input_.ndim - dim - 1) * 2 + [0, 1])
        input_ = before_gather(input_, parallel_mode, mod, padding)
        input_ = _gather(input_.contiguous(), parallel_mode, dim)
        input_ = after_gather(input_, parallel_mode, mod, dim)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mod_head == 0:
            return _split(grad_output.contiguous(), ctx.mode, ctx.dim), None, None, None, None, None
        # if the grad_output could not be splitted evenly, we should pass the split size
        grad_output = _split(grad_output.contiguous(), ctx.mode, ctx.dim, ctx.split_size)
        return grad_output, None, None, None, None, None


def gather_forward_split_backward(input_, parallel_mode, dim, div=0, mod=0, split_size=None):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim, div, mod, split_size)


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
    def forward(ctx, input_, parallel_mode, dim, div, mod, split_size):  # pylint: disable=W0613
        ctx.mode = parallel_mode
        ctx.dim = dim
        ctx.mod = mod

        # if the head could be even split
        if mod == 0:
            return _split(input_.contiguous(), parallel_mode, dim)

        # for ranks who have more head, we extract these heads before communicating
        # after communication, the more head could be instered to original position.

        input_ = _split(input_.contiguous(), parallel_mode, dim, split_size)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mod == 0:
            return _gather(grad_output.contiguous(), ctx.mode, ctx.dim), None, None, None, None, None
        padding = tuple([0] * (grad_output.ndim - ctx.dim - 1) * 2 + [0, 1])
        grad_output = before_gather(grad_output, ctx.mode, ctx.mod, padding)
        grad_output = _gather(grad_output.contiguous(), ctx.mode, ctx.dim)
        grad_output = after_gather(grad_output, ctx.mode, ctx.mod, ctx.dim)
        return grad_output, None, None, None, None, None


def split_forward_gather_backward(input_, parallel_mode, dim, div=0, mod=0, split_size=None):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim, div, mod, split_size)


class _EnableMemoryPoolForLlm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):  # pylint: disable=W0613
        gpc.llm_enable_memory = True
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=W0613
        gpc.llm_enable_memory = False
        return grad_output


def enable_memorypool_llm(input_):
    return _EnableMemoryPoolForLlm.apply(input_)


def uneven_all2all_gather_split(input_, gather_size, split_size, gather_dim, split_dim, tp_size, is_clone=False):
    if is_clone is True:
        input_ = input_.clone()

    # all_gather
    div_gather = gather_size // tp_size
    mod_gather = gather_size % tp_size
    uneven_split_size = get_split_size(tp_size, div_gather, mod_gather)
    input_ = gather_forward_split_backward(
        input_, ParallelMode.TENSOR, dim=gather_dim, div=div_gather, mod=mod_gather, split_size=uneven_split_size
    )

    # split
    div_split = split_size // tp_size
    mod_split = split_size % tp_size
    uneven_split_size = get_split_size(tp_size, div_split, mod_split)
    input_ = split_forward_gather_backward(
        input_, ParallelMode.TENSOR, dim=split_dim, div=div_split, mod=mod_split, split_size=uneven_split_size
    )

    return input_
