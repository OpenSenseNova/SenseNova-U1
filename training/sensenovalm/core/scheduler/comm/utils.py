# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0); originally adapted from
# ColossalAI (HPC-AI Tech, Apache-2.0).
# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/communication

from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.common import get_current_device

TensorShape = Union[torch.Size, List[int], Tuple[int]]

_dtype_to_int = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.half: 3,
    torch.int8: 4,
    torch.int16: 5,
    torch.int32: 6,
    torch.int: 7,
    torch.int64: 8,
    torch.bool: 9,
}

_int_to_dtype = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.half,
    4: torch.int8,
    5: torch.int16,
    6: torch.int32,
    7: torch.int,
    8: torch.int64,
    9: torch.bool,
}


def dtype_to_int(dtype: torch.dtype):
    assert dtype in _dtype_to_int
    return _dtype_to_int[dtype]


def int_to_dtype(value: int):
    assert value in _int_to_dtype
    return _int_to_dtype[value]


def send_meta_helper(obj, next_rank, tensor_kwargs):
    _dtype_int = dtype_to_int(obj.dtype)
    send_shape = torch.tensor(obj.size(), **tensor_kwargs)
    send_ndims = torch.tensor(len(obj.size()), **tensor_kwargs)
    send_dtype = torch.tensor(_dtype_int, **tensor_kwargs)
    send_requires_grad = torch.tensor(int(obj.requires_grad), **tensor_kwargs)
    dist.send(send_ndims, next_rank)
    dist.send(send_shape, next_rank)
    dist.send(send_dtype, next_rank)
    dist.send(send_requires_grad, next_rank)


def send_obj_meta(obj, next_rank=None):
    """Sends obj meta information before sending a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be sent before communications. This function
    synchronizes with :func:`recv_obj_meta`.

    Args:
        obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): obj to be sent.
        need_meta (bool, optional): If False, meta information won't be sent.
        next_rank (int): The rank of the next member in pipeline parallel group.

    Returns:
        bool: False
    """
    if next_rank is None:
        next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
    if isinstance(obj, torch.Tensor):
        send_obj_nums = torch.tensor(1, **tensor_kwargs)
        dist.send(send_obj_nums, next_rank)
        send_meta_helper(obj, next_rank, tensor_kwargs)
    else:
        send_obj_nums = torch.tensor(len(obj), **tensor_kwargs)
        dist.send(send_obj_nums, next_rank)
        for tensor_to_send in obj:
            send_meta_helper(tensor_to_send, next_rank, tensor_kwargs)


def recv_meta_helper(prev_rank, tensor_kwargs):
    recv_ndims = torch.empty((), **tensor_kwargs)
    dist.recv(recv_ndims, prev_rank)
    recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
    dist.recv(recv_shape, prev_rank)
    recv_dtype = torch.empty((), **tensor_kwargs)
    dist.recv(recv_dtype, prev_rank)
    recv_requires_grad = torch.empty((), **tensor_kwargs)
    dist.recv(recv_requires_grad, prev_rank)
    return torch.Size(recv_shape), int_to_dtype(recv_dtype.item()), bool(recv_requires_grad.item())


def recv_obj_meta(prev_rank=None) -> torch.Size:
    """Receives obj meta information before receiving a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be received before communications. This function
    synchronizes with :func:`send_obj_meta`.

    Args:
        obj_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the obj to be received.
        prev_rank (int): The rank of the source of the obj.

    Returns:
        Union[:class:`torch.Size`, List[:class:`torch.Size`]]: The shape of the obj to be received.
    """
    if prev_rank is None:
        prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    tensor_kwargs = {"dtype": torch.long, "device": get_current_device()}
    recv_obj_nums = torch.empty((), **tensor_kwargs)
    dist.recv(recv_obj_nums, prev_rank)
    if recv_obj_nums.item() == 1:
        obj_shape, obj_dtype, obj_requires_grad = recv_meta_helper(prev_rank, tensor_kwargs)
    else:
        obj_shape, obj_dtype, obj_requires_grad = [], [], []
        for _ in range(recv_obj_nums.item()):
            recv_shape, recv_dtype, recv_requires_grad = recv_meta_helper(prev_rank, tensor_kwargs)
            obj_shape.append(recv_shape)
            obj_dtype.append(recv_dtype)
            obj_requires_grad.append(recv_requires_grad)

    return obj_shape, obj_dtype, obj_requires_grad


def split_tensor_into_1d_equal_chunks(tensor: torch.Tensor, new_buffer=False) -> torch.Tensor:
    """Break a tensor into equal 1D chunks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be split before communication.
        new_buffer (bool, optional): Whether to use a new buffer to store sliced tensor.

    Returns:
        :class:`torch.Tensor`: The split tensor
    """
    partition_size = torch.numel(tensor) // gpc.get_world_size(ParallelMode.TENSOR)
    start_index = partition_size * gpc.get_local_rank(ParallelMode.TENSOR)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(partition_size, dtype=tensor.dtype, device=get_current_device(), requires_grad=False)
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered after communication.
    Returns:
        :class:`torch.Tensor`: The gathered tensor.
    """
    world_size = gpc.get_world_size(ParallelMode.TENSOR)
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype, device=get_current_device(), requires_grad=False)
    chunks = [gathered[i * numel : (i + 1) * numel] for i in range(world_size)]
    dist.all_gather(chunks, tensor, group=gpc.get_group(ParallelMode.TENSOR))
    return gathered
