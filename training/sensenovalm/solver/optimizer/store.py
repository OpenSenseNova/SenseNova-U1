#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0). Parameter / gradient
# bucket layout follows DeepSpeed ZeRO partitioning conventions.
# -*- encoding: utf-8 -*-

from typing import List

import torch
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc


class BaseStore:
    """
    Base Store
    """

    def __init__(self, dp_parallel_mode=ParallelMode.DATA):
        self._world_size = gpc.get_world_size(dp_parallel_mode)
        self._local_rank = gpc.get_local_rank(dp_parallel_mode)

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_rank(self):
        return self._local_rank


class BucketStore(BaseStore):
    """
    Bucket Store
    """

    def __init__(self, group_id, dp_parallel_mode, dtype):
        super().__init__(dp_parallel_mode)
        self._grads = dict()
        self._params = dict()
        self._num_elements_in_bucket = dict()
        self._dp_parallel_mode = dp_parallel_mode
        self._group_id = group_id
        self.dtype = dtype

        self.reset()

    def num_elements_in_bucket(self, reduce_rank: int = None):
        return self._num_elements_in_bucket[reduce_rank]

    def num_params_in_bucket(self, reduce_rank: int = None):
        return len(self._params[reduce_rank])

    def get_param_group_id(self):
        return self._group_id

    def get_dp_parallel_mode(self):
        return self._dp_parallel_mode

    def add_num_elements_in_bucket(self, num_elements, reduce_rank: int = None):
        self._num_elements_in_bucket[reduce_rank] += num_elements

    def add_grad(self, tensor, reduce_rank: int = None):
        self._grads[reduce_rank].append(tensor)

    def add_param(self, tensor, reduce_rank: int = None):
        self._params[reduce_rank].append(tensor)

    def reset(self):
        keys = [None] + list(range(self._world_size))
        self._grads = {rank: [] for rank in keys}
        self._params = {rank: [] for rank in keys}
        self._num_elements_in_bucket = {rank: 0 for rank in keys}

    def reset_by_rank(self, reduce_rank=None):
        self._grads[reduce_rank] = []
        self._params[reduce_rank] = []
        self._num_elements_in_bucket[reduce_rank] = 0

    def get_grad(self, reduce_rank: int = None):
        return self._grads[reduce_rank]

    def get_param(self, reduce_rank: int = None):
        return self._params[reduce_rank]


class GradientStore(BaseStore):
    """
    Gradient Store
    """

    def __init__(self, *args):
        super().__init__(*args)
        # bookkeeping data structures
        self._averaged_gradients = dict()

        # for backward reduction hooks
        self._grad_acc_objs = []

    def add_accumulate_grad_object(self, obj):
        """
        Keep :class:`AccumulateGrad` objects. If these objects are not kept, reduction hooks may not
        be attached successfully.

        :param obj: An object of :class:`AccumulateGrad` class
        :type obj: :class:`AccumulateGrad`
        """

        self._grad_acc_objs.append(obj)

    def get_averaged_gradients_by_group(self, group_id: int) -> List[Tensor]:
        """
        Return average gradients of a parameter group

        :param group_id: The index of parameter group
        :type group_id: int

        :return: Return the list of averaged gradients of a parameter group. Each element is a gradient,
            not a parameter.
        :rtype: List[torch.Tensor]
        """

        return self._averaged_gradients[group_id]

    def add_average_gradient_by_group(self, group_id: int, tensor: Tensor) -> None:
        """
        Append an average gradient to the list of averaged gradients of a parameter group

        :param group_id: The index of a parameter group
        :param tensor: A :class:`torch.Tensor` object
        :type group_id: int
        :type tensor: torch.Tensor

        """

        if group_id in self._averaged_gradients:
            self._averaged_gradients[group_id].append(tensor)
        else:
            self._averaged_gradients[group_id] = [tensor]

    def reset_average_gradients_by_group(self, group_id: int) -> None:
        """
        Reset the bookkeeping data structure for averaged gradients to an empty list

        :param group_id: The index of a parameter group
        :type group_id: int
        """

        self._averaged_gradients[group_id] = []


class ParameterStore(BaseStore):
    """
    Parameter Store
    """

    def __init__(self, dp_paralle_mode):
        super().__init__(dp_paralle_mode)
        # param partitioning data structures
        self._fp16_param_to_rank = dict()
        self._rank_groupid_to_fp16_param_list = dict()
        self._rank_group_id_to_flat_fp16_param = dict()

        # param reduction data structures
        self._is_param_reduced = dict()
        self._reduced_param = []

        self._bucket_reduced_param = {}
        self._bucket_reduced_grad = {}

    def set_param_to_rank(self, tensor: Tensor, rank: int) -> None:
        """
        Set the mapping between parameter to rank, each parameter should be owned by a rank.

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        :param rank: The rank of which the process is responsible for updating the parameter
        :type rank: int
        """
        if tensor not in self._fp16_param_to_rank:
            self._fp16_param_to_rank[tensor] = []

        self._fp16_param_to_rank[tensor].append(rank)

    def get_param_rank(self, tensor: Tensor) -> int:
        """
        Gives the rank which the parameter belongs to

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        """
        return self._fp16_param_to_rank[tensor]

    def add_fp16_param_list_by_rank_group(self, rank, group_id, tensor_list) -> None:
        if rank not in self._rank_groupid_to_fp16_param_list:
            self._rank_groupid_to_fp16_param_list[rank] = dict()

        if group_id not in self._rank_groupid_to_fp16_param_list[rank]:
            self._rank_groupid_to_fp16_param_list[rank][group_id] = []

        self._rank_groupid_to_fp16_param_list[rank][group_id].extend(tensor_list)

    def get_fp16_params_by_rank_group(self, rank, group_id) -> List[Tensor]:
        return self._rank_groupid_to_fp16_param_list[rank][group_id]

    def add_flat_fp16_param_by_rank_group(self, rank, group_id, tensor) -> None:
        if rank not in self._rank_group_id_to_flat_fp16_param:
            self._rank_group_id_to_flat_fp16_param[rank] = dict()

        self._rank_group_id_to_flat_fp16_param[rank][group_id] = tensor

    def get_flat_fp16_param_by_rank_group(self, rank, group_id) -> Tensor:
        return self._rank_group_id_to_flat_fp16_param[rank][group_id]

    def is_param_reduced(self, tensor):
        return self._is_param_reduced[tensor]

    def set_param_reduction_state(self, tensor, state):
        self._is_param_reduced[tensor] = state

    def get_param_reduction_states(self):
        return self._is_param_reduced

    def reset_previous_reduced_params(self):
        self._reduced_param = []

    def add_previous_reduced_param(self, tensor):
        self._reduced_param.append(tensor)

    def add_reduced_param_for_compute_norm(self, param):
        group_id = getattr(param, "group_id")
        if group_id not in self._bucket_reduced_param:
            self._bucket_reduced_param[group_id] = []
            self._bucket_reduced_grad[group_id] = []

        self._bucket_reduced_param[group_id].append(param)
        self._bucket_reduced_grad[group_id].append(param.grad)

    def get_reduced_param_for_compute_norm(self, group_id=0):
        if group_id not in self._bucket_reduced_param:
            return [], []
        return (
            self._bucket_reduced_param[group_id],
            self._bucket_reduced_grad[group_id],
        )

    def reset_reduced_data_for_compute_norm(self):
        self._bucket_reduced_param = {}
        self._bucket_reduced_grad = {}

    def clear_grads_of_previous_reduced_params(self):
        if len(self._reduced_param) > 0:
            for param in self._reduced_param:
                param.grad = None
            self.reset_previous_reduced_params()


class TensorBucket:
    """
    Tensor Bucket
    """

    def __init__(self, size):
        self._max_size = size
        self._current_size = 0
        self._bucket = []
        self._flat_tensor = None
        self._unflatten_and_copy_flag = False
        self.commu_handle = None

    @property
    def max_size(self):
        return self._max_size

    @property
    def current_size(self):
        return self._current_size

    def is_full_or_oversized(self):
        return self._current_size >= self._max_size

    def is_empty(self):
        return len(self._bucket) == 0

    def set_unflatten_and_copy_flag(self, flag):
        self._unflatten_and_copy_flag = flag

    def get_unflatten_and_copy_flag(self):
        return self._unflatten_and_copy_flag

    def get_flat_tensor(self):
        return self._flat_tensor

    def add_to_bucket(self, tensor, allow_oversize=False):
        tensor_size = tensor.numel()

        if not allow_oversize and self.will_exceed_max_size(tensor_size):
            msg = f"The param bucket max size {self._max_size} is exceeded" + f"by tensor (size {tensor_size})"
            raise RuntimeError(msg)

        self._bucket.append(tensor)
        self._current_size += tensor_size

    def will_exceed_max_size(self, tensor_size):
        expected_size = self._current_size + tensor_size
        return expected_size > self._max_size

    def get_bucket(self):
        return self._bucket

    def empty(self):
        self._bucket = []
        self._size = 0
        self._flat_tensor = None
        self.commu_handle = None

    def flatten(self):
        self._flat_tensor = _flatten_dense_tensors(self._bucket)
        if gpc.config.reduce_comm_dtype == torch.float32:
            self._flat_tensor = self._flat_tensor.to(torch.float32)

    def unflatten_and_copy(self, dp_group_size: int = -1):
        if self._unflatten_and_copy_flag:
            if gpc.config.reduce_comm_dtype == torch.float32:
                self._flat_tensor = self._flat_tensor.to(gpc.config.model.dtype)
            unflattened_tensor_list = _unflatten_dense_tensors(self._flat_tensor, self._bucket)
            for old, new in zip(self._bucket, unflattened_tensor_list):
                old.copy_(new)
                if dp_group_size != -1:
                    old.div_(dp_group_size)


class BucketStore_v2(BaseStore):
    """
    Bucket Store V2
    """

    def __init__(self, group_id, dp_parallel_mode, zero_mode=ParallelMode.ZERO1):
        super().__init__(dp_parallel_mode)
        self.zero_world_size = gpc.get_world_size(zero_mode)
        self.zero_local_rank = gpc.get_local_rank(zero_mode)
        self._dp_parallel_mode = dp_parallel_mode
        self._group_id = group_id
        self.reset_all()

    def get_param_group_id(self):
        return self._group_id

    def get_dp_parallel_mode(self):
        return self._dp_parallel_mode

    def reset_all(self) -> None:
        # init
        self._num_elements_in_bucket = 0
        # mapping gradient slices and parameter
        self.grad_to_param_mapping = dict()

        self._grad_in_bucket = dict()

        self._grad_current_rank_for_group = dict()
        self._param_list_for_group = dict()
        self._padding_size_for_group = dict()
        self.grad_to_param_mapping2 = dict()
        self.offset_list_for_group = dict()

        self._param_list = []
        self._padding_size = []
        for rank in range(self.zero_world_size):
            self._grad_in_bucket[rank] = []

        # offset_list records number of tensors in the bucket before each reduction
        self.offset_list = [0]

    def num_elements_in_bucket(self) -> int:
        """Return the total number of elements in bucket

        Returns:
            int: the total number of elements in bucket
        """

        return self._num_elements_in_bucket

    def reset_num_elements_in_bucket(self):
        """Set the number of elements in bucket to zero."""

        self._num_elements_in_bucket = 0

    def add_param_grad(self, param: Tensor, padding_size: int):
        """Add a param to bucket and record the padding size of a param for gradient padding

        Args:
            group_id (int): The index of a parameter group
            param (Tensor): The parameter
            padding_size (int): The padding size of the parameter
        """

        self._param_list.append(param)
        self._padding_size.append(padding_size)
        self._num_elements_in_bucket += param.numel() + padding_size

        # number of tensors in current bucket
        self.offset_list[-1] += 1

    def build_grad_in_bucket(self, comm_stream):
        """Organize parameters' gradient(padding and split), follows the parameters' splitting method

        Data structure of self._grad_in_bucket:
        {
        rank0: [grad0_rank0, grad1_rank0, ...]
        rank1: [grad0_rank1, grad1_rank1, ...]
        }
        """

        for param, padding_size in zip(self._param_list, self._padding_size):
            param.grad.record_stream(comm_stream)
            grad = param.grad.clone().detach().flatten()
            if padding_size > 0:
                with torch.no_grad():
                    grad = torch.nn.functional.pad(grad.view(-1), [0, padding_size])  # pylint: disable=E1102
            grad_list = grad.split(grad.numel() // self.zero_world_size)
            for rank in range(self.zero_world_size):
                grad_current_rank = grad_list[rank].clone().detach()
                self.grad_to_param_mapping[id(grad_current_rank)] = id(param)
                self._grad_in_bucket[rank].append(grad_current_rank)
            param.grad = None

        self.offset_list.append(0)

    def get_grad(self):
        """Return the dictionary of gradients slices, of which the keys are ranks

        Returns:
            Dict: The dictionary of gradients slices
        """

        return self._grad_in_bucket

    def get_param(self):
        return self._param_list

    def get_flatten_grad(self) -> Tensor:
        """Return the flattened gradients slices in the bucket, the data organization of the flattened tensor:
        [grad0_rank0, grad1_rank0, ..., grad_0_rank1, grad1_rank1, ....]

        Returns:
            Tensor: the flattened gradients slices in the bucket
        """

        flat_grad = []
        for grad_list in self._grad_in_bucket.values():
            flat_grad.append(_flatten_dense_tensors(grad_list))
        flat_grad = _flatten_dense_tensors(flat_grad)
        return flat_grad

    def get_param_id_of_grad(self, grad: Tensor) -> int:
        """Return the id of a parameter which the gradient slice belongs to

        Args:
            grad (Tensor): the gradient slice

        Returns:
            int: the id of a parameter which the gradient slice belongs to
        """

        return self.grad_to_param_mapping[id(grad)]

    def reset(self):
        """Reset the bucket storage after reduction, only release the tensors have been reduced"""
        cur_offset = self.offset_list.pop(0)
        self._param_list = self._param_list[cur_offset:]
        self._padding_size = self._padding_size[cur_offset:]
        for _ in range(cur_offset):
            del self.grad_to_param_mapping[next(iter(self.grad_to_param_mapping))]
        for rank in range(self.zero_world_size):
            self._grad_in_bucket[rank] = self._grad_in_bucket[rank][cur_offset:]


class GradientStore_v2(BaseStore):
    """
    Gradient Store V2
    """

    def __init__(self, *args, partition_grad: bool = False, zero_mode=ParallelMode.ZERO1):
        super().__init__(*args)
        """
        self._grads_of_params mapping the parameter and its gradient slices
        data structure:
        {
         group_id:{
            param_id: [grad_rank0, grad_rank1, ...]
          }
        }
        """
        self.zero_world_size = gpc.get_world_size(zero_mode)
        self.zero_local_rank = gpc.get_local_rank(zero_mode)
        self._grads_of_params = dict()
        # for zero2, it's `param_id: [grad_local_rank]`
        self._working_index = 0 if partition_grad else self.zero_local_rank

        self.grad_to_param_mapping = dict()

    def get_partitioned_gradients_by_param_id(self, group_id: int, param_id: int) -> List:
        """Return list of gradient slices of a specific parameter

        Args:
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter

        Returns:
            List: the list of gradient slices of a parameter.
        """

        if group_id in self._grads_of_params:
            if param_id in self._grads_of_params[group_id]:
                return self._grads_of_params[group_id][param_id]

        # the param has no grad, for instance, in layer drop
        return []

    def append_gradients_by_param_id(self, grad: Tensor, group_id: int, param_id: int):
        """Append a gradient slice to the parameter's gradient slice list

        Args:
            grad (Tensor): The gradient slice to append to list
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter
        """

        if group_id not in self._grads_of_params:
            self._grads_of_params[group_id] = dict()
        if param_id not in self._grads_of_params[group_id]:
            self._grads_of_params[group_id][param_id] = [grad]
        else:
            self._grads_of_params[group_id][param_id].append(grad)

        self.grad_to_param_mapping[id(grad)] = param_id

    def add_gradients_by_param_id(self, grad: Tensor, grad_idx: int, group_id: int, param_id: int):
        """Add a gradient slice on an existing slice of the parameter's gradient
        Used when no_sync is not activated.

        Args:
            grad (Tensor): The split gradient to append to list
            grad_idx (int): The index of the existing slice
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter
        """

        self._grads_of_params[group_id][param_id][grad_idx].add_(grad)

    def reset_grads_by_group_id(self, group_id: int):
        self._grads_of_params[group_id] = dict()


class ParameterStore_v2(BaseStore):
    """
    Parameter Store V2
    """

    def __init__(self, dp_parallel_mode):
        super().__init__(dp_parallel_mode)

        # record the padding size of each param
        self._padding_map = dict()

        # mapping working param and master param
        self.master_to_working_param = dict()
        self.working_to_master_param = dict()

        self._bucket_reduced_param = {}
        self._bucket_reduced_grad = {}

    def record_param_padding_size(self, param: Tensor, padding_size: int):
        """Record the padding size of a param

        Args:
            param (Tensor): The parameter
            padding_size (int): The padding size of the parameter
        """

        self._padding_map[id(param)] = padding_size

    def get_param_padding_size(self, param: Tensor) -> int:
        """Return the padding size of the parameter

        Args:
            param (Tensor): The parameter

        Returns:
            int: the padding size of the parameter
        """

        return self._padding_map[id(param)]

    def link_master_and_working_param(self, master_param: Tensor, working_param: Tensor):
        """Mapping master parameter and working parameter

        Args:
            master_param (Tensor): The parameter copy in optimizer
            working_param (Tensor): The parameter of the model
        """

        self.master_to_working_param[id(master_param)] = working_param
        self.working_to_master_param[id(working_param)] = master_param

    def add_reduced_param_for_compute_norm(self, param):
        group_id = getattr(param, "group_id")
        if group_id not in self._bucket_reduced_param:
            self._bucket_reduced_param[group_id] = []
            self._bucket_reduced_grad[group_id] = []

        self._bucket_reduced_param[group_id].append(param)
        self._bucket_reduced_grad[group_id].append(param.grad)

    def get_reduced_param_for_compute_norm(self, group_id=0):
        if group_id not in self._bucket_reduced_param:
            return [], []
        return (
            self._bucket_reduced_param[group_id],
            self._bucket_reduced_grad[group_id],
        )

    def reset_reduced_data_for_compute_norm(self):
        self._bucket_reduced_param = {}
        self._bucket_reduced_grad = {}
