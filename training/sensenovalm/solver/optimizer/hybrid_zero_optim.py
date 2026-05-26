#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import math
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.core.context import Config, ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.context.parallel_context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
)
from sensenovalm.core.parallel.comm.isp import ISPCommunicatorWrapper
from sensenovalm.core.parallel.comm.zero import ParamAsyncBcastHandler
from sensenovalm.model.modules.utils import is_moe_param
from sensenovalm.solver.optimizer.store import (
    BucketStore,
    GradientStore,
    ParameterStore,
    TensorBucket,
)
from sensenovalm.solver.optimizer.utils import (
    DynamicGradScaler,
    flatten,
    get_grad_accumulate_object,
    has_inf_or_nan,
    reduce_tensor,
    release_param_grad,
    split_half_float_double,
    sync_param,
)
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.megatron_timers import megatron_timer as timer
from sensenovalm.utils.parallel import is_using_isp, should_reduce_replica_param
from sensenovalm.utils.timeout import llm_timeout

from .base_optimizer import BaseOptimizer
from .utils import compute_norm

inf = math.inf
logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


class HybridZeroOptimizer(BaseOptimizer):
    """
    Hybrid Zero Optimizer.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        cpu_offload=False,
        grad_scal_cfg: Config = None,
        zero_cfg: Config = None,
        param_bcast_sync_handler: ParamAsyncBcastHandler = None,
        isp_communicator: ISPCommunicatorWrapper = None,
    ):
        # DynamicGradScaler related args
        if gpc.config.model.dtype is torch.float32:
            initial_scale = 1
        else:
            initial_scale = grad_scal_cfg.fp16.initial_scale
        min_scale = grad_scal_cfg.fp16.min_scale
        growth_interval = grad_scal_cfg.fp16.growth_interval
        growth_factor = grad_scal_cfg.growth_factor
        backoff_factor = grad_scal_cfg.backoff_factor
        hysteresis = grad_scal_cfg.hysteresis
        max_scale = grad_scal_cfg.max_scale

        # Zero related args
        reduce_bucket_size = zero_cfg.reduce_bucket_size
        clip_grad_norm = zero_cfg.clip_grad_norm
        self._overlap_sync_grad = zero_cfg.overlap_sync_grad
        self._overlap_sync_param = zero_cfg.overlap_sync_param
        self.use_isp = is_using_isp()
        self.moe_no_tp = getattr(gpc.config.parallel.expert, "no_tp", False)

        super().__init__(optim=optimizer)

        self._cpu_offload = cpu_offload
        self._zero_local_rank = []
        self._zero_world_size = []
        self._broadcast_parallel_mode = []

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(ParallelMode.ZERO1)
        parallel_mode = ParallelMode.WEIGHT_DATA if self.use_isp else ParallelMode.DATA
        self._grad_store = GradientStore(parallel_mode)
        self._bucket_store: dict = {}
        self._accum_grad_buckets: dict = {}
        self._bucket_in_progress = dict()

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        # self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            dtype=gpc.config.model.dtype,
        )
        self._found_overflow = torch.tensor([0], device=get_current_device(), dtype=torch.float32)

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        # need to record the rank in which parameter groups are not assigned parameters.
        self.param_group_has_params = []
        self.param_group_no_params_ranks = []
        self.padding_grad = torch.zeros([32], dtype=gpc.config.model.dtype, device=get_current_device())
        self.padding_tensor = torch.zeros([32], dtype=gpc.config.model.dtype, device=get_current_device())

        self.rank_unique_id = (
            f"gpus-{gpc.get_world_size(ParallelMode.GLOBAL)}_"
            + f"wp-{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            + f"tp-{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"dp-{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"pp-{gpc.get_local_rank(ParallelMode.PIPELINE)}_"
            + f"zo-{gpc.get_local_rank(ParallelMode.ZERO1)}.pt"
        )
        self.params_per_rank_id_dict = []
        self._param_bcast_sync_handler = param_bcast_sync_handler
        if self._overlap_sync_param:
            assert self._param_bcast_sync_handler is not None

        self._isp_communicator = isp_communicator

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # set the dtype for each param group
            param_group["dtype"] = group_params[0].dtype if len(group_params) != 0 else None

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            zero_mode = param_group["optimizer_mode"]
            self._zero_local_rank.append(gpc.get_local_rank(zero_mode))
            self._zero_world_size.append(gpc.get_world_size(zero_mode))
            # TODO _broadcast_parallel_mode is not only used in broadcast, maybe can change its name
            self._broadcast_parallel_mode.append(zero_mode)

            if self._is_moe_group(param_group):
                grad_reduce_mode = ParallelMode.EXPERT_DATA
            elif self.use_isp:
                grad_reduce_mode = ParallelMode.WEIGHT_DATA
            else:
                grad_reduce_mode = ParallelMode.DATA

            if param_group["dtype"] not in self._bucket_store:
                self._bucket_store[param_group["dtype"]] = dict()
                self._bucket_in_progress[param_group["dtype"]] = dict()
                self._accum_grad_buckets[param_group["dtype"]] = dict()
            if grad_reduce_mode not in self._bucket_store[param_group["dtype"]]:
                self._bucket_store[param_group["dtype"]][grad_reduce_mode] = BucketStore(
                    -1, grad_reduce_mode, param_group["dtype"]
                )
                self._bucket_in_progress[param_group["dtype"]][grad_reduce_mode] = []
                self._accum_grad_buckets[param_group["dtype"]][grad_reduce_mode] = BucketStore(
                    -1, grad_reduce_mode, param_group["dtype"]
                )

            # assign parameters to ranks the params in the list are sorted
            params_per_rank, no_params_ranks = self._partition_param_list(group_id, param_group)
            self.param_group_no_params_ranks.append(no_params_ranks)
            self.param_group_has_params.append(self._zero_local_rank[group_id] not in no_params_ranks)

            # store the mapping between param to rank each param should belong to only one rank.
            # we can skip the moe param and do not keep them in _param_store to save memory
            # (means we need to deal with moe param in a different way), but it will increase
            # complexity and reduce code readablity.
            for rank, params in enumerate(params_per_rank):
                # check whether any rank is not assigned params.
                if len(params) != 0:
                    self._param_store.add_fp16_param_list_by_rank_group(rank, group_id, params)
                    for param in params:
                        setattr(param, "group_id", group_id)
                        setattr(param, "grad_reduce_mode", grad_reduce_mode)
                        self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            for param in group_params:
                if param.requires_grad is False:
                    continue
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._zero_world_size[group_id]):
                # No flat fp16 buffer is allocated if the process has no parameters.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                    with torch.no_grad():
                        flat_tensor = flatten(tensor_list)
                    flat_tensor = flat_tensor.data.to(get_current_device())
                    self._param_store.add_flat_fp16_param_by_rank_group(rank, group_id, flat_tensor)
                    sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            # No flat fp32 buffer is allocated if the process has no parameters.
            # if self.param_group_has_params[group_id]:
            #     fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(
            #         self._zero_local_rank[group_id], group_id
            #     )
            #     fp32_flat_current_rank = fp16_flat_current_rank.float()
            #     device = "cpu" if self._cpu_offload else get_current_device()
            #     fp32_flat_current_rank = fp32_flat_current_rank.to(device)
            #     fp32_flat_current_rank.requires_grad = True
            #     self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

            #     # need to replace the params in the `params` field in the optimizer
            #     # so that when the optimizer calls step(), it only updates the tensors
            #     # managed by this data parallel rank
            #     param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            # for param in self._fp16_param_groups[group_id]:
            #     self._param_store.set_param_reduction_state(param, False)

        # pack param group with common dtype and zero broadcast mode
        """
        [
            {
                dtype:{
                    broadcast_mode:[]
                }
            },
            {
                dtype:{
                    broadcast_mode:[]
                }
            }
        ]
        """
        self.param_list_by_rank_dtype_bdc_mode = []
        for rank in range(max(self._zero_world_size)):

            _param_list_with_dtype_bdc_mode = {}
            for group_id in range(self.num_param_groups):
                # skip ranks not in this parameter group.
                if rank >= self._zero_world_size[group_id]:
                    continue
                # The following operations are performed only on the rank to which parameters are assigned.
                if rank in self.param_group_no_params_ranks[group_id]:
                    continue

                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)

                _dtype = self.optim.param_groups[group_id]["dtype"]
                _broadcast_mode = self._broadcast_parallel_mode[group_id]
                if _dtype not in _param_list_with_dtype_bdc_mode:
                    _param_list_with_dtype_bdc_mode[_dtype] = {}
                if _broadcast_mode not in _param_list_with_dtype_bdc_mode[_dtype]:
                    _param_list_with_dtype_bdc_mode[_dtype][_broadcast_mode] = []
                _param_list_with_dtype_bdc_mode[_dtype][_broadcast_mode].append(fp16_param)

            _flat_tensor_with_dtype_bdc_mode = {}
            for dtype, params_with_bdc_mode in _param_list_with_dtype_bdc_mode.items():
                _flat_tensor_with_bdc_mode = {}
                for bdc_mode, param_list in params_with_bdc_mode.items():
                    with torch.no_grad():
                        flat_tensor = flatten(param_list)
                    flat_tensor = flat_tensor.data.to(get_current_device())
                    sync_param(flat_tensor=flat_tensor, tensor_list=param_list)

                    # sync fp16 params again
                    _total_tensor_list = []
                    for group_id in range(self.num_param_groups):
                        if dtype != self.optim.param_groups[group_id]["dtype"]:
                            continue
                        if bdc_mode != self._broadcast_parallel_mode[group_id]:
                            continue
                        if rank >= self._zero_world_size[group_id]:
                            continue
                        if rank not in self.param_group_no_params_ranks[group_id]:
                            tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                            _total_tensor_list.extend(tensor_list)
                    sync_param(flat_tensor=flat_tensor, tensor_list=_total_tensor_list)

                    _flat_tensor_with_bdc_mode[bdc_mode] = flat_tensor

                _flat_tensor_with_dtype_bdc_mode[dtype] = _flat_tensor_with_bdc_mode

            self.param_list_by_rank_dtype_bdc_mode.append(_flat_tensor_with_dtype_bdc_mode)

        for group_id, param_group in enumerate(self.optim.param_groups):
            # create a copy of fp32 weights of the parameters for which this rank is responsible
            # No flat fp32 buffer is allocated if the process has no parameters.
            if self.param_group_has_params[group_id]:
                fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(
                    self._zero_local_rank[group_id], group_id
                )
                fp32_flat_current_rank = fp16_flat_current_rank.float()
                device = "cpu" if self._cpu_offload else get_current_device()
                fp32_flat_current_rank = fp32_flat_current_rank.to(device)
                fp32_flat_current_rank.requires_grad = True
                self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

                # need to replace the params in the `params` field in the optimizer
                # so that when the optimizer calls step(), it only updates the tensors
                # managed by this data parallel rank
                param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)

        assert len(self._fp16_param_groups) != 0

        # If a rank is not assigned any arguments, 'has_params' is False.
        self.has_params = sum(self.param_group_has_params) != 0
        # flag used to skip unnecessary gradient reduce operation when gradient accumulation is enabled.
        self.skip_grad_reduce = False

        self._attach_reduction_hook()

        if self._overlap_sync_param:
            self._param_bcast_sync_handler.register_sync_parameters_hook(self._isp_communicator)

    @property
    def zero_local_rank(self):
        return self._zero_local_rank

    @property
    def zero_world_size(self):
        return self._zero_world_size

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    @property
    def num_param_groups(self):
        return len(self._fp16_param_groups)

    def _partition_param_list(self, group_id, param_group):
        param_list = param_group["params"]
        zero_size = self._zero_world_size[group_id]
        no_params_ranks = []
        params_per_rank = [[] for _ in range(zero_size)]
        numel_per_rank = [0 for _ in range(zero_size)]

        if self._overlap_sync_param:
            (
                params_per_rank,
                no_params_ranks,
                numel_per_rank,
                _params_per_rank_id_dict,
            ) = self._param_bcast_sync_handler.partition_param_list(param_list=param_list, partition_size=zero_size)
            self.params_per_rank_id_dict.append(_params_per_rank_id_dict)
        else:
            self.params_per_rank_id_dict.append([[] for _ in range(zero_size)])
            sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)

            for i, param in enumerate(sorted_params):
                if param.requires_grad is False:
                    continue

                global_id = str(i)
                for j in range(len(param.size())):
                    global_id = "_".join([global_id, str(param.size()[j])])
                rank_to_go = numel_per_rank.index(min(numel_per_rank))
                params_per_rank[rank_to_go].append(param)
                self.params_per_rank_id_dict[-1][rank_to_go].append(global_id)
                numel_per_rank[rank_to_go] += param.numel()

            # check whether any rank is not assigned to parameters.
            for rank, params in enumerate(params_per_rank):
                if len(params) == 0:
                    no_params_ranks.append(rank)

        if gpc.is_rank_for_log():
            logger.info(  # pylint: disable=W1203
                f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}"
            )

        return params_per_rank, set(no_params_ranks)

    def _is_moe_group(self, param_group):
        return "moe" in param_group.keys() and param_group["moe"]

    # TODO check expert dp is correct when enable moe and overlap both
    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                # we should not reduce the param in moe
                if not param.requires_grad:
                    continue

                reduce_rank = None

                def _define_and_attach(param, reduce_rank=None):
                    reduction_func = partial(
                        self._store_and_try_reduce_grads_by_bucket,
                        param=param,
                        reduce_rank=reduce_rank,
                    )

                    reduce_scatter_checker = partial(
                        self._wait_reduce_scatter_and_accumulate_grads,
                        param=param,
                        reduce_rank=reduce_rank,
                    )

                    def reduction_layernorm_func():
                        # BUG: 8.0.RC1.alpha003 hccl allreduce AVG op will not perform averaging operation.
                        # So we use sum + div here when training on Ascend machines.
                        # op_type = (
                        #     torch.distributed.ReduceOp.SUM
                        #     if sensenovalm_accelerator.get_accelerator_backend()
                        #     in [AcceleratorType.NPU, AcceleratorType.DIPU, AcceleratorType.DITORCH]
                        #     else torch.distributed.ReduceOp.AVG
                        # )
                        parallel_mode = ParallelMode.WEIGHT if self.use_isp else ParallelMode.TENSOR
                        reduce_tensor(
                            param.grad,
                            dtype=None,
                            dst_rank=reduce_rank,
                            parallel_mode=parallel_mode,
                            async_op=False,
                        )

                    # define hook
                    # NOT IMPORTANT BUT GOOD TO KNOW:
                    # args here is not grad, but allow_unreacable and accumulate_grad
                    def reduce_grad_hook(*args):  # pylint: disable=W0613
                        if self.skip_grad_reduce is False:
                            reduction_func()

                    # define hook for real gradient accumulation.
                    def accum_grad_hook(*args):  # pylint: disable=W0613
                        reduce_scatter_checker()

                    # define hook for sequence_parallel
                    def extra_layernorm_reduce_grad_hook(*args):  # pylint: disable=W0613
                        if self.skip_grad_reduce is False:
                            reduction_layernorm_func()

                    # get the AccumulateGrad object of the param itself
                    # If these objects are not kept, reduction hooks may not be attached successfully.
                    if not hasattr(param, "evo_tensor"):
                        accum_grad_obj = get_grad_accumulate_object(param)
                        self._grad_store.add_accumulate_grad_object(accum_grad_obj)

                    # the grad of layernorm should be all-reduce across the global process group
                    # here is the first stage all-reduce in tp/wp process group
                    # the second stage all-reduce will be processed in reduce_grad_hook
                    if should_reduce_replica_param(param):
                        accum_grad_obj.register_hook(extra_layernorm_reduce_grad_hook)

                    # we should not only register for parameters which have isp_reduce_scatter_name attr.
                    # we must keep up with reduce_grad_hook.
                    if self._isp_communicator and (
                        (
                            is_moe_param(param)
                            and gpc.config.parallel.expert_weight.size > 1
                            and gpc.config.parallel.expert_weight.overlap
                        )
                        or (
                            not is_moe_param(param)
                            and gpc.config.parallel.weight.size > 1
                            and gpc.config.parallel.weight.overlap
                        )
                    ):
                        if hasattr(param, "evo_tensor"):
                            param.register_post_accumulate_grad_hook(accum_grad_hook)
                        else:
                            accum_grad_obj.register_hook(accum_grad_hook)

                    if self._overlap_sync_grad:
                        if hasattr(param, "evo_tensor"):
                            param.register_post_accumulate_grad_hook(reduce_grad_hook)
                        else:
                            accum_grad_obj.register_hook(reduce_grad_hook)

                _define_and_attach(param, reduce_rank)

    def accumulate_left_grads_after_backward(self):
        if self._isp_communicator is None:
            return

        for _, _buckets in self._accum_grad_buckets.items():
            for _, _bucket in _buckets.items():
                self._accum_grads_store_in_bucket(_bucket)

    def belongs_to_current_rank(self, param) -> bool:
        """
        Check whether a parameter is supposed to be updated by the process of the current rank

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor

        :return: True if the parameter should be updated by the current rank. Otherwise false.
        :rtype: bool
        """
        tensor_ranks = self._param_store.get_param_rank(param)
        group_id = getattr(param, "group_id")
        return gpc.get_local_rank(self._broadcast_parallel_mode[group_id]) in tensor_ranks

    def _accum_grads_store_in_bucket(self, bucket: BucketStore, reduce_rank: Optional[int] = None) -> None:
        for _param in bucket.get_param(reduce_rank):
            if not hasattr(_param, "isp_reduce_scatter_name"):
                continue

            # wait and accumulate gardient.
            _key = getattr(_param, "isp_reduce_scatter_name")
            _grad, _comm_handle = self._isp_communicator.reduce_scatter_handlers[_key]
            _comm_handle.wait()
            _param.grad.add_(_grad)

            # release cuda memory.
            _grad = None
            self._isp_communicator.reduce_scatter_handlers[_key] = None

        bucket.reset_by_rank(reduce_rank)

    def _wait_reduce_scatter_and_accumulate_grads(self, param, reduce_rank: Optional[int] = None):
        param_size = param.numel()

        grad_reduce_mode = getattr(param, "grad_reduce_mode")
        current_bucket = self._accum_grad_buckets[param.dtype][grad_reduce_mode]

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if current_bucket.num_elements_in_bucket(reduce_rank) + param_size > self._reduce_bucket_size:
            self._accum_grads_store_in_bucket(current_bucket, reduce_rank)

        # otherwise, add the parameter into bucket.
        current_bucket.add_num_elements_in_bucket(param_size, reduce_rank)
        current_bucket.add_param(param, reduce_rank)

    def _store_and_try_reduce_grads_by_bucket(self, param, reduce_rank=None):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        grad_reduce_mode = getattr(param, "grad_reduce_mode")
        current_bucket = self._bucket_store[param.dtype][grad_reduce_mode]

        if current_bucket.num_elements_in_bucket(reduce_rank) + param_size > self._reduce_bucket_size:
            self._reduce_grads_stored_in_bucket(current_bucket, reduce_rank)

        # the param must not be reduced to ensure correctness
        is_param_reduced = self._param_store.is_param_reduced(param)
        if is_param_reduced:
            msg = (
                f"Parameter of size ({param.size()}) has already been reduced, "
                + "duplicate reduction will lead to arithmetic incorrectness"
            )
            raise RuntimeError(msg)

        # the param must have grad for reduction
        assert param.grad is not None, f"Parameter of size ({param.size()}) has None grad, cannot be reduced"

        current_bucket.add_num_elements_in_bucket(param_size, reduce_rank)
        current_bucket.add_grad(param.grad, reduce_rank)
        current_bucket.add_param(param, reduce_rank)

    def _reduce_grads_stored_in_bucket(self, current_bucket, reduce_rank=None):
        # reduce grads
        self._reduce_grads_by_rank(
            reduce_rank=reduce_rank,
            grads=current_bucket.get_grad(reduce_rank=reduce_rank),
            bucket_size=current_bucket.num_elements_in_bucket(reduce_rank),
            group_id=current_bucket.get_param_group_id(),
            dtype=current_bucket.dtype,
            dp_parallel_mode=current_bucket.get_dp_parallel_mode(),
        )

        params_in_bucket = current_bucket.get_param(reduce_rank=reduce_rank)

        for param in params_in_bucket:
            # the is_param_reduced flag should be False showing that
            # this param is not reduced before calling self._reduce_grads_by_rank
            is_param_reduced = self._param_store.is_param_reduced(param)

            if is_param_reduced:
                msg = (
                    f"Parameter of size ({param.size()}) has been reduced, "
                    + "duplicate reduction will lead to arithmetic incorrectness"
                )
                raise RuntimeError(msg)

            # update the flag
            self._param_store.set_param_reduction_state(param, True)

            if self.belongs_to_current_rank(param):
                self._param_store.add_reduced_param_for_compute_norm(param)
            else:
                self._param_store.add_previous_reduced_param(param)

        current_bucket.reset_by_rank(reduce_rank)

    def _reduce_grads_by_rank(self, reduce_rank, grads, bucket_size, group_id, dtype, dp_parallel_mode):
        grad_buckets_by_dtype = split_half_float_double(grads)
        next_bucket_list = []


        op_type = torch.distributed.ReduceOp.SUM
        avg_size = -1

        # add parameters into bucket for reduction
        for tensor_list in grad_buckets_by_dtype:
            param_bucket = TensorBucket(size=bucket_size)
            for tensor in tensor_list:
                param_bucket.add_to_bucket(tensor, allow_oversize=True)
            if not param_bucket.is_empty():
                self._reduce_and_copy(
                    bucket=param_bucket,
                    reduce_rank=reduce_rank,
                    group_id=group_id,
                    dp_parallel_mode=dp_parallel_mode,
                    op_type=op_type,
                )
            next_bucket_list.append(param_bucket)

        # wait for the completion of previouce bucket list reduction, and do unflatten_and_copy()
        # here we can also overlap the communication with some memcpy operation caused by bucket.flatten()
        for bucket in self._bucket_in_progress[dtype][dp_parallel_mode]:
            bucket.commu_handle.wait()
            bucket.unflatten_and_copy(dp_group_size=avg_size)
            bucket.empty()
        self._bucket_in_progress[dtype][dp_parallel_mode] = []
        self._param_store.clear_grads_of_previous_reduced_params()

        # after the completion of bucket list reduction, add new buckets into _bucket_in_progress
        self._bucket_in_progress[dtype][dp_parallel_mode] = next_bucket_list.copy()

    def _reduce_and_copy(self, bucket: TensorBucket, reduce_rank, group_id, dp_parallel_mode, op_type):
        # flatten the tensors and do allreduce
        bucket.flatten()
        bucket.commu_handle = reduce_tensor(
            tensor=bucket.get_flat_tensor(),
            dtype=None,
            dst_rank=reduce_rank,
            parallel_mode=dp_parallel_mode,
            op_type=op_type,
        )

        # update the reduced tensor
        if reduce_rank is None or reduce_rank == self._zero_local_rank[group_id]:
            bucket.set_unflatten_and_copy_flag(flag=True)

    def _has_inf_or_nan(self, tensor):
        try:
            tensor_mean = float(tensor.mean())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if tensor_mean == float("inf") or tensor_mean == -float("inf"):
                return True
            return False

    def _sync_grad(self):
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False
        self._param_store.reset_reduced_data_for_compute_norm()

        # accumulate gradient
        avg_gradients = self._grad_store._averaged_gradients
        for group_id in range(self.num_param_groups):
            # the following operations are performed only on the rank to which parameters are assigned.
            if self._zero_local_rank[group_id] not in self.param_group_no_params_ranks[group_id]:
                param_group = self._param_store.get_fp16_params_by_rank_group(self._zero_local_rank[group_id], group_id)

                if group_id not in avg_gradients:
                    avg_gradients[group_id] = []

                param_idx = 0
                for param in param_group:
                    if param.grad is not None:
                        if len(avg_gradients[group_id]) == param_idx:
                            avg_gradients[group_id].append(param.grad)
                        else:
                            avg_gradients[group_id][param_idx].add_(param.grad)
                        param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for _, param_group in self._fp16_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.detach()
                    param.grad.zero_()
                else:
                    pass

    def backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

        # Gradients may not be fully synchronized here.

    def _compute_norm(self, group_id: int = 0):
        # compute norm for gradients that have been reduced
        params, grads = self._param_store.get_reduced_param_for_compute_norm(group_id=group_id)
        params_is_padding = False
        if len(params) == 0:
            params_is_padding = True
            dtype = self.param_groups[group_id]["dtype"]
            grads = [self.padding_grad.to(dtype)]
            params = [self.padding_tensor.to(dtype)]

            if self.optim.param_groups[group_id]["name"] in [
                "moe_gate_decay",
                "moe_gate_no_decay",
                "moe_coeff_decay",
                "moe_coeff_no_decay",
                "vit_decay",
                "vit_no_decay",
                "llm_decay",
                "llm_no_decay",
                "mlp_decay",
                "mlp_no_decay",
                "fm_modules_decay",
                "fm_modules_no_decay",
                "other_decay",
                "other_no_decay",
            ]:
                for param in params:
                    setattr(param, IS_REPLICA_ZERO_PARALLEL, True)
            elif self.optim.param_groups[group_id]["name"] in [
                "vit_tp_decay",
                "vit_tp_no_decay",
                "llm_tp_decay",
                "llm_tp_no_decay",
                "mlp_tp_decay",
                "mlp_tp_no_decay",
            ]:
                for param in params:
                    if self.use_isp:
                        setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)
                    else:
                        setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
            elif self.optim.param_groups[group_id]["name"] == "default":
                for param in params:
                    if self.use_isp:
                        setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)
                    else:
                        setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
            elif self.optim.param_groups[group_id]["name"] == "fp32":
                for param in params:
                    setattr(param, IS_REPLICA_ZERO_PARALLEL, True)
            elif self._is_moe_group(self.optim.param_groups[group_id]) or self.optim.param_groups[group_id]["name"] in [
                "moe_expert_no_decay",
                "moe_expert_decay",
            ]:
                for param in params:
                    if self.moe_no_tp:
                        setattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL, True)
                    elif self.use_isp:
                        setattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL, True)
                    else:
                        setattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL, True)
            else:
                raise NotImplementedError("unrecognized parameter group.")

        norm = 0
        if self._clip_grad_norm > 0:
            # this norm is before scaling, it will be very large
            norm = compute_norm(gradients=grads, parameters=params, zero_mode=self._broadcast_parallel_mode[group_id])

        if params_is_padding:
            for param in params:
                if hasattr(param, IS_REPLICA_ZERO_PARALLEL):
                    delattr(param, IS_REPLICA_ZERO_PARALLEL)
                if hasattr(param, IS_TENSOR_ZERO_PARALLEL):
                    delattr(param, IS_TENSOR_ZERO_PARALLEL)
                if hasattr(param, IS_WEIGHT_ZERO_PARALLEL):
                    delattr(param, IS_WEIGHT_ZERO_PARALLEL)
                if hasattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL):
                    delattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL)
                if hasattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL):
                    delattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL)
                if hasattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL):
                    delattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL)

        return norm

    @llm_timeout(func_name="optim_step")
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            Union[bool, float]: Whether the gradient is success updated, and the gradient.
        """
        assert closure is None, "closure is not supported by step()"

        # if not overlapping communication (no reduction hook is attached)
        # we need to manually reduce these gradients
        if not self._overlap_sync_grad:
            for group_id in range(len(self._fp16_param_groups)):
                for param in self._fp16_param_groups[group_id]:
                    # we should not reduce the param in moe
                    if param.grad is not None:
                        self._store_and_try_reduce_grads_by_bucket(param)

        # we need to reduce the gradients left in the communication bucket
        for _, buckets in self._bucket_store.items():
            for _, bucket in buckets.items():
                self._reduce_grads_stored_in_bucket(bucket, reduce_rank=None)

        if sensenovalm_accelerator.get_accelerator_backend() in [
            AcceleratorType.NPU,
            AcceleratorType.DIPU,
            AcceleratorType.DITORCH,
        ]:
            avg_size = gpc.get_world_size(ParallelMode.DATA)
        else:
            avg_size = -1

        # wait grads reduced and clear reduced grads
        for dtype, buckets_by_same_dtype in self._bucket_in_progress.items():
            for grad_reduce_mode, buckets in buckets_by_same_dtype.items():
                for bucket in buckets:
                    bucket.commu_handle.wait()
                    bucket.unflatten_and_copy(dp_group_size=avg_size)
                    bucket.empty()
                self._bucket_in_progress[dtype][grad_reduce_mode] = []  # pylint: disable=R1733
        self._param_store.clear_grads_of_previous_reduced_params()

        # compute norm for gradients in the last bucket
        total_norms = {}
        for group_id in range(self.num_param_groups):
            group_name = self.param_groups[group_id]["name"] if "name" in self.param_groups[group_id] else "default"
            group_name = f"{group_id}_{group_name}"
            total_norms[group_name] = self._compute_norm(group_id=group_id)

        timer("sync_grad").start()
        self._sync_grad()
        timer("sync_grad").stop()

        state, global_norms = self._step(closure=closure, norms=total_norms)

        return state, global_norms

    def _step(self, closure=None, norms=None):
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = False
        found_nan = False
        # if there is INF values in grades, compute_norm func would also returns -1
        # thus, we try to avoid call _check_overflow here
        # found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm

        if -1 in norms.values():
            found_inf = True

        if -2 in norms.values():
            found_nan = True

        loss_scale = float(self.loss_scale.item())  # backup
        if gpc.config.model.dtype is not torch.float32:
            self.grad_scaler.update(found_inf)

        # update loss scale if overflow occurs
        if found_inf:
            if gpc.is_rank_for_log():
                logger.warning("Overflow occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, norms

        if found_nan:
            if gpc.is_rank_for_log():
                logger.warning("Nan grad norm occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, norms
        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            # The following operations are performed only on the rank to which parameters are assigned.
            if not self.param_group_has_params[group_id]:
                continue

            # create flat gradient for the flat fp32 params
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)

            rank = self._zero_local_rank[group_id]
            fp16_params = self._param_store.get_fp16_params_by_rank_group(rank=rank, group_id=group_id)

            aligned_grads = []
            gi = 0
            num_g = len(gradients)

            with torch.no_grad():
                for p in fp16_params:
                    p_numel = p.numel()

                    if gi < num_g:
                        g = gradients[gi]
                        # Some pipelines store grads as views/flattened; compare numel only
                        if g is not None and g.numel() == p_numel:
                            aligned_grads.append(g)
                            gi += 1
                            continue

                    # Missing grad for this param => pad zeros
                    # Use p.data to get correct dtype/device/shape
                    aligned_grads.append(torch.zeros_like(p.data))

                # If there are leftover grads, something is inconsistent (order mismatch, or collection bug)
                if gi != num_g:
                    # Optional: print first few leftovers for debugging
                    leftover_shapes = [tuple(gradients[k].shape) for k in range(gi, min(num_g, gi + 8))]
                    raise RuntimeError(
                        f"[ZeRO-v1] Unconsumed grads after alignment: consumed={gi}, total={num_g}, "
                        f"leftover_shapes={leftover_shapes}"
                    )

            with torch.no_grad():
                flat_fp16_avg_grads = flatten(aligned_grads)

            self._grad_store.reset_average_gradients_by_group(group_id)
            gradients = None
            aligned_grads = None

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)
            flat_fp16_avg_grads = None  # release cuda memory

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            single_grad_partition_groups.append(self._fp32_flat_param_groups_of_current_rank[group_id].grad)
        # unscale and clip grads
        # get the global norm
        scaled_global_grad_norm = sum(list(norms.values())) ** 0.5
        if gpc.config.model.dtype is not torch.float32:
            if len(single_grad_partition_groups) != 0 and self._clip_grad_norm > 0:
                self._unscale_and_clip_grads_test(
                    single_grad_partition_groups,
                    scaled_global_grad_norm,
                    loss_scale,
                )

        # update the parameters
        timer("step").start()

        global_norm_groups = {}
        if self._clip_grad_norm > 0:
            for group_name, norm in norms.items():
                global_norm_groups[group_name] = norm**0.5


        # For those ranks that are not assigned parameters, we just wait for other ranks
        # to send them updated their own parameters.
        if self.has_params:
            self.optim.step()
            # release the fp32 grad
            release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())
            # update fp16 partition updated by the current rank
            for group_id in range(len(self._fp16_param_groups)):
                if self.param_group_has_params[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                        rank=self._zero_local_rank[group_id], group_id=group_id
                    )
                    fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                    fp16_param.data.copy_(fp32_param)
        sensenovalm_accelerator.synchronize()
        self.broadcast_params()

        timer("step").stop()

        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        for group_name, global_norm in global_norm_groups.items():
            global_norm_groups[group_name] = global_norm / loss_scale
        return True, global_norm_groups

    def broadcast_params(self):
        handles = []
        # traverse according to rank firstly, which is conducive to overlapping broadcast communication.
        for rank in range(max(self._zero_world_size)):
            packed_params_by_dtype_bdc_mode = self.param_list_by_rank_dtype_bdc_mode[rank]

            for _, packed_params_with_bdc_mode in packed_params_by_dtype_bdc_mode.items():
                for bdc_mode, packed_params in packed_params_with_bdc_mode.items():
                    _ranks = gpc.get_ranks_in_group(bdc_mode)
                    g_rank = _ranks[rank]
                    if len(_ranks) == 1:
                        continue

                    if self._overlap_sync_param:
                        self._param_bcast_sync_handler.add_bcast_closure(
                            rank,
                            partial(
                                dist.broadcast,
                                tensor=packed_params,
                                src=g_rank,
                                group=gpc.get_group(bdc_mode),
                                async_op=True,
                            ),
                        )
                    else:
                        handle = dist.broadcast(
                            packed_params,
                            src=g_rank,
                            group=gpc.get_group(bdc_mode),
                            async_op=True,
                        )
                        handles.append(handle)

        if self._overlap_sync_param:
            self._param_bcast_sync_handler.submit_bcast_async()
        else:
            for handle in handles:
                handle.wait()

    def submit_bcast_async(self, manual_submit_count: int = None, submit_all: bool = False) -> None:
        if self._param_bcast_sync_handler is not None:
            self._param_bcast_sync_handler.submit_bcast_async(
                manual_submit_count=manual_submit_count, submit_all=submit_all
            )

    ##################
    # FP16 Utilities #
    ##################

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            # The following operations are performed only on the rank to which parameters are assigned.
            if self._zero_local_rank[group_id] not in self.param_group_no_params_ranks[group_id]:
                for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                    if avg_grad is not None and has_inf_or_nan(avg_grad):
                        self._found_overflow.fill_(1.0)
                        break
        dist.all_reduce(
            self._found_overflow,
            op=dist.ReduceOp.MAX,
            group=gpc.get_group(ParallelMode.GLOBAL),
        )

        return self._found_overflow.item() > 0

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm_groups, loss_scale):
        # compute combined scale factor for this group
        combined_scale_groups = []

        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            for group_id, total_norm in enumerate(total_norm_groups):
                combined_scale_groups.append(loss_scale)
                clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
                if clip > 1.0:
                    combined_scale_groups[group_id] = clip * loss_scale

        for group_id, grad in enumerate(grad_groups_flat):
            grad.data.mul_(1.0 / combined_scale_groups[group_id])

    def _unscale_and_clip_grads_test(self, grad_groups_flat, total_norm, loss_scale):
        # compute combined scale factor for this group
        combined_scale = loss_scale
        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1.0:
                combined_scale = clip * loss_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / combined_scale)

    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        states["grad_scaler"] = grad_scaler
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        flat_fp32_weights = {}
        for group_id, param in self._fp32_flat_param_groups_of_current_rank.items():
            if self._zero_local_rank[group_id] not in self.param_group_no_params_ranks[group_id]:
                assert param.grad is None
                flat_fp32_weights[group_id] = param
        states["flat_fp32_weights"] = flat_fp32_weights
        states["zero_devide_optim_plan"] = self.params_per_rank_id_dict

        return states

    def load_state_dict(self, states):
        # TODO: Need to take into account the change in the number of DP.
        assert "grad_scaler" in states, "Not found grad_scaler state!"
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)
        optim_states = states["base_optim_states"]

        if gpc.config.get("only_load_lr", False):
            if gpc.is_rank_for_log():
                logger.info("Only load lr in param_groups, skip loading weights in optimizer...")
            for pg1, pg2 in zip(self.optim.param_groups, optim_states["param_groups"]):
                pg1["lr"] = pg2["lr"]
            return

        self.optim.load_state_dict(optim_states)

        # load fp32 model weight.
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_flat_param_groups_of_current_rank)
        for group_id, param in flat_fp32_weights.items():
            if self._zero_local_rank[group_id] not in self.param_group_no_params_ranks[group_id]:
                self_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                assert (
                    self_param.shape == param.shape
                ), f"The loaded parameter shape is inconsistent, {self_param.shape} != {param.shape}"
                self_param.data.copy_(param.data)

        # Load the fp16 model weights.
        for group_id in range(len(self._fp16_param_groups)):
            if self._zero_local_rank[group_id] not in self.param_group_no_params_ranks[group_id]:
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank=self._zero_local_rank[group_id], group_id=group_id
                )
                fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                fp16_param.data.copy_(fp32_param)

        if "zero_devide_optim_plan" in states:
            self.params_per_rank_id_dict = states["zero_devide_optim_plan"]

    def reload_zero_fp32_buff(self):
        # If we use AMP optimizer, we need to update its fp32 buffer as newly loaded weights value.
        # Or we must ensure that loading model weights must be done before zero is initialized.
        for group_id, param_group in enumerate(self.optim.param_groups):
            if self.param_group_has_params[group_id]:
                # flatten fp16 params have already been updated by 'load_model_checkpoint'
                fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(
                    self._zero_local_rank[group_id], group_id
                )
                # param_group["params"] is fp32 flatten optimizer states of this zero rank.
                param_group["params"][0].data.copy_(fp16_flat_current_rank.float())
