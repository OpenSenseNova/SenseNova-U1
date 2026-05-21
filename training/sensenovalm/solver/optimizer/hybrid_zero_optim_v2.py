# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0); ZeRO partitioning logic
# inspired by DeepSpeed (Microsoft, Apache-2.0) and implemented from
# scratch with our own design.
import math
from functools import partial
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from sensenovalm.core.context import Config, ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.context.parallel_context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
)
from sensenovalm.core.parallel.comm.zero import ParamAsyncBcastHandler
from sensenovalm.solver.optimizer.store import (
    BucketStore_v2,
    GradientStore_v2,
    ParameterStore_v2,
)
from sensenovalm.solver.optimizer.utils import (
    DynamicGradScaler,
    flatten,
    reduce_tensor,
    release_param_grad,
    sync_param,
)
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.parallel import is_using_isp, is_using_sequence_parallel

from .base_optimizer import BaseOptimizer
from .utils import compute_norm


def calculate_global_norm_from_list(global_norm_groups):
    """Compute total from a list of norms"""
    total_norm = 0.0
    for norm in global_norm_groups.values():
        total_norm += norm**2.0
    return math.sqrt(total_norm)


logger = get_logger(__file__)


class HybridZeroOptimizer_v2(BaseOptimizer):
    """Optimizer used for ZeRO-1 and ZeRO-2."""

    def __init__(
        self,
        optimizer: Optimizer,
        grad_scal_cfg: Config = None,
        zero_cfg: Config = None,
        param_bcast_sync_handler: ParamAsyncBcastHandler = None,
        isp_communicator=None,
        partition_grad: bool = False,  # zero 2
        cpu_offload: bool = False,  # cpu offload
        master_weights: bool = True,  # master weights
    ):
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
        self._reduce_bucket_size = zero_cfg.reduce_bucket_size
        self._all_gather_size = zero_cfg.all_gather_size
        self._clip_grad_norm = zero_cfg.clip_grad_norm
        self._overlap_sync_grad = zero_cfg.overlap_sync_grad
        self._overlap_sync_param = zero_cfg.overlap_sync_param
        self.use_isp = is_using_isp()

        self._param_bcast_sync_handler = param_bcast_sync_handler

        if self._overlap_sync_param:
            assert self._param_bcast_sync_handler is not None

        self._isp_communicator = isp_communicator

        super().__init__(optim=optimizer)

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._element_size = self.optim.param_groups[0]["params"][0].element_size()

        # stage 2
        self._partition_grads = partition_grad
        self._cpu_offload = cpu_offload

        # if process_group is none, will use the default one
        self._local_rank = gpc.get_local_rank(ParallelMode.DATA)
        self._world_size = gpc.get_world_size(ParallelMode.DATA)

        self._zero_local_rank = []
        self._zero_world_size = []
        self._zero_parallel_mode = []

        # working and master params for mixed precision training
        # master params: params that are splited into the current rank, fp32 params
        # working params: the original complete params, fp16 params
        self._working_param_groups = dict()
        self._master_param_groups_of_current_rank = dict()

        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )

        # master weights copy
        self._master_weights = master_weights

        # check argument conflict
        self._sanity_checks()

        # ParameterStore_v2 will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        parallel_mode = ParallelMode.WEIGHT_DATA if self.use_isp else ParallelMode.DATA
        self._param_store = ParameterStore_v2(ParallelMode.ZERO1)
        self._grad_store = GradientStore_v2(parallel_mode, partition_grad=partition_grad)
        self._bucket_store: List[BucketStore_v2] = []
        self._accum_grad_buckets: List[BucketStore_v2] = []

        self.rank_unique_id = (
            f"gpus-{gpc.get_world_size(ParallelMode.GLOBAL)}_"
            + f"wp-{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            + f"tp-{gpc.get_local_rank(ParallelMode.TENSOR)}_"
            + f"dp-{gpc.get_local_rank(ParallelMode.DATA)}_"
            + f"pp-{gpc.get_local_rank(ParallelMode.PIPELINE)}_"
            + f"zo-{gpc.get_local_rank(ParallelMode.ZERO1)}.pt"
        )

        self.zero_1_5 = False

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = []
            for param in param_group["params"]:
                if param.requires_grad:
                    setattr(param, "group_id", group_id)
                    group_params.append(param)

            param_group["dtype"] = group_params[0].dtype if len(group_params) != 0 else None

            zero_mode = param_group["optimizer_mode"]
            self._zero_local_rank.append(gpc.get_local_rank(zero_mode))
            self._zero_world_size.append(gpc.get_world_size(zero_mode))
            self._zero_parallel_mode.append(zero_mode)

            # add the working params to working_param_groups for bookkeeping
            self._working_param_groups[group_id] = group_params
            master_param_current_rank = self._create_master_param_current_rank(group_id, group_params)
            self._master_param_groups_of_current_rank[group_id] = master_param_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group["params"] = master_param_current_rank

            if self._is_moe_group(param_group):
                grad_reduce_mode = ParallelMode.EXPERT_DATA
            elif param_group["name"] != "embed_head" and self.use_isp:
                grad_reduce_mode = ParallelMode.WEIGHT_DATA
            else:
                grad_reduce_mode = ParallelMode.DATA
            self._bucket_store.append(BucketStore_v2(group_id, grad_reduce_mode, zero_mode=zero_mode))
            self._accum_grad_buckets.append(BucketStore_v2(group_id, grad_reduce_mode, zero_mode=zero_mode))

            if gpc.get_world_size(grad_reduce_mode) != gpc.get_world_size(zero_mode):
                self.zero_1_5 = True

        # initialize communication stream for
        # communication-computation overlapping
        self._comm_stream = torch.cuda.Stream(priority=0)

        self.skip_grad_reduce = False

        self._attach_reduction_hook()

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_param_groups(self):
        return len(self._working_param_groups)

    @property
    def loss_scale(self) -> float:
        return self.grad_scaler.scale.item()

    def _is_moe_group(self, param_group):
        return "moe" in param_group.keys() and param_group["moe"]

    def _wait_reduce_scatter_and_accumulate_grads(self, param):
        param_size = param.numel()

        group_id = getattr(param, "group_id")
        current_bucket = self._accum_grad_buckets[group_id]

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if current_bucket.num_elements_in_bucket() + param_size > self._reduce_bucket_size:
            self._accum_grads_store_in_bucket(current_bucket)

        # otherwise, add the parameter into bucket.
        current_bucket._num_elements_in_bucket += param.numel()
        current_bucket._param_list.append(param)

    def _accum_grads_store_in_bucket(self, bucket: BucketStore_v2) -> None:
        for _param in bucket.get_param():
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

        bucket.reset_all()

    def accumulate_left_grads_after_backward(self):
        if self._isp_communicator is None:
            return

        for group_id in range(self.num_param_groups):
            self._accum_grads_store_in_bucket(self._accum_grad_buckets[group_id])

    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    def _sanity_checks(self):
        for param_group in self.optim.param_groups:
            group_params = param_group["params"]
            for param in group_params:
                if not hasattr(param, "skip_zero_check") or param.skip_zero_check is False:
                    assert (
                        param.dtype == self._dtype
                    ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    def add_attr_for_splited_param(self, origin_param, splited_param_current_rank):

        if hasattr(origin_param, IS_TENSOR_ZERO_PARALLEL):
            value = getattr(origin_param, IS_TENSOR_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_TENSOR_ZERO_PARALLEL, value)

        if hasattr(origin_param, IS_WEIGHT_ZERO_PARALLEL):
            value = getattr(origin_param, IS_WEIGHT_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_WEIGHT_ZERO_PARALLEL, value)

        if hasattr(origin_param, IS_REPLICA_ZERO_PARALLEL):
            value = getattr(origin_param, IS_REPLICA_ZERO_PARALLEL)
            setattr(splited_param_current_rank, IS_REPLICA_ZERO_PARALLEL, value)

        if hasattr(origin_param, IS_TENSOR_EXPERT_DATA_PARALLEL):
            value = getattr(origin_param, IS_TENSOR_EXPERT_DATA_PARALLEL)
            setattr(splited_param_current_rank, IS_TENSOR_EXPERT_DATA_PARALLEL, value)

        if hasattr(origin_param, "block_name"):
            value = getattr(origin_param, "block_name")
            setattr(splited_param_current_rank, "block_name", value)

    def _create_master_param_current_rank(self, group_id, param_list):
        # split each param evenly by world size
        params_current_rank = []
        device = "cpu" if self._cpu_offload else get_current_device()
        zero_world_size = self._zero_world_size[group_id]

        for param in param_list:
            padding_size = (zero_world_size - param.numel() % zero_world_size) % zero_world_size
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                    # reset working params' ptr when no master weights
                    if self._master_weights is False:
                        param.data = padding_param[: param.numel()].view(param.shape)
                else:
                    padding_param = param.data.view(-1)

                splited_params = padding_param.split(padding_param.numel() // zero_world_size)
                splited_params = splited_params[self._zero_local_rank[group_id]]

                # use fp32 when master_weights is True
                if self._master_weights is True:
                    splited_param_current_rank = splited_params.detach().float().to(device)
                else:
                    splited_param_current_rank = splited_params

                self.add_attr_for_splited_param(param, splited_param_current_rank)

                params_current_rank.append(splited_param_current_rank)
                self._param_store.link_master_and_working_param(splited_param_current_rank, param)

        return params_current_rank

    #######################
    # Reduction Functions #
    #######################

    def _run_reduction(self):
        for group_id in range(self.num_param_groups):
            current_bucket = self._bucket_store[group_id]
            dp_parallel_mode = current_bucket.get_dp_parallel_mode()
            reduce_group = gpc.get_group(dp_parallel_mode)
            world_size = gpc.get_world_size(dp_parallel_mode)
            local_rank = gpc.get_local_rank(dp_parallel_mode)
            if current_bucket.num_elements_in_bucket() > 0:
                stream = self._comm_stream
                # waiting for ops in the default stream finishing
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    current_bucket.build_grad_in_bucket(stream)

                    flat_grads = current_bucket.get_flatten_grad()
                    flat_grads /= world_size

                    # ready to add other tensors to bucket
                    current_bucket.reset_num_elements_in_bucket()
                    group_id = current_bucket.get_param_group_id()

                    grad_dtype = flat_grads.dtype

                    if not self._partition_grads:
                        if not self.zero_1_5:
                            reduce_grads = torch.zeros(
                                flat_grads.numel() // self._zero_world_size[group_id],
                                dtype=grad_dtype,
                                device=get_current_device(),
                            )
                            dist.reduce_scatter_tensor(reduce_grads, flat_grads, group=reduce_group)

                            if reduce_grads.dtype != grad_dtype:
                                reduce_grads = reduce_grads.to(grad_dtype)

                            grad_in_bucket_current_rank = current_bucket.get_grad()[local_rank]
                            self._update_unpartitoned_grad(grad_in_bucket_current_rank, reduce_grads, group_id)
                        else:
                            # zero 1.5
                            dist.all_reduce(flat_grads, group=reduce_group)
                            if flat_grads.dtype != grad_dtype:
                                flat_grads = flat_grads.to(grad_dtype)
                            flat_grads_per_rank = flat_grads.split(
                                flat_grads.numel() // self._zero_world_size[group_id]
                            )
                            grad_in_bucket = current_bucket.get_grad()
                            self._update_unpartitoned_grad(grad_in_bucket.values(), flat_grads_per_rank, group_id)
                    else:
                        flat_grads_list = list(flat_grads.split(len(flat_grads) // world_size))
                        recieved_grad = torch.zeros_like(flat_grads_list[0])
                        dist.reduce_scatter(recieved_grad, flat_grads_list, group=reduce_group)

                        if recieved_grad.dtype != grad_dtype:
                            recieved_grad = recieved_grad.to(grad_dtype)

                        grad_in_bucket_current_rank = current_bucket.get_grad()[self._zero_local_rank[group_id]]
                        self._update_partitoned_grad(grad_in_bucket_current_rank, recieved_grad, group_id, 1)

                    current_bucket.reset()

    def _update_unpartitoned_grad(self, origin_grad_list: List, flat_grad_list: List, group_id: int) -> None:
        if not self.zero_1_5:
            sync_param(flat_grad_list, origin_grad_list)
            for grad in origin_grad_list:
                param_id = self._bucket_store[group_id].get_param_id_of_grad(grad)
                self._add_grad(grad, self._zero_world_size[group_id], group_id, param_id)
        else:
            for rank, grad_list in enumerate(origin_grad_list):
                sync_param(flat_grad_list[rank], grad_list)
                for grad in grad_list:
                    param_id = self._bucket_store[group_id].get_param_id_of_grad(grad)
                    self._add_grad(grad, self._zero_world_size[group_id], group_id, param_id, rank)

    def _update_partitoned_grad(
        self,
        origin_grad_list: List,
        flat_grad: torch.Tensor,
        group_id: int,
        partition_num: int,
    ) -> None:
        sync_param(flat_grad, origin_grad_list)
        for grad in origin_grad_list:
            param_id = self._bucket_store[group_id].get_param_id_of_grad(grad)
            self._add_grad(grad, partition_num, group_id, param_id)

    def _add_grad(
        self,
        grad: torch.Tensor,
        partition_num: int,
        group_id: int,
        param_id: int,
        rank: int = 0,
    ) -> None:
        if len(self._grad_store.get_partitioned_gradients_by_param_id(group_id, param_id)) < partition_num:
            self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)
        else:
            self._grad_store.add_gradients_by_param_id(grad, rank, group_id, param_id)

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self._bucket_store[group_id].num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or group_id != self._bucket_store[group_id].get_param_group_id()
        ):
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store[group_id].add_param_grad(param, padding_size)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False):
        assert not (
            self._partition_grads and self.skip_grad_reduce
        ), "ZeRO2(partition_grads) and no_sync are not compatible"

        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

    def backward_by_grad(self, tensor, grad):
        assert not (
            self._partition_grads and self.skip_grad_reduce
        ), "ZeRO2(partition_grads) and gradient accumulation(no_sync) are not compatible"

        torch.autograd.backward(tensor, grad)

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for _, param_group in self._working_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()
        for group_id in range(self.num_param_groups):
            self._bucket_store[group_id].reset_all()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure=None):
        assert closure is None, "closure is not supported by step()"

        self._reduce_grad(self._partition_grads)
        # clear reduced grads
        torch.cuda.synchronize()
        self.zero_grad()

        # record all grads for unscale and clip
        grad_partition_groups = []

        # sometimes not all params are 'really' working
        # for instance, when layer drop, the dropped layer has no grad
        # and should not be updated
        real_working_params = dict()
        real_master_params = dict()
        real_master_grads = dict()
        total_norms = {}

        for group_id in range(self.num_param_groups):
            master_params = self._master_param_groups_of_current_rank[group_id]
            real_working_params[group_id] = []
            real_master_params[group_id] = []
            real_master_grads[group_id] = []
            grad_index = 0 if not self.zero_1_5 else self._zero_local_rank[group_id]

            for splited_param in master_params:
                working_param = self._param_store.master_to_working_param[id(splited_param)]
                # if a working param requires grad and has no grad
                # it is not 'really' working, e.g. the droped layer
                # else the splited grad should be attached to the splited param
                grads = self._grad_store.get_partitioned_gradients_by_param_id(group_id, id(working_param))
                if len(grads) > 0:
                    real_working_params[group_id].append(working_param)
                    grad = grads[grad_index]
                    # no need to copy fp32 grad if master_weights is False
                    if self._master_weights:
                        grad = grad.to(splited_param.dtype).to(splited_param.device)
                    splited_param.grad = grad
                    grad_partition_groups.append(grad)
                    real_master_params[group_id].append(splited_param)
                    real_master_grads[group_id].append(splited_param.grad)

            # compute norm
            param_group = real_master_params[group_id]
            working_grads = real_master_grads[group_id]

            group_name = self.param_groups[group_id]["name"] if "name" in self.param_groups[group_id] else "default"
            group_name = f"{group_id}_{group_name}"
            total_norms[group_name] = self._compute_norm(
                group_id=group_id, gradients=working_grads, parameters=param_group
            )

            self._grad_store.reset_grads_by_group_id(group_id)

            # update the params in the optimizer
            self.optim.param_groups[group_id]["params"] = real_master_params[group_id]

        # check norm
        found_inf = False
        found_nan = False

        if -1 in total_norms.values():
            found_inf = True

        if -2 in total_norms.values():
            found_nan = True

        if gpc.config.model.dtype is not torch.float32:
            self.grad_scaler.update(found_inf)

        # update loss scale if overflow occurs
        if found_inf:
            if gpc.is_rank_for_log():
                logger.warning("Overflow occurs, please check it.")
            self._grad_store._grads_of_params = dict()
            self.zero_grad()
            return False, total_norms

        if found_nan:
            if gpc.is_rank_for_log():
                logger.warning("Nan grad norm occurs, please check it.")
            self._grad_store._grads_of_params = dict()
            self.zero_grad()
            return False, total_norms

        global_norm_groups = {}
        if self._clip_grad_norm > 0:
            for group_name, norm in total_norms.items():
                global_norm_groups[group_name] = norm**0.5

        # unscale and clip grads
        global_norm_l2 = calculate_global_norm_from_list(global_norm_groups)
        self._unscale_and_clip_grads(grad_partition_groups, global_norm_l2)

        # update the parameters
        self.optim.step()

        # release the grad
        grad_partition_groups = []
        for group_id in range(self.num_param_groups):
            release_param_grad(self._master_param_groups_of_current_rank[group_id])

        # update working partition updated by the current rank
        device = get_current_device()
        handles = []
        gathered_params_list = []
        working_params_list = []
        master_params_list = []
        for group_id in range(self.num_param_groups):
            if self._zero_world_size[group_id] > 1:
                master_working_param = self.optim.param_groups[group_id]["params"]

                if len(master_working_param) == 0:
                    continue

                # do all_gather at fused block granularity
                # In this way, param_overlap is available
                all_gather_master_params = []
                all_gather_working_params = []
                sum_numel_size = 0
                for idx in range(len(master_working_param)):
                    working_param = real_working_params[group_id][idx]
                    block_name = master_working_param[idx].block_name
                    # for the same block, all params are arranged in consecutive order
                    # when enter next block, check numel_size to determine whether to execute all_gather
                    if idx > 0 and block_name != master_working_param[idx - 1].block_name:
                        if sum_numel_size >= self._all_gather_size:
                            self.all_gather_params(
                                group_id,
                                all_gather_master_params,
                                all_gather_working_params,
                                gathered_params_list,
                                working_params_list,
                                master_params_list,
                                handles,
                                device,
                            )
                            all_gather_master_params = []
                            all_gather_working_params = []
                            sum_numel_size = 0

                    sum_numel_size += master_working_param[idx].numel() * self._element_size
                    all_gather_master_params.append(master_working_param[idx])
                    all_gather_working_params.append(working_param)

                # clear the last fused block
                if len(all_gather_master_params) > 0:
                    self.all_gather_params(
                        group_id,
                        all_gather_master_params,
                        all_gather_working_params,
                        gathered_params_list,
                        working_params_list,
                        master_params_list,
                        handles,
                        device,
                    )
                    all_gather_master_params = []
                    all_gather_working_params = []
            else:
                # if zero_world_size==1, directly update working param with master param
                for working_param, master_param in zip(real_working_params[group_id], real_master_params[group_id]):
                    working_param.data.copy_(master_param.view_as(working_param))

        if not self._overlap_sync_param:
            for gather_idx in range(len(handles)):
                handles[gather_idx].wait()
                # reorganize gatherd params to update working param
                # [[A1, B1], [A2, B2]] -> [[A1.reshape, A2.reshape], [B1.reshape, B2.reshape]]
                master_params_all_gather = master_params_list[gather_idx]
                gathered_params = gathered_params_list[gather_idx]
                all_splited_param_list = []
                offset = 0
                for p in master_params_all_gather:
                    param_size = p.numel()
                    all_splited_param = []
                    for all_params in gathered_params:
                        split_params = all_params[offset : offset + param_size].reshape(p.shape)
                        all_splited_param.append(split_params)
                    offset += param_size
                    all_splited_param_list.append(all_splited_param)

                # Update working parameters
                for working_param, all_splited_param in zip(working_params_list[gather_idx], all_splited_param_list):
                    working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].view_as(working_param))

        for group_id in range(self.num_param_groups):
            self.optim.param_groups[group_id]["params"] = self._master_param_groups_of_current_rank[group_id]

        for group_name, global_norm in global_norm_groups.items():
            global_norm_groups[group_name] = global_norm / float(self.loss_scale)

        return True, global_norm_groups

    def all_gather_params(
        self,
        group_id,
        all_gather_master_params,
        all_gather_working_params,
        gathered_params_list,
        working_params_list,
        master_params_list,
        handles,
        device,
    ):
        # fuse params to do all_gather
        handle, gathered_params = self.gather_fused_params(
            all_gather_master_params,
            self._zero_world_size[group_id],
            gpc.get_group(self._zero_parallel_mode[group_id]),
            device,
        )
        if self._overlap_sync_param:
            self._param_bcast_sync_handler.add_allgather_handle(
                handle,
                all_gather_master_params,
                all_gather_working_params,
                gathered_params,
                all_gather_working_params[0].block_name,
            )
        else:
            gathered_params_list.append(gathered_params)
            working_params_list.append(all_gather_working_params)
            master_params_list.append(all_gather_master_params)
            handles.append(handle)

    def gather_fused_params(self, params, world_size, group, device):
        # Flatten and concatenate all parameters into a single tensor
        flattened_params = torch.cat([p.view(-1) for p in params]).to(device).to(self._dtype)

        # Prepare the buffer for all_gather
        gathered_params = [
            torch.empty_like(flattened_params, device=device, dtype=self._dtype) for _ in range(world_size)
        ]
        # Perform the all_gather operation
        handle = dist.all_gather(gathered_params, flattened_params, group=group, async_op=True)

        return handle, gathered_params

    def _compute_norm(self, group_id, gradients, parameters):

        if len(parameters) == 0:
            return 0

        norm = 0
        if self._clip_grad_norm > 0:
            # this norm is before scaling, it will be very large
            norm = compute_norm(
                gradients=gradients, parameters=parameters, zero_mode=self._zero_parallel_mode[group_id]
            )

        return norm

    #############################
    # Mixed Precision Utilities #
    #############################

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm_groups):
        # compute combined scale factor for this group
        div_scale = float(self.loss_scale)
        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            clip = ((total_norm_groups / div_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1:
                div_scale = clip * div_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1.0 / div_scale)

    ############################
    # Gradient Synchronization #
    ############################

    # this method is used to sync gradient manually
    def _sync_grad(self):
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                if param.requires_grad and param.grad is not None:
                    self._add_to_bucket(param, group_id)

        self._run_reduction()

    def _reduce_grad(self, partition_grad):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not partition_grad and not self._overlap_sync_grad:
            self._sync_grad()
        else:
            self._run_reduction()

    ##############
    # State Dict #
    ##############

    def state_dict(self) -> Dict:
        states = {}

        grad_scaler = self.grad_scaler.state_dict()
        states["grad_scaler"] = grad_scaler
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        master_current_weights = {}
        for group_id, params in self._master_param_groups_of_current_rank.items():
            master_current_weights[group_id] = params
        states["master_current_weights"] = master_current_weights

        return states

    def load_state_dict(self, states: Dict):
        """Load state dict, requires the state_dict be the pytorch form

        Args:
            state_dict (dict): A pytorch form state_dict
        """
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

        master_current_weights = states["master_current_weights"]
        for group_id, params in master_current_weights.items():
            if len(params) > 0:
                self_params = self._master_param_groups_of_current_rank[group_id]
                assert len(self_params) == len(
                    params
                ), f"The loaded parameter shape is inconsistent, {self_params.shape} != {params.shape}"
                for self_param, param in zip(self_params, params):
                    self_param.data.copy_(param.data)

    def reload_zero_fp32_buff(self):
        for group_id, param_group in enumerate(self.optim.param_groups):
            if len(param_group["params"]) > 0:
                for master_param in param_group["params"]:
                    working_param = self._param_store.master_to_working_param[id(master_param)]
                    padding_size = self._param_store.get_param_padding_size(working_param)

                    with torch.no_grad():
                        if padding_size > 0:
                            padding_param = torch.nn.functional.pad(working_param.data.view(-1), [0, padding_size])
                        else:
                            padding_param = working_param.data.view(-1)

                        splited_params = padding_param.split(padding_param.numel() // self._zero_world_size[group_id])
                        splited_params = splited_params[self._zero_local_rank[group_id]]
                        splited_param_current_rank = splited_params.detach().float()

                    master_param.data.copy_(splited_param_current_rank)

    ################
    # Overlap Hook #
    ################

    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._working_param_groups[group_id]
            for param in param_group:
                # we should not reduce the param in moe
                if not param.requires_grad:
                    continue

                reduce_rank = None

                def _define_and_attach(param, reduce_rank=None):
                    # pylint: disable=W0640
                    def grad_handler(group_id, param):
                        # if run with no_sync context, would not sync grad when backward
                        if not self.skip_grad_reduce:
                            self._add_to_bucket(param, group_id)

                    reduce_scatter_checker = partial(
                        self._wait_reduce_scatter_and_accumulate_grads,
                        param=param,
                    )

                    def reduction_layernorm_func():
                        handle = reduce_tensor(
                            param.grad,
                            dtype=None,
                            dst_rank=reduce_rank,
                            parallel_mode=ParallelMode.WEIGHT if self.use_isp else ParallelMode.TENSOR,
                        )
                        handle.wait()

                    # define hook for real gradient accumulation.

                    def accum_grad_hook(*args):  # pylint: disable=W0613
                        reduce_scatter_checker()

                    # define hook for sequence_parallel
                    def extra_layernorm_reduce_grad_hook(*args):  # pylint: disable=W0613
                        if self.skip_grad_reduce is False:
                            reduction_layernorm_func()

                    # the grad of layernorm should be all-reduce across the global process group
                    # here is the first stage all-reduce in tp/wp process group
                    # the second stage all-reduce will be processed in reduce_grad_hook
                    if (
                        is_using_sequence_parallel()
                        and hasattr(param, IS_REPLICA_ZERO_PARALLEL)
                        and getattr(param, IS_REPLICA_ZERO_PARALLEL) is True
                    ):
                        param.register_post_accumulate_grad_hook(extra_layernorm_reduce_grad_hook)

                    # we should not only register for parameters which have isp_reduce_scatter_name attr.
                    # we must keep up with reduce_grad_hook.
                    if (
                        self._isp_communicator
                        and self._isp_communicator.overlap
                        and gpc.config.parallel.weight.size > 1
                    ):
                        param.register_post_accumulate_grad_hook(accum_grad_hook)

                    if self._overlap_sync_grad:
                        param.register_post_accumulate_grad_hook(
                            partial(grad_handler, group_id)
                        )  # pylint: disable=W0640

                _define_and_attach(param, reduce_rank)
