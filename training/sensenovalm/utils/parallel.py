#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist

from sensenovalm.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.modules.utils import is_gate_param, is_moe_param
from sensenovalm.utils.utils import TensorParallelMode


def is_using_sequence_parallel():
    return (
        isinstance(gpc.config.parallel["tensor"], dict)
        and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) != TensorParallelMode.mtp.name
        and gpc.config.parallel["tensor"]["size"] > 1
    )


def is_using_isp():
    return (
        isinstance(gpc.config.parallel["tensor"], dict)
        and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.isp.name
    )


def is_replica_zero_parallel_parameter(p):
    return hasattr(p, IS_REPLICA_ZERO_PARALLEL) and getattr(p, IS_REPLICA_ZERO_PARALLEL)


def is_tensor_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and not is_using_isp()
        and hasattr(p, IS_TENSOR_ZERO_PARALLEL)
        and getattr(p, IS_TENSOR_ZERO_PARALLEL)
    )


def is_weight_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.WEIGHT)
        and is_using_isp()
        and hasattr(p, IS_WEIGHT_ZERO_PARALLEL)
        and getattr(p, IS_WEIGHT_ZERO_PARALLEL)
    )


def is_tensor_expert_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and hasattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
        and getattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
    )


def is_weight_expert_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.WEIGHT)
        and hasattr(p, IS_WEIGHT_EXPERT_DATA_PARALLEL)
        and getattr(p, IS_WEIGHT_EXPERT_DATA_PARALLEL)
    )


def is_replica_expert_data_parallel_parameter(p):
    return hasattr(p, IS_REPLICA_EXPERT_DATA_PARALLEL) and getattr(p, IS_REPLICA_EXPERT_DATA_PARALLEL)


def should_reduce_replica_param(p):
    _reduce = False

    if not is_replica_zero_parallel_parameter(p):
        return _reduce

    # for replica parameter
    if gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.mtp.name:
        _reduce = False
    elif gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) in (
        TensorParallelMode.msp.name,
        TensorParallelMode.fsp.name,
    ):
        _reduce = gpc.is_using_parallel_mode(ParallelMode.TENSOR)
    elif gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.isp.name:
        _reduce = gpc.is_using_parallel_mode(ParallelMode.WEIGHT)

    if not is_gate_param(p):
        return _reduce

    # for moe gate parameter
    if gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.mtp.name:
        _reduce = gpc.is_using_parallel_mode(ParallelMode.TENSOR) and getattr(
            gpc.config.parallel.expert, "no_tp", False
        )

    return _reduce


def sync_model_param(model):
    r"""Make sure data parameters are consistent during Data Parallel Mode.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    sync_moe_param = gpc.is_using_parallel_mode(ParallelMode.EXPERT_DATA)
    sync_parallel_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    for param in model.parameters():
        if getattr(param, "is_expert", False):
            if sync_moe_param:
                ranks = gpc.get_ranks_in_group(ParallelMode.EXPERT_DATA)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.EXPERT_DATA))
        else:
            ranks = gpc.get_ranks_in_group(sync_parallel_mode)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(sync_parallel_mode))


def sync_model_replica_param_group(model):
    r"""This function is changed from colossalai, which is ``sync_model_param``.

    We modified this function to make sure it only sync IS_REPLICA_ZERO_PARALLEL parameters in tp or wp process group.
    This function is used to make sure parameters that are not splitted are the same across each rank.
    For example, parameters like RMSNorm, LayerNorm...

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """

    parallel_mode = ParallelMode.WEIGHT if is_using_isp() else ParallelMode.TENSOR
    if gpc.is_using_parallel_mode(parallel_mode):
        for param in model.parameters():
            if is_replica_zero_parallel_parameter(param):
                ranks = gpc.get_ranks_in_group(parallel_mode)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(parallel_mode))


def get_parallel_log_file_name():
    if gpc.is_rank_for_log():
        fn_prefix = "main_"  # Indicates a rank with more output information
    else:
        fn_prefix = ""

    if is_using_isp():
        log_file_name = (
            f"{fn_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
            f"wp={gpc.get_local_rank(ParallelMode.WEIGHT)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
        )
    else:
        log_file_name = (
            f"{fn_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
            f"tp={gpc.get_local_rank(ParallelMode.TENSOR)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
        )

    return log_file_name


def check_parallel_statistic_equality(model):
    for name, params in model.named_parameters():
        if "bias" in name:
            continue
        named_mean = params.to(dtype=torch.float64).mean()
        named_std = params.to(dtype=torch.float64).std()
        if not is_moe_param(params):
            if is_tensor_zero_parallel_parameter(params):
                check_statistic_equality(name, named_mean, ParallelMode.TENSOR, eq=False, is_mean=True)
                check_statistic_equality(name, named_std, ParallelMode.TENSOR, eq=False, is_mean=False)
            elif is_weight_zero_parallel_parameter(params):
                check_statistic_equality(name, named_mean, ParallelMode.WEIGHT_DATA, eq=True, is_mean=True)
                check_statistic_equality(name, named_std, ParallelMode.WEIGHT_DATA, eq=True, is_mean=False)
            elif is_replica_zero_parallel_parameter(params):
                check_statistic_equality(name, named_mean, ParallelMode.WEIGHT, eq=True, is_mean=True)
                check_statistic_equality(name, named_std, ParallelMode.WEIGHT_DATA, eq=True, is_mean=False)
                check_statistic_equality(name, named_mean, ParallelMode.TENSOR, eq=True, is_mean=True)
                check_statistic_equality(name, named_std, ParallelMode.DATA, eq=True, is_mean=False)
            elif is_tensor_expert_data_parallel_parameter(params):
                check_statistic_equality(name, named_mean, ParallelMode.TENSOR, eq=False, is_mean=True)
                check_statistic_equality(name, named_std, ParallelMode.TENSOR, eq=False, is_mean=False)

        if is_tensor_expert_data_parallel_parameter(params) or is_weight_expert_data_parallel_parameter(params):
            # for moe param, we always check "in EXPERT_DATA" mode (e.g. wp)
            check_statistic_equality(name, named_mean, ParallelMode.EXPERT_DATA, eq=True, is_mean=True)
            check_statistic_equality(name, named_std, ParallelMode.EXPERT_DATA, eq=True, is_mean=False)
        elif gpc.weight_parallel_size >= gpc.tensor_parallel_size:
            check_statistic_equality(name, named_mean, ParallelMode.WEIGHT_DATA, eq=True, is_mean=True)
            check_statistic_equality(name, named_std, ParallelMode.WEIGHT_DATA, eq=True, is_mean=False)
        else:
            check_statistic_equality(name, named_mean, ParallelMode.DATA, eq=True, is_mean=True)
            check_statistic_equality(name, named_std, ParallelMode.DATA, eq=True, is_mean=False)


def check_statistic_equality(name, value, mode, eq=False, is_mean=True, rtol=1e-2, atol=1e-2):
    """
    Check if statistical values are equal across different processes in a distributed setting.

    Args:
        name (str): Name of the statistic being checked
        value (torch.Tensor): Value to check
        mode: Communication mode
        eq (bool): If True, raises error when values are different; if False, only warns
        is_mean (bool): If True, checking mean values; if False, checking std values
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison
    """
    group_world_size = gpc.get_world_size(mode)

    if group_world_size > 1:
        group_local_rank = gpc.get_local_rank(mode)
        group = gpc.get_group(mode)
        value_list = [torch.zeros_like(value)] * group_world_size
        dist.all_gather(value_list, value, group)

        for i in range(group_local_rank + 1, group_world_size):
            values_are_close = torch.allclose(value_list[i], value, rtol=rtol, atol=atol)

            if eq and not values_are_close:
                # When eq=True, raise error if values are NOT close
                error_message = (
                    f"On {str(mode)}, "
                    f"{'mean' if is_mean else 'std'} values of {name} "
                    f"are different between "
                    f"rank{group_local_rank}({gpc.get_global_rank()}):{value} "
                    f"and rank{i}:{value_list[i]}"
                )
                raise AssertionError(error_message)
            # elif not eq and values_are_close:
            #     # When eq=False, warn if values are NOT close
            #     print(
            #         f"On {str(mode)}, "
            #         f"{'mean' if is_mean else 'std'} values of {name} "
            #         f"are close between "
            #         f"rank{group_local_rank}({gpc.get_global_rank()}):{value} "
            #         f"and rank{i}:{value_list[i]}",
            #         flush=True,
            #     )
