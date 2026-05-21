#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import gc
import math

import torch
import torch.distributed as dist
from torch.utils import benchmark

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.utils.common import get_current_device
from sensenovalm.utils.logger import get_logger

try:
    import GPUtil
    import psutil
except ImportError:
    GPUtil, psutil = None, None

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


# Gloabl cuda cache flush counter
n_caching_allocator_flushes = 0


def empty_cache_and_diag(batch_count, bcast_sumbit_hook, interval=50):
    """empty cuda cache and run diag bench or tests."""
    if interval <= 0:
        interval = 50

    cuda_memory_analyze(batch_count, batch_count % int(interval) == 0 or batch_count <= 5)

    if batch_count % int(interval) == 0:
        bcast_sumbit_hook(manual_submit_count=8)
        # there is no need to do diag on the first batch
        if batch_count > 0:
            if gpc.is_rank_for_log():
                logger.info("Empty Cache and Diagnosis GPU/NCCL/Timer ...")
        # do empty_cache after the bench
        sensenovalm_accelerator.empty_cache()
        # do garbage collection
        gc.collect()


def benchmark_forward(
    test_fn,
    *inputs,
    repeats=100,
    amp=True,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type=sensenovalm_accelerator.get_backend_name(), dtype=amp_dtype, enabled=amp):
            test_fn(*inputs, **kwinputs)

    bench_timer = benchmark.Timer(
        stmt="test_fn_amp(*inputs, **kwinputs)",
        globals={"test_fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    used_time = bench_timer.timeit(repeats)
    return used_time.mean


def flops(batch, seqlen, headdim, nheads, time_f):
    """Compute the flops value of a GPU with give flashattention function"""

    flop = 4 * batch * seqlen**2 * nheads * headdim
    return (flop / time_f / 10**12) if not math.isnan(time_f) else 0.0


def get_gpu_temperature():
    """Get current GPU temperature."""
    try:
        gpu_id = sensenovalm_accelerator.get_device_id()
    except AssertionError:
        gpu_id = -1

    if GPUtil is not None and gpu_id >= 0 and sensenovalm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
        gpus = GPUtil.getGPUs()
        gpu_temperature = gpus[gpu_id].temperature
    else:
        gpu_temperature = -1

    return gpu_temperature


def get_cpu_temperature():
    """Get current CPU temperature."""

    if psutil is not None:
        cpu_temperature = psutil.sensors_temperatures()["coretemp"][0].current
    else:
        cpu_temperature = -1

    return cpu_temperature


def warmup_process_group():
    # Prevent OOM from nccl communication.
    if dist.is_initialized():
        buffer = torch.ones([64], device=get_current_device())
        if gpc.is_initialized(ParallelMode.DATA):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.DATA))
        if gpc.is_initialized(ParallelMode.TENSOR):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.TENSOR))
        if gpc.is_initialized(ParallelMode.PIPELINE):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.PIPELINE))
        if gpc.is_initialized(ParallelMode.ZERO1):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.ZERO1))
        if gpc.is_initialized(ParallelMode.ZERO3_DP):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.ZERO3_DP))
        if gpc.is_initialized(ParallelMode.EXPERT_DATA):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.EXPERT_DATA))
        if gpc.is_initialized(ParallelMode.EXPERT):
            dist.all_reduce(buffer, group=gpc.get_group(ParallelMode.EXPERT))

        dist.barrier()
        del buffer
        sensenovalm_accelerator.empty_cache()


def cuda_memory_analyze(step=0, print_mm_suage=False):
    """
    Useful utility functions migrated from deepseped.
    """

    global n_caching_allocator_flushes

    g_rank = gpc.get_global_rank()
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    dp_rank = gpc.get_local_rank(ParallelMode.DATA)
    rank_id = f"Rank:{g_rank}-tp{tp_rank}-pp{pp_rank}-dp{dp_rank}"

    if print_mm_suage and gpc.get_local_rank(ParallelMode.DATA) == 0:
        logger.info(
            f"{rank_id}: Step {step}: "
            f"Allocated {round(sensenovalm_accelerator.memory_allocated() / (1024 * 1024 * 1024),4 )} GB, "
            f"Max_Allocated {round(sensenovalm_accelerator.max_memory_allocated() / (1024 * 1024 * 1024),4)} GB, "
            f"Reserved {round(sensenovalm_accelerator.memory_reserved()/ (1024 * 1024 * 1024),4)} GB, "
            f"Max_Reserved {round(sensenovalm_accelerator.max_memory_reserved()/ (1024 * 1024 * 1024),4)} GB "
        )

        sensenovalm_accelerator.reset_peak_memory_stats()

    # warn user about caching allocator flushes
    memory_stats = sensenovalm_accelerator.memory_stats()
    alloc_retries = memory_stats.get("num_alloc_retries")
    if alloc_retries is None:
        alloc_retries = 0
    if alloc_retries > n_caching_allocator_flushes:
        retry_count = alloc_retries - n_caching_allocator_flushes
        if gpc.get_global_rank() == 0:
            logger.warning(
                f"{rank_id}: pytorch allocator cache flushes {retry_count} times since last step."
                "this happens when there is high memory pressure and is detrimental to "
                "performance. if this is happening frequently consider adjusting "
                "settings to reduce memory consumption. If you are unable to "
                "make the cache flushes go away consider adding "
                "sensenovalm_accelerator.empty_cache() calls in your training loop to ensure "
                "that all ranks flush their caches at the same time"
            )
        n_caching_allocator_flushes = alloc_retries
