# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
"""
shard strategies for parallel
"""

from typing import Callable

import torch
from torch import nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.comm.utils import _gather, _split
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.utils import TensorParallelMode

logger = get_logger(__file__)


def _split_data_for_sequence_parallel(data, label):
    _seq_dim = 1  # [batch, seqlen, ...]
    _indexes_seq_dim = 0  # [seqlen, ...]

    # NOTICE: since cu_seqlens is used by attention, it should not be splited.
    # NOTICE: indexes are only used by rotary embedding. There are a few cases:
    # 1. msp/fsp: After wqkv computation, the hidden states are complete along the sequence dimension,
    #    so we should use the complete indexes when computing the rotary embedding.
    # 2. isp: After wqkv computation, the hidden states are segmented along the sequence dimension,
    #    so we need to segment the indexes accordingly.
    if (
        "indexes" in data
        and gpc.config.parallel["tensor"].get("mode", TensorParallelMode.mtp.name) == TensorParallelMode.isp.name
    ):
        data["indexes"] = _split(data["indexes"], ParallelMode.TENSOR, dim=_indexes_seq_dim)

    # NOTICE: For compatibility where the shape of position_ids is [batch, seqlen, ...]
    if "inject_info" in gpc.config.model and gpc.config.model.inject_info.get("data_helper", False):
        _position_ids_seq_dim = 1
        data["position_ids"] = _split(data["position_ids"], ParallelMode.TENSOR, dim=_position_ids_seq_dim)

    data["input_ids"] = _split(data["input_ids"], ParallelMode.TENSOR, dim=_seq_dim)


    return data, label


def _split_data_for_2D_sequence_parallel(data, label):
    if gpc.config.parallel.sequence_2D.enable is False or gpc.get_world_size(ParallelMode.TENSOR) <= 1:
        return data, label

    assert len(data.keys()) == 3 and "input_ids" in data and "indexes" in data and "max_seqlen" in data

    sp_size = gpc.get_world_size(ParallelMode.TENSOR)
    hp_size = gpc.get_world_size(ParallelMode.HEAD)
    cp_size = gpc.get_world_size(ParallelMode.CONTEXT)
    hp_rank = gpc.get_local_rank(ParallelMode.HEAD)
    cp_rank = gpc.get_local_rank(ParallelMode.CONTEXT)
    stride = 2

    assert len(data["input_ids"].shape) == 2
    assert len(data["indexes"].shape) == 1
    assert len(label.shape) == 2
    seq_dim = 1
    data["input_ids"] = data["input_ids"].view(
        *data["input_ids"].shape[0:seq_dim],
        2 * sp_size,
        data["input_ids"].shape[seq_dim] // (2 * sp_size),
    )
    _index_seq_dim = 0
    data["indexes"] = data["indexes"].view(
        2 * sp_size,
        data["indexes"].shape[_index_seq_dim] // (2 * sp_size),
    )
    label = label.view(
        *label.shape[0:seq_dim],
        2 * sp_size,
        label.shape[seq_dim] // (2 * sp_size),
    )

    # get selected index
    if hp_size == sp_size:
        # rank0:0,1 rank1:2,3 rank2:4,5 rank3:6,7 rank4:8,9 rank5:10,11 rank6:12,13 rank7:14,15
        index = torch.tensor([hp_rank * stride, (hp_rank * stride + 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
    elif cp_size == sp_size:
        # rank0:0,15 rank1:1,14 rank2:2,13 rank3:3,12 rank4:4,11 rank5:5,10 rank6:6,9 rank7:7,8
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
    else:
        """
        hp_size=2 cp_size=4 head-first
        rank0:0,1 rank1:14,15 rank2:2,3 rank3:12,13 rank4:4,5 rank5:10,11 rank6:6,7 rank7:8,9

        hp_size=4 cp_size=2 head-first
        rank0:0,1 rank1:2,3 rank2:12,13 rank3:14,15 rank4:4,5 rank5:6,7 rank6:8,9 rank7:10,11

        hp_size=2 cp_size=4 context-first
        rank0:0,1 rank1:2,3 rank2:4,5 rank3:6,7 rank4:14,15 rank5:12,13 rank6:10,11 rank7:8,9

        hp_size=4 cp_size=2 context-first
        rank0:0,1 rank1:4,5 rank2:2,3 rank3:6,7 rank4:12,13 rank5:8,9 rank6:14,15 rank7:10,11
        """
        if hp_rank < (hp_size // 2):
            _index = hp_rank * stride + cp_rank * hp_size
        else:
            _index = (2 * sp_size - 2) - (hp_size - hp_rank - 1) * stride - cp_rank * hp_size

        index = torch.tensor([_index, _index + 1], device="cpu", pin_memory=True).cuda(non_blocking=True)

    data["input_ids"] = data["input_ids"].index_select(seq_dim, index)
    data["input_ids"] = data["input_ids"].view(
        *data["input_ids"].shape[0:seq_dim], -1, *data["input_ids"].shape[(seq_dim + 2) :]
    )
    data["indexes"] = data["indexes"].index_select(_index_seq_dim, index)
    data["indexes"] = data["indexes"].view(
        *data["indexes"].shape[0:_index_seq_dim], -1, *data["indexes"].shape[(_index_seq_dim + 2) :]
    )
    label = label.index_select(seq_dim, index)
    label = label.view(*label.shape[0:seq_dim], -1, *label.shape[(seq_dim + 2) :])

    # if gpc.config.model.parallel_output is False:
    label = _gather(label, ParallelMode.TENSOR, dim=seq_dim)

    return data, label


def split_data_for_sequence_parallel(data, label):
    if gpc.config.parallel.sequence_2D.enable is False:
        return _split_data_for_sequence_parallel(data, label)

    return _split_data_for_2D_sequence_parallel(data, label)


# The head layer in ISP mode is actually a special case,
# and we would prefer a unified segmentation and communication logic.
def get_tensor_split_parallel_mode(is_expert=False) -> ParallelMode:
    tp_mode = gpc.config.parallel.tensor.mode

    if tp_mode == TensorParallelMode.isp.name and not is_expert:
        return ParallelMode.WEIGHT
    elif tp_mode == TensorParallelMode.isp.name and is_expert:
        return ParallelMode.EXPERT_WEIGHT
    elif tp_mode != TensorParallelMode.isp.name and is_expert and gpc.config.parallel.expert.no_tp:
        return ParallelMode.EXPERT_TENSOR
    else:
        return ParallelMode.TENSOR


def get_head_parallel_mode() -> ParallelMode:
    return ParallelMode.TENSOR


def get_parallel_strategies_split_mode(linear_name: str) -> str:
    tp_mode = gpc.config.parallel.tensor.mode

    if linear_name in ("head", "output"):
        return "head"
    if linear_name in ("gate"):
        return "gate"  # for MoE model
    elif linear_name in ("wqkv", "wq", "wk", "wv", "wkv", "w1", "w3", "w13"):
        return "column"
    elif linear_name in ("fc1", "fc2", "linear_1", "linear_2"):  # for vit model
        return "column"
    elif linear_name in ("wo", "out_proj", "w2") and tp_mode == TensorParallelMode.isp.name:
        return "column"
    elif linear_name in ("wo", "out_proj", "w2"):
        return "row"
    elif linear_name in ("grouped_w1", "grouped_w2", "grouped_w3") and tp_mode == "isp":
        return "grouped_wp"
    elif linear_name in ("grouped_w1", "grouped_w3"):
        return "grouped_column"
    elif linear_name in ("grouped_w2"):
        return "grouped_row"
    else:
        return "unknown"


def partition_uniform(num_items: int, pipeline_parallel_size: int, num_chunks: int):
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            raise ValueError("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    return parts


def pipeline_parallel_sharding_wrapper(
    num_layers: int,
    num_chunks: int,
    model_builder: Callable,
    device: torch.device,
    pipeline_size: int,
    pipeline_rank: int,
    **kwargs,
):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """


    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]

    if gpc.is_rank_for_log():
        logger.info("The layer sharding is %r.", all_parts)

    models = []

    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start

        chunk = model_builder(**kwargs).to(device)
        setattr(chunk, "first_layer", start)
        setattr(chunk, "last_layer", end)
        gpc.config.model.first_layer = start
        gpc.config.model.last_layer = end
        gpc.config.model.num_layers_for_pp = kwargs["num_layers"]

        models.append(chunk)


    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model
