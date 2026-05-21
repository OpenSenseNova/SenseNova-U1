# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
from typing import List, Union

from torch import nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from sensenovalm.model.base_model import BaseModel
from sensenovalm.model.registry import model_initializer
from sensenovalm.utils.common import filter_kwargs, get_current_device
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


def create_model(model_type, model_conf) -> Union[nn.Module, List[nn.Module]]:

    # for InternEvo
    # kwargs = dict(gpc.config.model)

    kwargs = model_conf

    num_layers = kwargs.pop("num_layers")
    num_chunks = kwargs.pop("num_chunks", 1)
    pipeline_size = kwargs.pop("pipeline_size", 1)
    pipeline_rank = kwargs.pop("pipeline_rank", 0)

    # TODO: fix use_flash_attn parameter config
    kwargs.pop("use_flash_attn", False)
    kwargs.pop("apply_post_layer_norm", False)
    kwargs.pop("embed_split_hidden", True)

    kwargs["checkpoint"] = float(kwargs.get("checkpoint", False))
    kwargs["device"] = get_current_device()

    model_buidler = model_initializer.get_module(module_name=model_type)
    kwargs = filter_kwargs(model_buidler, kwargs)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        kwargs["first"] = kwargs["last"] = True
        kwargs["start_layer_idx"] = 0
        kwargs["num_layers"] = num_layers
        model = model_buidler(**kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
        gpc.config.model.first_layer = 0
        gpc.config.model.last_layer = num_layers
        gpc.config.model.num_layers_for_pp = kwargs["num_layers"]
    else:
        model = pipeline_parallel_sharding_wrapper(
            num_layers, num_chunks, model_buidler, pipeline_size=pipeline_size, pipeline_rank=pipeline_rank, **kwargs
        )

    if not isinstance(model, BaseModel) and gpc.is_rank_for_log():
        logger.warning(f"To load/save huggingface ckpt, built-in model should inherited from {BaseModel.__name__}")

    return model
