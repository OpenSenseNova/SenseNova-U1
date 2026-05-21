# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer import TrainState


def get_train_state(dataloader):
    if gpc.config.data.type in [
        "multimodal_streaming",
        "multimodal_packed_streaming",
        "random_multimodal_packed_streaming",
    ]:
        trainstate = TrainState(gpc.config, dataloader.batch_sampler)
    else:
        raise ValueError(f"dataset type {gpc.config.data.type} is not supported")
    return trainstate
