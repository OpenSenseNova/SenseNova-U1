# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
from collections import defaultdict

from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer import TrainState
from sensenovalm.utils.utils import DataType


class StreamingTrainState(TrainState):
    """
    This state should be selected when using streaming-style datasets. Because streaming-style datasets
    need to save the state of the dataset rather than the state of the sampler.
    """

    def __init__(self, config, batch_sampler, _init_marker=None) -> None:
        super().__init__(config, batch_sampler=batch_sampler, _init_marker=_init_marker)
        self.data_state_dict = {
            "rng_state": None,
            "multiple_packed_states": {},
            "consumed_samples": {},
            "used_epochs": defaultdict(int),
            "epochs_to_use": defaultdict(int),
            # Note that each data rank also holds a copy of this state;
            "dataset_consumed_tokens": defaultdict(int),
            "sample_offsets_to_last_worker": [],
        }


def get_train_state(dataloader=None) -> TrainState:
    """
    Args:
        dataloader (torch.utils.data.Dataset):

    Raises:
        ValueError: Only support tokenized/streaming dataset.

    Returns:
        TrainState
    """
    batch_sampler = dataloader.batch_sampler if dataloader is not None else None
    # initialize and resume train state
    if gpc.config.data.type in [
        DataType.tokenized.name,
        DataType.megatron.name,
        DataType.mocked.name,
        DataType.multimodal_streaming.name,
        DataType.multimodal_packed_streaming.name,
    ]:
        train_state = TrainState(gpc.config, batch_sampler, _init_marker="get_train_state")
    elif gpc.config.data.type == DataType.streaming.name:
        train_state = StreamingTrainState(gpc.config, batch_sampler, _init_marker="get_train_state")
    else:
        raise ValueError(f"dataset type {gpc.config.data.type} is not supported")

    # For conveniently obtaining batch_count through gpc.
    gpc.train_state = train_state
    return train_state
