# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Uses `get_length_grouped_indices` from HuggingFace transformers
# (HuggingFace Inc., Apache-2.0).
import sys
from typing import Optional

from transformers.trainer_pt_utils import get_length_grouped_indices

from sensenovalm.data.tokenized.batch_sampler import StaticBatchSampler
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


class StreamingStaticBatchSampler:
    """
    StreamingStaticBatchSampler is used for the training process.
    """

    def __init__(
        self, batch_size: int = 1, rampup_batch_size: Optional[str] = None, micro_bsz: int = 1
    ):  # pylint: disable=W0613
        if rampup_batch_size:
            # In the process increase to batch_size
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1

        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every

        self.batch_size = batch_size
        self.batch_count = 0

    def __len__(self):
        return sys.maxsize

    def __iter__(self):
        while True:
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            yield [0] * cur_batch_size
            self.batch_count += 1

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
        }
        return states

    def load_state_dict(self, states):
        for name in ("raw_rampup_batch_size",):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.batch_count = states["batch_count"]

    def copy(self):
        copy_sampler = StreamingStaticBatchSampler(self.batch_size, self.raw_rampup_batch_size)

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler


class LengthGroupedSampler(StaticBatchSampler):
    """
    LengthGroupedSampler
    """

    def __init__(self, datasets, **kwargs):
        lengths = []
        for dataset in datasets:
            lengths.extend(dataset.length)
        self.lengths = lengths

        super().__init__(datasets=datasets, **kwargs)
        self.rng_state = self.rng.get_state()

        assert len(self.lengths) == self.num_samples
        assert not self.raw_rampup_batch_size, f"'{self.raw_rampup_batch_size}' is not supported!"

    def get_indices(self, old_indices=None):
        if self.lengths is None:
            return super().get_indices(old_indices=old_indices)

        if old_indices is not None:
            raise NotImplementedError(f"{old_indices=} is not supported yet.")

        num_samples = self.num_samples // (self.batch_size * self.data_world_size)
        num_samples = num_samples * self.batch_size * self.data_world_size

        self.indices = get_length_grouped_indices(
            lengths=self.lengths,
            batch_size=self.batch_size,
            mega_batch_mult=self.data_world_size,
        )[:num_samples]
