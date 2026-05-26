#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import math
import random
from collections import defaultdict
from typing import Iterator, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)

T_co = TypeVar("T_co", covariant=True)


class DataParallelSampler(Sampler):
    """A data sampler for distributed data parallelism.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The Dataset for sampling.
        shuffle (bool, optional): Whether to shuffle data, defaults to False.
        seed (int, optional): The random seed used for sampling, defaults to 0.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.num_replicas = gpc.get_world_size(ParallelMode.DATA)
        self.rank = gpc.get_local_rank(ParallelMode.DATA)
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # update for next epoch so that there is no need to call
            # set_epoch manually
            self.epoch += 1
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class StaticBatchSampler:
    """
    A static batch sampler that generates batches with a fixed micro-batch size.

    Args:
        num_samples (int): The total number of samples in the dataset.
        batch_size (int): The batch size for the current rank. Defaults to 192.
        rampup_batch_size (str): A string with three space-separated integers representing the
                                 starting batch size, the increment, and the number of steps between
                                 each increment. For example, "192 24 8" means that the batch size
                                 starts at 192 and increases by 24 every 8 steps. Defaults to
                                 "6 2 8", which corresponds to a batch size of 2 for the first 6 steps.
        micro_bsz (int): The micro-batch size. Defaults to 2.
        seed (int): The random seed for shuffling the indices. Defaults to 0.
        drop_last (bool): If True, drop the last incomplete batch. Currently only supports True. Defaults to True.
        data_rank (int): The rank of the current process in the data parallel group. Defaults to 0.
        data_world_size (int): The number of processes in the data parallel group. Defaults to 1.
    """

    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            # In the process increase to batch_size
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        if gpc.is_initialized(ParallelMode.PIPELINE):
            assert (
                batch_size - self.start_bsz
            ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
            assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.start_bsz % micro_bsz == 0
            ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.bsz_incre % micro_bsz == 0
            ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.num_samples = sum([len(ds) for ds in datasets])

        self.get_indices()  # get data

    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"

        else:
            old_indices = np.array([])

        # indices includes len(old_indices) but not self.num_samples
        indices = np.arange(len(old_indices), self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        indices = np.concatenate([old_indices, indices]).astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
            yield batch

        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = StaticBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler


class WeightedSampler:
    """
    Sampling from datasets with different weights.
    Save/Load is supported to resume the sampling with fully deterministic
    Args:
        datasets (list): list of packed_datasets
        dataset_weights (list): list of float numbers, the sampling weights of the datasets
        seed (int): random seed
    """

    def __init__(
        self,
        datasets,
        dataset_weights,
        seed=0,
    ):
        assert len(datasets) == len(dataset_weights), f"{len(datasets)} != {len(dataset_weights)}"
        self.datasets = datasets
        self.dataset_names = [ds.get_dataset_name() for ds in datasets]
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.dataset_seeds = [seed + i for i in range(len(datasets))]

        self.dataset_weights = dataset_weights
        self.num_datasets = len(datasets)

        sum_weight = sum(dataset_weights)
        self.probabilities = np.array(dataset_weights) / sum_weight
        self.iter_rng = np.random.RandomState(seed)
        self.indices = [None for _ in range(len(self.dataset_names))]
        self.current_pos = [0 for _ in range(len(self.dataset_names))]
        self.consumed_samples = [0 for _ in range(len(self.dataset_names))]
        self.cum_lengths = np.cumsum(self.dataset_lengths)
        for idx in range(len(datasets)):
            seed = self.dataset_seeds[idx]
            self.shuffle_dataset(idx, seed)

    def get_num_samples(self):
        num_samples = []
        for prob, length in zip(self.probabilities, self.dataset_lengths):
            num_samples.append(int(length / (prob + 1e-8)))
        return min(num_samples)

    def get_global_index(self, dataset_idx, idx):
        if dataset_idx == 0:
            return idx
        else:
            return self.cum_lengths[dataset_idx - 1] + idx

    def shuffle_dataset(self, didx, seed):
        self.dataset_seeds[didx] = seed
        rng = np.random.RandomState(seed)
        indices = np.arange(self.dataset_lengths[didx])
        rng.shuffle(indices)
        self.indices[didx] = indices
        self.current_pos[didx] = 0

    def next_sample_in_dataset(self, didx):
        xidx = self.current_pos[didx]
        xidx = self.indices[didx][xidx]
        self.current_pos[didx] += 1
        self.consumed_samples[didx] += 1
        if self.current_pos[didx] >= self.dataset_lengths[didx]:
            seed = (self.dataset_seeds[didx] + 1) % 65536
            self.shuffle_dataset(didx, seed)
        return self.get_global_index(didx, xidx)

    def next_batch(self, batch_size=1):
        dataset_idxes = self.iter_rng.choice(self.num_datasets, size=batch_size, p=self.probabilities)
        indices = []
        for i in range(batch_size):
            didx = dataset_idxes[i]
            indices.append(self.next_sample_in_dataset(didx))
        return indices

    def state_dict(self):
        states = {
            "rng_state": self.iter_rng.get_state(),
            "dataset_names": self.dataset_names,
            "dataset_seeds": self.dataset_seeds,
            "dataset_lengths": self.dataset_lengths,
            "dataset_weights": self.dataset_weights,
            "current_pos": self.current_pos,
            "consumed_samples": self.consumed_samples,
        }
        return states

    def load_state_dict(self, states):
        self.iter_rng.set_state(states["rng_state"])

        for didx, name in enumerate(self.dataset_names):
            try:
                st_didx = states["dataset_names"].index(name)
            except ValueError:
                continue
            assert (
                self.dataset_lengths[didx] == states["dataset_lengths"][st_didx]
            ), f"Size of dataset {name} is changed!"
            new_seed = states["dataset_seeds"][st_didx]
            # updating indices
            if new_seed != self.dataset_seeds[didx]:
                self.shuffle_dataset(didx, new_seed)
            self.current_pos[didx] = states["current_pos"][st_didx]
            self.consumed_samples[didx] = states["consumed_samples"][st_didx]
            # NO NEED to load dataset_weights, set in gpc.configs
            # self.dataset_weights[didx] = states['dataset_weights'][st_didx]
        assert len(self.probabilities) == self.num_datasets, f"{len(self.probabilities)}, {self.num_datasets}"
        assert len(self.dataset_weights) == self.num_datasets, f"{len(self.dataset_weights)}, {self.num_datasets}"
        for st_didx, name in enumerate(states["dataset_names"]):
            try:
                didx = self.dataset_names.index(name)
                continue
            except ValueError:
                self.dataset_names.append(name)
                self.dataset_seeds.append(states["dataset_seeds"][st_didx])
                self.dataset_lengths.append(states["dataset_lengths"][st_didx])
                self.dataset_weights.append(0)  # set 0 for used dataset but not loaded
                self.current_pos.append(states["current_pos"][st_didx])
                self.consumed_samples.append(states["consumed_samples"][st_didx])

        sum_weight = sum(self.dataset_weights)
        self.probabilities = np.array(self.dataset_weights) / sum_weight
        self.num_datasets = len(self.dataset_names)
        assert len(self.probabilities) == self.num_datasets, f"{len(self.probabilities)}, {self.num_datasets}"

    def update_dataset_weights(self, new_weights):
        raise NotImplementedError

    def get_consumed_info(self):
        res = {}
        for didx, name in enumerate(self.dataset_names):
            res[name] = self.consumed_samples[didx]
        return res

    def convert_to_epoch(self, data_info):
        res = defaultdict(float)
        sub_res = defaultdict(float)
        lengths = defaultdict(int)
        sub_lengths = defaultdict(int)
        for didx, name in enumerate(self.dataset_names):
            part1, part2 = name.split("/")[-3:-1]
            res[part1] += data_info[name]
            sub_res[part2] += data_info[name]
            lengths[part1] += self.dataset_lengths[didx]
            sub_lengths[part2] += self.dataset_lengths[didx]
        for name in res:
            res[name] = res[name] / lengths[name]
        for name in sub_res:
            sub_res[name] = sub_res[name] / sub_lengths[name]
        return res, sub_res

    def get_batch_info(self, batch_size):
        res = {}
        for didx, name in enumerate(self.dataset_names):
            res[name] = self.probabilities[didx] * batch_size
        return res

    def get_length_info(self):
        res = {}
        for didx, name in enumerate(self.dataset_names):
            res[name] = self.dataset_lengths[didx]
        return res


class StaticBatchSamplerWithWeights:
    """
    A batch sampler that generates batches with a rampup batch size and with sampling weights for each dataset.
    Args:
        datasets (list): A list of datasets to sample from.
        dataset_weights (list): A list of weights for each dataset.
        batch_size (int): The batch size to use for each batch. Defaults to 192.
        rampup_batch_size (str): A string with three space-separated integers representing the
                                 starting batch size, the increment, and the number of steps between
                                 each increment. For example, "192 24 8" means that the batch size
                                 starts at 192 and increases by 24 every 8 steps. Defaults to
                                 "6 2 8", which corresponds to a batch size of 2 for the first 6 steps.
        micro_bsz (int): The micro batch size to use. Defaults to 2.
        seed (int): The random seed to use. Defaults to 0.
        drop_last (bool): Whether to drop the last batch if it's smaller than the batch size. Defaults to True.
        data_rank (int): The rank of the current process in distributed training. Defaults to 0.
        data_world_size (int): The number of processes in distributed training. Defaults to 1.
    """

    def __init__(
        self,
        datasets,
        dataset_weights,
        batch_size=192,
        rampup_batch_size="",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            # Increase the batch size to this value during this process.
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        if gpc.is_initialized(ParallelMode.PIPELINE):
            assert (
                batch_size - self.start_bsz
            ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
            assert (
                self.start_bsz // micro_bsz >= 4
            ), f"Must have more start samples:`{self.start_bsz}` with micro_bsz:\
    `{micro_bsz}`, so that the pipeline can run correctly"
            assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.start_bsz % micro_bsz == 0
            ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.bsz_incre % micro_bsz == 0
            ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.weight_sampler = WeightedSampler(datasets, dataset_weights, seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.num_samples = self.weight_sampler.get_num_samples()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:  # still in rampup mode
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"
            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        while 1:
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            total_batch_size = cur_batch_size * self.data_world_size
            indices = self.weight_sampler.next_batch(total_batch_size)
            batch = indices[self.data_rank * cur_batch_size : (self.data_rank + 1) * cur_batch_size]
            yield batch
            self.num_consumed_samples_in_epoch += len(batch)
            self.batch_count += 1

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "epoch": self.epoch,
            "data_world_size": self.data_world_size,
            "batch_count": self.batch_count,
            "weight_sampler": self.weight_sampler.state_dict(),
        }

        return states

    def convert_to_epoch(self, data_info):
        return self.weight_sampler.convert_to_epoch(data_info)

    def get_consumed_info(self):
        return self.weight_sampler.get_consumed_info()

    def get_batch_info(self):
        return self.weight_sampler.get_batch_info(self.batch_size * self.data_world_size)

    def get_length_info(self):
        return self.weight_sampler.get_length_info()

    def get_finished_info(self, total_steps):
        assert total_steps >= self.batch_count
        batch_sizes = []
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        for i in range(ramp_steps):
            batch_sizes.extend([i * self.bsz_incre + self.start_bsz] * self.incre_every)
        num_samples = sum(batch_sizes[self.batch_count : total_steps]) * self.data_world_size
        info = self.weight_sampler.get_batch_info(num_samples)
        consumed_info = self.get_consumed_info()
        for k, v in consumed_info.items():
            info[k] += v
        if len(batch_sizes) >= total_steps:
            return info
        add_info = self.weight_sampler.get_batch_info(
            (total_steps - len(batch_sizes)) * self.batch_size * self.data_world_size
        )
        for k, v in add_info.items():
            info[k] += v
        return info

    def load_state_dict(self, states):
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.weight_sampler.load_state_dict(states["weight_sampler"])

    def copy(self):
        sample_weight_dict = getattr(gpc.config, "sample_weight_dict", None)
        copy_sampler = StaticBatchSamplerWithWeights(
            self.weight_sampler.datasets,
            # self.weight_sampler.dataset_weights,
            # use the old weight to init the copied sampler
            get_dataset_weights(sample_weight_dict, datasets=self.weight_sampler.datasets),
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            0,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler


def get_dpsampler_dataloader(
    dataset,
    shuffle=False,
    seed=1024,
    add_sampler=True,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    **kwargs,
):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and gpc.is_using_parallel_mode(ParallelMode.DATA):
        sampler = DataParallelSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker(worker_id):  # pylint: disable=W0613
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )


def get_dataset_weights(weight_dict, datasets):
    """
    This function calculates the sampling weight of each dataset based on the given weight dictionary.
    Args:
        weight_dict (dict): A dictionary containing the weights for each dataset.
        datasets (list): A list of datasets to calculate the weights for.
    Returns:
        A list containing the sampling weights for each dataset.
    """
    folder_mapping = {}  # key: folder name, value: list of dataset indices
    lang_mapping = {}  # key: language name, value: list of folder names
    lang_folder_mapping = {}
    folder_length = defaultdict(int)
    lang_length = defaultdict(int)
    dup_keys = []
    for idx, ds in enumerate(datasets):
        name = ds.get_dataset_name()
        length = len(ds)
        lang_name, dir_name = name.split("/")[-3:-1]
        if lang_name in lang_mapping:
            lang_mapping[lang_name].add(dir_name)
        else:
            lang_mapping[lang_name] = set([dir_name])
        lang_length[lang_name] += length
        if dir_name not in folder_mapping:
            folder_mapping[dir_name] = [idx]
        else:
            folder_mapping[dir_name].append(idx)
        folder_length[dir_name] += length
        if dir_name in lang_folder_mapping:
            if lang_name != lang_folder_mapping[dir_name]:
                dup_keys.append(dir_name)
                if gpc.is_rank_for_log():
                    logger.error(
                        f"duplicated folder {dir_name} in both {lang_name} and {lang_folder_mapping[dir_name]}!"
                    )
        else:
            lang_folder_mapping[dir_name] = lang_name

    if dup_keys:
        raise ValueError(f"please check train dataset folder:{dup_keys}")

    if weight_dict is None:
        weight_dict = {n: folder_length[n] for n in folder_mapping}

    # sanity check
    not_matched_keys = []
    for key in weight_dict:
        if key in lang_mapping or key in folder_mapping or weight_dict[key] == 0:
            continue
        not_matched_keys.append(key)
    if not_matched_keys:
        for key in not_matched_keys:
            if gpc.is_rank_for_log():
                logger.error(f"sample_weight for {key} is set but no dataset matched!")
        raise ValueError(f"please check sample_weight_dict:{not_matched_keys}")

    new_weight_dict = weight_dict.copy()
    for lang_name, dir_list in lang_mapping.items():  # if language weight is set, set subfolder weights
        if lang_name in weight_dict:
            del new_weight_dict[lang_name]
            lengths = []
            for dir_name in dir_list:
                ds_length = sum([len(datasets[idx]) for idx in folder_mapping[dir_name]])
                lengths.append(ds_length)
                assert (
                    dir_name not in weight_dict
                ), f"Can not set weights\
                        for {lang_name} and subfolder {dir_name} at same time!"
            total_length = sum(lengths)
            for idx, dir_name in enumerate(dir_list):
                new_weight_dict[dir_name] = weight_dict[lang_name] * (lengths[idx] / total_length)
    weight_dict = new_weight_dict

    for dir_name in folder_mapping:
        if dir_name not in weight_dict:
            if gpc.is_rank_for_log():
                logger.warning(f"dataset weight for {dir_name} is NOT set, SET to 0!")
            weight_dict[dir_name] = 0

    folder_weights = {}
    for dir_name, ds_indices in folder_mapping.items():
        ds_list = [datasets[i] for i in ds_indices]
        lengths = [len(ds) for ds in ds_list]
        total_length = sum(lengths)
        folder_weights[dir_name] = [weight_dict[dir_name] * (le / total_length) for le in lengths]

    dataset_weights = [0 for _ in range(len(datasets))]
    for dir_name, ds_indices in folder_mapping.items():
        for i, idx in enumerate(ds_indices):
            dataset_weights[idx] = folder_weights[dir_name][i]

    # for i, w in enumerate(dataset_weights):
    # assert w != 0, f"ds[{i}] weight 0!"
    return dataset_weights
