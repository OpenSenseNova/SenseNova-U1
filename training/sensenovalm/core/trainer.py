#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import json
import os
from collections import deque
from typing import Iterable, Optional

from sensenovalm.core.engine import Engine
from sensenovalm.core.scheduler import (
    BaseScheduler,
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
)


class TrainState:
    """
    The TrainState class is used to record the current state of training.

    Args:
        train_dl (DataLoader): The DataLoader object used for training.
    """

    def __init__(self, config, batch_sampler, _init_marker=None) -> None:
        """
        Args:
            config (Config): sensenovalm config
            batch_sampler (torch.utils.data.Sampler): Because the dataloader loading is
            asynchronous and prefetched, the batch_sampler state maintained inside the
            dataloader are faster then the actual training progress, so we copy the
            batch_sampler as the anchor point of ckpt reload.
        """
        if _init_marker != "get_train_state":
            raise RuntimeError(
                "If you want to instantiate class 'TrainState', \
please use the interface 'sensenovalm.data.train_state.get_train_state()' instead of instantiating it directly."
            )

        # The number of batches produced by the data iterator
        self.batch_count: int = 0
        # Used to store the number of samples consumed in the current epoch
        self.num_consumed_samples_in_epoch: int = 0
        # Total number of tokens consumed
        self.num_consumed_tokens: int = 0
        self.consumed_samples: int = 0
        # Number of batches skipped due to inf or nan values
        self.inf_nan_skip_batches: int = 0
        # Records the number of updates, skipped batches and inf batches are not counted
        self.step_count: int = 0

        self.num_gen_samples_t2i: int = 0
        self.num_gen_samples_editing: int = 0
        self.num_gen_samples_interleave: int = 0
        self.num_consumed_images: int = 0
        self.num_consumed_gen_images: int = 0
        self.num_consumed_text_tokens: int = 0
        self.num_padding_tokens: int = 0

        # Total step count
        self.total_steps: int = config.data.total_steps
        self.total_epochs: int = getattr(config.data, "total_epochs", -1)

        # resume tensorboard folder, need load from checkpoint or set manually.
        self.resume_tb_folder = config.resume_tb_folder

        self.tensorboard_folder = config.tensorboard_folder

        # learning rate
        self.lr = config.adam.lr

        # smapler state
        if batch_sampler is not None:
            self.init_batch_sampler(batch_sampler)

        # ds state
        self.ds_state = {}

        # training state
        self.curr_epoch = 0

        # tgs statistic
        self.tgs_statistic = {
            "sum_step": 0,
            "sum_tg": 0,
            "total_time": 0,
            "sum_last_tg_10": 0,
            "sum_last_time_10": 0,
            "sum_last_tg_50": 0,
            "sum_last_time_50": 0,
            "SMA_tg_50": 0,
            "SMA_time_50": 0,
            "SMA_tg_50_list": deque(),
            "SMA_time_50_list": deque(),
            "sum_tgs": 0,
            "last_tgs_10": 0,
            "last_tgs_50": 0,
        }

    def init_batch_sampler(self, batch_sampler):
        """
        Args:
            batch_sampler (torch.utils.data.Sampler): sampler.
        """
        # make a copy of batch_sampler.
        if hasattr(batch_sampler, "copy"):
            self.batch_sampler = batch_sampler.copy()
        else:
            from copy import deepcopy

            self.batch_sampler = deepcopy(batch_sampler)

        # Iterator for the batch sampler
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __str__(self) -> str:
        """Returns a string representation of the training state in JSON format."""
        info = {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "consumed_samples": self.consumed_samples,
            "num_consumed_images": self.num_consumed_images,
            "num_consumed_gen_images": self.num_consumed_gen_images,
            "num_gen_samples_t2i": self.num_gen_samples_t2i,
            "num_gen_samples_editing": self.num_gen_samples_editing,
            "num_gen_samples_interleave": self.num_gen_samples_interleave,
            "num_consumed_text_tokens": self.num_consumed_text_tokens,
            "num_padding_tokens": self.num_padding_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
        }
        return json.dumps(info, indent=4, sort_keys=True)

    def load_state_dict(self, other_stuffs):
        """
        Resumes training from a checkpoint.

        Args:
            other_stuffs (dict): Other information needed to resume training.
        """
        self.num_consumed_samples_in_epoch = other_stuffs["num_consumed_samples_in_epoch"]
        self.num_consumed_tokens = other_stuffs["num_consumed_tokens"]
        self.consumed_samples = other_stuffs.get("consumed_samples", 0)
        self.inf_nan_skip_batches = other_stuffs["inf_nan_skip_batches"]

        # Because the ckpt save occurs after updating 'step_count',
        # there is no need to increment 'step_count' here (Does our step count start from 0 ?),
        # However, 'batch_count' is updating before ckpt storage, so it need to inc 1 when resume.
        self.batch_count = other_stuffs["batch_count"] + 1  # here you need to shift a batch backward
        self.step_count = other_stuffs.get("step_count", self.batch_count)

        # resume tensorboard from older tensorboard_folder
        self.resume_tb_folder = other_stuffs.get("tensorboard_folder", None)

        self.num_consumed_images = other_stuffs.get("num_consumed_images", 0)
        self.num_consumed_gen_images = other_stuffs.get("num_consumed_gen_images", 0)
        self.num_gen_samples_t2i = other_stuffs.get("num_gen_samples_t2i", 0)
        self.num_gen_samples_editing = other_stuffs.get("num_gen_samples_editing", 0)
        self.num_gen_samples_interleave = other_stuffs.get("num_gen_samples_interleave", 0)
        self.num_consumed_text_tokens = other_stuffs.get("num_consumed_text_tokens", 0)
        self.num_padding_tokens = other_stuffs.get("num_padding_tokens", 0)

        self.ds_state = other_stuffs.get("ds_state", {})
        self.curr_epoch = other_stuffs.get("curr_epoch", 0)

    def state_dict(self):
        tensorboard_folder = self.tensorboard_folder
        return {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "consumed_samples": self.consumed_samples,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
            "tensorboard_folder": tensorboard_folder,
            "num_consumed_images": self.num_consumed_images,
            "num_consumed_gen_images": self.num_consumed_gen_images,
            "num_gen_samples_t2i": self.num_gen_samples_t2i,
            "num_gen_samples_editing": self.num_gen_samples_editing,
            "num_gen_samples_interleave": self.num_gen_samples_interleave,
            "num_consumed_text_tokens": self.num_consumed_text_tokens,
            "num_padding_tokens": self.num_padding_tokens,
            "ds_state": self.ds_state,
            "curr_epoch": self.curr_epoch,
        }


class Trainer:
    """This is a class tending for easy deployments of users' training and evaluation instead of
    writing their own scripts.

    Args:
        engine (:class:`Engine`): Engine responsible for the process function.
        schedule (:class:`BaseScheduler`, optional): Runtime schedule. Defaults to None.
    """

    def __init__(
        self,
        engine: Engine,
        schedule: Optional[BaseScheduler] = None,
    ):
        """Initializes the Trainer class.

        Args:
            engine (Engine): The engine responsible for the process function.
            schedule (Optional[BaseScheduler], optional): The runtime schedule. Defaults to None.
        """
        self._engine = engine

        # build schedule
        if schedule is None:
            self._schedule = NonPipelineScheduler()
        else:
            assert isinstance(
                schedule, BaseScheduler
            ), f"expected schedule to be of type BaseSchedule, but got {type(schedule)}"
            self._schedule = schedule

        self._schedule.pre_processing(self._engine)

    @property
    def engine(self):
        """Returns the engine that responsible for managing the training and evaluation process."""
        return self._engine

    @property
    def schedule(self):
        """Returns the runtime scheduler."""
        return self._schedule

    @property
    def uses_pipeline(self):
        """Returns whether the pipeline parallel is used or not."""
        return isinstance(self._schedule, (PipelineScheduler, InterleavedPipelineScheduler))

    def train(self):
        """Sets the model to training mode."""
        self._engine.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self._engine.eval()

    def zero_grad(self):
        """Sets the gradient of all parameters in the model to zero."""
        self._engine.zero_grad()

    def step(self):
        """Executes the parameter update step."""
        return self._engine.step()

    def execute_schedule(self, data_iter: Iterable, **kwargs):
        """Runs the forward, loss computation, and backward for the model.
        Returns a tuple of (output, label, loss).

        Args:
            data_iter (Iterable): The data iterator.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss).
        """
        return self._schedule.forward_backward_step(self._engine, data_iter, **kwargs)
