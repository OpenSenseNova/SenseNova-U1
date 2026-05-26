#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from typing import Any, Callable, Iterable, List, Optional

import torch

from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.engine import Engine
from sensenovalm.model.moe import SenseNovaVLMoE
from sensenovalm.model.utils import _submodule_filter
from sensenovalm.utils.common import (
    SchedulerHook,
    check_data_is_packed,
    conditional_context,
    update_e_score_correction_bias,
)
from sensenovalm.utils.logger import get_logger
from sensenovalm.utils.timeout import llm_timeout

from .base_scheduler import BaseScheduler

logger = get_logger(__file__)


class NonPipelineScheduler(BaseScheduler):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data
            and returns a tuple in the form of (data, label), and it will be executed in load_batch.
        gradient_accumulation_steps(int, optional): the steps of gradient accumulation, 1 for disable
            gradient accumulation.

    Examples:
        >>> # this shows an example of customized data_process_func
        >>> def data_process_func(dataloader_output):
        >>>     item1, item2, item3 = dataloader_output
        >>>     data = (item1, item2)
        >>>     label = item3
        >>>     return data, label
    """

    def __init__(
        self,
        data_process_func: Callable = None,
        gradient_accumulation_size: int = 1,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        self._grad_accum_size = gradient_accumulation_size
        self._grad_accum_offset = 0

        self._hooks = scheduler_hooks

        super().__init__(data_process_func)

    def pre_processing(self, engine: Engine):
        """Performs actions before running the schedule.

        Args:
           engine (sensenovalm.core.Engine): SenseNovaLM engine for training and inference.
        """
        pass

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _load_accum_batch(self, data: Any, label: Any):
        """Loads a batch of data and label for gradient accumulation.

        Args:
            data (Any): The data to be loaded.
            label (Any): The label to be loaded.
        """

        _data, _label = self._load_micro_batch(
            data=data, label=label, offset=self._grad_accum_offset, bsz_stride=self._bsz_stride
        )
        self._grad_accum_offset += self._bsz_stride

        if self.data_process_func:
            _data, _label = self.data_process_func(_data, _label)

        return _data, _label

    def _call_engine_mtp_criterion(self, engine: Engine, outputs: Any, labels: Any):
        """Calls the engine's criterion with the given outputs and labels.
        Args:
            engine (sensenovalm.core.Engine): SenseNovaLM engine for training and inference.
            outputs (Any): The outputs from the model, can be of type torch.Tensor, list, tuple, or dict.
            labels (Any): The labels for the outputs, can be of type torch.Tensor, list, tuple, or dict.
        """
        assert isinstance(
            outputs, (torch.Tensor, list, tuple, dict)
        ), f"Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}"

        mtp_losses = []
        for i, (output, label) in enumerate(zip(outputs, labels)):
            if isinstance(output, torch.Tensor):
                output = (output,)
            if isinstance(label, torch.Tensor):
                label = (label,)

            self._call_hooks("before_criterion", output, label)
            if isinstance(output, (tuple, list)) and isinstance(label, (tuple, list)):
                mtp_loss = engine.mtp_criterions[i](*output, *label)
            elif isinstance(output, (tuple, list)) and isinstance(label, dict):
                mtp_loss = engine.mtp_criterions[i](*output, **label)
            elif isinstance(output, dict) and isinstance(label, dict):
                mtp_loss = engine.mtp_criterions[i](**output, **label)
            elif isinstance(output, dict) and isinstance(label, (list, tuple)):
                raise ValueError(f"Expected labels to be a dict when the model outputs are dict, but got {type(label)}")
            else:
                raise TypeError(
                    f"Expected model outputs and labels to be of type torch.Tensor ' \
                    '(which is auto-converted to tuple), list, tuple, or dict, ' \
                    'but got {type(output)} (model outputs) and {type(label)} (labels)"
                )
            self._call_hooks("after_criterion", mtp_loss)
            mtp_losses.append(mtp_loss)

        return mtp_losses

    def _train_one_batch(
        self,
        data: Any,
        label: Any,
        engine: Engine,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output: bool = False,
        scale_loss: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (sensenovalm.core.Engine): SenseNovaLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output (bool, optional): Output will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        extra_losses, _extra_loss, losses_for_log_only = [], 0.0, {}
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output, mtp_outputs, *_extra_losses = self._call_engine(engine, data)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)
            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss = self._call_engine_criterion(
                    engine,
                    output,
                    label,
                    loss_weight=data.pop("loss_weight", None),
                    loss_reduction_all_gather=data.pop("loss_reduction_all_gather", False),
                )
                self._call_hooks("after_criterion", loss)

                if hasattr(gpc.config.model, "num_mtp_layers") and gpc.config.model.num_mtp_layers > 0:
                    mtp_labels = []
                    for i in range(gpc.config.model.num_mtp_layers):
                        mtp_labels.append(
                            torch.cat(
                                [
                                    label[:, i + 1 :],
                                    torch.full((label.size(0), i + 1), -100, dtype=label.dtype, device=label.device),
                                ],
                                dim=1,
                            )
                        )
                    mtp_losses = self._call_engine_mtp_criterion(engine, mtp_outputs, mtp_labels)
                    mtp_loss = sum(mtp_losses) * gpc.config.loss.mtp_loss_coeff
                    mtp_loss /= scale_loss
                    extra_losses.append(mtp_loss)
                    _extra_loss += mtp_loss
                else:
                    mtp_loss = None

                # cal extra_loss if _extra_losses has eles and put the result metric into extra_losses
                for other_loss in _extra_losses:
                    if other_loss is None:
                        continue

                    if isinstance(other_loss, list):
                        if len(other_loss) == 0:
                            continue
                        assert all(isinstance(x, (int, float, torch.Tensor)) for x in other_loss)
                        other_loss_sum = sum(other_loss) / scale_loss
                    elif isinstance(other_loss, (int, float, torch.Tensor)):
                        other_loss_sum = other_loss / scale_loss
                    elif isinstance(other_loss, dict):
                        # these losses are only for logging
                        losses_for_log_only = other_loss
                        for v in losses_for_log_only.values():
                            v /= scale_loss
                    else:
                        assert False, "Extra loss can only be Tensor or Tensor list type"

                    if not isinstance(other_loss, dict):
                        extra_losses.append(other_loss_sum)
                        _extra_loss += other_loss_sum

                loss /= scale_loss

        # clear output before backward for releasing memory resource
        if not return_output:
            output = None

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            engine.backward(loss + _extra_loss)
            self._call_hooks("after_backward", None)

        if not return_loss:
            loss, extra_losses, losses_for_log_only = None, None, None

        return output, loss, extra_losses, losses_for_log_only

    @llm_timeout(func_name="nopp_forward_backward_step")
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (sensenovalm.core.Engine): SenseNovaLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        # actual_batch_size is micro_num when training,
        # actual_batch_size is micro_num * micro_bsz when evaluating
        batch_data, actual_batch_size = engine.load_batch(data_iter)

        if check_data_is_packed(batch_data):
            micro_num = actual_batch_size
        else:
            micro_num = actual_batch_size // gpc.config.data["micro_bsz"]

        self._grad_accum_size = micro_num  # Rampup or variable bsz size.
        self._bsz_stride = actual_batch_size // self._grad_accum_size

        data, label = batch_data

        loss = 0 if return_loss else None
        accm_extra_losses = []
        accm_losses_for_log_only = {}
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if engine.optimizer is not None:
                if _current_accum_step == self._grad_accum_size - 1:
                    engine.optimizer.skip_grad_reduce = False
                else:
                    engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss, _extra_losses, _losses_for_log_only = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, return_output_label, self._grad_accum_size
            )
            if return_loss:
                loss += _loss
                # process extra loss
                if len(accm_extra_losses) == 0:
                    accm_extra_losses = [0] * len(_extra_losses)
                accm_extra_losses = [
                    accm_loss + extra_loss for accm_loss, extra_loss in zip(accm_extra_losses, _extra_losses)
                ]
                
                if len(accm_losses_for_log_only) == 0:
                    for loss_key in _losses_for_log_only:
                        accm_losses_for_log_only[loss_key] = 0
                    for loss_key in _losses_for_log_only:
                        accm_losses_for_log_only[loss_key] += _losses_for_log_only[loss_key]

            if return_output_label:
                outputs.append(_output)
                labels.append(_label)


        if not return_output_label:
            outputs, labels = None, None

        if hasattr(gpc.config.model, "moe_kwargs") and gpc.config.model.moe_kwargs.get("num_experts", 1) > 1:
            expert_counts_list = []
            expert_biases_list = []
            update_rate = getattr(gpc.config.model.moe_layer_kwargs, "update_rate", 0.001)
            if getattr(gpc.config.model.moe_layer_kwargs, "aux_free", False):
                for layer in _submodule_filter(engine.model, SenseNovaVLMoE):
                    expert_counts_list.append(layer.moe_layer.expert_counts)
                    expert_biases_list.append(layer.moe_layer.e_score_correction_bias)
                expert_biases_list = update_e_score_correction_bias(expert_counts_list, expert_biases_list, update_rate)
                moe_layer_id = 0
                for layer in _submodule_filter(engine.model, SenseNovaVLMoE):
                    layer.moe_layer.e_score_correction_bias.data.copy_(expert_biases_list[moe_layer_id])
                    torch.zero_(layer.moe_layer.expert_counts)
                    moe_layer_id += 1

        return_tuple = [outputs, labels, loss]
        if accm_extra_losses:
            return_tuple += accm_extra_losses
        if accm_losses_for_log_only:
            return_tuple += [accm_losses_for_log_only]
        return tuple(return_tuple)
