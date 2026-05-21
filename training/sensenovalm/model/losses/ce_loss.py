#!/usr/bin/env python
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from torch import nn

from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.model.ops.cross_entropy import new_cross_entropy
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)


class FlashGPTLMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, parallel_output=True, label_smoothing=0, ce_loss_weight=1.0):
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                if gpc.is_rank_for_log():
                    print(f"use label_smoothing: {label_smoothing}")
        else:
            label_smoothing = 0

        self.label_smoothing = label_smoothing
        self.loss_fn = new_cross_entropy(
            reduction="mean",
            label_smoothing=self.label_smoothing,
            parallel_output=parallel_output,
            inplace_backward=True,
        )

        self.loss_weight_fn = new_cross_entropy(
            reduction="none",
            label_smoothing=self.label_smoothing,
            parallel_output=parallel_output,
            inplace_backward=True,
        )

        self.ce_loss_weight = ce_loss_weight

    def forward(self, *args, loss_weight=None, loss_reduction_all_gather=False):
        if len(args) >= 3:
            # residual is to match prenorm
            logits, *_, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)

        if loss_weight is None:
            loss = self.loss_fn(
                shift_logits, shift_labels
            )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
            # calculated through the calculation range, and -100 must be outside this range, so there is no problem
            #
        else:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=shift_labels.device)
            loss = self.loss_weight_fn(shift_logits, shift_labels)
            weight_sum = loss_weight.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(weight_sum, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.DATA))

            loss = loss * loss_weight
            loss = loss.sum() / weight_sum

        loss = loss * self.ce_loss_weight

        if torch.isnan(loss) and shift_labels.max() == -100:
            # check if target all -100
            loss = shift_logits[-1, :].sum() * 0.0

        return loss
