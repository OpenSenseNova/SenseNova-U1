# --------------------------------------------------------
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from Megatron-DeepSpeed (Microsoft / NVIDIA), NVIDIA
# Apex (BSD-3-Clause), and NVIDIA NeMo (Apache-2.0).
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause / Apache-2.0 portions retain their
# respective licenses.
# Upstream references:
#   https://github.com/microsoft/Megatron-DeepSpeed
#   https://github.com/NVIDIA/apex
#   https://github.com/NVIDIA/NeMo
# --------------------------------------------------------
"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the cross entropy operators.
"""

import torch
import torch.distributed as dist
from torch import nn

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()


# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/core/ \
# sequence_parallel/cross_entropy.py
class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):
    """
    Cross Entropy module for isp.
    """

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, reduction, label_smoothing=0.0):  # pylint: disable=W0613
        sp_size = gpc.get_world_size(ParallelMode.TENSOR)

        # reshape
        # vocab_seq_parallel_logits: [B * (S/P), V] -> [B, S/P, V]
        # target: [B * S/P] -> [B, S/P]
        bsz = gpc.config.data.micro_bsz if gpc.config.data.use_packed_dataset is False else 1
        vocab_seq_parallel_logits = vocab_seq_parallel_logits.view(bsz, -1, gpc.config.model.vocab_size)
        target = target.view(bsz, -1)

        # transpose
        # vocab_seq_parallel_logits: [B, S/P, V] -> [S/P, B, V]
        # target: [B, S/P] -> [S/P, B]
        # return: [S, B]
        vocab_seq_parallel_logits = vocab_seq_parallel_logits.transpose(0, 1).contiguous()
        target = target.transpose(0, 1).contiguous()

        ctx.seqlen = vocab_seq_parallel_logits.size(0) * sp_size
        batch_size = vocab_seq_parallel_logits.size(1)

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction="none")

        loss_all = torch.empty(
            ctx.seqlen, batch_size, dtype=vocab_seq_parallel_logits.dtype, device=vocab_seq_parallel_logits.device
        )

        torch.distributed.all_gather_into_tensor(loss_all, loss, group=gpc.get_group(ParallelMode.TENSOR))

        # [s b] => [b, s]
        loss_all = loss_all.transpose(0, 1).contiguous()

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        # transpose
        grad_output = grad_output.transpose(0, 1).contiguous()

        step_seqlen = ctx.seqlen // gpc.get_world_size(ParallelMode.TENSOR)
        sp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        grad_output_part = grad_output[step_seqlen * sp_rank : step_seqlen * (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        # transpose
        grad_input = grad_input.transpose(0, 1).contiguous()
        # reshape
        grad_input = grad_input.view(-1, gpc.config.model.vocab_size)

        return grad_input, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=gpc.get_group(ParallelMode.DATA))
    averaged_losses = averaged_losses / gpc.get_world_size(ParallelMode.DATA)

    return averaged_losses


class VocabSequenceParallelCrossEntropyLoss(nn.Module):
    """
    Cross Entropy module for isp.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0,
        process_group=None,
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.process_group = process_group

    def loss_mean_func(self, output_tensor):
        losses = output_tensor.float()
        loss = torch.sum(losses.view(-1)) / losses.numel()

        # TODO: allreduce loss in dp group

        return loss

    def forward(self, _input, target):
        assert _input.is_cuda and target.is_cuda

        _loss_list = vocab_sequence_parallel_cross_entropy(_input, target, self.label_smoothing)

        if self.reduction == "mean":
            loss = self.loss_mean_func(_loss_list)
            return loss

        return _loss_list.view(-1)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Adapt from: https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
    Supports vocab parallel loss calculation, but does not support inplace backward.
    NOTE: This class is different from the original Apex implementation. Apex will calculate the loss of
        ignore_index and flashCrossEntropy will set it to 0. InterEvo adapts the second approach.
    """

    @staticmethod
    @sensenovalm_accelerator.amp.custom_fwd
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, process_group=None):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        # get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        if process_group is not None and dist.get_world_size(process_group) > 1:
            rank = dist.get_rank(process_group)
            part_len = vocab_parallel_logits.shape[-1]
            vocab_start_index, vocab_end_index = part_len * rank, part_len * (rank + 1)
        else:
            vocab_start_index, vocab_end_index = 0, vocab_parallel_logits.shape[-1]

        # vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        ignore_mask = target == -100
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # All reduce is needed to get the chunks from other GPUs.
        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        sum_exp_logits = torch.log(sum_exp_logits)
        loss = sum_exp_logits - predicted_logits
        loss[ignore_mask] = 0.0

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, ignore_mask)

        return loss

    @staticmethod
    @sensenovalm_accelerator.amp.custom_bwd
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d, ignore_mask = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax  # s_{k}
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        grad_input[ignore_mask] = 0.0  # set ignore token loss as 0.

        return grad_input, None, None, None


class CrossEntropyApexVocabParallel(nn.Module):
    """Adapt from: https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
    Supports vocab parallel loss calculation, but does not support inplace backward.
    """

    def __init__(
        self, ignore_index=-100, reduction="mean", label_smoothing=0.0, process_group=None, inplace_backward=False
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        assert inplace_backward is False, "does not support inplace backward"
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.process_group = process_group

    def forward(self, vocab_parallel_logits, target):

        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, self.label_smoothing, self.process_group)
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss


def flash_loss(
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    process_group=None,
    inplace_backward=False,  # pylint:disable=W0613
):
    try:
        from flash_attn.losses.cross_entropy import (
            CrossEntropyLoss as FlashCrossEntropyLoss,
        )

        flash_cross_entropy_impl = True
    except (ModuleNotFoundError, ImportError):
        flash_cross_entropy_impl = False

    assert (
        gpc.config.model.get("use_flash_attn", False) and flash_cross_entropy_impl
    ), "Only flash cross entropy support parallel_output"

    assert (
        sensenovalm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
    ), "flash cross entropy only support gpu backend"

    return FlashCrossEntropyLoss(
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        process_group=process_group,
    )


# TODO: ops是否需要实现更加统一的形式
def new_cross_entropy(
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0,
    parallel_output: bool = False,
    **kwargs,
):

    if parallel_output:

        return CrossEntropyApexVocabParallel(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            process_group=gpc.get_group(ParallelMode.TENSOR),
        )
    else:
        if gpc.is_rank_for_log():
            logger.warning(
                "Use nn.CrossEntropyLoss rather than flashattn CrossEntropyLoss."
                "parallel_output must be set false. Please note this!"
            )
        kwargs.pop("inplace_backward", None)
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing, **kwargs
        )
