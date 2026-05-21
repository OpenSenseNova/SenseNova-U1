# --------------------------------------------------------
# Derived from InternEvo (OpenGVLab, Apache-2.0).
# Portions adapted from Flash-Attention (Tri Dao, BSD-3-Clause).
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause portions retain their license.
# Upstream reference: https://github.com/Dao-AILab/flash-attention
# --------------------------------------------------------
"""
Linear Modules
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import init

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.parallel.shard import (
    get_head_parallel_mode,
    get_parallel_strategies_split_mode,
    get_tensor_split_parallel_mode,
)
from sensenovalm.model.ops.linear import (
    gmm_backward_op,
    gmm_forward_op,
    linear_bias_wgrad_op,
    linear_forward_op,
    linear_igrad_op,
)
from sensenovalm.utils.logger import get_logger

if TYPE_CHECKING:
    from sensenovalm.core.parallel.comm.isp import WPCommunicator
    from sensenovalm.core.parallel.comm.tensor import TPCommunicator

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()

custom_bwd = sensenovalm_accelerator.return_custom_bwd()
custom_fwd = sensenovalm_accelerator.return_custom_fwd()


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class SPFusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for tensor parallel in flash-attn implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        communicator: TPCommunicator,
        return_residual=False,
    ):
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.communicator = communicator

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        # parallel strategy-specific communication callback 1-1.
        # see more details in the communicator for different parallel strategies.
        # we want to kick off the all_gather early, before weight dtype conversion.
        total_x, handle_x = communicator.input_hook(x, async_op=True)

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()

        # wait for x has been gathered.
        handle_x.wait()

        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")

        output = linear_forward_op(total_x, weight, bias)

        # parallel strategy-specific communication callback 2.
        # see more details in the communicator for different parallel strategies.
        output, _ = communicator.output_hook(output, async_op=False)

        saved_x = None if ctx.compute_weight_gradient is False else total_x if communicator.save_total_input() else x
        ctx.save_for_backward(saved_x, weight, bias)

        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        communicator: TPCommunicator = ctx.communicator

        # parallel strategy-specific communication callback 3.
        # see more details in the communicator for different parallel strategies.
        grad_output, _ = communicator.grad_output_hook(grad_output, async_op=False)
        grad_output = grad_output.contiguous()

        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()

        x, weight, bias = ctx.saved_tensors

        # parallel strategy-specific communication callback 1-2.
        # see more details in the communicator for different parallel strategies.
        if ctx.needs_input_grad[1]:
            x, handle_x = communicator.input_hook(x, async_op=True, is_forward=False)

        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            grad_input = linear_igrad_op(weight, grad_output, None if not ctx.return_residual else grad_input)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            # parallel strategy-specific communication callback 4.
            # see more details in the communicator for different parallel strategies.
            grad_input, handle_grad_input = communicator.grad_input_hook(grad_input, async_op=True)
        else:
            grad_input = None

        # computes gradinets for weight and bias if necessary
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient

            # wait for x has been gathered
            handle_x.wait()

            x = x.reshape(batch_dim, x.shape[-1])
            if (
                gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                and gpc.config.parallel["pipeline"].get("zero_bubble", False)
                and not gpc.is_first_rank(ParallelMode.PIPELINE)
            ):
                from sensenovalm.core.scheduler.pipeline_scheduler import WeightGradStore

                WeightGradStore.put(weight, bias, x, grad_output, ctx.needs_input_grad[2], linear_bias_wgrad_op)
                grad_weight, grad_bias = None, None
            else:
                grad_weight, grad_bias = linear_bias_wgrad_op(x, grad_output, ctx.needs_input_grad[2])

        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        # wait for grad_input has been gathered
        handle_grad_input.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


# Q: Should we unify WPFusedDenseFunc and SPFusedDenseFunc, as well as the related communicator interface?
# A: Currently, WPFusedDenseFunc and SPFusedDenseFunc have significant differences in their computation logic
#    and communication interfaces, so they should not be unified.
class WPFusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for weigth parallel, which is optimized based on flash implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        module: nn.Module,
        communicator: WPCommunicator,
        return_residual=False,
    ):
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.module = module
        ctx.communicator = communicator

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        total_weight = communicator.weight_hook(weight, module=module)
        total_bias = bias if bias is None else communicator.weight_hook(bias, module=module, is_bias=True)

        if torch.is_autocast_enabled():
            total_weight = total_weight.to(dtype=torch.get_autocast_gpu_dtype())
            if total_bias:
                total_bias.to(dtype=torch.get_autocast_gpu_dtype())

        total_weight = total_weight.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *total_weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")

        output = linear_forward_op(x, total_weight, total_bias)

        # release memory
        del total_weight
        del total_bias

        # parallel strategy-specific communication callback 2.
        # see more details in the communicator for different parallel strategies.
        # gather seq dim when head parallel_output is False
        if hasattr(communicator, "output_hook"):
            output, _ = communicator.output_hook(output, async_op=False)

        saved_x = None if ctx.compute_weight_gradient is False else x
        ctx.save_for_backward(saved_x, weight, bias)

        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        module: nn.Module = ctx.module
        communicator: WPCommunicator = ctx.communicator
        x, weight, bias = ctx.saved_tensors

        # parallel strategy-specific communication callback 3.
        # see more details in the communicator for different parallel strategies.
        if hasattr(communicator, "grad_output_hook"):
            grad_output, _ = communicator.grad_output_hook(grad_output, async_op=False)

        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()

        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        total_weight = communicator.weight_hook(weight, module=module)

        is_using_ZB = (
            gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
            and gpc.config.parallel["pipeline"].get("zero_bubble", False)
            and not gpc.is_first_rank(ParallelMode.PIPELINE)
        )

        # compute weight grad
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            x = x.reshape(batch_dim, x.shape[-1])
            if is_using_ZB:
                from sensenovalm.core.scheduler.pipeline_scheduler import WeightGradStore

                WeightGradStore.put(
                    weight, bias, x, grad_output, ctx.needs_input_grad[2], linear_bias_wgrad_op, communicator.grad_hook
                )
                grad_weight, grad_bias = None, None
            else:
                grad_weight, grad_bias = linear_bias_wgrad_op(
                    x,
                    grad_output,
                    ctx.needs_input_grad[2],
                )

                grad_weight, grad_weight_sync = communicator.grad_hook(
                    grad_weight, async_op=True, module=module, is_bias=False
                )
                if grad_bias is not None:
                    grad_bias, grad_bias_sync = communicator.grad_hook(
                        grad_bias, async_op=True, module=module, is_bias=True
                    )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        if ctx.needs_input_grad[0]:
            grad_input = linear_igrad_op(total_weight, grad_output, None if not ctx.return_residual else grad_input)
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        else:
            grad_input = None

        del total_weight

        if ctx.needs_input_grad[1] and not is_using_ZB:
            grad_weight_sync.wait()
            if grad_bias is not None:
                grad_bias_sync.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


class GroupedGemmSPFusedDenseFunc(torch.autograd.Function):
    "Grouped Gemm FusedDenseFunc for tensor parallel"

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        batch_sizes: torch.Tensor,
        backend: str,
    ):
        if backend == "bmm":
            assert x.dim() == 3, f"bmm only support 3d input (e, c, m), but got: {x.shape}"
        elif backend == "gmm":
            assert x.dim() == 2, f"gmm only support 2d input (s, m), but got: {x.shape}"
            assert batch_sizes is not None, "batch_sizes should be provided for gmm"
        else:
            raise NotImplementedError(f"Invalid backend: {backend}")

        input_numel = x.numel()
        if input_numel == 0:
            backend = "bmm"

        ctx.compute_weight_gradient = weight.requires_grad
        ctx.backend = backend

        saved_x = None if ctx.compute_weight_gradient is False else x
        ctx.save_for_backward(saved_x, weight, batch_sizes)

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        if backend == "gmm":
            output = gmm_forward_op(x, weight, batch_sizes)
        else:
            if input_numel == 0:
                # if inp is empty, reshape to make grad flow.
                # inp shape: (0, hdim)
                weight = weight.view(x.shape[-1], -1)

            output = torch.matmul(x, weight)

        assert len(output.shape) == len(x.shape)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        backend = ctx.backend

        grad_output = grad_output.contiguous()
        x, weight, batch_sizes = ctx.saved_tensors
        grad_input, grad_weight = None, None

        if grad_output.numel() == 0:
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(x)
            return grad_input, grad_weight, None, None, None, None, None

        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if backend == "gmm":
                grad_input, grad_weight = gmm_backward_op(x, grad_output, batch_sizes, input_weight=weight)
            else:
                grad_weight = torch.matmul(x.transpose(-1, -2), grad_output)

        if ctx.needs_input_grad[0]:
            if backend == "gmm":
                if grad_input is None:
                    grad_input, _ = gmm_backward_op(grad_output, weight, batch_sizes, is_grad_input=True)
            else:
                grad_input = torch.matmul(grad_output, weight.transpose(-1, -2))

        return grad_input, grad_weight, None, None, None, None, None


class GroupedGemmWPFusedDenseFunc(torch.autograd.Function):
    "Grouped Gemm FusedDenseFunc for weigth parallel."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        module: nn.Module,
        communicator: WPCommunicator,
        batch_sizes: torch.Tensor,
        backend: str,
        full_weight_shape: torch.Size,
    ):
        assert full_weight_shape is not None, "full_weight_shape should be provided"
        if backend == "bmm":
            assert x.dim() == 3, f"bmm only support 3d input (e, c, m), but got: {x.shape}"
        elif backend == "gmm":
            assert x.dim() == 2, f"gmm only support 2d input (s, m), but got: {x.shape}"
            assert batch_sizes is not None, "batch_sizes should be provided for gmm"
        else:
            raise NotImplementedError(f"Invalid backend: {backend}")

        input_numel = x.numel()
        if input_numel == 0:
            backend = "bmm"

        ctx.compute_weight_gradient = weight.requires_grad
        ctx.module = module
        ctx.communicator = communicator
        ctx.backend = backend
        ctx.full_weight_shape = full_weight_shape

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        total_weight = communicator.weight_hook(weight, module=module)
        total_weight = total_weight.reshape(full_weight_shape)

        if torch.is_autocast_enabled():
            total_weight = total_weight.to(dtype=torch.get_autocast_gpu_dtype())
        total_weight = total_weight.contiguous()

        if backend == "gmm":
            output = gmm_forward_op(x, total_weight, batch_sizes)
        else:
            if input_numel == 0:
                # if inp is empty, reshape to make grad flow.
                # inp shape: (0, hdim)
                total_weight = total_weight.view(x.shape[-1], -1)

            output = torch.matmul(x, total_weight)

        # release memory
        del total_weight

        saved_x = None if ctx.compute_weight_gradient is False else x
        ctx.save_for_backward(saved_x, weight, batch_sizes)

        assert len(output.shape) == len(x.shape)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        module: nn.Module = ctx.module
        communicator: WPCommunicator = ctx.communicator
        x, weight, batch_sizes = ctx.saved_tensors
        backend = ctx.backend
        full_weight_shape = ctx.full_weight_shape

        if grad_output.numel() == 0:
            if ctx.needs_input_grad[1]:
                total_weight_shape = torch.Size(
                    (full_weight_shape.numel() // full_weight_shape[-1], full_weight_shape[-1])
                )
                grad_weight = torch.zeros(total_weight_shape, dtype=weight.dtype, device=weight.device)
                grad_weight, grad_weight_sync = communicator.grad_hook(
                    grad_weight, async_op=True, module=module, is_bias=False
                )
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(x)
            if ctx.needs_input_grad[1]:
                grad_weight_sync.wait()
            return grad_input, grad_weight, None, None, None, None, None

        grad_output = grad_output.contiguous()
        total_weight = communicator.weight_hook(weight, module=module)
        total_weight = total_weight.reshape(full_weight_shape)
        grad_input, grad_weight = None, None

        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if backend == "gmm":
                grad_input, grad_weight = gmm_backward_op(x, grad_output, batch_sizes, input_weight=total_weight)
            else:
                grad_weight = torch.matmul(x.transpose(-1, -2), grad_output)
            grad_weight = grad_weight.view(-1, grad_weight.shape[-1])
            grad_weight, grad_weight_sync = communicator.grad_hook(
                grad_weight, async_op=True, module=module, is_bias=False
            )

        if ctx.needs_input_grad[0]:
            if backend == "gmm":
                if grad_input is None:
                    grad_input, _ = gmm_backward_op(grad_output, total_weight, batch_sizes, is_grad_input=True)
            else:
                grad_input = torch.matmul(grad_output, total_weight.transpose(-1, -2))

        del total_weight

        if ctx.needs_input_grad[1]:
            grad_weight_sync.wait()

        return grad_input, grad_weight, None, None, None, None, None


def fused_dense_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    communicator: Union[TPCommunicator, WPCommunicator],
    module: Optional[nn.Module] = None,
    bias: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    use_grouped_linear: bool = False,
    **kwargs,
):
    if communicator.communication_mode() == "wp":
        if not use_grouped_linear:
            return WPFusedDenseFunc.apply(
                x,
                weight,
                bias,
                module,
                communicator,
                return_residual,
            )
        else:
            batch_sizes = kwargs.pop("batch_sizes", None)
            backend = kwargs.pop("backend", "gmm")
            full_weight_shape = kwargs.pop("full_weight_shape", None)
            return GroupedGemmWPFusedDenseFunc.apply(
                x,
                weight,
                module,
                communicator,
                batch_sizes,
                backend,
                full_weight_shape,
            )
    else:  # mtp, msp, and fsp
        if not use_grouped_linear:
            return SPFusedDenseFunc.apply(
                x,
                weight,
                bias,
                communicator,
                return_residual,
            )
        else:
            # TODO: support grouped linear for mtp, msp, and fsp
            batch_sizes = kwargs.pop("batch_sizes", None)
            backend = kwargs.pop("backend", "gmm")
            return GroupedGemmSPFusedDenseFunc.apply(
                x,
                weight,
                batch_sizes,
                backend,
            )


class ParallelLinearWithCommExt(nn.Linear):
    """
    Parallel linear with commuication extention.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        split_mode (str): The split mode. It can be "none", "column", or "row".
    """

    # class level communicator variable.
    _communicator = None

    @classmethod
    def register_cls_communicator(cls, communicator):
        cls._communicator = communicator

    def register_communicator(self, communicator):
        """
        override the class default communicator for a parallel linear instance
        """
        self._communicator = communicator

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_mode: ParallelMode,
        bias: bool = True,
        multiple_of: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        split_mode: str = "none",
    ) -> None:
        assert split_mode in ("none", "column", "row"), f"unknown split_mode {split_mode}"

        world_size = gpc.get_world_size(parallel_mode)
        rank = gpc.get_local_rank(parallel_mode)

        if split_mode != "none":
            split_features = out_features if split_mode == "column" else in_features
            multiple = split_features // multiple_of
            # We want to split @multiple across world_size, but it could be an uneven split
            # uneven split is forbidden
            div = multiple // world_size
            mod = multiple % world_size
            assert mod == 0, "linear module uneven split is forbidden"
            # The first @mod ranks get @div + 1 copies, the rest get @div copies
            local_multiple = div + int(rank < mod)

        if split_mode == "column":
            super().__init__(in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype)
        elif split_mode == "row":
            super().__init__(local_multiple * multiple_of, out_features, bias=bias, device=device, dtype=dtype)
        else:
            super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor, batch_sizes: torch.Tensor = None) -> torch.Tensor:  # pylint: disable=W0622
        _class_name = self.__class__.__name__
        assert self._communicator is not None, f"{_class_name} should register with a communicator first."

        mixer_kwargs = {}
        use_grouped_linear = getattr(self, "is_grouped_linear", False)
        if use_grouped_linear:
            mixer_kwargs = {
                "batch_sizes": batch_sizes,
                "backend": self.backend,
                "full_weight_shape": self.full_weight_shape if hasattr(self, "full_weight_shape") else None,
            }
        return fused_dense_func(
            input,
            self.weight,
            communicator=self._communicator,
            module=self,
            bias=self.bias,
            use_grouped_linear=use_grouped_linear,
            **mixer_kwargs,
        )


def reset_isp_linear_parameters(module) -> None:
    world_size = gpc.get_world_size(ParallelMode.WEIGHT)

    # init module weight and bias with full size
    full_weight = torch.zeros(
        module.out_features * world_size, module.in_features, dtype=module.weight.dtype, device=module.weight.device
    )
    init.kaiming_uniform_(full_weight, a=math.sqrt(5))
    if module.bias is not None:
        full_bias = torch.zeros(
            module.out_features * world_size, dtype=module.weight.dtype, device=module.weight.device
        )
        fan_in, _ = init._calculate_fan_in_and_fan_out(full_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(full_bias, -bound, bound)

    # broadcast and reset
    parallel_mode = ParallelMode.WEIGHT
    ranks = gpc.get_ranks_in_group(parallel_mode)
    dist.broadcast(full_weight, src=ranks[0], group=gpc.get_group(parallel_mode))
    weight_local = full_weight.chunk(world_size, dim=0)[gpc.get_local_rank(parallel_mode)]
    module.weight.data.copy_(weight_local.data)
    if module.bias is not None:
        dist.broadcast(full_bias, src=ranks[0], group=gpc.get_group(parallel_mode))
        bias_local = full_bias.chunk(world_size, dim=0)[gpc.get_local_rank(parallel_mode)]
        module.bias.data.copy_(bias_local.data)


class ColumnParallelLinear(ParallelLinearWithCommExt):
    """
    ColumnParallelLinear

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        is_expert: bool = False,
    ) -> None:
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")

        parallel_mode = get_tensor_split_parallel_mode(is_expert=is_expert)
        super().__init__(
            in_features, out_features, parallel_mode, bias=bias, device=device, dtype=dtype, split_mode="column"
        )


class RowParallelLinear(ParallelLinearWithCommExt):
    """
    RowParallelLinear

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        is_expert: bool = False,
    ) -> None:
        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")

        parallel_mode = get_tensor_split_parallel_mode(is_expert=is_expert)
        rank = gpc.get_local_rank(parallel_mode)
        super().__init__(
            in_features,
            out_features,
            parallel_mode,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
            split_mode="row",
        )


class ScaleColumnParallelLinear(ParallelLinearWithCommExt):
    """
    ScaleColumnParallelLinear.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
        norm_head (bool): Normalize the output embedding in order to let the calculation of logits not affected by
            the norm of embedding. The implementation is referred to baichuan2,
            see https://huggingface.co/baichuan-inc/Baichuan2-7B-Base for more information. False by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
        norm_head: bool = False,
    ) -> None:
        if norm_head:
            logger.info("Notice that norm head is enabled to normalize head weight.")

        parallel_mode = get_tensor_split_parallel_mode()
        super().__init__(
            in_features, out_features, parallel_mode, bias=bias, device=device, dtype=dtype, split_mode="column"
        )

        self.weight_scale = weight_scale
        self.norm_head = norm_head
        self.first_eval_flag = True
        self.tmp_weight = None

    def forward(self, input):  # pylint: disable=W0622
        _class_name = self.__class__.__name__
        assert self._communicator is not None, f"{_class_name} should register with a communicator first."

        if self.weight_scale == 1:
            weight = self.weight
        else:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()

        if self.norm_head:
            if self.training:
                if not self.first_eval_flag:
                    self.first_eval_flag = True
                    self.tmp_weight = None
                # We normalized the output Embedding so that the dot product
                # is not affected by the norm of embedding. Ref: https://arxiv.org/pdf/2309.10305.pdf
                weight = nn.functional.normalize(weight)
            else:
                if self.first_eval_flag:
                    # cache l2 norm of head to accelerate infer.
                    self.first_eval_flag = False
                    self.tmp_weight = nn.functional.normalize(weight)

                weight = self.tmp_weight

        return fused_dense_func(
            input,
            weight,
            communicator=self._communicator,
            module=self,
            bias=self.bias,
        )


class RewardModelLinear(ScaleColumnParallelLinear):
    """
    RewardModelLinear.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, weight_scale)

        # broadcast parameters for reward model head layer.
        parallel_mode = get_head_parallel_mode()
        process_group = gpc.get_group(parallel_mode)
        dist.broadcast(self.weight, gpc.get_ranks_in_group(parallel_mode)[0], process_group)
        if bias:
            dist.broadcast(self.bias, gpc.get_ranks_in_group(parallel_mode)[0], process_group)


class GroupedParallelLinearWithCommExt(ParallelLinearWithCommExt):
    """
    Parallel linear with commuication extention.
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        split_mode (str): The split mode. It can be "none", "column", or "row".
    """

    def __init__(  # pylint: disable=W0231, W0233
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        parallel_mode: ParallelMode,
        backend: str = "gmm",
        multiple_of: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        split_mode: str = "none",
    ) -> None:
        nn.Module.__init__(self)
        assert split_mode in ("none", "column", "row", "weight"), f"unknown split_mode {split_mode}"
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        world_size = gpc.get_world_size(parallel_mode)
        rank = gpc.get_local_rank(parallel_mode)
        split_features_dict = {"column": out_features, "row": in_features, "weight": num_groups * in_features}
        if split_mode != "none":
            split_features = split_features_dict[split_mode]
            multiple = split_features // multiple_of
            # We want to split @multiple across world_size, but it could be an uneven split
            div = multiple // world_size
            mod = multiple % world_size
            # The first @mod ranks get @div + 1 copies, the rest get @div copies
            local_multiple = div + int(rank < mod)
        if split_mode == "column":
            self.weight = nn.Parameter(
                torch.empty(num_groups, in_features, local_multiple * multiple_of, device=device, dtype=dtype)
            )
        elif split_mode == "row":
            self.weight = nn.Parameter(
                torch.empty(num_groups, local_multiple * multiple_of, out_features, device=device, dtype=dtype)
            )
        elif split_mode == "weight":
            self.weight = nn.Parameter(
                torch.empty(local_multiple * multiple_of, out_features, device=device, dtype=dtype)
            )
        else:  # none
            self.weight = nn.Parameter(torch.empty(num_groups, in_features, out_features, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.is_grouped_linear = True
        self.backend = backend


class GroupedColumnLinear(GroupedParallelLinearWithCommExt):
    """
    GroupedSPLinear
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        num_groups (int): number of groups.
        backend (str): backend used for the grouped linear. It can be "gmm" or "bmm".
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        backend: str = "gmm",
        device: torch.device = None,
        dtype: torch.dtype = None,
        is_expert: bool = True,
    ):
        parallel_mode = get_tensor_split_parallel_mode(is_expert=is_expert)
        super().__init__(
            in_features,
            out_features,
            num_groups,
            parallel_mode,
            backend,
            device=device,
            dtype=dtype,
            split_mode="column",
        )
        world_size = gpc.get_world_size(parallel_mode)
        assert world_size == 1, "GroupedSPLinear not support tensor parallel yet."


class GroupedRowLinear(GroupedParallelLinearWithCommExt):
    """
    GroupedSPLinear
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        num_groups (int): number of groups.
        backend (str): backend used for the grouped linear. It can be "gmm" or "bmm".
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        backend: str = "gmm",
        device: torch.device = None,
        dtype: torch.dtype = None,
        is_expert: bool = True,
    ):
        parallel_mode = get_tensor_split_parallel_mode(is_expert=is_expert)
        super().__init__(
            in_features, out_features, num_groups, parallel_mode, backend, device=device, dtype=dtype, split_mode="row"
        )
        world_size = gpc.get_world_size(parallel_mode)
        assert world_size == 1, "GroupedSPLinear not support tensor parallel yet."


class GroupedWPLinear(GroupedParallelLinearWithCommExt):
    """
    GroupedWPLinear
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        num_groups (int): number of groups.
        backend (str): backend used for the grouped linear. It can be "gmm" or "bmm".
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        backend: str = "gmm",
        device: torch.device = None,
        dtype: torch.dtype = None,
        is_expert: bool = True,
    ):
        parallel_mode = get_tensor_split_parallel_mode(is_expert=is_expert)
        super().__init__(
            in_features,
            out_features,
            num_groups,
            parallel_mode,
            backend,
            device=device,
            dtype=dtype,
            split_mode="weight",
        )
        self.full_weight_shape = torch.Size((num_groups, in_features, out_features))


def new_linear(
    name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    multiple_of=1,
    device=None,
    dtype=None,
    is_reward: bool = False,
    weight_scale: int = 1,
    norm_head: bool = False,
    is_expert: bool = False,
    **kwargs,
) -> nn.Linear:

    name = str.lower(name)
    manual_select_class: Optional[str] = kwargs.get("manual_select_class", None)

    if manual_select_class is not None:
        assert manual_select_class in (
            "head",
            "column",
            "row",
        ), f"unknown manual selection {manual_select_class} for creating a linear."

    # use caller manual selection if it is provided.
    split_mode = manual_select_class if manual_select_class is not None else get_parallel_strategies_split_mode(name)

    if split_mode == "head":
        if is_reward:
            return RewardModelLinear(
                in_features,
                out_features,
                bias,
                device,
                dtype,
                weight_scale,
            )
        else:
            return ScaleColumnParallelLinear(
                in_features,
                out_features,
                bias,
                device,
                dtype,
                weight_scale=weight_scale,
                norm_head=norm_head,
            )
    elif split_mode == "column":
        return ColumnParallelLinear(
            in_features,
            out_features,
            bias,
            multiple_of,
            device,
            dtype,
            is_expert,
        )
    elif split_mode == "row":
        return RowParallelLinear(
            in_features,
            out_features,
            bias,
            multiple_of,
            device,
            dtype,
            is_expert,
        )
    elif split_mode == "grouped_wp":
        return GroupedWPLinear(
            in_features,
            out_features,
            kwargs["num_groups"],
            kwargs["backend"],
            device,
            dtype,
            is_expert,
        )
    elif split_mode == "grouped_column":
        return GroupedColumnLinear(
            in_features,
            out_features,
            kwargs["num_groups"],
            kwargs["backend"],
            device,
            dtype,
            is_expert,
        )
    elif split_mode == "grouped_row":
        return GroupedRowLinear(
            in_features,
            out_features,
            kwargs["num_groups"],
            kwargs["backend"],
            device,
            dtype,
            is_expert,
        )
    elif split_mode == "gate":
        return nn.Linear(
            in_features,
            out_features,
            bias,
        )
    else:
        err_msg = (
            f"Parallel strategies for linear is unsupported, which is named as {name}.\n"
            + "Consider use manual_select_class parameter to select a linear class manually."
        )

        raise ValueError(err_msg)
