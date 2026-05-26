# --------------------------------------------------------
# Derived from NVIDIA Apex (NVIDIA, BSD-3-Clause).
# Copyright (c) NVIDIA Corporation.
# Modifications copyright (c) SenseNovaLM contributors. Licensed under
# Apache-2.0; original BSD-3-Clause portions retain their license.
# Upstream reference:
#   https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm
# --------------------------------------------------------

import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from sensenovalm.accelerator import AcceleratorType, get_accelerator
from sensenovalm.utils.logger import get_logger

logger = get_logger(__file__)
sensenovalm_accelerator = get_accelerator()

try:
    from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine

    apex_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
    apex_rmsnorm_impl = False

try:
    from deeplink_ext.internevo_ops import MixedFusedRMSNorm as _RMSNormDIPU

    deeplink_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    deeplink_rmsnorm_impl = False

try:
    from torch_npu import npu_rms_norm

    torchnpu_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    torchnpu_rmsnorm_impl = False


def manual_rms_norm(my_input, weight, normalized_shape, eps, add_unit_offset=False):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)

    if add_unit_offset:
        return (1 + weight) * my_input
    else:
        return weight * my_input


class _RMSNorm(torch.nn.Module):
    """A generic module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5, add_unit_offset=False):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.add_unit_offset = add_unit_offset
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        if apex_rmsnorm_impl:
            _norm_func = mixed_dtype_fused_rms_norm_affine
            return _norm_func(_input, self.weight, self.normalized_shape, self.eps)
        else:
            _norm_func = manual_rms_norm
            return _norm_func(_input, self.weight, self.normalized_shape, self.eps, self.add_unit_offset)

    def reset_parameters(self):
        if self.add_unit_offset:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, "


class _RMSNormNPU(torch.nn.Module):
    """A custom NPU module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.reset_parameters()
        self.rmsorm_npu_forward = npu_rms_norm

    def forward(self, _input: torch.Tensor):
        weight_fp32 = self.weight.to(torch.float32)
        input_fp32 = _input.to(torch.float32)
        output = self.rmsorm_npu_forward(input_fp32, gamma=weight_fp32, epsilon=self.eps)[0].to(self.weight.dtype)
        return output

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, ".format(**self.__dict__)


# TODO: Support deeplink in a more unified manner
backend = sensenovalm_accelerator.get_accelerator_backend()
if backend in [AcceleratorType.DIPU, AcceleratorType.DITORCH] and deeplink_rmsnorm_impl:
    RMSNorm = _RMSNormDIPU
elif backend == AcceleratorType.NPU and torchnpu_rmsnorm_impl:
    RMSNorm = _RMSNormNPU
else:
    RMSNorm = _RMSNorm
