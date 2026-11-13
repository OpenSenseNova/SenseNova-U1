"""FP8 per-channel min/max quantization, adapted from LightX2V.

Source: lightx2v/utils/quant_utils.py::quant_naive_inplace
Commit: 3aebc7f885f84e2f3c4276617caf669f4e858326 (Apache-2.0).
Modified: return a new quantized tensor and explicit float32 scales.
Uses native PyTorch E4M3FN rounding, not the qtorch simulation path.
"""

import torch


@torch.no_grad()
def quantize_weight(weight):
    weight = weight.float()
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 448.0
    quantized = (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale, {}
