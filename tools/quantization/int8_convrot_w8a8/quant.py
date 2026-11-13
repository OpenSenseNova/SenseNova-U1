"""INT8 ConvRot adapter, adapted from LightX2V's QuantWeightINT8ConvRot.

Source: tools/convert/quant/quant.py
Commit: 3aebc7f885f84e2f3c4276617caf669f4e858326 (Apache-2.0).
Modified: plain function; no registry, unrelated quantizers or framework imports.
"""

import torch
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout


@torch.no_grad()
def quantize_weight(weight):
    group = next((g for g in (256, 64, 16) if weight.shape[1] % g == 0), None)
    if group is None:
        raise ValueError(f"Unsupported ConvRot width: {weight.shape}")
    q, params = TensorWiseINT8Layout.quantize(
        weight.float().contiguous(),
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=group,
        stochastic_rounding=0,
    )
    return q.contiguous(), params.scale.float().reshape(-1, 1), {"group_size": group}
