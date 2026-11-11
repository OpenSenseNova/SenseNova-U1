#!/usr/bin/env python3
"""Check pinned software and every visible CUDA GPU; does not load model weights."""

import argparse
import sys
from importlib.metadata import version

import torch
import torch.nn.functional as F

import sensenova_u1

if not __debug__:
    raise SystemExit("Run without -O or PYTHONOPTIMIZE")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpus", type=int, choices=(1, 2, 4), help="Require exactly this many visible GPUs")
args = parser.parse_args()
expected = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "transformers": "5.15.0",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.27.0",
    "safetensors": "0.8.0",
    "pillow": "12.0.0",
}
assert sys.version_info[:2] == (3, 11), sys.version
print("Python:", sys.executable, sys.version)
for name, want in expected.items():
    got = version(name)
    print(name, got)
    assert got == want, (name, got, want)
print("Package:", sensenova_u1.__file__)
assert torch.cuda.is_available(), "CUDA unavailable"
count = torch.cuda.device_count()
assert count > 0, "No visible CUDA GPU"
if args.gpus is not None:
    assert count == args.gpus, f"Expected {args.gpus} visible GPUs, got {count}"
print("CUDA runtime:", torch.version.cuda, "Visible GPUs:", count)
for i in range(count):
    with torch.cuda.device(i), torch.inference_mode():
        assert torch.cuda.is_bf16_supported(including_emulation=False), f"GPU {i} requires native BF16 support"
        print("GPU", i, torch.cuda.get_device_properties(i))
        x = torch.randn(32, 32, device=f"cuda:{i}", dtype=torch.bfloat16)
        y = x @ x
        q = torch.randn(1, 2, 32, 64, device=f"cuda:{i}", dtype=torch.bfloat16)
        out = F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize(i)
        assert bool(torch.isfinite(y).all())
        assert bool(torch.isfinite(out).all())
        del x, y, q, out
sensenova_u1.set_attn_backend("sdpa")
assert sensenova_u1.effective_attn_backend() == "sdpa"
print("ENVIRONMENT_CHECK_OK", f"visible_gpus={count}")
