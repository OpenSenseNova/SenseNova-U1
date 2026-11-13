"""Merge BF16 shards without changing tensor names, shapes or values."""

import argparse

import torch
from convert_utils import finish, prepare_output, tensors
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    partial = prepare_output(a.model_path, a.output)
    state = {}
    for name, tensor in tensors(a.model_path):
        if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
            raise ValueError(f"Expected finite BF16 tensor: {name}")
        state[name] = tensor
    if len(state) != 1116:
        raise ValueError(f"Expected 1116 U1.6 tensors, got {len(state)}")
    save_file(state, str(partial), metadata={"format": "pt"})
    finish(partial, a.output, a.model_path, format="bf16", tensors=len(state))


if __name__ == "__main__":
    main()
