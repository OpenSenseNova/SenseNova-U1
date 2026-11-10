"""Quantize a SenseNova-U1 Safetensors checkpoint to a GGUF file.

Tensor names stay the same as in the checkpoint, so the file loads with
`--gguf_checkpoint`. Weights are read one tensor at a time and written to disk
as they are quantized, so peak memory stays small.

- 2-D weights of `*_proj.weight` (and `lm_head.weight`) become the chosen quant type.
- `embed_tokens.weight` and `lm_head.weight` stay in F16 (checked for overflow).
- All other tensors (norms, biases, convolutions) are stored as F32.

Usage:
    python scripts/quantize_to_gguf.py MODEL_DIR OUT.gguf [--qtype Q5_0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType as QT
from gguf import GGUFWriter
from gguf.quants import quant_shape_to_byte_shape, quantize
from safetensors import safe_open

QTYPES = {"Q4_0": QT.Q4_0, "Q4_1": QT.Q4_1, "Q5_0": QT.Q5_0, "Q5_1": QT.Q5_1, "Q8_0": QT.Q8_0}
FILE_TYPES = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 8, "Q5_1": 9, "Q8_0": 7}
BLOCK = 32
F16_KEEP = ("embed_tokens.weight", "lm_head.weight")


def plan(name: str, shape: tuple[int, ...], qtype: QT) -> QT:
    """Pick the storage type for one tensor."""
    if name.endswith(F16_KEEP):
        return QT.F16
    if name.endswith("_proj.weight") or "_proj_mot_gen.weight" in name:
        if len(shape) == 2 and shape[1] % BLOCK == 0:
            return qtype
    return QT.F32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--qtype", default="Q5_0", choices=sorted(QTYPES))
    args = ap.parse_args()
    qtype = QTYPES[args.qtype]

    root = Path(args.model_dir)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))

    # Pass 1: names, shapes and target types (headers only, no weights read).
    entries: list[tuple[str, str, QT, tuple[int, ...]]] = []
    for shard in shards:
        with safe_open(root / shard, "pt") as f:
            for name in f.keys():
                shape = tuple(f.get_slice(name).get_shape())
                entries.append((shard, name, plan(name, shape, qtype), shape))

    writer = GGUFWriter(args.out, "sensenova_u1")
    writer.add_name("SenseNova U1.5 8B MoT")
    writer.add_file_type(FILE_TYPES[args.qtype])
    for _, name, t, shape in entries:
        if t == QT.F32:
            writer.add_tensor_info(name, shape, np.dtype("float32"), int(np.prod(shape)) * 4)
        elif t == QT.F16:
            writer.add_tensor_info(name, shape, np.dtype("float16"), int(np.prod(shape)) * 2)
        else:
            byte_shape = quant_shape_to_byte_shape(shape, t)
            writer.add_tensor_info(name, byte_shape, np.dtype("uint8"), int(np.prod(byte_shape)), raw_dtype=t)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Pass 2: stream tensors in the same order as the tensor-info table.
    handles: dict[str, object] = {}
    n_q = 0
    try:
        for i, (shard, name, t, _) in enumerate(entries):
            if shard not in handles:
                for h in handles.values():
                    h.__exit__(None, None, None)
                handles = {shard: safe_open(root / shard, "pt").__enter__()}
            w = handles[shard].get_tensor(name).to(torch.float32)
            if t == QT.F32:
                arr = w.numpy()
            elif t == QT.F16:
                if w.abs().max() > 65000:
                    raise ValueError(f"{name}: values overflow float16")
                arr = w.to(torch.float16).numpy()
            else:
                arr = quantize(w.numpy(), t)
                n_q += 1
            writer.write_tensor_data(arr)
            del w, arr
            if i % 100 == 0:
                print(f"[{i}/{len(entries)}] {name} -> {t.name}", flush=True)
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    writer.close()
    print(f"done: {len(entries)} tensors, {n_q} quantized as {args.qtype} -> {args.out}")


if __name__ == "__main__":
    main()
