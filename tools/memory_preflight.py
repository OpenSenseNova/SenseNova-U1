#!/usr/bin/env python3
"""Read-only U1.5 resource/configuration preflight; does not load model tensors.

This is a conservative official-CLI guard, not an inference or GGUF compatibility
test. Run it using the same Python environment that will run inference.
"""

import argparse
import collections
import json
import math
import pathlib
import re
import struct
import sys

GIB = 1024**3


def read_inventory(directory):
    directory = pathlib.Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    tensors, shards = {}, []
    for filename in sorted(set(index["weight_map"].values())):
        path = directory / filename
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"{filename}: incomplete safetensors header")
            length = struct.unpack("<Q", raw)[0]
            if length > 32 * 1024 * 1024:
                raise ValueError(f"{filename}: implausibly large metadata header")
            header = json.loads(f.read(length))
        payload_bytes = path.stat().st_size - 8 - length
        ranges = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or index["weight_map"].get(name) != filename:
                raise ValueError(f"{name}: duplicate tensor or index/shard mismatch")
            lo, hi = info["data_offsets"]
            if not 0 <= lo <= hi <= payload_bytes:
                raise ValueError(f"{name}: out-of-range payload offset")
            ranges.append((lo, hi))
            info["nbytes"] = hi - lo
            info["elements"] = math.prod(info["shape"])
            item_bytes = {
                "BF16": 2,
                "F16": 2,
                "F32": 4,
                "F64": 8,
                "I8": 1,
                "U8": 1,
                "I16": 2,
                "I32": 4,
                "I64": 8,
                "BOOL": 1,
            }.get(info["dtype"])
            if item_bytes is None or item_bytes * info["elements"] != info["nbytes"]:
                raise ValueError(f"{name}: unsupported dtype or shape/byte-size mismatch")
            tensors[name] = info
        ranges.sort()
        if ranges and (ranges[0][0] != 0 or ranges[-1][1] != payload_bytes):
            raise ValueError(f"{filename}: incomplete payload coverage")
        if any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f"{filename}: overlapping/gapped tensor payloads")
        shards.append(
            {
                "file": filename,
                "file_bytes": path.stat().st_size,
                "payload_bytes": payload_bytes,
                "tensor_count": len(ranges),
            }
        )
    if set(index["weight_map"]) != set(tensors):
        raise ValueError("Index and shard tensor sets differ")
    groups = collections.Counter()
    dtypes = collections.Counter()
    per_layer = collections.defaultdict(collections.Counter)
    for name, info in tensors.items():
        nbytes = info["nbytes"]
        dtypes[info["dtype"]] += nbytes
        match = re.match(r"language_model\.model\.layers\.(\d+)\.(.*)", name)
        if match:
            layer, relative = match.groups()
            parts = relative.split(".")
            if any(part.endswith("_mot_gen") for part in parts[:-1]):
                group = "generation"
            else:
                prefix = f"language_model.model.layers.{layer}."
                pair = False
                for i in range(len(parts) - 1):
                    candidate = parts.copy()
                    candidate[i] += "_mot_gen"
                    if prefix + ".".join(candidate) in tensors:
                        pair = True
                        break
                group = "understanding" if pair else "shared"
            per_layer[layer][group] += nbytes
            groups["decoder_" + group] += nbytes
        elif name.startswith(("language_model.model.embed_tokens.", "language_model.lm_head.")):
            groups["prefix_only"] += nbytes
        else:
            groups["other_non_decoder"] += nbytes
    total = sum(info["nbytes"] for info in tensors.values())
    if total != index["metadata"]["total_size"]:
        raise ValueError("Index total_size and shard payload total differ")
    return {
        "tensor_count": len(tensors),
        "elements": sum(x["elements"] for x in tensors.values()),
        "payload_bytes": total,
        "payload_gib": total / GIB,
        "shards": shards,
        "dtype_payload_bytes": dict(dtypes),
        "group_payload_bytes": dict(groups),
        "per_decoder_layer_bytes": dict(sorted(per_layer.items(), key=lambda x: int(x[0]))),
        "minimum_pinned_backing_bytes": total - groups["other_non_decoder"],
        "note": "Header arithmetic only, not runtime peak, tensor-value check, or SHA256 validation.",
    }


def host_memory():
    mem = {}
    p = pathlib.Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            key, value = line.split(":", 1)
            if value.strip().endswith("kB"):
                mem[key] = int(value.split()[0]) * 1024
    out = {key: mem.get(key) for key in ["MemTotal", "MemAvailable", "SwapTotal"]}
    for filename in ["memory.max", "memory.current", "memory.swap.max"]:
        p = pathlib.Path("/sys/fs/cgroup") / filename
        if p.exists():
            raw = p.read_text().strip()
            out["cgroup_" + filename] = int(raw) if raw.isdigit() else raw
    available = out.get("MemAvailable")
    limit, used = out.get("cgroup_memory.max"), out.get("cgroup_memory.current")
    if isinstance(limit, int) and isinstance(used, int):
        cg_free = max(0, limit - used)
        available = min(available, cg_free) if available is not None else cg_free
    out["effective_available_bytes"] = available
    return out


def combination_errors(mode, gguf, lora, device_map, device):
    errors = []
    if mode != "full" and device_map:
        errors.append("Layer offload and device_map are incompatible; choose one.")
    if gguf and device_map:
        errors.append("Official GGUF loader does not implement device_map dispatch.")
    if gguf and lora:
        errors.append("Official GGUF + ordinary LoRA merge is not an accepted configuration.")
    if device.split(":")[0] not in ("cuda", "xpu"):
        errors.append(
            "This preflight accepts only CUDA/XPU deployments; CPU/MPS are outside its validation matrix and cannot use official layer offload."
        )
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--vram-mode", choices=["full", "fast", "balanced", "low"], default="balanced")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gguf")
    p.add_argument("--lora")
    p.add_argument("--device-map")
    p.add_argument("--output")
    args = p.parse_args()
    errors = combination_errors(args.vram_mode, args.gguf, args.lora, args.device_map, args.device)
    warnings = []
    facts = {"requested_configuration": vars(args), "host_memory": host_memory()}
    try:
        import torch

        facts["torch_version"] = torch.__version__
        facts["cuda_available"] = torch.cuda.is_available()
        facts["cuda_device_count"] = torch.cuda.device_count()
        facts["xpu_available"] = hasattr(torch, "xpu") and torch.xpu.is_available()
        try:
            target = torch.device(args.device)
        except (RuntimeError, ValueError) as exc:
            errors.append(f"Invalid target device: {exc}")
            target = None
        backend = target.type if target is not None else None
        if backend in ("cuda", "xpu"):
            mod = getattr(torch, backend, None)
            index = target.index if target.index is not None else 0
            if mod is None or not mod.is_available() or index >= mod.device_count():
                errors.append(f"Target accelerator {args.device} is unavailable.")
            else:
                props = mod.get_device_properties(index)
                facts["accelerator"] = {"name": props.name, "total_memory_bytes": props.total_memory}
                if backend == "cuda":
                    with torch.cuda.device(index):
                        facts["bf16_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
                    if not facts["bf16_supported"]:
                        errors.append("Native BF16 support is unavailable; do not silently change precision.")
                    if args.device_map:
                        warnings.append(
                            "This preflight checks only the selected GPU; verify every device in the actual hf_device_map separately."
                        )
    except ImportError as exc:
        errors.append(f"PyTorch import failed: {exc}")
    if args.gguf:
        path = pathlib.Path(args.gguf)
        if not path.is_file():
            errors.append("GGUF file is missing.")
        else:
            facts["gguf_file_bytes"] = path.stat().st_size
        warnings.append("GGUF tensor names/shapes/types and task output require separate validation.")
    else:
        try:
            inv = read_inventory(args.model_dir)
            facts["weight_inventory"] = inv
            available = facts["host_memory"]["effective_available_bytes"]
            if args.vram_mode != "full" and available is not None and available < inv["minimum_pinned_backing_bytes"]:
                errors.append(
                    "Available host memory is below fixed BF16 pinned backing alone; offload is not safe to start."
                )
            if (
                args.vram_mode == "full"
                and not args.device_map
                and facts.get("accelerator", {}).get("total_memory_bytes", float("inf")) < inv["payload_bytes"]
            ):
                errors.append("Single-device VRAM is smaller than unquantized weight payload alone.")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"Checkpoint header validation failed: {exc}")
    warnings.append(
        "Passing this preflight is not proof of sufficient RAM/VRAM, model quality, or successful inference."
    )
    facts.update(
        errors=errors, warnings=warnings, status="blocked" if errors else "preflight_passed_not_inference_verified"
    )
    serialized = json.dumps(facts, ensure_ascii=False, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(serialized + "\n")
    print(serialized)
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
