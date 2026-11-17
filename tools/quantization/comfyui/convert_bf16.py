"""Losslessly repack a SenseNova HF checkpoint for ComfyUI.

Usage: python convert_bf16.py HF_DIR OUTPUT [--report REPORT]
Only tensor storage is repacked; names, dtypes and values are unchanged.
"""

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from safetensors import safe_open


def read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def repack(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    header, shards = {}, []
    offset = 0
    for filename in sorted(set(weight_map.values())):
        path = (source / filename).resolve()
        if path.parent != source:
            raise ValueError(f"Shard is outside checkpoint directory: {filename}")
        tensors, start = read_header(path)
        # Let the Safetensors reader validate shapes, dtypes and storage offsets.
        with safe_open(path, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != {k for k, v in weight_map.items() if v == filename}:
                raise ValueError(f"Shard keys do not match the HF index: {filename}")
        size = path.stat().st_size - start
        for key, spec in tensors.items():
            if key in header:
                raise ValueError(f"Duplicate tensor: {key}")
            begin, end = spec["data_offsets"]
            header[key] = {**spec, "data_offsets": [offset + begin, offset + end]}
        shards.append((path, start, size))
        offset += size
    if not header or set(header) != set(weight_map):
        raise ValueError("Checkpoint has no tensors or does not match the HF index")
    if "total_size" in index.get("metadata", {}) and index["metadata"]["total_size"] != offset:
        raise ValueError("Checkpoint byte count does not match the HF index")
    header["__metadata__"] = {
        "format": "pt",
        "architecture": "sensenova-u1-mot",
        "source_checkpoint": str(source),
        "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "conversion": "lossless-shard-repack-v1",
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    source_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for path, start, size in shards:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = size
                    while remaining:
                        data = stream.read(min(8 * 1024 * 1024, remaining))
                        if not data:
                            raise ValueError(f"Shard truncated during conversion: {path}")
                        source_digest.update(data)
                        target.write(data)
                        remaining -= len(data)
                print(f"Copied {path.name}: {size:,} bytes", flush=True)
            target.flush()
            os.fsync(target.fileno())
        with safe_open(temporary, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(weight_map):
                raise ValueError("Converted checkpoint has different tensor keys")
        output_digest, payload_digest = hashlib.sha256(), hashlib.sha256()
        with open(temporary, "rb") as stream:
            output_digest.update(stream.read(8 + len(encoded)))
            while data := stream.read(8 * 1024 * 1024):
                output_digest.update(data)
                payload_digest.update(data)
        if payload_digest.digest() != source_digest.digest():
            raise ValueError("Converted tensor bytes differ from source")
        # Publish only a fully verified artifact, without replacing existing files.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    report = {
        "source": str(source),
        "output": str(output),
        "shards": len(shards),
        "tensors": len(weight_map),
        "tensor_bytes": offset,
        "file_bytes": output.stat().st_size,
        "sha256": output_digest.hexdigest(),
        "tensor_payload_sha256": payload_digest.hexdigest(),
        "source_payload_sha256": source_digest.hexdigest(),
        "bit_exact": True,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = repack(args.source, args.output)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
