from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sensenova_u1 import check_checkpoint_compatibility


def add_offload_args(parser: argparse.ArgumentParser) -> None:
    """Add Transformers/Accelerate device-map loading flags to an example CLI."""
    parser.add_argument(
        "--device_map",
        default=None,
        help=(
            "Optional Transformers device_map, e.g. 'auto', 'balanced', "
            "'balanced_low_0', or 'sequential'. When set, the model is loaded "
            "with Accelerate dispatch and is not moved again with .to(device)."
        ),
    )
    parser.add_argument(
        "--max_memory",
        default=None,
        help=(
            "Optional per-device memory limits for --device_map, either JSON "
            "or comma-separated KEY=VALUE pairs, e.g. '0=20GiB,cpu=64GiB'."
        ),
    )
    parser.add_argument(
        "--offload_folder",
        default=None,
        help="Folder for disk offload when --device_map places modules on 'disk'.",
    )
    parser.add_argument(
        "--offload_state_dict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forwarded to AutoModel.from_pretrained for CPU/disk offload loading.",
    )


def parse_max_memory(spec: str | None) -> dict[int | str, str] | None:
    if not spec:
        return None

    stripped = spec.strip()
    if stripped.startswith("{"):
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            raise ValueError("--max_memory JSON must be an object")
        return {_coerce_memory_key(k): str(v) for k, v in raw.items()}

    result: dict[int | str, str] = {}
    for item in stripped.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --max_memory item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        result[_coerce_memory_key(key.strip())] = value.strip()
    return result or None


def _coerce_memory_key(key: object) -> int | str:
    if isinstance(key, int):
        return key
    key_str = str(key)
    return int(key_str) if key_str.isdigit() else key_str


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str = "cuda",
    device_map: str | dict[str, Any] | None = None,
    max_memory: str | dict[int | str, str] | None = None,
    offload_folder: str | None = None,
    offload_state_dict: bool | None = None,
):
    """Load a SenseNova-U1 checkpoint with optional Accelerate offload."""
    config = AutoConfig.from_pretrained(model_path)
    check_checkpoint_compatibility(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    parsed_max_memory = parse_max_memory(max_memory) if isinstance(max_memory, str) else max_memory
    if parsed_max_memory:
        load_kwargs["max_memory"] = parsed_max_memory
    if offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = offload_folder
    if offload_state_dict is not None:
        load_kwargs["offload_state_dict"] = offload_state_dict

    model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
    if not device_map:
        model = model.to(device)
    return model, tokenizer


def infer_input_device(model: torch.nn.Module, fallback: str = "cuda") -> torch.device:
    """Pick a usable device for tensors passed into a dispatched model."""
    for param in model.parameters():
        if param.device.type not in {"cpu", "meta"}:
            return param.device
    return torch.device(fallback)
