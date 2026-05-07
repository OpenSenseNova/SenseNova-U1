"""Shared model + tokenizer loader for SenseNova-U1.

Centralises the ``AutoConfig`` / ``AutoTokenizer`` / ``AutoModel`` calls used
by the example scripts and the ComfyUI app, and adds an optional GGUF
checkpoint override.

Usage:

    from sensenova_u1.utils import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_path="sensenova/SenseNova-U1-8B-MoT",
        dtype=torch.bfloat16,
        device="cuda",
    )

    # GGUF override (config / tokenizer still come from `model_path`):
    model, tokenizer = load_model_and_tokenizer(
        model_path="sensenova/SenseNova-U1-8B-MoT",
        dtype=torch.bfloat16,
        device="cuda",
        gguf_checkpoint="/path/to/SenseNova-U1-8B-MoT-Q5_K_M.gguf",
    )
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _resolve_local_model_path(model_path: str) -> str:
    """Resolve a HF id to its cached snapshot directory when offline.

    Mirrors transformers' fall-back behaviour but skips the up-front HEAD
    request that times out on offline machines. Returns the input unchanged
    if the path already exists or no cached snapshot is found.
    """
    if Path(model_path).exists():
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        return model_path


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = "cuda",
    gguf_checkpoint: str | None = None,
    device_map: str | None = None,
    max_memory: str | dict[int | str, str] | None = None,
    offload_folder: str | None = None,
    offload_state_dict: bool | None = None,
) -> tuple[nn.Module, Any]:
    """Build a SenseNova-U1 model + tokenizer pair.

    ``model_path`` always provides the config and tokenizer (HF id or local
    directory containing ``config.json``).

    Weight loading branches on ``gguf_checkpoint``:

    - ``None``: standard ``AutoModel.from_pretrained(model_path, ...)``.
      The ``device_map`` / ``max_memory`` / ``offload_*`` accelerate kwargs
      apply on this path; when ``device_map`` is ``None`` the model is
      ``.to(device)`` after loading.
    - ``"*.gguf"``: build a meta-init model from the config and inject
      dequantizing weights from the GGUF file via the diffusers quantizer.
      The accelerate offload kwargs are ignored on this path.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .. import check_checkpoint_compatibility

    model_path = _resolve_local_model_path(model_path)
    config = AutoConfig.from_pretrained(model_path)
    check_checkpoint_compatibility(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if gguf_checkpoint is not None:
        model = _load_from_gguf(config, gguf_checkpoint, dtype=dtype, device=device)
    else:
        model_kwargs: dict[str, Any] = {"config": config, "torch_dtype": dtype}
        if device_map:
            model_kwargs["device_map"] = device_map
            parsed_max_memory = _normalize_max_memory(max_memory)
            if parsed_max_memory:
                model_kwargs["max_memory"] = parsed_max_memory
            if offload_folder:
                model_kwargs["offload_folder"] = offload_folder
            if offload_state_dict is not None:
                model_kwargs["offload_state_dict"] = offload_state_dict

        model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        if not device_map and device is not None:
            model = model.to(device)

    return model, tokenizer


def _normalize_max_memory(value: str | dict | None) -> dict[int | str, str]:
    """Accept either a parsed mapping or the comma-separated CLI form ``"0=20GiB,cpu=64GiB"``."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    result: dict[int | str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError("max_memory entries must look like 0=20GiB,cpu=64GiB.")
        key, memory = item.split("=", 1)
        key = key.strip()
        memory = memory.strip()
        if not key or not memory:
            raise RuntimeError("max_memory entries must include both device and memory.")
        result[int(key) if key.isdigit() else key] = memory
    return result


def _load_from_gguf(
    config,
    gguf_checkpoint: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None,
) -> nn.Module:
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise RuntimeError("GGUF loading requires `accelerate`; install it in your environment.") from exc

    from transformers import AutoModel

    from .gguf_loader import load_gguf_checkpoint, set_gguf2meta_model

    print(f"[gguf] loading quantized checkpoint from {gguf_checkpoint}")
    with init_empty_weights():
        model = AutoModel.from_config(config)

    state_dict = load_gguf_checkpoint(gguf_checkpoint)
    print(f"[gguf] parsed {len(state_dict)} tensors")
    target_device = torch.device(device) if isinstance(device, str) else device
    # set_gguf2meta_model places weights on `target_device` while injecting;
    # callers that ultimately want a different device can `.to()` afterwards.
    set_gguf2meta_model(model, state_dict, dtype, target_device)

    n_gguf_linear = sum(1 for m in model.modules() if type(m).__name__ == "GGUFLinear")
    print(f"[gguf] {n_gguf_linear} GGUFLinear modules active (dequantized at forward time)")
    if n_gguf_linear == 0:
        print("[gguf] WARNING: no GGUFLinear modules found — quantizer hook did not run as expected")

    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model.eval()
