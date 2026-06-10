#!/usr/bin/env python3
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""Convert an internevo LoRA checkpoint into a PEFT-compatible adapter.

The training-side saver (``sensenovalm/checkpoint/checkpoint_manager.py``,
LoRA short-circuit) writes a tiny ``lora_state.pt`` file containing:

    {
        "lora_state_dict": {"<module>.lora_A.weight": Tensor, ...},
        "lora_config":     {"r": 16, "alpha": 32, "dropout": 0.0,
                             "target_prefixes": ["fm_modules."],
                             "target_leaf_names": [...]},
        "step":            int,
        "num_lora_tensors": int,
    }

This script reads that file and emits a PEFT-style adapter directory::

    <output_dir>/
        adapter_model.safetensors    # weight tensors with PEFT key naming
        adapter_config.json          # rank/alpha/target_modules/etc.

The result can then be loaded with::

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "<output_dir>")

Notes
-----
* The PEFT key convention is ``base_model.model.<base_qualname>.lora_A.weight``;
  the training-side key is just ``<base_qualname>.lora_A.weight``. We rewrite
  the prefix during export.
* Because we only insert LoRA under ``fm_modules.*``, the resulting
  ``target_modules`` list is a flat de-duplicated set of leaf attribute names.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_lora_to_hf")


def _load_lora_payload(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        raise SystemExit(f"LoRA checkpoint not found: {ckpt_path}")
    logger.info(f"Loading LoRA payload from {ckpt_path} ...")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "lora_state_dict" not in payload:
        raise SystemExit(
            f"{ckpt_path} does not look like a LoRA checkpoint (missing 'lora_state_dict')."
        )
    return payload


def _remap_to_peft(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite ``<qualname>.lora_A.weight`` → ``base_model.model.<qualname>.lora_A.default.weight``.

    PEFT's default adapter name is ``default``; including it makes the resulting
    files load cleanly without specifying an adapter name.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        # Identify the "lora_A" / "lora_B" component and insert ``.default``
        # between it and ``.weight``.
        if ".lora_A.weight" in key:
            new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        elif ".lora_B.weight" in key:
            new_key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        else:
            logger.warning(f"Unexpected key in LoRA state, copied as-is: {key}")
            new_key = key
        # PEFT prefixes everything with ``base_model.model.``.
        new_key = f"base_model.model.{new_key}"
        out[new_key] = tensor.contiguous()
    return out


def _derive_target_modules(state: Dict[str, torch.Tensor]) -> List[str]:
    """Extract the unique leaf attribute names that got LoRA applied."""
    names = set()
    for key in state:
        # key looks like '<...>.<leaf>.lora_A.weight'
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            continue
        head = key.split(".lora_")[0]
        leaf = head.rsplit(".", 1)[-1]
        names.add(leaf)
    return sorted(names)


def _write_adapter_config(
    output_dir: Path,
    lora_config: dict,
    target_modules: List[str],
    base_model_name_or_path: str,
) -> Path:
    cfg = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": int(lora_config.get("alpha", 16)),
        "lora_dropout": float(lora_config.get("dropout", 0.0)),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": int(lora_config.get("r", 8)),
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": None,
    }
    path = output_dir / "adapter_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def _save_safetensors(state: Dict[str, torch.Tensor], output_dir: Path) -> Path:
    out_path = output_dir / "adapter_model.safetensors"
    try:
        from safetensors.torch import save_file  # noqa: WPS433
    except ImportError as e:
        raise SystemExit(
            "safetensors is required for export. Install with: pip install safetensors"
        ) from e
    # safetensors disallows shared storage — clone every tensor defensively.
    state = {k: v.detach().clone().contiguous() for k, v in state.items()}
    save_file(state, str(out_path))
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Path to the lora_state.pt produced by training.")
    p.add_argument("--tgt", required=True, help="Output directory for the PEFT adapter.")
    p.add_argument(
        "--base_model_name_or_path",
        default="SenseNova/SenseNova-U1-8B-MoT-SFT",
        help="String stored in adapter_config.base_model_name_or_path. "
             "Only used as metadata — the LoRA still has to be loaded onto whatever model you initialize.",
    )
    args = p.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    tgt = Path(args.tgt).expanduser().resolve()
    tgt.mkdir(parents=True, exist_ok=True)

    payload = _load_lora_payload(src)
    raw_state = payload["lora_state_dict"]
    lora_config = payload.get("lora_config", {})

    target_modules = _derive_target_modules(raw_state)
    if not target_modules:
        raise SystemExit("Could not derive target_modules from state dict — keys malformed?")
    logger.info(f"Detected {len(target_modules)} target module names: {target_modules}")

    peft_state = _remap_to_peft(raw_state)
    safetensors_path = _save_safetensors(peft_state, tgt)
    config_path = _write_adapter_config(tgt, lora_config, target_modules, args.base_model_name_or_path)

    logger.info(f"Wrote adapter weights to {safetensors_path}")
    logger.info(f"Wrote adapter config  to {config_path}")
    logger.info(
        f"PEFT load example:\n"
        f"    from peft import PeftModel\n"
        f"    model = PeftModel.from_pretrained(base_model, '{tgt}')\n"
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(str(e))
        sys.exit(1)
