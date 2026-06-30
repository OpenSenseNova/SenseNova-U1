#!/usr/bin/env python3
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""Convert an internevo LoRA checkpoint into a PEFT-compatible adapter.

The training-side saver (``sensenovalm/checkpoint/checkpoint_manager.py``)
writes a ``lora_state.pt`` file containing::

    {
        "lora_state_dict": {"<module>.lora_A.weight": Tensor, ...},
        "lora_config":     {"r": ..., "alpha": ..., "dropout": ...,
                             "target_prefixes": [...], "target_leaf_names": [...]},
        "step":            int,
        "num_lora_tensors": int,
    }

This script reads that file and emits a PEFT-style adapter directory::

    <output_dir>/
        adapter_model.safetensors    # weight tensors with PEFT key naming
        adapter_config.json          # rank/alpha/target_modules/etc.

loadable with::

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "<output_dir>")
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_lora_to_hf")

# The training-side keys carry a leading "model." from the AMP wrapper.
_WRAPPER_PREFIX = "model."


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
    """Rewrite ``[model.]<qualname>.lora_{A,B}.weight`` → ``base_model.model.<qualname>.lora_{A,B}.weight``.

    PEFT checkpoints store keys *without* the adapter name —
    ``set_peft_model_state_dict`` re-inserts ``.default.`` at load time.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            logger.warning(f"Skipping unexpected key in LoRA state: {key}")
            continue
        key = key.removeprefix(_WRAPPER_PREFIX)
        out[f"base_model.model.{key}"] = tensor.contiguous()
    return out


def _derive_target_modules(state: Dict[str, torch.Tensor]) -> List[str]:
    """Full module paths that received LoRA.

    Full paths (rather than leaf attribute names) keep PEFT's suffix matching
    exact — leaf names like the ``"0"`` of ``fm_head.0`` would otherwise match
    every module whose qualname ends in ``.0``.
    """
    names = set()
    for key in state:
        if ".lora_A." in key or ".lora_B." in key:
            names.add(key.removeprefix(_WRAPPER_PREFIX).split(".lora_")[0])
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
        from safetensors.torch import save_file
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
    logger.info(f"Detected {len(target_modules)} target modules: {target_modules}")

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
