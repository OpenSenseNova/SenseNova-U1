from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

import torch

import sensenova_u1
from examples.t2i.inference import SenseNovaU1T2I

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "test.jsonl"
DEFAULT_ASPECT_RATIO = "1:1"
RATIO_LONG_SIDE = 2048
DEFAULT_ATTN_BACKEND = "auto"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BizGenEval images with SenseNova-U1.")
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or HF model id.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="BizGenEval JSONL path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-norm", default="none", choices=["none", "global", "channel", "cfg_zero_star"])
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-backend", default=DEFAULT_ATTN_BACKEND, choices=["auto", "flash", "sdpa"])
    return parser.parse_args()


def _load_items(*, data_path: Path) -> list[dict[str, Any]]:
    items = []
    with data_path.open(encoding="utf-8") as f:
        for prompt_id, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_prompt_id"] = prompt_id
            items.append(item)
    return items


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _parse_ratio(value: object) -> tuple[int, int] | None:
    try:
        if isinstance(value, str):
            lower = value.lower()
            if "x" in lower:
                rw, rh = [int(x) for x in lower.split("x", 1)]
            elif ":" in value:
                rw, rh = [int(x) for x in value.split(":", 1)]
            else:
                return None
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            rw, rh = int(value[0]), int(value[1])
        else:
            return None
    except Exception:
        return None
    if rw <= 0 or rh <= 0:
        return None
    return rw, rh


def _dims_from_ratio_long_side(
    ratio: tuple[int, int],
    long_side: int,
    factor: int = 32,
) -> tuple[int, int]:
    rw, rh = ratio
    if rw >= rh:
        width = max(factor, _round_by_factor(long_side, factor))
        height = max(factor, _round_by_factor(long_side * rh / rw, factor))
    else:
        height = max(factor, _round_by_factor(long_side, factor))
        width = max(factor, _round_by_factor(long_side * rw / rh, factor))
    return int(width), int(height)


def _resolve_image_size(item: dict[str, Any], *, default_aspect_ratio: str, ratio_long_side: int) -> tuple[int, int]:
    ratio = _parse_ratio(item.get("aspect_ratio")) or _parse_ratio(default_aspect_ratio)
    if ratio is None:
        raise ValueError(
            f"Failed to resolve aspect ratio for prompt_id={item.get('_prompt_id')}: "
            f"aspect_ratio={item.get('aspect_ratio')!r}, default_aspect_ratio={default_aspect_ratio!r}"
        )
    return _dims_from_ratio_long_side(ratio, ratio_long_side)


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(data_path=data_path)
    print(f"[bizgeneval] loaded {len(items)} items from {data_path}")

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[bizgeneval] attn_backend={args.attn_backend!r} effective={sensenova_u1.effective_attn_backend()!r}")

    engine = SenseNovaU1T2I(
        model_path=args.model_path,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
    )

    generated = 0
    skipped = 0

    for item in items:
        prompt_id = int(item["_prompt_id"])
        width, height = _resolve_image_size(
            item,
            default_aspect_ratio=DEFAULT_ASPECT_RATIO,
            ratio_long_side=RATIO_LONG_SIDE,
        )
        out_path = output_dir / f"{prompt_id:04d}.png"
        if out_path.exists():
            skipped += 1
            continue

        images = engine.generate(
            item["prompt"],
            image_size=(width, height),
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            timestep_shift=args.timestep_shift,
            num_steps=args.num_steps,
            batch_size=1,
            seed=args.seed,
        )
        images[0].save(out_path)
        generated += 1
        print(
            f"[saved] prompt_id={prompt_id} "
            f"size={width}x{height} domain={item.get('domain')} "
            f"dimension={item.get('dimension')} -> {out_path}"
        )

    print(f"[bizgeneval] done: items={len(items)} generated={generated} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
