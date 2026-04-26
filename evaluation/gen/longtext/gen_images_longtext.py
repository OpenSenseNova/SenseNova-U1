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
from examples.t2i.inference import SUPPORTED_RESOLUTIONS, SenseNovaU1T2I

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EN_DATA_PATH = DEFAULT_DATA_DIR / "text_prompts.jsonl"
DEFAULT_ZH_DATA_PATH = DEFAULT_DATA_DIR / "text_prompts_zh.jsonl"
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_ATTN_BACKEND = "auto"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LongText images with SenseNova-U1.")
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or HF model id.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images.")
    parser.add_argument("--data-path", default=None, help="LongText JSONL path.")
    parser.add_argument(
        "--lang", default="en", choices=["en", "zh"], help="Default data split when --data-path is not set."
    )
    parser.add_argument(
        "--aspect-ratio", default=DEFAULT_ASPECT_RATIO, help="Output aspect ratio key in SUPPORTED_RESOLUTIONS."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-norm", default="none", choices=["none", "global", "channel", "cfg_zero_star"])
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-backend", default=DEFAULT_ATTN_BACKEND, choices=["auto", "flash", "sdpa"])
    return parser.parse_args()


def _default_data_path(lang: str) -> Path:
    return DEFAULT_ZH_DATA_PATH if lang == "zh" else DEFAULT_EN_DATA_PATH


def _load_items(*, data_path: Path) -> list[dict[str, Any]]:
    items = []
    with data_path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_idx"] = line_idx
            if "prompt_id" not in item:
                item["prompt_id"] = line_idx
            items.append(item)
    return items


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _resolve_image_size(
    *,
    aspect_ratio: str,
    supported_resolutions: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    if aspect_ratio not in supported_resolutions:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio!r}. Supported: {sorted(supported_resolutions)}")
    return supported_resolutions[aspect_ratio]


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path).resolve() if args.data_path else _default_data_path(args.lang).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(data_path=data_path)
    print(f"[longtext] loaded {len(items)} items from {data_path}")

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[longtext] attn_backend={args.attn_backend!r} effective={sensenova_u1.effective_attn_backend()!r}")

    engine = SenseNovaU1T2I(
        model_path=args.model_path,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
    )

    generated = 0
    skipped = 0
    width, height = _resolve_image_size(
        aspect_ratio=args.aspect_ratio,
        supported_resolutions=SUPPORTED_RESOLUTIONS,
    )

    for item in items:
        prompt_id = int(item["prompt_id"])
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
            f"size={width}x{height} category={item.get('category')} "
            f"length={item.get('length')} text_length={item.get('text_length')} -> {out_path}"
        )

    print(f"[longtext] done: items={len(items)} generated={generated} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
