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

SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_ATTN_BACKEND = "auto"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IGenBench images with SenseNova-U1.")
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or HF model id.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images.")
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR), help="IGenBench directory with per-item JSON files."
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


def _load_items(*, data_dir: Path) -> list[dict[str, Any]]:
    items = []
    for path in sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem)):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_data_path"] = str(path)
        items.append(item)
    return items


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _resolve_image_size() -> tuple[int, int]:
    return SUPPORTED_RESOLUTIONS[DEFAULT_ASPECT_RATIO]


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = _resolve_image_size()
    items = _load_items(data_dir=data_dir)
    print(f"[igenbench] loaded {len(items)} items from {data_dir}")

    import sensenova_u1
    from examples.t2i.inference import SenseNovaU1T2I

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[igenbench] attn_backend={args.attn_backend!r} effective={sensenova_u1.effective_attn_backend()!r}")

    engine = SenseNovaU1T2I(
        model_path=args.model_path,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
    )

    generated = 0
    skipped = 0

    for item in items:
        prompt_id = int(item["id"])
        prompt = item["t2i_prompt"]
        out_path = output_dir / f"{prompt_id:04d}.png"
        if out_path.exists():
            skipped += 1
            continue

        images = engine.generate(
            prompt,
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
        print(f"[saved] prompt_id={prompt_id} size={width}x{height} chart_type={item.get('chart_type')} -> {out_path}")

    print(f"[igenbench] done: items={len(items)} generated={generated} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
