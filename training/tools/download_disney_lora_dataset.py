"""Download a tiny Disney-style dataset and convert it for LoRA training.

Fetches `nastyafairypro/disney_princess_stickers` from the Hugging Face Hub —
49 sticker-style images with detailed English captions, MIT licensed, ~13 MB —
which is small enough to verify the whole LoRA pipeline end to end in minutes.

Usage (from the ``training/`` directory):

    python tools/download_disney_lora_dataset.py

Optionally pick a different output root or HF endpoint (e.g. a mirror):

    HF_ENDPOINT=https://hf-mirror.com \\
    python tools/download_disney_lora_dataset.py --output_dir data/disney_lora

The script writes ``images/`` + sidecar ``.txt`` captions, then reuses
``tools/prepare_lora_dataset.py`` to emit ``annotations.jsonl`` and the meta
JSON. When it finishes it prints the ``mm_data_path`` export line to use with
``shell/train_u1/8B_lora.sh``.

Only ``pyarrow`` and ``Pillow`` are required (both ship with ``datasets``).
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

DATASET_ID = "nastyafairypro/disney_princess_stickers"
PARQUET_PATH = "data/train-00000-of-00001.parquet"
TRIGGER_WORD = "disney_princess"


def _download_parquet(dst: Path) -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{DATASET_ID}/resolve/main/{PARQUET_PATH}"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as r, dst.open("wb") as f:
        f.write(r.read())
    print(f"Saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def _extract(parquet: Path, raw_dir: Path) -> int:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet)
    images = table.column("image").to_pylist()
    prompts = table.column("prompt").to_pylist()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        pil = Image.open(io.BytesIO(img["bytes"]))
        name = f"disney_princess_{i:03d}.png"
        pil.save(raw_dir / name)
        caption = " ".join((prompt or "").split())
        (raw_dir / f"disney_princess_{i:03d}.txt").write_text(caption, encoding="utf-8")
    print(f"Extracted {len(images)} images + captions to {raw_dir}")
    return len(images)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="data/disney_princess_lora", help="Output dataset directory.")
    p.add_argument("--repeat_time", type=int, default=40, help="Dataset replay factor (49 imgs -> ~2k samples/epoch at 40).")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet = output_dir / "disney_princess_stickers.parquet"
    if not parquet.exists():
        _download_parquet(parquet)
    _extract(parquet, raw_dir)

    prepare = Path(__file__).resolve().parent / "prepare_lora_dataset.py"
    cmd = [
        sys.executable, str(prepare),
        "--image_dir", str(raw_dir),
        "--output_dir", str(output_dir),
        "--trigger_word", TRIGGER_WORD,
        "--dataset_name", "disney_princess",
        "--repeat_time", str(args.repeat_time),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    meta = output_dir / "disney_princess_meta.json"
    print("\nDone. To launch LoRA training:")
    print(f"  export mm_data_path={meta.resolve()}")
    print("  bash shell/train_u1/8B_lora.sh")


if __name__ == "__main__":
    main()
