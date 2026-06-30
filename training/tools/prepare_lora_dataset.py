#!/usr/bin/env python3
# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
"""Convert a folder of images into a SenseNova-U1 LoRA training dataset.

Usage
-----
Run from the ``training/`` directory. Minimum invocation (uses sidecar
``.txt`` captions if present, otherwise just the trigger word):

    python tools/prepare_lora_dataset.py \\
        --image_dir /path/to/style_images \\
        --output_dir data/pixar_lora \\
        --trigger_word "in pixar style" \\
        --dataset_name pixar_style

With automatic captioning via BLIP (recommended for ~50–500 images — needs a
GPU and ``pip install transformers torch pillow``):

    python tools/prepare_lora_dataset.py \\
        --image_dir /path/to/style_images \\
        --output_dir data/pixar_lora \\
        --trigger_word "in pixar style" \\
        --auto_caption blip \\
        --repeat_time 20

Outputs
-------
``<output_dir>/`` will contain:

* ``images/``       — symlinked or copied source images.
* ``annotations.jsonl`` — one JSON object per line, matching the
  ``{"id", "image", "conversations"}`` schema consumed by
  ``sensenovavl/data/dataset.py``.
* ``<dataset_name>_meta.json`` — the top-level meta file that the
  ``mm_data_path`` env var should point at.

Notes
-----
* Captions used at training time: ``{trigger_word}. {auto_or_sidecar_caption}``.
* All samples are tagged ``task=t2i`` so they're routed through the text→image
  flow-matching path (which is the only path LoRA touches by default).
* The script is intentionally dependency-light: only ``Pillow`` is required at
  minimum. ``transformers`` is imported lazily and only when ``--auto_caption``
  is set.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_lora_dataset")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
DEFAULT_CAPTION = "A high-quality illustration."


# --------------------------------------------------------------------------- #
# Captioners
# --------------------------------------------------------------------------- #
class _BlipCaptioner:
    """BLIP image captioning. Dependencies imported lazily."""

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large", device: Optional[str] = None):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BLIP captioner '{model_id}' on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    @staticmethod
    def _open(path: Path):
        from PIL import Image
        return Image.open(path).convert("RGB")

    def __call__(self, image_path: Path) -> str:
        img = self._open(image_path)
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=40)
        caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return caption or DEFAULT_CAPTION


def _build_captioner(mode: str) -> Optional[_BlipCaptioner]:
    return _BlipCaptioner() if mode == "blip" else None


# --------------------------------------------------------------------------- #
# Conversion logic
# --------------------------------------------------------------------------- #
def _iter_images(image_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p


def _sidecar_caption(image_path: Path) -> Optional[str]:
    """Look for ``image.png`` -> ``image.txt`` next to the image (Kohya/Civitai
    convention)."""
    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        try:
            text = sidecar.read_text(encoding="utf-8", errors="replace").strip()
            return text or None
        except OSError as e:
            logger.warning(f"Could not read sidecar {sidecar}: {e}")
    return None


def _copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            # Fall through to copy on platforms / filesystems that disallow symlinks.
            pass
    shutil.copy2(src, dst)


def _format_prompt(trigger: str, caption: str) -> str:
    trigger = trigger.strip().rstrip(".")
    caption = caption.strip()
    if not caption:
        return f"{trigger}."
    # Light de-duplication: don't put the trigger twice if BLIP already says it.
    if trigger.lower() in caption.lower():
        return caption
    return f"{trigger}. {caption}"


def build_dataset(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"--image_dir does not exist or is not a directory: {image_dir}")

    images_out = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    captioner = _build_captioner(args.auto_caption)

    records: List[dict] = []
    seen = 0
    skipped = 0
    for src in _iter_images(image_dir):
        seen += 1
        rel = src.relative_to(image_dir)
        dst = images_out / rel
        try:
            _copy_or_link(src, dst, symlink=args.symlink)
        except Exception as e:
            logger.warning(f"Skipping {src}: {e}")
            skipped += 1
            continue

        caption = _sidecar_caption(src)
        if caption is None and captioner is not None:
            try:
                caption = captioner(src)
            except Exception as e:
                logger.warning(f"Auto-caption failed for {src}: {e}")
        prompt = _format_prompt(args.trigger_word, caption or "")

        records.append(
            {
                "id": str(rel).replace(os.sep, "/"),
                "image": str(rel).replace(os.sep, "/"),
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "<image>"},
                ],
            }
        )

    if not records:
        raise SystemExit(f"No usable images found under {image_dir}.")

    # Write the annotations jsonl
    annotation_path = output_dir / "annotations.jsonl"
    with annotation_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} samples to {annotation_path}")

    # Write the dataset meta json — this is what mm_data_path points at.
    meta = {
        args.dataset_name: {
            "root": str(images_out),
            "annotation": str(annotation_path),
            "length": len(records),
            "repeat_time": args.repeat_time,
            "task": "t2i",
        }
    }
    meta_path = output_dir / f"{args.dataset_name}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Wrote meta JSON to {meta_path}. "
        f"Set this in your launcher: export mm_data_path={meta_path}"
    )
    if skipped:
        logger.info(f"Skipped {skipped} of {seen} discovered images.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_dir", required=True, help="Folder of style images (.png/.jpg/.jpeg/.webp/.bmp/.tiff).")
    p.add_argument("--output_dir", required=True, help="Where to write images/, annotations.jsonl and *_meta.json.")
    p.add_argument("--trigger_word", default="in custom style",
                   help='Style-trigger phrase prepended to every caption (e.g. "in pixar style"). '
                        'Used at inference time as the unique style cue.')
    p.add_argument("--dataset_name", default="lora_style", help="Logical dataset name (key in the meta JSON).")
    p.add_argument("--repeat_time", type=int, default=20,
                   help="How many times the loader replays the dataset per epoch. "
                        "Small style sets need a large value — 20 is a sane default for 100-image sets.")
    p.add_argument("--auto_caption", default="none", choices=("none", "blip"),
                   help="Auto-captioner. 'none' uses only the trigger word + sidecar .txt if present. "
                        "'blip' loads Salesforce/blip-image-captioning-large.")
    p.add_argument("--no_symlink", dest="symlink", action="store_false", default=True,
                   help="Copy images into output_dir/images/ instead of symlinking them (the default).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        build_dataset(args)
    except Exception as e:  # noqa: BLE001 — top-level CLI catch-all
        logger.error(str(e))
        sys.exit(1)
