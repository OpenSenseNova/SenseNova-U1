from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sensenova_u1
from sensenova_u1 import check_checkpoint_compatibility
from sensenova_u1.models.neo_unify.utils import smart_resize
from sensenova_u1.utils import DEFAULT_IMAGE_PATCH_SIZE, InferenceProfiler

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42

# Output H and W must be divisible by this factor so the model's image-token
# grid has integer dimensions. It is the architectural ``patch_size *
# merge_size`` constant of the image-generation head and is NOT tunable; we
# pull the same value the profiler uses to count tokens.
_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE

# Pixel-budget defaults for the fallback "derive output resolution from the
# input image" path (only used when --width / --height are not given).
# They mirror the values the editing model was trained / evaluated at – see
# ``it2i_generate`` -> ``load_image_native`` in
# ``sensenova_u1/models/neo_unify/modeling_neo_chat.py``.
DEFAULT_MIN_PIXELS = 512 * 512
DEFAULT_MAX_PIXELS = 2048 * 2048


def _set_seed(seed: int) -> None:
    """Make sampling reproducible across python / numpy / torch (+ all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Invert the (img - mean) / std normalization back to [0, 1]."""
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    """Convert a [B, 3, H, W] float tensor in normalized space to a list of PIL images."""
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _load_input_image(path: str | Path) -> Image.Image:
    """Load a PIL RGB image, compositing RGBA onto an opaque white canvas."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _coerce_image_paths(value: object) -> list[str]:
    """Normalize the JSONL ``image`` field to a list of string paths."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _check_grid_divisible(width: int, height: int) -> None:
    """Fail fast if (W, H) is not a multiple of the image-token grid factor.

    The editing model's generation head slices the output canvas into
    ``(H / _IMAGE_GRID_FACTOR, W / _IMAGE_GRID_FACTOR)`` tokens; non-integer
    grids silently corrupt the sampling loop.
    """
    if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
        raise SystemExit(
            f"[editing] output resolution ({width}x{height}) must be a multiple "
            f"of {_IMAGE_GRID_FACTOR} on both axes (image-token grid factor)."
        )


def _resolve_output_size(
    input_images: Sequence[Image.Image],
    *,
    explicit: tuple[int, int] | None,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Pick the output (W, H) for this edit.

    The editing model generates at whatever resolution the caller asks for
    in ``image_size`` (it is decoupled from the input image – the VLM tower
    reshapes the input independently via ``load_image_native``). Two
    caller-side strategies are supported:

    * ``explicit`` is ``(W, H)``: honor it verbatim (after a %factor check).
    * ``explicit`` is ``None``: match the first input image's aspect ratio
      and pixel budget via ``smart_resize``. This is the natural default
      for "edit in place" flows and keeps the generation grid inside the
      range the model saw during training.
    """
    if explicit is not None:
        width, height = explicit
        _check_grid_divisible(width, height)
        return width, height

    w, h = input_images[0].size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resized_w, resized_h


def _explicit_size_from_sample(sample: dict) -> tuple[int, int] | None:
    """Per-sample output-size override for JSONL mode.

    If the line supplies both ``width`` and ``height`` use them (mirrors
    the t2i script's ``_resolve_size``). Otherwise ``None`` = fall back to
    the CLI / smart_resize default.
    """
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return None


class SenseNovaU1Editing:
    """Thin wrapper around ``AutoModel.from_pretrained`` for image editing.

    Because ``sensenova_u1`` has already registered the config / model with
    transformers at import time, no ``trust_remote_code=True`` is needed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        config = AutoConfig.from_pretrained(model_path)
        check_checkpoint_compatibility(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, config=config, torch_dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        image_size: tuple[int, int],
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
    ) -> list[Image.Image]:
        # The model auto-prepends ``<image>\n`` placeholders when the prompt
        # has fewer ``<image>`` tags than there are input images, so no
        # client-side prompt massaging is needed here.
        output = self.model.it2i_generate(
            self.tokenizer,
            prompt,
            list(images),
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            batch_size=batch_size,
        )
        return _to_pil(output)


def _save_images(
    images: Sequence[Image.Image],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(out_path)
        print(f"[saved] {out_path}")
        return
    for i, img in enumerate(images):
        p = out_path.with_name(f"{out_path.stem}_{i}{out_path.suffix}")
        img.save(p)
        print(f"[saved] {p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image editing (it2i) inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. OpenSenseNova/SenseNova-U1-Mini) or a local path.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prompt",
        help="Edit instruction. When the prompt does not include an ``<image>`` "
        "placeholder, the model prepends one per input image automatically. "
        "Requires --image.",
    )
    src.add_argument(
        "--jsonl",
        help='JSONL file, one sample per line. Required: {"prompt": str, '
        '"image": str | list[str]}. Optional: {"width": int, "height": int, '
        '"seed": int, "type": str}. When "width" and "height" are both '
        "present they override --width / --height for that sample.",
    )

    p.add_argument(
        "--image",
        nargs="+",
        metavar="PATH",
        help="One or more input image paths (only used with --prompt).",
    )

    p.add_argument("--output", default="output.png", help="Output path when using --prompt.")
    p.add_argument("--output_dir", default="outputs", help="Output directory when using --jsonl.")

    p.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Explicit output width in pixels. Must be given together with --height, "
            f"and must be a multiple of {_IMAGE_GRID_FACTOR} (the image-token grid factor). "
            "When both --width and --height are omitted the output resolution is "
            "derived from the input image (aspect ratio preserved, pixels clamped to "
            "[--min_pixels, --max_pixels])."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Explicit output height in pixels. See --width. Must be a multiple of {_IMAGE_GRID_FACTOR}.",
    )
    p.add_argument(
        "--min_pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help=(
            f"Lower pixel budget for the input-derived output resolution "
            f"(default: {DEFAULT_MIN_PIXELS}). Smaller inputs are upscaled at least to "
            "this total pixel count. Ignored when --width / --height are given."
        ),
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=(
            f"Upper pixel budget for the input-derived output resolution "
            f"(default: {DEFAULT_MAX_PIXELS}). Larger inputs are downscaled so that "
            "H*W stays at or below this cap. Ignored when --width / --height are given."
        ),
    )

    p.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Text CFG weight. Higher values track the edit instruction more aggressively.",
    )
    p.add_argument(
        "--img_cfg_scale",
        type=float,
        default=1.0,
        help=(
            "Image CFG weight (default: 1.0 = image CFG disabled). Raise above 1.0 "
            "to keep the output closer to the input reference; useful when edits "
            "drift too far from the source. Typical tuning range: 1.0 - 2.0."
        ),
    )
    p.add_argument(
        "--cfg_norm",
        default="none",
        choices=["none", "global", "channel"],
        help=(
            "Classifier-free guidance rescaling mode. 'none' (default) is classical CFG; "
            "'global'/'channel' rescale the CFG output back to the conditional norm "
            "(globally / per-channel). Unlike t2i, 'cfg_zero_star' is not supported here."
        ),
    )
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducible sampling (default: {DEFAULT_SEED}). "
            "In --jsonl mode, a per-sample `seed` field in the JSONL overrides this."
        ),
    )

    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help=(
            "Attention kernel used by the Qwen3 layers. "
            "'auto' picks flash-attn when it's importable and falls back to SDPA "
            "otherwise. 'flash' hard-requires flash-attn; 'sdpa' forces torch SDPA "
            "even when flash-attn is installed (useful for A/B-ing outputs)."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print timing stats: model load time, average per-image generation "
            f"time, and the same time normalized per image token (patch size = "
            f"{DEFAULT_IMAGE_PATCH_SIZE})."
        ),
    )

    args = p.parse_args()
    if args.prompt is not None and not args.image:
        p.error("--prompt requires at least one --image.")
    if args.jsonl is not None and args.image:
        p.error("--image is only valid with --prompt; in --jsonl mode, put 'image' in the JSONL.")
    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be given together (or both omitted).")
    if args.width is not None:
        if args.width % _IMAGE_GRID_FACTOR or args.height % _IMAGE_GRID_FACTOR:
            p.error(
                f"--width / --height must each be a multiple of {_IMAGE_GRID_FACTOR} (got {args.width}x{args.height})."
            )
    return args


def main() -> None:
    args = parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(enabled=args.profile, device=args.device)

    with profiler.time_load():
        engine = SenseNovaU1Editing(args.model_path, device=args.device, dtype=dtype)

    cfg_interval = tuple(args.cfg_interval)
    cli_explicit_size: tuple[int, int] | None = (args.width, args.height) if args.width is not None else None

    if args.prompt is not None:
        images = [_load_input_image(p) for p in args.image]
        w, h = _resolve_output_size(
            images,
            explicit=cli_explicit_size,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        _set_seed(args.seed)
        with profiler.time_generate(w, h, args.batch_size):
            outputs = engine.edit(
                args.prompt,
                images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
            )
        _save_images(outputs, Path(args.output))
        profiler.report()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_kw):  # type: ignore[no-redef]
            return x

    for i, sample in enumerate(tqdm(samples, desc="Editing")):
        paths = _coerce_image_paths(sample["image"])
        images = [_load_input_image(p) for p in paths]
        w, h = _resolve_output_size(
            images,
            explicit=_explicit_size_from_sample(sample) or cli_explicit_size,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        _set_seed(int(sample.get("seed", args.seed)))
        with profiler.time_generate(w, h, 1):
            outputs = engine.edit(
                sample["prompt"],
                images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                batch_size=1,
            )
        tag = sample.get("type")
        stem = f"{i + 1:04d}" + (f"_{tag}" if tag else "") + f"_{w}x{h}.png"
        outputs[0].save(out_dir / stem)

    profiler.report()


if __name__ == "__main__":
    main()
