from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

import sensenova_u1
from sensenova_u1.models.neo_unify.utils import smart_resize
from sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    InferenceProfiler,
    add_offload_args,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer,
    make_offload_ctx,
    save_compare,
    vram_mode_to_prefetch_count,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42

# Output H / W must be divisible by this (= patch_size * merge_size).
_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE

# Aspect ratio is preserved, total output pixels are normalized to this target.
DEFAULT_TARGET_PIXELS = 2048 * 2048
DEFAULT_INPUT_MAX_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512


def _auto_input_max_pixels(num_images: int) -> int:
    full_res_image_budget = 2
    if num_images <= full_res_image_budget:
        return DEFAULT_INPUT_MAX_PIXELS
    total_budget = full_res_image_budget * DEFAULT_INPUT_MAX_PIXELS
    return max(MIN_INPUT_MAX_PIXELS, total_budget // max(1, num_images))


def _resolve_input_max_pixels(value: str | None, num_images: int) -> int | None:
    if value is None:
        return None
    if value == "auto":
        return _auto_input_max_pixels(num_images)
    try:
        input_max_pixels = int(value)
    except ValueError as exc:
        raise SystemExit("--input_max_pixels must be an integer or 'auto'.") from exc
    if input_max_pixels < MIN_INPUT_MAX_PIXELS:
        side = int(math.sqrt(MIN_INPUT_MAX_PIXELS))
        raise SystemExit(f"--input_max_pixels must be >= {MIN_INPUT_MAX_PIXELS} ({side}*{side}).")
    return input_max_pixels


def _resize_to_max_budget(img: Image.Image, input_max_pixels: int) -> Image.Image:
    resized_h, resized_w = smart_resize(
        height=img.height,
        width=img.width,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=input_max_pixels,
        max_pixels=input_max_pixels,
    )
    if (resized_w, resized_h) == img.size:
        return img
    return img.resize((resized_w, resized_h), Image.LANCZOS)


def _print_input_resize_hint(num_images: int, input_max_pixels: int | None, source: str, do_resize: bool) -> None:
    if not do_resize:
        print("[editing] resize-to-budget disabled; model preprocessing will still enforce its input limits.")
        return
    if input_max_pixels is None:
        return
    side = int(math.sqrt(input_max_pixels))
    print(
        f"[editing] {num_images} input image(s); {source} input_max_pixels={input_max_pixels} "
        f"(about {side}x{side} per image, aspect ratio preserved)."
    )


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _load_input_image(
    path: str | Path,
    *,
    do_resize: bool,
    input_max_pixels: int | None,
) -> Image.Image:
    """Load as RGB; flatten RGBA onto white so the generator sees a clean canvas."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
    img = img.convert("RGB")
    if do_resize and input_max_pixels is not None:
        img = _resize_to_max_budget(img, input_max_pixels)
    return img


def _coerce_image_paths(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise SystemExit(f"Expected a boolean value for do_resize, got {value!r}.")


def _check_grid_divisible(width: int, height: int) -> None:
    if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
        raise SystemExit(
            f"[editing] output resolution ({width}x{height}) must be a multiple "
            f"of {_IMAGE_GRID_FACTOR} on both axes (image-token grid factor)."
        )


def _resolve_output_size(
    input_images: Sequence[Image.Image],
    *,
    explicit: tuple[int, int] | None,
    target_pixels: int,
) -> tuple[int, int]:
    """Explicit (W, H) wins; else match the first input's aspect ratio and
    normalize the total pixel count to ``target_pixels``."""
    if explicit is not None:
        width, height = explicit
        _check_grid_divisible(width, height)
        return width, height

    w, h = input_images[0].size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_w, resized_h


def _explicit_size_from_sample(sample: dict) -> tuple[int, int] | None:
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return None


class SenseNovaU1Editing:
    """Thin wrapper calling ``model.it2i_generate`` on top of ``AutoModel``."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gguf_checkpoint: str | None = None,
        device_map: str | None = None,
        max_memory: str | None = None,
        vram_mode: str = DEFAULT_VRAM_MODE,
    ) -> None:
        self.device = device
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )

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
        think_mode: bool = False,
        seed: int = 0,
    ) -> tuple[list[Image.Image], str]:
        with make_offload_ctx(self.model, self.prefetch_count, self.device) as offloaded:
            output = offloaded.it2i_generate(
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
                think_mode=think_mode,
                seed=seed,
            )
        if think_mode:
            return _to_pil(output[0]), output[1]
        return _to_pil(output), ""


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
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )
    p.add_argument(
        "--lora_path",
        required=False,
        default=None,
        help="HuggingFace Hub id or a local path to a lora model.",
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
            f"and must be a multiple of {_IMAGE_GRID_FACTOR}. "
            "When both --width and --height are omitted the output resolution is "
            "derived from the first input image: aspect ratio preserved, total "
            "pixels normalized to --target_pixels."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Explicit output height in pixels. See --width. Must be a multiple of {_IMAGE_GRID_FACTOR}.",
    )
    p.add_argument(
        "--target_pixels",
        type=int,
        default=DEFAULT_TARGET_PIXELS,
        help=(
            f"Target pixel count for the auto-derived output resolution "
            f"(default: {DEFAULT_TARGET_PIXELS} = 2048*2048). The first input "
            "image's aspect ratio is preserved and H*W is rescaled to match "
            f"this target, which is a multiple of {_IMAGE_GRID_FACTOR}. "
            "Ignored when --width / --height are given."
        ),
    )
    p.add_argument(
        "--input_max_pixels",
        default="auto",
        help=(
            "Maximum pixels per input/reference image before vision encoding. "
            "Use an integer (for example 1048576 for 1024*1024) or 'auto' to "
            "keep up to two inputs at 2048*2048 and divide that total budget across more inputs. "
            "Default: auto."
        ),
    )
    p.add_argument(
        "--do_resize",
        "--do-resize",
        dest="do_resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to resize input/reference images to the input pixel budget before model preprocessing. "
            "Enabled by default. If disabled, the model's native image preprocessing still applies its "
            "own size limits."
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
        help=("Image CFG weight (default: 1.0 = image CFG disabled)."),
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
    add_offload_args(p)
    p.add_argument(
        "--gguf_checkpoint",
        default=None,
        help=(
            "Optional path to a .gguf quantized checkpoint. When set, the dequantizing "
            "diffusers GGUF Linear layer is used instead of safetensors weights. "
            "Requires the [gguf] extra (gguf>=0.10.0, diffusers>=0.30.0)."
        ),
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
            "Print timing and CUDA memory stats: model load time, average "
            "per-image generation time, peak GPU memory, and the same time "
            f"normalized per image token (patch size = {DEFAULT_IMAGE_PATCH_SIZE})."
        ),
    )
    p.add_argument(
        "--think",
        action="store_true",
        help=(
            "Enable think mode (chain-of-thought reasoning). The model will "
            "reason about the edit before generating the output image. "
            "The thinking content is printed to stdout and saved to a "
            "``<stem>_think.txt`` file next to the output image."
        ),
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Also save a side-by-side ``[inputs... | output]`` montage with the "
            "prompt rendered below, written next to the plain output as "
            "``<stem>_compare.png``. Useful for eyeballing edits without an "
            "external image viewer."
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

    profiler = InferenceProfiler(
        enabled=args.profile,
        device=args.device,
        config={
            "vram_mode": args.vram_mode,
            "attn_backend": sensenova_u1.effective_attn_backend(),
            "dtype": args.dtype,
            "gguf": args.gguf_checkpoint,
        },
    )

    with profiler.time_load():
        engine = SenseNovaU1Editing(
            args.model_path,
            device=args.device,
            dtype=dtype,
            gguf_checkpoint=args.gguf_checkpoint,
            device_map=args.device_map,
            max_memory=args.max_memory,
            vram_mode=args.vram_mode,
        )

    if args.lora_path is not None:
        print(f"load lora {args.lora_path}")
        engine.model = load_and_merge_lora_weight_from_safetensors(engine.model, args.lora_path)

    cfg_interval = tuple(args.cfg_interval)
    cli_explicit_size: tuple[int, int] | None = (args.width, args.height) if args.width is not None else None

    if args.prompt is not None:
        input_max_pixels = _resolve_input_max_pixels(args.input_max_pixels, len(args.image))
        _print_input_resize_hint(
            len(args.image), input_max_pixels, args.input_max_pixels or "model-default", args.do_resize
        )
        images = [_load_input_image(p, do_resize=args.do_resize, input_max_pixels=input_max_pixels) for p in args.image]
        w, h = _resolve_output_size(
            images,
            explicit=cli_explicit_size,
            target_pixels=args.target_pixels,
        )
        # _set_seed(args.seed)
        with profiler.time_generate(w, h, args.batch_size):
            outputs, think_text = engine.edit(
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
                think_mode=args.think,
                seed=args.seed,
            )
        out_path = Path(args.output)
        _save_images(outputs, out_path)
        if think_text:
            print(f"[think] {think_text}")
            think_path = out_path.with_name(f"{out_path.stem}_think.txt")
            think_path.write_text(think_text, encoding="utf-8")
            print(f"[saved] {think_path}")
        if args.compare:
            save_compare(out_path, images, outputs[0], args.prompt)
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
        sample_input_max_pixels = sample.get("input_max_pixels", args.input_max_pixels)
        sample_do_resize = _coerce_bool(sample.get("do_resize", args.do_resize))
        input_max_pixels = _resolve_input_max_pixels(str(sample_input_max_pixels), len(paths))
        _print_input_resize_hint(
            len(paths),
            input_max_pixels,
            str(sample_input_max_pixels or "model-default"),
            sample_do_resize,
        )
        images = [_load_input_image(p, do_resize=sample_do_resize, input_max_pixels=input_max_pixels) for p in paths]
        w, h = _resolve_output_size(
            images,
            explicit=_explicit_size_from_sample(sample) or cli_explicit_size,
            target_pixels=args.target_pixels,
        )
        # _set_seed(int(sample.get("seed", args.seed)))
        with profiler.time_generate(w, h, 1):
            outputs, think_text = engine.edit(
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
                think_mode=args.think,
                seed=args.seed,
            )
        tag = sample.get("type")
        stem = f"{i + 1:04d}" + (f"_{tag}" if tag else "") + f"_{w}x{h}.png"
        sample_out = out_dir / stem
        outputs[0].save(sample_out)
        if think_text:
            think_path = sample_out.with_suffix(".think.txt")
            think_path.write_text(think_text, encoding="utf-8")
        if args.compare:
            save_compare(sample_out, images, outputs[0], sample["prompt"])

    profiler.report()


if __name__ == "__main__":
    main()
