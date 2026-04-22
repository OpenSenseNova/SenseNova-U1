from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sensenova_u1
from sensenova_u1 import check_checkpoint_compatibility
from sensenova_u1.utils import DEFAULT_IMAGE_PATCH_SIZE, InferenceProfiler

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42

SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}

DEFAULT_RESOLUTION = "16:9"
DEFAULT_WIDTH, DEFAULT_HEIGHT = SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION]


def _warn_if_unsupported(width: int, height: int) -> None:
    if (width, height) in SUPPORTED_RESOLUTIONS.values():
        return
    buckets = ", ".join(f"{r}->{w}x{h}" for r, (w, h) in SUPPORTED_RESOLUTIONS.items())
    print(
        f"[warn] ({width}x{height}) is outside the trained resolution set; "
        f"quality may degrade. Supported buckets: {buckets}"
    )


# Interleave inference requires a system prompt that describes the
# think / no-think protocol expected by the model during training.
DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""


def _set_seed(seed: int) -> None:
    """Make sampling reproducible across python / numpy / torch (+ all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _round_by(n: int, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by(n: int, factor: int) -> int:
    return math.ceil(n / factor) * factor


def _floor_by(n: int, factor: int) -> int:
    return math.floor(n / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 512 * 512,
    max_pixels: int = (4 * 2048 * 2048) // 8,
) -> tuple[int, int]:
    """Return ``(h, w)`` that are divisible by ``factor``, keep aspect ratio,
    and fall inside ``[min_pixels, max_pixels]``.

    Adapted from the Qwen2.5-VL utility used by the training pipeline so
    generated-image sizes stay in the buckets the model saw during SFT.
    """
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, _round_by(height, factor))
    w_bar = max(factor, _round_by(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by(height / beta, factor))
        w_bar = max(factor, _floor_by(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return h_bar, w_bar


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> Image.Image:
    """Convert a single [1, 3, H, W] normalized tensor to a PIL image."""
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr[0])


class SenseNovaU1Interleave:
    """Thin wrapper around ``AutoModel.from_pretrained`` for interleaved text+image generation.

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
    def generate(
        self,
        prompt: str,
        input_images: Sequence[Image.Image] = (),
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        think_mode: bool = True,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        seed: int = 0,
    ) -> tuple[str, list[Image.Image]]:
        text, image_tensors = self.model.interleave_gen(
            self.tokenizer,
            prompt,
            images=list(input_images),
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            system_message=system_message,
            think_mode=think_mode,
            seed=seed,
        )
        return text, [_to_pil(img) for img in image_tensors]


def _load_input_images(paths: Sequence[str], image_root: str = "") -> list[Image.Image]:
    """Load images from ``paths``. When ``image_root`` is set, it is prepended
    to any non-absolute path; absolute paths are used as-is."""
    images: list[Image.Image] = []
    for p in paths:
        resolved = p if not image_root or Path(p).is_absolute() else str(Path(image_root) / p)
        if not Path(resolved).exists():
            raise FileNotFoundError(f"input image not found: {resolved}")
        images.append(Image.open(resolved).convert("RGB"))
    return images


def _resolve_image_size(
    input_images: Sequence[Image.Image],
    fallback_w: int,
    fallback_h: int,
) -> tuple[int, int]:
    """Pick generation (W, H). With input images, follow the first one so
    edits stay aligned (snapped to 32-aligned buckets via ``smart_resize``).
    Without input images, use the caller-provided fallback as-is — it is
    expected to already be one of ``SUPPORTED_RESOLUTIONS``."""
    if input_images:
        w, h = input_images[0].size
        resized_h, resized_w = smart_resize(h, w)
        return resized_w, resized_h
    return fallback_w, fallback_h


def _save_outputs(
    text: str,
    images: Sequence[Image.Image],
    out_dir: Path,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    print(f"[saved] {text_path}")
    for i, img in enumerate(images):
        img_path = out_dir / f"{stem}_image_{i}.png"
        img.save(img_path)
        print(f"[saved] {img_path}")


def _sample_images(sample: dict, image_root: str = "") -> list[Image.Image]:
    """Load ``sample['image']`` (or ``sample['images']``). Relative paths are
    resolved against ``image_root`` when provided. Missing key is treated as
    no input images."""
    paths = sample.get("image") or sample.get("images") or []
    return _load_input_images(paths, image_root=image_root)


def _extract_prompt(sample: dict) -> str:
    """Accept both flat ``{"prompt": ...}`` and ShareGPT-style
    ``{"conversations": [{"from": "human", "value": ...}, ...]}``."""
    if "prompt" in sample:
        return sample["prompt"]
    for conv in sample.get("conversations", []):
        if conv.get("from") == "human":
            return conv["value"]
    raise ValueError("sample has no 'prompt' and no human turn in 'conversations'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interleaved text+image inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. OpenSenseNova/SenseNova-U1-Mini) or a local path.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Generate from a single text prompt.")
    src.add_argument(
        "--jsonl",
        help=(
            'JSONL file, one sample per line. Required: {"prompt": ...} or '
            '{"conversations": [{"from": "human", "value": ...}, ...]}. '
            'Optional: {"image": [paths], "width": W, "height": H, "seed": S, '
            '"think_mode": bool}. '
            "If 'image' is set, output size follows the first input image "
            "(via smart_resize); 'width'/'height' are used only for "
            "text-only samples."
        ),
    )
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help=(
            "Path to an input image (repeatable). Only valid with --prompt. "
            "The prompt should contain a matching '<image>' placeholder per image."
        ),
    )
    p.add_argument(
        "--image_root",
        default="",
        help=(
            "Directory prepended to relative image paths in --jsonl samples. "
            "Absolute paths (in the jsonl or --image) are used as-is."
        ),
    )

    p.add_argument("--output_dir", default="outputs", help="Directory for generated text + images.")
    p.add_argument(
        "--stem",
        default="sample",
        help="Filename stem when using --prompt. Generated files are <stem>.txt and <stem>_image_<i>.png.",
    )

    p.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        choices=list(SUPPORTED_RESOLUTIONS.keys()),
        help=(
            f"Aspect-ratio bucket used when no input image is provided "
            f"(default: {DEFAULT_RESOLUTION} -> "
            f"{SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION][0]}x"
            f"{SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION][1]}). "
            "Overridden by --width/--height when both are set."
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Explicit fallback width. Overrides --resolution when both --width and --height are set.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Explicit fallback height. Overrides --resolution when both --width and --height are set.",
    )
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--img_cfg_scale", type=float, default=1.0)
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument(
        "--think_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable <think></think> reasoning before the final answer. On by default; pass --no-think_mode to disable.",
    )
    p.add_argument(
        "--system_message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="Override the default interleave system prompt.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducible sampling (default: {DEFAULT_SEED}). "
            "In --jsonl mode, a per-sample `seed` field overrides this."
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
            "Attention kernel used by the Qwen3 layers. 'auto' picks flash-attn when importable and falls back to SDPA."
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(enabled=args.profile, device=args.device)
    with profiler.time_load():
        engine = SenseNovaU1Interleave(args.model_path, device=args.device, dtype=dtype)

    cfg_interval = tuple(args.cfg_interval)
    out_dir = Path(args.output_dir)

    if args.width is not None and args.height is not None:
        fallback_w, fallback_h = args.width, args.height
        _warn_if_unsupported(fallback_w, fallback_h)
    else:
        fallback_w, fallback_h = SUPPORTED_RESOLUTIONS[args.resolution]

    # Single-sample inference: --prompt + optional --image (repeatable).
    if args.prompt is not None:
        print("prompt:", args.prompt)
        input_images = _load_input_images(args.image)
        w, h = _resolve_image_size(input_images, fallback_w, fallback_h)
        # _set_seed(args.seed)
        with profiler.time_generate(w, h, 1):
            text, images = engine.generate(
                args.prompt,
                input_images=input_images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                think_mode=args.think_mode,
                system_message=args.system_message,
                seed=args.seed,
            )
        print(f"[text] {text}")
        _save_outputs(text, images, out_dir, args.stem)
        profiler.report()
        return

    # Batch inference: one sample per line in --jsonl.
    with open(args.jsonl) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_kw):  # type: ignore[no-redef]
            return x

    results_path = out_dir / "results.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as rf:
        for i, sample in enumerate(tqdm(samples, desc="interleave")):
            prompt = _extract_prompt(sample)
            input_images = _sample_images(sample, image_root=args.image_root)
            if input_images:
                # When the sample ships input images, always follow their
                # size (via smart_resize); any per-sample width/height is
                # treated as a no-input-image fallback only.
                w, h = _resolve_image_size(input_images, fallback_w, fallback_h)
            elif "width" in sample and "height" in sample:
                w, h = int(sample["width"]), int(sample["height"])
                _warn_if_unsupported(w, h)
            else:
                w, h = fallback_w, fallback_h
            think_mode = bool(sample.get("think_mode", args.think_mode))
            # _set_seed(int(sample.get("seed", args.seed)))

            with profiler.time_generate(w, h, 1):
                text, images = engine.generate(
                    prompt,
                    input_images=input_images,
                    image_size=(w, h),
                    cfg_scale=args.cfg_scale,
                    img_cfg_scale=args.img_cfg_scale,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    think_mode=think_mode,
                    system_message=args.system_message,
                    seed=args.seed,
                )

            stem = f"{i + 1:04d}" + ("_think" if think_mode else "_no_think")
            _save_outputs(text, images, out_dir, stem)
            rf.write(
                json.dumps(
                    {
                        "index": i,
                        "prompt": prompt,
                        "text": text,
                        "images": [f"{stem}_image_{j}.png" for j in range(len(images))],
                        "width": w,
                        "height": h,
                        "think_mode": think_mode,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rf.flush()

    print(f"[saved] {results_path}")
    profiler.report()


if __name__ == "__main__":
    main()
