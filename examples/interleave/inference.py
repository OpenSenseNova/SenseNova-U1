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

import sensenova_u1
from sensenova_u1.utils import (
    DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
    DEFAULT_FAST_VRAM_FRACTION,
    DEFAULT_FAST_VRAM_HEADROOM_GIB,
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    InferenceProfiler,
    add_offload_args,
    best_available_device,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer_for_format,
    make_offload_ctx,
    seed_all_accelerators,
    vram_mode_keeps_generation_resident,
    vram_mode_to_prefetch_count,
)
from sensenova_u1.utils.resolutions import (
    add_resolution_args,
    calculate_image_size,
    print_output_size,
    resolve_cli_image_size,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42

# Interleave inference requires a system prompt that describes the
# think / no-think protocol expected by the model during training.
DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""


def _set_seed(seed: int) -> None:
    """Make sampling reproducible across python / numpy / torch (+ every available accelerator backend)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_all_accelerators(seed)


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
        model_format: str = "auto",
        gguf_checkpoint: str | None = None,
        device_map: str | None = None,
        max_memory: str | None = None,
        vram_mode: str = DEFAULT_VRAM_MODE,
        fast_vram_fraction: float = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | None = None,
    ) -> None:
        self.device = device
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.fast_vram_fraction = fast_vram_fraction
        self.fast_vram_headroom_gib = fast_vram_headroom_gib
        self.fast_activation_reserve_gib = fast_activation_reserve_gib
        self.fast_vram_budget_gib = fast_vram_budget_gib
        self.model, self.tokenizer = load_model_and_tokenizer_for_format(
            model_path,
            model_format=model_format,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        input_images: Sequence[Image.Image] = (),
        image_size: tuple[int, int] = calculate_image_size(),
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        max_images: int = 10,
        think_mode: bool = True,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        seed: int = 0,
    ) -> tuple[str, list[Image.Image]]:
        with make_offload_ctx(
            self.model,
            self.prefetch_count,
            self.device,
            keep_generation_resident=vram_mode_keeps_generation_resident(self.vram_mode),
            fast_vram_fraction=self.fast_vram_fraction,
            fast_vram_headroom_gib=self.fast_vram_headroom_gib,
            fast_activation_reserve_gib=self.fast_activation_reserve_gib,
            fast_vram_budget_gib=self.fast_vram_budget_gib,
        ) as offloaded:
            text, image_tensors = offloaded.interleave_gen(
                self.tokenizer,
                prompt,
                images=list(input_images),
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                max_images=max_images,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
                verbose=True,
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
    fallback_size: tuple[int, int],
) -> tuple[int, int]:
    """Pick generation (W, H). With input images, follow the first one so
    edits stay aligned (snapped to 32-aligned buckets via ``smart_resize``).
    Without input images, use the caller-provided fallback as-is."""
    if input_images:
        w, h = input_images[0].size
        resized_h, resized_w = smart_resize(h, w)
        return resized_w, resized_h
    return fallback_size


def _save_outputs(
    text: str,
    images: Sequence[Image.Image],
    out_dir: Path,
    stem: str,
    input_images: Sequence[Image.Image] = (),
    prompt: str = "",
) -> list[str]:
    """Persist the prompt + model output + generated images and (optionally)
    the user-supplied input images so a result can be reproduced from disk
    alone. Returns the relative filenames of saved input images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"{stem}.txt"
    if prompt:
        text_path.write_text(
            f"# PROMPT\n{prompt}\n\n# OUTPUT\n{text}\n",
            encoding="utf-8",
        )
    else:
        text_path.write_text(text, encoding="utf-8")
    print(f"[saved] {text_path}")
    input_names: list[str] = []
    for i, img in enumerate(input_images):
        name = f"{stem}_input_{i}.png"
        img_path = out_dir / name
        img.save(img_path)
        input_names.append(name)
        print(f"[saved] {img_path}")
    for i, img in enumerate(images):
        img_path = out_dir / f"{stem}_image_{i}.png"
        img.save(img_path)
        print(f"[saved] {img_path}")
    return input_names


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
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )
    p.add_argument(
        "--lora_path",
        required=False,
        default=None,
        help="HuggingFace Hub id or a local path to a lora model.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Generate from a single text prompt.")
    src.add_argument(
        "--jsonl",
        help=(
            'JSONL file, one sample per line. Required: {"prompt": ...} or '
            '{"conversations": [{"from": "human", "value": ...}, ...]}. '
            'Optional: {"image": [paths], "width": W, "height": H, '
            '"think_mode": bool}. '
            "With input images and no CLI size options, use the first input image to derive the size. "
            "Otherwise, a per-sample width and height override the CLI size."
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

    add_resolution_args(p)
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
    p.add_argument("--max_images", type=int, default=10, help="Maximum number of generated images (default: 10).")
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
        help=f"Random seed for each sample in this run (default: {DEFAULT_SEED}).",
    )

    p.add_argument(
        "--device",
        default=str(best_available_device()),
        help="Compute device, e.g. 'cuda', 'cuda:0', 'xpu', 'xpu:0', 'cpu'. Defaults to the best available accelerator.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--model_format",
        default="auto",
        choices=["auto", "bf16", "fp8", "int8-convrot", "gguf"],
        help=(
            "Native checkpoint format. 'auto' detects SenseNova native INT8/FP8 metadata "
            "and leaves ordinary HF/Safetensors checkpoints on the standard loader."
        ),
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
            "Attention kernel used by the Qwen3 layers. 'auto' picks flash-attn when importable and falls back to SDPA."
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
    args = p.parse_args()
    if args.max_images < 1:
        p.error("--max_images must be at least 1.")
    args.image_size = resolve_cli_image_size(p, args)
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
            "fast_vram_fraction": args.fast_vram_fraction,
            "fast_vram_headroom_gib": args.fast_vram_headroom_gib,
            "fast_activation_reserve_gib": args.fast_activation_reserve_gib,
            "fast_vram_budget_gib": args.fast_vram_budget_gib,
            "attn_backend": sensenova_u1.effective_attn_backend(),
            "dtype": args.dtype,
            "model_format": args.model_format,
            "gguf": args.gguf_checkpoint,
        },
    )
    with profiler.time_load():
        engine = SenseNovaU1Interleave(
            args.model_path,
            device=args.device,
            dtype=dtype,
            model_format=args.model_format,
            gguf_checkpoint=args.gguf_checkpoint,
            device_map=args.device_map,
            max_memory=args.max_memory,
            vram_mode=args.vram_mode,
            fast_vram_fraction=args.fast_vram_fraction,
            fast_vram_headroom_gib=args.fast_vram_headroom_gib,
            fast_activation_reserve_gib=args.fast_activation_reserve_gib,
            fast_vram_budget_gib=args.fast_vram_budget_gib,
        )

    if args.lora_path is not None:
        print(f"load lora {args.lora_path}")
        engine.model = load_and_merge_lora_weight_from_safetensors(engine.model, args.lora_path)

    cfg_interval = tuple(args.cfg_interval)
    out_dir = Path(args.output_dir)

    fallback_size = args.image_size or calculate_image_size()

    # Single-sample inference: --prompt + optional --image (repeatable).
    if args.prompt is not None:
        print("prompt:", args.prompt)
        input_images = _load_input_images(args.image)
        w, h = args.image_size or _resolve_image_size(input_images, fallback_size)
        # _set_seed(args.seed)
        print_output_size(w, h)
        with profiler.time_generate(w, h, 1) as gen:
            text, images = engine.generate(
                args.prompt,
                input_images=input_images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                max_images=args.max_images,
                think_mode=args.think_mode,
                system_message=args.system_message,
                seed=args.seed,
            )
        profiler.update_last_batch(len(images))
        print(f"[text] {text}")
        _save_outputs(
            text,
            images,
            out_dir,
            args.stem,
            input_images=input_images,
            prompt=args.prompt,
        )
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
            if input_images and args.image_size is None:
                w, h = _resolve_image_size(input_images, fallback_size)
            elif "width" in sample and "height" in sample:
                w, h = int(sample["width"]), int(sample["height"])
            else:
                w, h = fallback_size
            think_mode = bool(sample.get("think_mode", args.think_mode))
            # _set_seed(int(sample.get("seed", args.seed)))

            print_output_size(w, h, sample_number=i + 1)
            with profiler.time_generate(w, h, 1) as gen:
                text, images = engine.generate(
                    prompt,
                    input_images=input_images,
                    image_size=(w, h),
                    cfg_scale=args.cfg_scale,
                    img_cfg_scale=args.img_cfg_scale,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    max_images=args.max_images,
                    think_mode=think_mode,
                    system_message=args.system_message,
                    seed=args.seed,
                )
            profiler.update_last_batch(len(images))

            stem = f"{i + 1:04d}" + ("_think" if think_mode else "_no_think")
            input_names = _save_outputs(
                text,
                images,
                out_dir,
                stem,
                input_images=input_images,
                prompt=prompt,
            )
            rf.write(
                json.dumps(
                    {
                        "index": i,
                        "prompt": prompt,
                        "text": text,
                        "input_images": input_names,
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
