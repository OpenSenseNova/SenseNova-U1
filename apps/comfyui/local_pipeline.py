from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from .image_utils import comfy_image_to_pil, pil_to_comfy_image
except ImportError:  # pragma: no cover - supports direct imports during tests
    from image_utils import comfy_image_to_pil, pil_to_comfy_image

LOGGER = logging.getLogger(__name__)


def _vram_snapshot(label: str, *, device: str = "cuda", reset_peak: bool = False) -> None:
    """Log allocated/reserved/peak CUDA memory plus pinned-host stats with ``label``.

    Used to trace the VRAM growth that shows up under
    ``vram_mode='balanced'`` inside ComfyUI but not when the same code runs
    via ``examples/t2i/inference.py``. Cheap (~1 ms) and never raises so it
    can be sprinkled liberally; falls back to a no-op when CUDA is missing.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        dev = torch.device(device)
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev) / (1024**3)
        peak = torch.cuda.max_memory_allocated(dev) / (1024**3)
        LOGGER.info(
            "[vram] %-44s | alloc=%6.2f GiB  reserved=%6.2f GiB  peak=%6.2f GiB",
            label,
            alloc,
            reserved,
            peak,
        )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(dev)
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.debug("vram snapshot %r failed: %s", label, exc)


@contextmanager
def _progress_hook(model: Any, total_steps: int):
    """Temporarily wrap ``model.unpatchify`` so each call advances a
    ComfyUI :class:`ProgressBar`.

    ``unpatchify`` is invoked exactly once at the end of every sampling
    step in t2i / it2i / interleave generation, so it is a precise and
    non-invasive progress signal that does not require modifying the
    model code. If ``comfy.utils.ProgressBar`` is unavailable (e.g. tests
    outside ComfyUI), we still install the wrapper and emit a log line
    with the final step count so users get feedback on the terminal.
    """

    pbar = None
    try:
        from comfy.utils import ProgressBar  # type: ignore[import-not-found]

        pbar = ProgressBar(max(1, int(total_steps)))
    except Exception:  # pragma: no cover - ComfyUI runtime not present
        pbar = None

    if not hasattr(model, "unpatchify"):
        yield
        return

    original = model.unpatchify
    counter = {"n": 0}

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        counter["n"] += 1
        if pbar is not None:
            try:
                pbar.update(1)
            except Exception:
                pass
        _vram_snapshot(f"sampling step {counter['n']}/{total_steps}")
        # Log a heartbeat at every multiple of total_steps so users can see
        # multi-image interleave progress past the saturated bar.
        if total_steps and counter["n"] % total_steps == 0:
            LOGGER.info(
                "SenseNova U1 sampling: image #%d ready (%d steps).",
                counter["n"] // total_steps,
                total_steps,
            )
        return out

    model.unpatchify = wrapped
    try:
        yield
    finally:
        try:
            del model.unpatchify  # restore the class-level binding
        except AttributeError:
            try:
                model.unpatchify = original
            except Exception:
                pass
        if counter["n"]:
            LOGGER.info(
                "SenseNova U1 sampling: %d step(s) completed (target=%d).",
                counter["n"],
                total_steps,
            )


LOCAL_MODEL_TYPE = "SENSENOVA_U1_LOCAL_MODEL"
INTERLEAVE_RESULT_TYPE = "SENSENOVA_INTERLEAVE_RESULT"

DEFAULT_SEED = 42
DEFAULT_SOURCE_PATH = ""
DEFAULT_TARGET_PIXELS = 2048 * 2048
DEFAULT_IMAGE_PATCH_SIZE = 32
DEFAULT_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. "
    "Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in both reasoning "
    "and the final answer."
)

T2I_RESOLUTIONS: dict[str, tuple[int, int]] = {
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

INTERLEAVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
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

T2I_RESOLUTION_OPTIONS = tuple(f"{width}x{height}|{ratio}" for ratio, (width, height) in T2I_RESOLUTIONS.items())
INTERLEAVE_RESOLUTION_OPTIONS = tuple(
    f"{width}x{height}|{ratio}" for ratio, (width, height) in INTERLEAVE_RESOLUTIONS.items()
)
DTYPE_OPTIONS = ("bfloat16", "float16", "float32")
CFG_NORM_OPTIONS = ("none", "global", "channel", "cfg_zero_star")
ATTN_BACKEND_OPTIONS = ("auto", "flash", "sdpa")
DEVICE_MAP_OPTIONS = ("none", "auto", "balanced", "balanced_low_0", "sequential")

VRAM_MODE_OPTIONS = ("full", "low", "balanced")
DEFAULT_VRAM_MODE = "full"

# vram_mode -> prefetch_count (the underlying knob on the layer-offload wrapper)
# 0 = no offload, 1 = synchronous, >=2 = async prefetch this many layers ahead.
# Absolute VRAM is workload-dependent (KV cache grows with image/text count in
# interleave mode), so modes describe the *mechanism*, not a fixed budget.
_VRAM_MODE_TO_PREFETCH: dict[str, int] = {
    "full": 0,  # no offload, whole model on GPU
    "low": 1,  # sync per-layer offload, smallest weight footprint, slowest
    "balanced": 2,  # async prefetch, overlaps H2D with compute
}

DEFAULT_LAYERS_ATTR = "language_model.model.layers"

_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)


@dataclass
class LocalGenerationResult:
    images: Any
    text: str
    think_text: str
    metadata: dict[str, Any]
    interleave_result: dict[str, Any] | None = None


class SenseNovaU1LocalModel:
    def __init__(
        self,
        *,
        model_path: str,
        sensenova_u1_src: str = "",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_backend: str = "auto",
        device_map: str = "none",
        max_memory: str = "",
        gguf_checkpoint: str = "",
        vram_mode: str = DEFAULT_VRAM_MODE,
    ) -> None:
        if not model_path.strip():
            raise RuntimeError("Local model_path cannot be empty.")
        if vram_mode not in _VRAM_MODE_TO_PREFETCH:
            raise RuntimeError(f"Unsupported vram_mode={vram_mode!r}. Choose one of {VRAM_MODE_OPTIONS}.")
        prefetch_count = _VRAM_MODE_TO_PREFETCH[vram_mode]

        injected_path = _maybe_add_source_path(sensenova_u1_src)
        model_path = _resolve_local_model_path(model_path)
        torch = _import_torch()
        sensenova_u1, load_model_and_tokenizer, _ = _import_sensenova_u1()

        if attn_backend not in ATTN_BACKEND_OPTIONS:
            raise RuntimeError(f"Unsupported attention backend: {attn_backend}")
        sensenova_u1.set_attn_backend(attn_backend)

        torch_dtype = _resolve_dtype(torch, dtype)
        normalized_device_map = None if device_map == "none" else device_map
        normalized_gguf = gguf_checkpoint.strip() or None
        offloading = prefetch_count > 0
        if offloading and normalized_device_map:
            LOGGER.warning(
                "SenseNova U1 loader: vram_mode=%r overrides device_map=%r "
                "(layer offload is incompatible with accelerate placement).",
                vram_mode,
                normalized_device_map,
            )
            normalized_device_map = None
        if normalized_gguf and normalized_device_map:
            # diffusers' GGUF quantizer skips accelerate sharding — let the user know.
            raise RuntimeError("gguf_checkpoint cannot be combined with a device_map; pick one.")
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.attn_backend = attn_backend
        self.gguf_checkpoint = normalized_gguf or ""
        self.vram_mode = vram_mode
        self.prefetch_count = int(prefetch_count)
        self.effective_attn_backend = sensenova_u1.effective_attn_backend()
        _vram_snapshot(f"loader: pre-load (vram_mode={vram_mode})", device=device, reset_peak=True)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=torch_dtype,
            device=device,
            device_map=normalized_device_map,
            max_memory=max_memory or None,
            gguf_checkpoint=normalized_gguf,
            for_offload=offloading,
        )
        _vram_snapshot(f"loader: post-load (for_offload={offloading})", device=device)
        _maybe_remove_source_path(injected_path)

    @property
    def info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "attn_backend": self.attn_backend,
            "effective_attn_backend": self.effective_attn_backend,
            "gguf_checkpoint": self.gguf_checkpoint,
            "vram_mode": self.vram_mode,
            "prefetch_count": self.prefetch_count,
        }

    def _offload_ctx(self):
        """Return a context manager that yields the model wrapped for layer
        offload, or a no-op nullcontext yielding ``self.model`` when offload
        is disabled (``prefetch_count == 0``)."""
        if self.prefetch_count == 0:
            return contextlib.nullcontext(self.model)

        torch = _import_torch()
        from sensenova_u1.utils import offload_layers_async, offload_layers_sync

        target = torch.device(self.device)
        if self.prefetch_count == 1:
            inner = offload_layers_sync(self.model, DEFAULT_LAYERS_ATTR, target)
        else:
            inner = offload_layers_async(self.model, DEFAULT_LAYERS_ATTR, target, prefetch_count=self.prefetch_count)
        return self._instrumented_offload_ctx(inner)

    @contextmanager
    def _instrumented_offload_ctx(self, inner):
        """Wrap the offload context manager so we can snapshot VRAM at the
        boundaries that matter when diagnosing leaks across repeated runs in
        ComfyUI: just before the wrapper is built, just after, and on the way
        out (after teardown + ``model.to('cpu')`` + ``empty_cache``).
        """
        _vram_snapshot(
            f"offload_ctx: enter (prefetch_count={self.prefetch_count}, vram_mode={self.vram_mode})",
            device=self.device,
            reset_peak=True,
        )
        try:
            with inner as offloaded:
                _vram_snapshot("offload_ctx: wrapper ready", device=self.device)
                yield offloaded
                _vram_snapshot("offload_ctx: forward done (pre-teardown)", device=self.device)
        finally:
            _vram_snapshot("offload_ctx: exit (post-teardown+empty_cache)", device=self.device)

    def text_to_image(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        cfg_interval: tuple[float, float],
        num_steps: int,
        batch_size: int,
        seed: int,
        think_mode: bool,
    ) -> LocalGenerationResult:
        if not prompt.strip():
            raise RuntimeError("Text-to-image prompt cannot be empty.")

        _check_cfg_interval(cfg_interval)
        torch = _import_torch()
        with (
            torch.inference_mode(),
            self._offload_ctx() as offloaded,
            _progress_hook(self.model, num_steps),
        ):
            out = offloaded.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=(width, height),
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
        else:
            tensor = out
            think_text = ""
        return LocalGenerationResult(
            images=_batch_tensor_to_comfy_image(tensor),
            text="",
            think_text=think_text,
            metadata={
                **self.info,
                "task": "text-to-image",
                "width": width,
                "height": height,
                "seed": seed,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "think_mode": think_mode,
            },
        )

    def edit_image(
        self,
        *,
        prompt: str,
        input_image: Any,
        width: int | None,
        height: int | None,
        target_pixels: int,
        cfg_scale: float,
        img_cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        cfg_interval: tuple[float, float],
        num_steps: int,
        batch_size: int,
        seed: int,
        think_mode: bool,
    ) -> LocalGenerationResult:
        if not prompt.strip():
            raise RuntimeError("Image editing prompt cannot be empty.")
        if cfg_norm == "cfg_zero_star":
            raise RuntimeError("cfg_zero_star is only supported for local text-to-image.")

        pil_image = comfy_image_to_pil(input_image)
        # Match the Terminal pipeline by upsampling small inputs to the same
        # pixel budget before they hit the model; otherwise edits on sub-2K
        # images come out noticeably softer than `examples/editing/inference.py`.
        pil_image = _resize_input_to_budget(pil_image, target_pixels)
        out_width, out_height = _resolve_edit_size(
            pil_image,
            width=width,
            height=height,
            target_pixels=target_pixels,
        )
        _check_cfg_interval(cfg_interval)
        torch = _import_torch()

        with (
            torch.inference_mode(),
            self._offload_ctx() as offloaded,
            _progress_hook(self.model, num_steps),
        ):
            out = offloaded.it2i_generate(
                self.tokenizer,
                prompt,
                [pil_image],
                image_size=(out_width, out_height),
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
        else:
            tensor = out
            think_text = ""
        return LocalGenerationResult(
            images=_batch_tensor_to_comfy_image(tensor),
            text="",
            think_text=think_text,
            metadata={
                **self.info,
                "task": "image-editing",
                "width": out_width,
                "height": out_height,
                "seed": seed,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "target_pixels": target_pixels,
                "think_mode": think_mode,
            },
        )

    def interleave(
        self,
        *,
        prompt: str,
        input_image: Any | None,
        width: int,
        height: int,
        cfg_scale: float,
        img_cfg_scale: float,
        timestep_shift: float,
        cfg_interval: tuple[float, float],
        num_steps: int,
        seed: int,
        think_mode: bool,
        system_message: str,
    ) -> LocalGenerationResult:
        if not prompt.strip():
            raise RuntimeError("Interleave prompt cannot be empty.")

        _, _, smart_resize = _import_sensenova_u1()
        input_images: list[Image.Image] = []
        if input_image is not None:
            pil_image = comfy_image_to_pil(input_image)
            resized_height, resized_width = smart_resize(pil_image.height, pil_image.width)
            width, height = resized_width, resized_height
            input_images.append(pil_image)

        _check_cfg_interval(cfg_interval)
        torch = _import_torch()
        # Interleave can emit multiple images, each running num_steps sampling
        # steps. The bar saturates at the first image; subsequent images are
        # tracked via LOGGER (one line per completed image).
        with (
            torch.inference_mode(),
            self._offload_ctx() as offloaded,
            _progress_hook(self.model, num_steps),
        ):
            text, image_tensors = offloaded.interleave_gen(
                self.tokenizer,
                prompt,
                images=input_images,
                image_size=(width, height),
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
            )
        images = [_single_tensor_to_pil(tensor) for tensor in image_tensors]
        metadata = {
            **self.info,
            "task": "interleave",
            "width": width,
            "height": height,
            "seed": seed,
            "num_steps": num_steps,
            "think_mode": think_mode,
            "num_output_images": len(image_tensors),
        }
        interleave_result = build_interleave_result(
            text=text,
            num_images=len(images),
            metadata=metadata,
        )
        if not images:
            images = [Image.new("RGB", (1, 1), (0, 0, 0))]
        return LocalGenerationResult(
            images=_pil_images_to_comfy_batch(images),
            text=text,
            think_text=interleave_result["think_text"],
            metadata=metadata,
            interleave_result=interleave_result,
        )


def default_source_path() -> str:
    env_path = os.environ.get("SENSENOVA_U1_SRC", "")
    if env_path:
        return env_path
    repo_src = Path(__file__).resolve().parents[2] / "src"
    if repo_src.is_dir():
        return str(repo_src)
    return DEFAULT_SOURCE_PATH


def parse_resolution_option(value: str) -> tuple[int, int]:
    size = value.split("|", 1)[0].strip()
    width, height = size.split("x", 1)
    return int(width), int(height)


def output_to_tuple(result: LocalGenerationResult) -> tuple[Any, str, str, str]:
    return (
        result.images,
        result.text,
        result.think_text,
        json.dumps(result.metadata, ensure_ascii=False),
    )


def interleave_output_to_tuple(result: LocalGenerationResult) -> tuple[Any, str, str, str, dict[str, Any]]:
    interleave_result = result.interleave_result or build_interleave_result(
        text=result.text,
        num_images=0,
        metadata=result.metadata,
    )
    return (
        result.images,
        result.text,
        result.think_text,
        json.dumps(result.metadata, ensure_ascii=False),
        interleave_result,
    )


def build_interleave_result(
    *,
    text: str,
    num_images: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    parts = _parse_interleave_parts(text, num_images)
    think_text = "\n\n".join(part["text"] for part in parts if part["type"] == "think")
    return {
        "version": 1,
        "parts": parts,
        "text": text,
        "think_text": think_text,
        "num_images": num_images,
        "metadata": metadata,
    }


def interleave_result_to_markdown(result: dict[str, Any], *, include_think: bool = True) -> str:
    lines: list[str] = []
    for part in result.get("parts", []):
        part_type = part.get("type")
        if part_type == "think":
            if include_think:
                lines.extend(["<details><summary>think</summary>", "", str(part.get("text", "")), "", "</details>"])
        elif part_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                lines.append(text)
        elif part_type == "image":
            lines.append(f"[image:{int(part.get('index', 0))}]")
    return "\n\n".join(line for line in lines if line != "")


def _parse_interleave_parts(text: str, num_images: int) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    image_index = 0
    chunks = text.split("<image>")
    num_image_tags = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        _append_text_and_think_parts(parts, chunk)
        if index < num_image_tags:
            if image_index < num_images:
                parts.append({"type": "image", "index": image_index})
            else:
                parts.append({"type": "image", "index": image_index, "missing": True})
            image_index += 1
    while image_index < num_images:
        parts.append({"type": "image", "index": image_index})
        image_index += 1
    return parts


def _append_text_and_think_parts(parts: list[dict[str, Any]], chunk: str) -> None:
    remaining = chunk
    while remaining:
        start = remaining.find("<think>")
        if start < 0:
            _append_text_part(parts, remaining)
            return
        _append_text_part(parts, remaining[:start])
        after_start = start + len("<think>")
        end = remaining.find("</think>", after_start)
        if end < 0:
            think_text = remaining[after_start:]
            remaining = ""
        else:
            think_text = remaining[after_start:end]
            remaining = remaining[end + len("</think>") :]
        if think_text.strip():
            parts.append({"type": "think", "text": think_text.strip()})


def _append_text_part(parts: list[dict[str, Any]], text: str) -> None:
    if text.strip():
        parts.append({"type": "text", "text": text.strip()})


def _maybe_add_source_path(source_path: str) -> list[str]:
    """Inject source_path into sys.path for this session only; returns the
    injected path so _maybe_remove_source_path can undo it."""
    source_path = source_path.strip()
    if not source_path:
        source_path = default_source_path()
    if not source_path:
        return []

    path = Path(source_path).expanduser()
    if path.name != "src" and (path / "src").is_dir():
        path = path / "src"
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        return [path_str]
    return []


def _maybe_remove_source_path(injected: list[str]) -> None:
    """Remove paths injected by _maybe_add_source_path, keeping any the user
    may have added independently."""
    for p in injected:
        if p in sys.path:
            sys.path.remove(p)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Local SenseNova-U1 inference requires PyTorch in ComfyUI.") from exc
    return torch


def _import_sensenova_u1():
    try:
        import sensenova_u1
        from sensenova_u1.models.neo_unify.utils import smart_resize
        from sensenova_u1.utils import load_model_and_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local SenseNova-U1 inference requires the sensenova_u1 package. "
            "Install this repository into the ComfyUI Python environment, set "
            "SENSENOVA_U1_SRC, or fill the loader's sensenova_u1_src input."
        ) from exc
    return sensenova_u1, load_model_and_tokenizer, smart_resize


def _resolve_local_model_path(model_path: str) -> str:
    if Path(model_path).exists():
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        return model_path


def _resolve_dtype(torch, dtype: str):
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported dtype: {dtype}") from exc


def _denorm(x):
    torch = _import_torch()
    mean = torch.tensor(_NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _single_tensor_to_pil(tensor) -> Image.Image:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return _tensor_batch_to_pil(tensor)[0]


def _tensor_batch_to_pil(batch) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a).convert("RGB") for a in arr]


def _batch_tensor_to_comfy_image(batch):
    images = _tensor_batch_to_pil(batch)
    return _pil_images_to_comfy_batch(images)


def _pil_images_to_comfy_batch(images: list[Image.Image]):
    torch = _import_torch()
    tensors = [pil_to_comfy_image(image) for image in images]
    return torch.cat(tensors, dim=0)


def _resize_input_to_budget(image: Image.Image, target_pixels: int) -> Image.Image:
    """Match the Terminal pipeline (`examples/editing/inference.py`):
    rescale the source image so its total pixels equal ``target_pixels``,
    keeping aspect ratio, snapping H/W to the model's grid factor, and using
    LANCZOS resampling. Without this step a small input (e.g. 1024x1024)
    would be passed through to the model as-is, costing visible detail.
    """
    _, _, smart_resize = _import_sensenova_u1()
    resized_height, resized_width = smart_resize(
        height=image.height,
        width=image.width,
        factor=DEFAULT_IMAGE_PATCH_SIZE,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    if (resized_width, resized_height) == image.size:
        return image
    return image.resize((resized_width, resized_height), Image.LANCZOS)


def _resolve_edit_size(
    image: Image.Image,
    *,
    width: int | None,
    height: int | None,
    target_pixels: int,
) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise RuntimeError("width and height must be provided together.")
        _check_grid_divisible(width, height)
        return width, height

    _, _, smart_resize = _import_sensenova_u1()
    resized_height, resized_width = smart_resize(
        height=image.height,
        width=image.width,
        factor=DEFAULT_IMAGE_PATCH_SIZE,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_width, resized_height


def _check_grid_divisible(width: int, height: int) -> None:
    if width % DEFAULT_IMAGE_PATCH_SIZE or height % DEFAULT_IMAGE_PATCH_SIZE:
        raise RuntimeError(
            f"Output resolution ({width}x{height}) must be a multiple of {DEFAULT_IMAGE_PATCH_SIZE} on both axes."
        )


def _check_cfg_interval(cfg_interval: tuple[float, float]) -> None:
    lo, hi = cfg_interval
    if not 0.0 <= lo <= hi <= 1.0:
        raise RuntimeError("cfg_interval must satisfy 0.0 <= start <= end <= 1.0.")


def target_pixels_from_megapixels(megapixels: float) -> int:
    minimum = DEFAULT_IMAGE_PATCH_SIZE * DEFAULT_IMAGE_PATCH_SIZE
    return max(minimum, math.floor(megapixels * 1_000_000))
