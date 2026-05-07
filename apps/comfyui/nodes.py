from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from comfy_api.latest import ComfyExtension, io

try:
    from .api_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        VISION_MODELS,
        SenseNovaClient,
    )
    from .image_utils import (
        comfy_batch_to_pil_images,
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
    )
    from .local_pipeline import (
        ATTN_BACKEND_OPTIONS,
        CFG_NORM_OPTIONS,
        DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
        DEFAULT_SEED,
        DEFAULT_VRAM_MODE,
        DEVICE_MAP_OPTIONS,
        DTYPE_OPTIONS,
        INTERLEAVE_RESOLUTION_OPTIONS,
        INTERLEAVE_RESULT_TYPE,
        LOCAL_MODEL_TYPE,
        T2I_RESOLUTION_OPTIONS,
        VRAM_MODE_OPTIONS,
        SenseNovaU1LocalModel,
        default_source_path,
        interleave_output_to_tuple,
        interleave_result_to_markdown,
        output_to_tuple,
        parse_resolution_option,
        target_pixels_from_megapixels,
    )
    from .prompt_utils import load_prompt_template
except ImportError:  # pragma: no cover - supports direct imports during tests
    from api_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        VISION_MODELS,
        SenseNovaClient,
    )
    from image_utils import (
        comfy_batch_to_pil_images,
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
    )
    from local_pipeline import (
        ATTN_BACKEND_OPTIONS,
        CFG_NORM_OPTIONS,
        DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
        DEFAULT_SEED,
        DEFAULT_VRAM_MODE,
        DEVICE_MAP_OPTIONS,
        DTYPE_OPTIONS,
        INTERLEAVE_RESOLUTION_OPTIONS,
        INTERLEAVE_RESULT_TYPE,
        LOCAL_MODEL_TYPE,
        T2I_RESOLUTION_OPTIONS,
        VRAM_MODE_OPTIONS,
        SenseNovaU1LocalModel,
        default_source_path,
        interleave_output_to_tuple,
        interleave_result_to_markdown,
        output_to_tuple,
        parse_resolution_option,
        target_pixels_from_megapixels,
    )
    from prompt_utils import load_prompt_template

CATEGORY = "SenseNova"
LOCAL_CATEGORY = f"{CATEGORY}/Local"
VISION_SYSTEM_PROMPT = "You are a careful vision assistant. Describe only visible details."
BUILDER_PROMPT_TEMPLATE = "builder_prompt.txt"
LOGGER = logging.getLogger(__name__)

LocalModelIO = io.Custom(LOCAL_MODEL_TYPE)
InterleaveResultIO = io.Custom(INTERLEAVE_RESULT_TYPE)


_GGUF_FOLDER_CANDIDATES: tuple[str, ...] = ("gguf", "diffusion_models")


def _list_gguf_options() -> list[str]:
    """Combo options for SenseNovaU1LocalLoader.gguf_checkpoint.

    Always starts with an empty string (= no GGUF, load via safetensors), then
    every `.gguf` filename found under any registered folder in
    ``_GGUF_FOLDER_CANDIDATES`` (`gguf` for the dedicated layout, plus the
    stock ComfyUI `diffusion_models` folder where ComfyUI-GGUF style packs
    live). Returns just ``[""]`` when folder_paths is unavailable or no
    matching files exist, so the schema still loads cleanly outside ComfyUI.
    """
    found: set[str] = set()
    try:
        import folder_paths

        for folder in _GGUF_FOLDER_CANDIDATES:
            try:
                files = folder_paths.get_filename_list(folder)
            except Exception:
                continue
            for f in files:
                if f.lower().endswith(".gguf"):
                    found.add(f)
    except Exception:
        pass
    return ["", *sorted(found)]


def _resolve_gguf_choice(value: str) -> str:
    """Map a Combo selection back to an absolute path.

    Searches the configured folders in order; the first registered folder
    that contains the file wins. If the value isn't a registered filename
    (e.g. workflow JSON edited to point at a literal path), it is returned
    unchanged so SenseNovaU1LocalModel can treat it as an absolute path.
    """
    if not value:
        return ""
    try:
        import folder_paths

        for folder in _GGUF_FOLDER_CANDIDATES:
            try:
                full = folder_paths.get_full_path(folder, value)
            except Exception:
                continue
            if full:
                return full
    except Exception:
        pass
    return value


_LOCAL_MODEL_CACHE: dict[tuple, SenseNovaU1LocalModel] = {}


def _evict_model_cache(keep_key: tuple | None = None) -> None:
    to_evict = [k for k in _LOCAL_MODEL_CACHE if k != keep_key]
    for k in to_evict:
        old = _LOCAL_MODEL_CACHE.pop(k)
        try:
            del old.model
        except Exception:
            pass
        try:
            del old.tokenizer
        except Exception:
            pass
        del old
    if to_evict:
        # Force a GC pass *before* empty_cache so any tensors waiting on
        # cyclic refs / lingering hooks actually drop their CUDA memory back
        # to the caching allocator. Without this, empty_cache() can't reclaim
        # the old model's VRAM and the next load OOMs partway through inference.
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Old model may have been CPU-pinned (vram_mode != "full");
                # release the pinned host blocks too.
                if hasattr(torch._C, "_host_emptyCache"):
                    torch._C._host_emptyCache()
        except Exception:
            pass
        LOGGER.info("SenseNova U1 loader: evicted %d cached model(s) from VRAM.", len(to_evict))


class SenseNovaChat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaChat",
            display_name="SenseNova Chat",
            category=CATEGORY,
            inputs=[
                io.String.Input("text", multiline=True, default=""),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="You are a helpful assistant. Answer clearly and concisely.",
                ),
                io.Combo.Input("model", options=list(CHAT_MODELS), default=CHAT_MODELS[0]),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.1),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=1, max=65536),
                io.Int.Input("timeout", default=120, min=10, max=600),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="usage_json"),
                io.String.Output(display_name="raw_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        text: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> io.NodeOutput:
        client = SenseNovaClient.from_env()
        result = client.chat(
            text=text,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return io.NodeOutput(
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaImageGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaImageGenerate",
            display_name="SenseNova Image Generate",
            category=CATEGORY,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("model", options=list(IMAGE_MODELS), default=IMAGE_MODELS[0]),
                io.Combo.Input("size", options=list(IMAGE_SIZE_OPTIONS), default=IMAGE_SIZE_OPTIONS[0]),
                io.Int.Input("timeout", default=300, min=30, max=900),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="image_base64"),
                io.String.Output(display_name="image_url"),
                io.String.Output(display_name="raw_json"),
                io.String.Output(display_name="image_info"),
            ],
        )

    @classmethod
    def execute(cls, prompt: str, model: str, size: str, timeout: int) -> io.NodeOutput:
        client = SenseNovaClient.from_env()
        result = client.generate_image(prompt=prompt, model=model, size=size, timeout=timeout)
        image = image_bytes_to_comfy_image(result.image_bytes)
        image_info = comfy_image_info(image)
        LOGGER.info(
            "SenseNova image generated: bytes=%s; url=%s; %s",
            len(result.image_bytes),
            bool(result.image_url),
            image_info,
        )
        return io.NodeOutput(
            image,
            result.image_base64,
            result.image_url,
            json.dumps(result.raw, ensure_ascii=False),
            image_info,
        )


class SenseNovaPromptBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaPromptBuilder",
            display_name="SenseNova Prompt Builder",
            category=CATEGORY,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=load_prompt_template(BUILDER_PROMPT_TEMPLATE),
                ),
                io.Combo.Input("model", options=list(CHAT_MODELS), default=CHAT_MODELS[0]),
                io.Float.Input("temperature", default=0.3, min=0.0, max=2.0, step=0.1),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=1, max=65536),
                io.Int.Input("timeout", default=120, min=10, max=600),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="usage_json"),
                io.String.Output(display_name="raw_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> io.NodeOutput:
        client = SenseNovaClient.from_env()
        result = client.chat(
            text=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return io.NodeOutput(
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaVisionURL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaVisionURL",
            display_name="SenseNova Vision URL",
            category=CATEGORY,
            inputs=[
                io.String.Input("image_url", default=""),
                io.String.Input("prompt", multiline=True, default="Describe this image."),
                io.String.Input("system_prompt", multiline=True, default=VISION_SYSTEM_PROMPT),
                io.Combo.Input("model", options=list(VISION_MODELS), default=VISION_MODELS[0]),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.1),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=1, max=65536),
                io.Int.Input("timeout", default=120, min=10, max=600),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="usage_json"),
                io.String.Output(display_name="raw_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_url: str,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> io.NodeOutput:
        client = SenseNovaClient.from_env()
        result = client.vision_chat(
            image_url=image_url,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return io.NodeOutput(
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaVisionImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaVisionImage",
            display_name="SenseNova Vision Image",
            category=CATEGORY,
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default="Describe this image."),
                io.String.Input("system_prompt", multiline=True, default=VISION_SYSTEM_PROMPT),
                io.Combo.Input("model", options=list(VISION_MODELS), default=VISION_MODELS[0]),
                io.Float.Input("temperature", default=0.2, min=0.0, max=2.0, step=0.1),
                io.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("max_tokens", default=2048, min=1, max=65536),
                io.Int.Input("timeout", default=120, min=10, max=600),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="usage_json"),
                io.String.Output(display_name="raw_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> io.NodeOutput:
        client = SenseNovaClient.from_env()
        result = client.vision_chat(
            image_url=comfy_image_to_png_data_url(image),
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return io.NodeOutput(
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaU1LocalLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1LocalLoader",
            display_name="SenseNova U1 Local Loader",
            category=LOCAL_CATEGORY,
            inputs=[
                io.String.Input(
                    "model_path",
                    default="sensenova/SenseNova-U1-8B-MoT",
                    tooltip="HuggingFace model id or local checkpoint directory.",
                ),
                io.String.Input(
                    "sensenova_u1_src",
                    default=default_source_path(),
                    tooltip="Optional SenseNova-U1 source checkout or src directory.",
                ),
                io.String.Input("device", default="cuda"),
                io.Combo.Input("dtype", options=list(DTYPE_OPTIONS), default="bfloat16"),
                io.Combo.Input("attn_backend", options=list(ATTN_BACKEND_OPTIONS), default="auto"),
                io.Combo.Input(
                    "device_map",
                    options=list(DEVICE_MAP_OPTIONS),
                    default="none",
                    tooltip=(
                        "Multi-GPU sharding via accelerate. 'none' = single device "
                        "(default). auto/balanced/balanced_low_0/sequential split layers "
                        "across all visible GPUs. For *single-GPU VRAM reduction* use "
                        "vram_mode instead — they are mutually exclusive."
                    ),
                ),
                io.String.Input(
                    "max_memory",
                    default="",
                    tooltip=(
                        "Per-device memory budget for device_map (e.g. 0=20GiB,1=20GiB,cpu=64GiB). "
                        "Only relevant when device_map != 'none'."
                    ),
                ),
                io.Combo.Input(
                    "vram_mode",
                    options=list(VRAM_MODE_OPTIONS),
                    default=DEFAULT_VRAM_MODE,
                    tooltip=(
                        "Single-GPU layer-offload mode (controls weight residency only; "
                        "activations / KV cache grow with workload — especially in interleave "
                        "mode where each generated image enlarges the cache).\n"
                        "  full     — no offload, whole model on GPU, fastest (default)\n"
                        "  low      — synchronous per-layer CPU<->GPU swap, smallest weight\n"
                        "             footprint, slowest\n"
                        "  balanced — async prefetch, overlaps H2D with compute, faster than low\n"
                        "Anything other than 'full' forces device_map='none' (use device_map "
                        "for multi-GPU sharding instead)."
                    ),
                ),
                io.Combo.Input(
                    "gguf_checkpoint",
                    options=_list_gguf_options(),
                    default="",
                    tooltip=(
                        "Optional .gguf quantized checkpoint, picked from "
                        "`<comfyui>/models/gguf/` or `<comfyui>/models/diffusion_models/`. "
                        "Empty (default) loads safetensors via from_pretrained. When set, weights "
                        "are loaded via the diffusers GGUF quantizer; device_map must be 'none'. "
                        "Requires the [gguf] extra (gguf>=0.10.0, diffusers>=0.30.0). Restart "
                        "ComfyUI to refresh the list after dropping new files into either folder."
                    ),
                ),
            ],
            outputs=[
                LocalModelIO.Output(display_name="u1_model"),
                io.String.Output(display_name="model_info_json"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model_path: str,
        sensenova_u1_src: str,
        device: str,
        dtype: str,
        attn_backend: str,
        device_map: str,
        max_memory: str,
        vram_mode: str,
        gguf_checkpoint: str,
    ) -> str:
        key = (
            model_path.strip(),
            sensenova_u1_src.strip(),
            device.strip(),
            dtype,
            attn_backend,
            device_map,
            max_memory.strip(),
            vram_mode,
            _resolve_gguf_choice(gguf_checkpoint.strip()),
        )
        return hashlib.sha256(str(key).encode()).hexdigest()

    @classmethod
    def execute(
        cls,
        model_path: str,
        sensenova_u1_src: str,
        device: str,
        dtype: str,
        attn_backend: str,
        device_map: str,
        max_memory: str,
        vram_mode: str,
        gguf_checkpoint: str,
    ) -> io.NodeOutput:
        resolved_gguf = _resolve_gguf_choice(gguf_checkpoint.strip())
        cache_key = (
            model_path.strip(),
            sensenova_u1_src.strip(),
            device.strip(),
            dtype,
            attn_backend,
            device_map,
            max_memory.strip(),
            vram_mode,
            resolved_gguf,
        )
        if cache_key not in _LOCAL_MODEL_CACHE:
            _evict_model_cache()
            if resolved_gguf:
                LOGGER.info(
                    "SenseNova U1 loader: loading %s with GGUF checkpoint %s",
                    model_path,
                    resolved_gguf,
                )
            else:
                LOGGER.info("SenseNova U1 loader: loading model from %s", model_path)
            _LOCAL_MODEL_CACHE[cache_key] = SenseNovaU1LocalModel(
                model_path=model_path,
                sensenova_u1_src=sensenova_u1_src,
                device=device,
                dtype=dtype,
                attn_backend=attn_backend,
                device_map=device_map,
                max_memory=max_memory,
                gguf_checkpoint=resolved_gguf,
                vram_mode=vram_mode,
            )
        else:
            LOGGER.info("SenseNova U1 loader: reusing cached model for %s", model_path)
        model = _LOCAL_MODEL_CACHE[cache_key]
        return io.NodeOutput(model, json.dumps(model.info, ensure_ascii=False))


class SenseNovaU1LocalTextToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1LocalTextToImage",
            display_name="SenseNova U1 Local Text to Image",
            category=LOCAL_CATEGORY,
            inputs=[
                LocalModelIO.Input("u1_model"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input(
                    "resolution",
                    options=list(T2I_RESOLUTION_OPTIONS),
                    default=T2I_RESOLUTION_OPTIONS[0],
                ),
                io.Float.Input("cfg_scale", default=4.0, min=0.0, max=20.0, step=0.1),
                io.Combo.Input("cfg_norm", options=list(CFG_NORM_OPTIONS), default="none"),
                io.Float.Input("timestep_shift", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("cfg_interval_start", default=0.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("cfg_interval_end", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("num_steps", default=50, min=1, max=200),
                io.Int.Input("batch_size", default=1, min=1, max=16),
                io.Int.Input("seed", default=DEFAULT_SEED, min=0, max=2**31 - 1),
                io.Boolean.Input("think_mode", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="text"),
                io.String.Output(display_name="think_text"),
                io.String.Output(display_name="metadata_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        u1_model: SenseNovaU1LocalModel,
        prompt: str,
        resolution: str,
        cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        cfg_interval_start: float,
        cfg_interval_end: float,
        num_steps: int,
        batch_size: int,
        seed: int,
        think_mode: bool,
    ) -> io.NodeOutput:
        width, height = parse_resolution_option(resolution)
        result = u1_model.text_to_image(
            prompt=prompt,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=(cfg_interval_start, cfg_interval_end),
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
            think_mode=think_mode,
        )
        LOGGER.info("SenseNova U1 local T2I generated: %s", comfy_image_info(result.images))
        return io.NodeOutput(*output_to_tuple(result))


class SenseNovaU1LocalImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1LocalImageEdit",
            display_name="SenseNova U1 Local Image Edit",
            category=LOCAL_CATEGORY,
            inputs=[
                LocalModelIO.Input("u1_model"),
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Boolean.Input("auto_size", default=True),
                io.Int.Input("width", default=2048, min=32, max=8192, step=32),
                io.Int.Input("height", default=2048, min=32, max=8192, step=32),
                io.Float.Input(
                    "target_megapixels",
                    default=4.194304,
                    min=0.25,
                    max=32.0,
                    step=0.25,
                ),
                io.Float.Input("cfg_scale", default=4.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("img_cfg_scale", default=1.0, min=0.0, max=20.0, step=0.1),
                io.Combo.Input("cfg_norm", options=list(CFG_NORM_OPTIONS[:-1]), default="none"),
                io.Float.Input("timestep_shift", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("cfg_interval_start", default=0.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("cfg_interval_end", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("num_steps", default=50, min=1, max=200),
                io.Int.Input("batch_size", default=1, min=1, max=16),
                io.Int.Input("seed", default=DEFAULT_SEED, min=0, max=2**31 - 1),
                io.Boolean.Input("think_mode", default=False, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="text"),
                io.String.Output(display_name="think_text"),
                io.String.Output(display_name="metadata_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        u1_model: SenseNovaU1LocalModel,
        image,
        prompt: str,
        auto_size: bool,
        width: int,
        height: int,
        target_megapixels: float,
        cfg_scale: float,
        img_cfg_scale: float,
        cfg_norm: str,
        timestep_shift: float,
        cfg_interval_start: float,
        cfg_interval_end: float,
        num_steps: int,
        batch_size: int,
        seed: int,
        think_mode: bool = False,
    ) -> io.NodeOutput:
        result = u1_model.edit_image(
            prompt=prompt,
            input_image=image,
            width=None if auto_size else width,
            height=None if auto_size else height,
            target_pixels=target_pixels_from_megapixels(target_megapixels),
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=(cfg_interval_start, cfg_interval_end),
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
            think_mode=think_mode,
        )
        LOGGER.info("SenseNova U1 local edit generated: %s", comfy_image_info(result.images))
        return io.NodeOutput(*output_to_tuple(result))


class SenseNovaU1LocalInterleave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1LocalInterleave",
            display_name="SenseNova U1 Local Interleave",
            category=LOCAL_CATEGORY,
            inputs=[
                LocalModelIO.Input("u1_model"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input(
                    "resolution",
                    options=list(INTERLEAVE_RESOLUTION_OPTIONS),
                    default=INTERLEAVE_RESOLUTION_OPTIONS[1],
                ),
                io.String.Input(
                    "system_message",
                    multiline=True,
                    default=DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
                ),
                io.Float.Input("cfg_scale", default=4.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("img_cfg_scale", default=1.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("timestep_shift", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input("cfg_interval_start", default=0.0, min=0.0, max=1.0, step=0.05),
                io.Float.Input("cfg_interval_end", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("num_steps", default=50, min=1, max=200),
                io.Int.Input("seed", default=DEFAULT_SEED, min=0, max=2**31 - 1),
                io.Boolean.Input("think_mode", default=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="text"),
                io.String.Output(display_name="think_text"),
                io.String.Output(display_name="metadata_json"),
                InterleaveResultIO.Output(display_name="interleave_result"),
            ],
        )

    @classmethod
    def execute(
        cls,
        u1_model: SenseNovaU1LocalModel,
        prompt: str,
        resolution: str,
        system_message: str,
        cfg_scale: float,
        img_cfg_scale: float,
        timestep_shift: float,
        cfg_interval_start: float,
        cfg_interval_end: float,
        num_steps: int,
        seed: int,
        think_mode: bool,
        image=None,
    ) -> io.NodeOutput:
        width, height = parse_resolution_option(resolution)
        result = u1_model.interleave(
            prompt=prompt,
            input_image=image,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            timestep_shift=timestep_shift,
            cfg_interval=(cfg_interval_start, cfg_interval_end),
            num_steps=num_steps,
            seed=seed,
            think_mode=think_mode,
            system_message=system_message,
        )
        LOGGER.info("SenseNova U1 local interleave generated: %s", comfy_image_info(result.images))
        return io.NodeOutput(*interleave_output_to_tuple(result))


class SenseNovaInterleavePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaInterleavePreview",
            display_name="SenseNova Interleave Preview",
            category=LOCAL_CATEGORY,
            is_output_node=True,
            inputs=[
                InterleaveResultIO.Input("interleave_result"),
                io.Boolean.Input("include_think", default=False),
                io.Image.Input("images", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="markdown"),
            ],
        )

    @classmethod
    def execute(
        cls,
        interleave_result: dict,
        include_think: bool,
        images=None,
    ) -> io.NodeOutput:
        markdown = interleave_result_to_markdown(interleave_result, include_think=include_think)
        saved_images: list[dict[str, str]] = _save_preview_images(images) if images is not None else []

        # Structured parts let the frontend render text and images in their
        # original interleaved order instead of stacking them.
        parts_payload: list[dict[str, Any]] = []
        for part in interleave_result.get("parts", []):
            ptype = part.get("type")
            if ptype == "think" and not include_think:
                continue
            if ptype in ("text", "think"):
                text = str(part.get("text", "")).strip()
                if text:
                    parts_payload.append({"type": ptype, "text": text})
            elif ptype == "image":
                idx = int(part.get("index", 0))
                img = saved_images[idx] if 0 <= idx < len(saved_images) else None
                if img is None:
                    parts_payload.append({"type": "image", "index": idx, "missing": True})
                else:
                    parts_payload.append(
                        {
                            "type": "image",
                            "index": idx,
                            "filename": img.get("filename", ""),
                            "subfolder": img.get("subfolder", ""),
                            "image_type": img.get("type", "temp"),
                        }
                    )

        # The custom `parts` field is consumed by web/sensenova_interleave_preview.js;
        # `text` mirrors the legacy v1 ui shape.
        return io.NodeOutput(
            markdown,
            ui={"text": [markdown], "parts": parts_payload},
        )


def _save_preview_images(images) -> list[dict[str, str]]:
    managed_by_comfyui = False
    try:
        import folder_paths

        output_dir = Path(folder_paths.get_temp_directory())
        managed_by_comfyui = True
    except Exception:
        output_dir = Path(tempfile.gettempdir()) / "sensenova_comfyui_preview"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not managed_by_comfyui:
        for stale in output_dir.glob("sensenova_interleave_*.png"):
            try:
                stale.unlink()
            except OSError:
                pass

    saved: list[dict[str, str]] = []
    for index, image in enumerate(comfy_batch_to_pil_images(images)):
        filename = f"sensenova_interleave_{uuid.uuid4().hex}_{index:03d}.png"
        image.save(output_dir / filename, format="PNG")
        saved.append({"filename": filename, "subfolder": "", "type": "temp"})
    return saved


class SenseNovaExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaChat,
            SenseNovaImageGenerate,
            SenseNovaPromptBuilder,
            SenseNovaVisionURL,
            SenseNovaVisionImage,
            SenseNovaU1LocalLoader,
            SenseNovaU1LocalTextToImage,
            SenseNovaU1LocalImageEdit,
            SenseNovaU1LocalInterleave,
            SenseNovaInterleavePreview,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()
