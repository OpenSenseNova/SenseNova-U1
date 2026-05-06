from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

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
        DEVICE_MAP_OPTIONS,
        DTYPE_OPTIONS,
        INTERLEAVE_RESOLUTION_OPTIONS,
        INTERLEAVE_RESULT_TYPE,
        LOCAL_MODEL_TYPE,
        T2I_RESOLUTION_OPTIONS,
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
        DEVICE_MAP_OPTIONS,
        DTYPE_OPTIONS,
        INTERLEAVE_RESOLUTION_OPTIONS,
        INTERLEAVE_RESULT_TYPE,
        LOCAL_MODEL_TYPE,
        T2I_RESOLUTION_OPTIONS,
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
VISION_SYSTEM_PROMPT = "You are a careful vision assistant. Describe only visible details."
BUILDER_PROMPT_TEMPLATE = "builder_prompt.txt"
LOGGER = logging.getLogger(__name__)

_LOCAL_MODEL_CACHE: dict[tuple, SenseNovaU1LocalModel] = {}


def _evict_model_cache(keep_key: tuple | None = None) -> None:
    to_evict = [k for k in _LOCAL_MODEL_CACHE if k != keep_key]
    for k in to_evict:
        old = _LOCAL_MODEL_CACHE.pop(k)
        try:
            del old.model
        except Exception:
            pass
        del old
    if to_evict:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        LOGGER.info("SenseNova U1 loader: evicted %d cached model(s) from VRAM.", len(to_evict))


class SenseNovaChat:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "usage_json", "raw_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant. Answer clearly and concisely.",
                    },
                ),
                "model": (list(CHAT_MODELS), {"default": CHAT_MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    def run(
        self,
        text: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ):
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
        return (
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaImageGenerate:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_base64", "image_url", "raw_json", "image_info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(IMAGE_MODELS), {"default": IMAGE_MODELS[0]}),
                "size": (list(IMAGE_SIZE_OPTIONS), {"default": IMAGE_SIZE_OPTIONS[0]}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 900}),
            }
        }

    def run(self, prompt: str, model: str, size: str, timeout: int):
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
        return (
            image,
            result.image_base64,
            result.image_url,
            json.dumps(result.raw, ensure_ascii=False),
            image_info,
        )


class SenseNovaPromptBuilder:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "usage_json", "raw_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": load_prompt_template(BUILDER_PROMPT_TEMPLATE),
                    },
                ),
                "model": (list(CHAT_MODELS), {"default": CHAT_MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    def run(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ):
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
        return (
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaVisionURL:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "usage_json", "raw_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": VISION_SYSTEM_PROMPT,
                    },
                ),
                "model": (list(VISION_MODELS), {"default": VISION_MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    def run(
        self,
        image_url: str,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ):
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
        return (
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaVisionImage:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "usage_json", "raw_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": VISION_SYSTEM_PROMPT,
                    },
                ),
                "model": (list(VISION_MODELS), {"default": VISION_MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    def run(
        self,
        image,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ):
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
        return (
            result.text,
            json.dumps(result.usage, ensure_ascii=False),
            json.dumps(result.raw, ensure_ascii=False),
        )


class SenseNovaU1LocalLoader:
    CATEGORY = f"{CATEGORY}/Local"
    RETURN_TYPES = (LOCAL_MODEL_TYPE, "STRING")
    RETURN_NAMES = ("u1_model", "model_info_json")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": "sensenova/SenseNova-U1-8B-MoT",
                        "tooltip": "HuggingFace model id or local checkpoint directory.",
                    },
                ),
                "sensenova_u1_src": (
                    "STRING",
                    {
                        "default": default_source_path(),
                        "tooltip": "Optional SenseNova-U1 source checkout or src directory.",
                    },
                ),
                "device": ("STRING", {"default": "cuda"}),
                "dtype": (list(DTYPE_OPTIONS), {"default": "bfloat16"}),
                "attn_backend": (list(ATTN_BACKEND_OPTIONS), {"default": "auto"}),
                "device_map": (list(DEVICE_MAP_OPTIONS), {"default": "none"}),
                "max_memory": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional accelerate max_memory, e.g. 0=20GiB,cpu=64GiB.",
                    },
                ),
                "offload_folder": ("STRING", {"default": ""}),
                "offload_state_dict": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        model_path: str,
        sensenova_u1_src: str,
        device: str,
        dtype: str,
        attn_backend: str,
        device_map: str,
        max_memory: str,
        offload_folder: str,
        offload_state_dict: bool,
    ) -> str:
        key = (
            model_path.strip(),
            sensenova_u1_src.strip(),
            device.strip(),
            dtype,
            attn_backend,
            device_map,
            max_memory.strip(),
            offload_folder.strip(),
            offload_state_dict,
        )
        return hashlib.sha256(str(key).encode()).hexdigest()

    def load(
        self,
        model_path: str,
        sensenova_u1_src: str,
        device: str,
        dtype: str,
        attn_backend: str,
        device_map: str,
        max_memory: str,
        offload_folder: str,
        offload_state_dict: bool,
    ):
        cache_key = (
            model_path.strip(),
            sensenova_u1_src.strip(),
            device.strip(),
            dtype,
            attn_backend,
            device_map,
            max_memory.strip(),
            offload_folder.strip(),
            offload_state_dict,
        )
        if cache_key not in _LOCAL_MODEL_CACHE:
            _evict_model_cache()
            LOGGER.info("SenseNova U1 loader: loading model from %s", model_path)
            _LOCAL_MODEL_CACHE[cache_key] = SenseNovaU1LocalModel(
                model_path=model_path,
                sensenova_u1_src=sensenova_u1_src,
                device=device,
                dtype=dtype,
                attn_backend=attn_backend,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
            )
        else:
            LOGGER.info("SenseNova U1 loader: reusing cached model for %s", model_path)
        model = _LOCAL_MODEL_CACHE[cache_key]
        return model, json.dumps(model.info, ensure_ascii=False)


class SenseNovaU1LocalTextToImage:
    CATEGORY = f"{CATEGORY}/Local"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "text", "think_text", "metadata_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "u1_model": (LOCAL_MODEL_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (
                    list(T2I_RESOLUTION_OPTIONS),
                    {"default": T2I_RESOLUTION_OPTIONS[0]},
                ),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_norm": (list(CFG_NORM_OPTIONS), {"default": "none"}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_interval_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "cfg_interval_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seed": ("INT", {"default": DEFAULT_SEED, "min": 0, "max": 2**31 - 1}),
                "think_mode": ("BOOLEAN", {"default": False}),
            }
        }

    def run(
        self,
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
    ):
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
        return output_to_tuple(result)


class SenseNovaU1LocalImageEdit:
    CATEGORY = f"{CATEGORY}/Local"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "text", "think_text", "metadata_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "u1_model": (LOCAL_MODEL_TYPE,),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "auto_size": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 32}),
                "target_megapixels": (
                    "FLOAT",
                    {"default": 4.194304, "min": 0.25, "max": 32.0, "step": 0.25},
                ),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "img_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_norm": (list(CFG_NORM_OPTIONS[:-1]), {"default": "none"}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_interval_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "cfg_interval_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seed": ("INT", {"default": DEFAULT_SEED, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "think_mode": ("BOOLEAN", {"default": False}),
            },
        }

    def run(
        self,
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
    ):
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
        return output_to_tuple(result)


class SenseNovaU1LocalInterleave:
    CATEGORY = f"{CATEGORY}/Local"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", INTERLEAVE_RESULT_TYPE)
    RETURN_NAMES = ("images", "text", "think_text", "metadata_json", "interleave_result")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "u1_model": (LOCAL_MODEL_TYPE,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (
                    list(INTERLEAVE_RESOLUTION_OPTIONS),
                    {"default": INTERLEAVE_RESOLUTION_OPTIONS[1]},
                ),
                "system_message": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_INTERLEAVE_SYSTEM_MESSAGE},
                ),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "img_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_interval_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "cfg_interval_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "seed": ("INT", {"default": DEFAULT_SEED, "min": 0, "max": 2**31 - 1}),
                "think_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def run(
        self,
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
    ):
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
        return interleave_output_to_tuple(result)


class SenseNovaInterleavePreview:
    CATEGORY = f"{CATEGORY}/Local"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interleave_result": (INTERLEAVE_RESULT_TYPE,),
                "include_think": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    def run(self, interleave_result: dict, include_think: bool, images=None):
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

        return {
            "ui": {"text": [markdown], "parts": parts_payload},
            "result": (markdown,),
        }


NODE_CLASS_MAPPINGS = {
    "SenseNovaChat": SenseNovaChat,
    "SenseNovaImageGenerate": SenseNovaImageGenerate,
    "SenseNovaPromptBuilder": SenseNovaPromptBuilder,
    "SenseNovaVisionURL": SenseNovaVisionURL,
    "SenseNovaVisionImage": SenseNovaVisionImage,
    "SenseNovaU1LocalLoader": SenseNovaU1LocalLoader,
    "SenseNovaU1LocalTextToImage": SenseNovaU1LocalTextToImage,
    "SenseNovaU1LocalImageEdit": SenseNovaU1LocalImageEdit,
    "SenseNovaU1LocalInterleave": SenseNovaU1LocalInterleave,
    "SenseNovaInterleavePreview": SenseNovaInterleavePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseNovaChat": "SenseNova Chat",
    "SenseNovaImageGenerate": "SenseNova Image Generate",
    "SenseNovaPromptBuilder": "SenseNova Prompt Builder",
    "SenseNovaVisionURL": "SenseNova Vision URL",
    "SenseNovaVisionImage": "SenseNova Vision Image",
    "SenseNovaU1LocalLoader": "SenseNova U1 Local Loader",
    "SenseNovaU1LocalTextToImage": "SenseNova U1 Local Text to Image",
    "SenseNovaU1LocalImageEdit": "SenseNova U1 Local Image Edit",
    "SenseNovaU1LocalInterleave": "SenseNova U1 Local Interleave",
    "SenseNovaInterleavePreview": "SenseNova Interleave Preview",
}


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
