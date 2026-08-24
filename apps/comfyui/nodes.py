from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from comfy_api.latest import ComfyExtension, io

try:
    from .api_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        MULTIMODAL_CHAT_MODELS,
        VISION_MODELS,
        SenseNovaClient,
    )
    from .image_utils import (
        comfy_batch_to_pil_images,
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
        pil_to_png_data_url,
    )
    from .local_pipeline import (
        ATTN_BACKEND_OPTIONS,
        CFG_NORM_OPTIONS,
        DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        DEFAULT_FAST_VRAM_FRACTION,
        DEFAULT_FAST_VRAM_HEADROOM_GIB,
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
        default_device,
        default_source_path,
        interleave_output_to_tuple,
        interleave_result_to_markdown,
        output_to_tuple,
        parse_resolution_option,
    )
    from .prompt_utils import load_prompt_template
except ImportError:  # pragma: no cover - supports direct imports during tests
    from api_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        MULTIMODAL_CHAT_MODELS,
        VISION_MODELS,
        SenseNovaClient,
    )
    from image_utils import (
        comfy_batch_to_pil_images,
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
        pil_to_png_data_url,
    )
    from local_pipeline import (
        ATTN_BACKEND_OPTIONS,
        CFG_NORM_OPTIONS,
        DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        DEFAULT_FAST_VRAM_FRACTION,
        DEFAULT_FAST_VRAM_HEADROOM_GIB,
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
        default_device,
        default_source_path,
        interleave_output_to_tuple,
        interleave_result_to_markdown,
        output_to_tuple,
        parse_resolution_option,
    )
    from prompt_utils import load_prompt_template

CATEGORY = "SenseNova"
LOCAL_CATEGORY = f"{CATEGORY}/Local"
VISION_SYSTEM_PROMPT = "You are a careful vision assistant. Describe only visible details."
BUILDER_PROMPT_TEMPLATE = "builder_prompt.txt"
LOGGER = logging.getLogger(__name__)

LocalModelIO = io.Custom(LOCAL_MODEL_TYPE)
InterleaveResultIO = io.Custom(INTERLEAVE_RESULT_TYPE)
EDIT_OUTPUT_SIZE_TYPE = "SENSENOVA_U1_EDIT_OUTPUT_SIZE"
EditOutputSizeIO = io.Custom(EDIT_OUTPUT_SIZE_TYPE)

EDIT_SIZE_AUTO_4MP = "Auto · 4MP (Recommended)"
EDIT_SIZE_AUTO_2MP = "Auto · 2MP"
EDIT_SIZE_AUTO_1MP = "Auto · 1MP"
EDIT_SIZE_MATCH_INPUT = "Match First Input"
EDIT_SIZE_CUSTOM = "Custom"
EDIT_OUTPUT_SIZE_OPTIONS = (
    EDIT_SIZE_AUTO_4MP,
    EDIT_SIZE_AUTO_2MP,
    EDIT_SIZE_AUTO_1MP,
    EDIT_SIZE_MATCH_INPUT,
    EDIT_SIZE_CUSTOM,
)
_EDIT_SIZE_PRESET_PIXELS = {
    EDIT_SIZE_AUTO_4MP: 2048 * 2048,
    EDIT_SIZE_AUTO_2MP: 2048 * 2048 // 2,
    EDIT_SIZE_AUTO_1MP: 1024 * 1024,
}


_GGUF_FOLDER_CANDIDATES: tuple[str, ...] = ("gguf", "diffusion_models")
_SENSENOVA_MODEL_FOLDER = "sensenova"
_SENSENOVA_ARTIFACT_SUFFIXES = {".safetensors", ".sft", ".gguf"}
_HF_OPTION_PREFIX = "HF | "
_LOCAL_OPTION_PREFIX = "Local | "
_AUTO_RESOURCES = "Auto"
_OFFICIAL_MODEL_IDS = (
    "sensenova/SenseNova-U1.5-8B-MoT",
    "sensenova/SenseNova-U1.5-8B-MoT-SFT",
    "sensenova/SenseNova-U1.5-8B-MoT-Preview",
    "sensenova/SenseNova-U1-8B-MoT",
)
_MAX_PROMPT_BUILDER_IMAGES = 10


@lru_cache(maxsize=1)
def _list_remote_model_options() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        catalog = SenseNovaClient.from_env().list_models(timeout=5)
    except Exception as exc:
        LOGGER.warning(
            "Unable to load SenseNova models from the API (%s); using built-in options.",
            exc.__class__.__name__,
        )
        return CHAT_MODELS, MULTIMODAL_CHAT_MODELS, IMAGE_MODELS

    chat_models = catalog.chat_models or CHAT_MODELS
    multimodal_chat_models = catalog.multimodal_chat_models or MULTIMODAL_CHAT_MODELS
    image_models = catalog.image_models or IMAGE_MODELS
    if not catalog.chat_models or not catalog.multimodal_chat_models or not catalog.image_models:
        LOGGER.warning("SenseNova API returned an incomplete model catalog; using built-in options where needed.")
    return chat_models, multimodal_chat_models, image_models


def _list_chat_model_options() -> tuple[str, ...]:
    return _list_remote_model_options()[0]


def _list_image_model_options() -> tuple[str, ...]:
    return _list_remote_model_options()[2]


def _list_multimodal_chat_model_options() -> tuple[str, ...]:
    return _list_remote_model_options()[1]


def _prompt_builder_image_urls(images: io.Autogrow.Type | None) -> list[str]:
    pil_images = []
    for image_batch in (images or {}).values():
        pil_images.extend(comfy_batch_to_pil_images(image_batch))
        if len(pil_images) > _MAX_PROMPT_BUILDER_IMAGES:
            raise RuntimeError(f"SenseNova Prompt Builder accepts at most {_MAX_PROMPT_BUILDER_IMAGES} images.")
    return [pil_to_png_data_url(image) for image in pil_images]


def _sensenova_model_roots() -> tuple[Path, ...]:
    try:
        import folder_paths

        return tuple(Path(path).expanduser() for path in folder_paths.get_folder_paths(_SENSENOVA_MODEL_FOLDER))
    except Exception:
        return ()


def _is_complete_sensenova_model(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if (path / "model.safetensors").is_file():
        return True

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        weight_map = json.loads(index_path.read_text()).get("weight_map")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    for filename in set(weight_map.values()):
        if not isinstance(filename, str):
            return False
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or not (path / relative).is_file():
            return False
    return True


def _list_sensenova_model_options() -> list[str]:
    found: set[str] = set()
    for root in _sensenova_model_roots():
        if not root.is_dir():
            continue
        try:
            config_paths = root.rglob("config.json")
            for config_path in config_paths:
                model_directory = config_path.parent
                relative = model_directory.relative_to(root)
                if relative == Path(".") or any(part.startswith(".") for part in relative.parts):
                    continue
                if _is_complete_sensenova_model(model_directory):
                    found.add(relative.as_posix())
            for artifact_path in root.rglob("*"):
                if not artifact_path.is_file() or artifact_path.suffix.lower() not in _SENSENOVA_ARTIFACT_SUFFIXES:
                    continue
                if _is_complete_sensenova_model(artifact_path.parent):
                    continue
                relative = artifact_path.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                found.add(relative.as_posix())
        except OSError:
            continue
    return ["", *sorted(found)]


def _resolve_sensenova_model_choice(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Invalid SenseNova model selection: {value!r}")

    for root in _sensenova_model_roots():
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if _is_complete_sensenova_model(candidate) or (
            candidate.is_file() and candidate.suffix.lower() in _SENSENOVA_ARTIFACT_SUFFIXES
        ):
            return str(candidate)
    raise RuntimeError(
        f"SenseNova model {value!r} was not found under any registered ComfyUI models/{_SENSENOVA_MODEL_FOLDER} folder."
    )


def _resolve_model_path(model_path: str, local_model: str) -> str:
    if local_model.strip():
        return _resolve_sensenova_model_choice(local_model)
    return model_path.strip()


def _cached_sensenova_repo_ids() -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir

        return {
            repo.repo_id
            for repo in scan_cache_dir().repos
            if repo.repo_type == "model" and "sensenova" in repo.repo_id.lower()
        }
    except Exception:
        return set()


def _list_model_weight_options() -> list[str]:
    hf_ids = set(_OFFICIAL_MODEL_IDS) | _cached_sensenova_repo_ids()
    hf_options = [f"{_HF_OPTION_PREFIX}{repo_id}" for repo_id in sorted(hf_ids)]
    local_options = [f"{_LOCAL_OPTION_PREFIX}{relative}" for relative in _list_sensenova_model_options() if relative]
    preferred = f"{_HF_OPTION_PREFIX}{_OFFICIAL_MODEL_IDS[0]}"
    return [preferred, *[option for option in hf_options if option != preferred], *local_options]


def _list_model_resource_options() -> list[str]:
    hf_ids = set(_OFFICIAL_MODEL_IDS) | _cached_sensenova_repo_ids()
    local_resources: set[str] = set()
    for root in _sensenova_model_roots():
        if not root.is_dir():
            continue
        try:
            for config_path in root.rglob("config.json"):
                relative = config_path.parent.relative_to(root)
                if relative != Path(".") and not any(part.startswith(".") for part in relative.parts):
                    local_resources.add(f"{_LOCAL_OPTION_PREFIX}{relative.as_posix()}")
        except OSError:
            continue
    return [
        _AUTO_RESOURCES,
        *[f"{_HF_OPTION_PREFIX}{repo_id}" for repo_id in sorted(hf_ids)],
        *sorted(local_resources),
    ]


def _resolve_model_weight_choice(value: str) -> str:
    if value.startswith(_HF_OPTION_PREFIX):
        return value.removeprefix(_HF_OPTION_PREFIX).strip()
    if value.startswith(_LOCAL_OPTION_PREFIX):
        return _resolve_sensenova_model_choice(value.removeprefix(_LOCAL_OPTION_PREFIX))
    raise RuntimeError(f"Invalid model_weights selection: {value!r}.")


def _resolve_model_resource_choice(value: str) -> str:
    if value == _AUTO_RESOURCES:
        return ""
    if value.startswith(_HF_OPTION_PREFIX):
        return value.removeprefix(_HF_OPTION_PREFIX).strip()
    if not value.startswith(_LOCAL_OPTION_PREFIX):
        raise RuntimeError(f"Invalid model_resources selection: {value!r}.")

    relative = Path(value.removeprefix(_LOCAL_OPTION_PREFIX).strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Invalid local model_resources selection: {value!r}.")
    for root in _sensenova_model_roots():
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if (candidate / "config.json").is_file():
            return str(candidate)
    raise RuntimeError(f"Model resources {value!r} were not found under models/sensenova.")


def _list_lora_options() -> list[str]:
    try:
        import folder_paths

        return [
            "",
            *sorted(name for name in folder_paths.get_filename_list("loras") if name.lower().endswith(".safetensors")),
        ]
    except Exception:
        return [""]


def _resolve_lora_choice(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        import folder_paths

        resolved = folder_paths.get_full_path("loras", value)
        if resolved:
            return resolved
    except Exception:
        pass
    return value


def _lora_signature(value: str) -> tuple[str, int, int]:
    resolved = _resolve_lora_choice(value)
    if not resolved:
        return ("", 0, 0)
    try:
        stat = Path(resolved).stat()
        return (resolved, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (resolved, 0, 0)


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


def _normalize_fast_settings(
    fast_vram_fraction: float | str | None,
    fast_vram_headroom_gib: float | str | None,
    fast_activation_reserve_gib: float | str | None,
    fast_vram_budget_gib: float | str | None,
) -> tuple[float, float, float, float]:
    def value_or_default(value: float | str | None, default: float) -> float:
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return float(value)

    return (
        value_or_default(fast_vram_fraction, DEFAULT_FAST_VRAM_FRACTION),
        value_or_default(fast_vram_headroom_gib, DEFAULT_FAST_VRAM_HEADROOM_GIB),
        value_or_default(fast_activation_reserve_gib, DEFAULT_FAST_ACTIVATION_RESERVE_GIB),
        value_or_default(fast_vram_budget_gib, 0.0),
    )


def _resolve_edit_output_size(
    output_size: dict[str, Any] | None,
) -> tuple[int | None, int | None, int | None]:
    """Normalize the typed size config; disconnected means Auto 4MP."""
    if output_size is None:
        return None, None, _EDIT_SIZE_PRESET_PIXELS[EDIT_SIZE_AUTO_4MP]

    if not isinstance(output_size, dict):
        raise RuntimeError("output_size must come from a SenseNova U1 Edit Output Size node.")
    mode = output_size.get("mode")
    if mode == "auto":
        target_pixels = output_size.get("target_pixels")
        if not isinstance(target_pixels, int) or target_pixels <= 0:
            raise RuntimeError("Auto output_size requires a positive target_pixels value.")
        return None, None, target_pixels
    if mode == "match_input":
        return None, None, None
    if mode == "custom":
        custom_width = output_size.get("width")
        custom_height = output_size.get("height")
        if not isinstance(custom_width, int) or not isinstance(custom_height, int):
            raise RuntimeError("Custom output_size requires integer width and height values.")
        return custom_width, custom_height, None
    raise RuntimeError(f"Unsupported output_size mode: {mode!r}.")


_LOCAL_MODEL_CACHE: dict[tuple, SenseNovaU1LocalModel] = {}
_LOCAL_MODEL_CACHE_KEY_ATTR = "_sensenova_u1_cache_key"


def _make_local_model_cache_key(
    *,
    model_path: str,
    model_resources: str,
    sensenova_u1_src: str,
    device: str,
    dtype: str,
    attn_backend: str,
    device_map: str,
    max_memory: str,
    vram_mode: str,
    fast_vram_fraction: float,
    fast_vram_headroom_gib: float,
    fast_activation_reserve_gib: float,
    fast_vram_budget_gib: float,
    resolved_gguf: str,
    lora_signature: tuple[str, int, int],
    lora_strength: float,
) -> tuple:
    return (
        model_path,
        model_resources,
        sensenova_u1_src,
        device,
        dtype,
        attn_backend,
        device_map,
        max_memory,
        vram_mode,
        fast_vram_fraction,
        fast_vram_headroom_gib,
        fast_activation_reserve_gib,
        fast_vram_budget_gib,
        resolved_gguf,
        lora_signature,
        lora_strength,
    )


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
        chat_models = _list_chat_model_options()
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
                io.Combo.Input("model", options=list(chat_models), default=chat_models[0]),
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
        image_models = _list_image_model_options()
        return io.Schema(
            node_id="SenseNovaImageGenerate",
            display_name="SenseNova Image Generate",
            category=CATEGORY,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("model", options=list(image_models), default=image_models[0]),
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
        chat_models = _list_chat_model_options()
        return io.Schema(
            node_id="SenseNovaPromptBuilder",
            display_name="SenseNova Prompt Builder",
            category=CATEGORY,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[
                            "image1",
                            "image2",
                            "image3",
                            "image4",
                            "image5",
                            "image6",
                            "image7",
                            "image8",
                            "image9",
                            "image10",
                        ],
                        min=0,
                    ),
                    tooltip=(
                        "Optional ordered reference images. Select a model whose input_modalities includes image."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=load_prompt_template(BUILDER_PROMPT_TEMPLATE),
                ),
                io.Combo.Input("model", options=list(chat_models), default=chat_models[0]),
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
        images: io.Autogrow.Type | None = None,
    ) -> io.NodeOutput:
        image_urls = _prompt_builder_image_urls(images)
        if image_urls:
            multimodal_models = _list_multimodal_chat_model_options()
            if model not in multimodal_models:
                supported_models = ", ".join(multimodal_models)
                raise RuntimeError(
                    f"SenseNova model {model!r} does not support image input. Choose one of: {supported_models}."
                )
        client = SenseNovaClient.from_env()
        result = client.chat(
            text=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            image_urls=image_urls,
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


def _new_model_loader_cache_key(
    *,
    model_weights: str,
    model_resources: str,
    lora_name: str,
    lora_strength: float,
    device: str,
    dtype: str,
    attn_backend: str,
    device_map: str,
    max_memory: str,
    vram_mode: str,
    fast_vram_fraction: float | str,
    fast_vram_headroom_gib: float | str,
    fast_activation_reserve_gib: float | str,
    fast_vram_budget_gib: float | str,
) -> tuple:
    fast_vram_fraction, fast_vram_headroom_gib, fast_activation_reserve_gib, fast_vram_budget_gib = (
        _normalize_fast_settings(
            fast_vram_fraction,
            fast_vram_headroom_gib,
            fast_activation_reserve_gib,
            fast_vram_budget_gib,
        )
    )
    return _make_local_model_cache_key(
        model_path=_resolve_model_weight_choice(model_weights),
        model_resources=_resolve_model_resource_choice(model_resources),
        sensenova_u1_src="",
        device=device.strip(),
        dtype=dtype,
        attn_backend=attn_backend,
        device_map=device_map,
        max_memory=max_memory.strip(),
        vram_mode=vram_mode,
        fast_vram_fraction=fast_vram_fraction,
        fast_vram_headroom_gib=fast_vram_headroom_gib,
        fast_activation_reserve_gib=fast_activation_reserve_gib,
        fast_vram_budget_gib=fast_vram_budget_gib,
        resolved_gguf="",
        lora_signature=_lora_signature(lora_name),
        lora_strength=float(lora_strength),
    )


class SenseNovaU1LoraSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = _list_lora_options()
        return io.Schema(
            node_id="SenseNovaU1LoraSelector",
            display_name="SenseNova U1 LoRA Selector",
            category=LOCAL_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=lora_options,
                    default=lora_options[0],
                    tooltip="Select a SenseNova LoRA from ComfyUI models/loras.",
                ),
            ],
            outputs=[io.String.Output(display_name="lora_name")],
        )

    @classmethod
    def execute(cls, lora_name: str) -> io.NodeOutput:
        return io.NodeOutput(lora_name)


class SenseNovaU1EditOutputSize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1EditOutputSize",
            display_name="SenseNova U1 Edit Output Size",
            category=LOCAL_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=list(EDIT_OUTPUT_SIZE_OPTIONS),
                    default=EDIT_SIZE_AUTO_4MP,
                    tooltip=(
                        "Auto presets preserve the first input image's aspect ratio. "
                        "Match First Input uses its native pixel count; Custom uses width and height."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=2048,
                    min=32,
                    max=8192,
                    step=32,
                    advanced=True,
                    tooltip="Used only by the Custom preset.",
                ),
                io.Int.Input(
                    "height",
                    default=2048,
                    min=32,
                    max=8192,
                    step=32,
                    advanced=True,
                    tooltip="Used only by the Custom preset.",
                ),
            ],
            outputs=[EditOutputSizeIO.Output(display_name="output_size")],
        )

    @classmethod
    def execute(cls, preset: str, width: int, height: int) -> io.NodeOutput:
        if preset in _EDIT_SIZE_PRESET_PIXELS:
            config = {"mode": "auto", "target_pixels": _EDIT_SIZE_PRESET_PIXELS[preset]}
        elif preset == EDIT_SIZE_MATCH_INPUT:
            config = {"mode": "match_input"}
        elif preset == EDIT_SIZE_CUSTOM:
            config = {"mode": "custom", "width": width, "height": height}
        else:
            raise RuntimeError(f"Unsupported edit output size preset: {preset!r}.")
        return io.NodeOutput(config)


class SenseNovaU1ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        weight_options = _list_model_weight_options()
        resource_options = _list_model_resource_options()
        return io.Schema(
            node_id="SenseNovaU1ModelLoader",
            display_name="SenseNova U1 Model Loader",
            category=LOCAL_CATEGORY,
            inputs=[
                io.Combo.Input(
                    "model_weights",
                    options=weight_options,
                    default=weight_options[0],
                    tooltip="Base weights from Hugging Face/cache or ComfyUI models/sensenova.",
                ),
                io.Combo.Input(
                    "model_resources",
                    options=resource_options,
                    default=_AUTO_RESOURCES,
                    tooltip="Config and tokenizer source. Auto uses the weight artifact metadata/location.",
                ),
                io.String.Input(
                    "lora_name",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional SenseNova LoRA filename from ComfyUI models/loras, "
                        "or a path supplied by another STRING node."
                    ),
                ),
                io.Float.Input(
                    "lora_strength",
                    default=1.0,
                    min=-4.0,
                    max=4.0,
                    step=0.05,
                    optional=True,
                ),
                io.String.Input(
                    "device",
                    default=default_device(),
                    optional=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "dtype",
                    options=list(DTYPE_OPTIONS),
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "attn_backend",
                    options=list(ATTN_BACKEND_OPTIONS),
                    default="auto",
                    optional=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "vram_mode",
                    options=list(VRAM_MODE_OPTIONS),
                    default=DEFAULT_VRAM_MODE,
                    optional=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "device_map",
                    options=list(DEVICE_MAP_OPTIONS),
                    default="none",
                    optional=True,
                    advanced=True,
                ),
                io.String.Input("max_memory", default="", optional=True, advanced=True),
                io.String.Input(
                    "fast_vram_fraction",
                    default=str(DEFAULT_FAST_VRAM_FRACTION),
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "fast_vram_headroom_gib",
                    default=str(DEFAULT_FAST_VRAM_HEADROOM_GIB),
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "fast_activation_reserve_gib",
                    default=str(DEFAULT_FAST_ACTIVATION_RESERVE_GIB),
                    optional=True,
                    advanced=True,
                ),
                io.String.Input(
                    "fast_vram_budget_gib",
                    default="0.0",
                    optional=True,
                    advanced=True,
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
        model_weights: str,
        model_resources: str = _AUTO_RESOURCES,
        lora_name: str = "",
        lora_strength: float = 1.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_backend: str = "auto",
        vram_mode: str = DEFAULT_VRAM_MODE,
        device_map: str = "none",
        max_memory: str = "",
        fast_vram_fraction: float | str = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float | str = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float | str = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | str = 0.0,
    ) -> str:
        key = _new_model_loader_cache_key(
            model_weights=model_weights,
            model_resources=model_resources,
            lora_name=lora_name,
            lora_strength=lora_strength,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
            device_map=device_map,
            max_memory=max_memory,
            vram_mode=vram_mode,
            fast_vram_fraction=fast_vram_fraction,
            fast_vram_headroom_gib=fast_vram_headroom_gib,
            fast_activation_reserve_gib=fast_activation_reserve_gib,
            fast_vram_budget_gib=fast_vram_budget_gib,
        )
        return hashlib.sha256(str(key).encode()).hexdigest()

    @classmethod
    def execute(
        cls,
        model_weights: str,
        model_resources: str = _AUTO_RESOURCES,
        lora_name: str = "",
        lora_strength: float = 1.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_backend: str = "auto",
        vram_mode: str = DEFAULT_VRAM_MODE,
        device_map: str = "none",
        max_memory: str = "",
        fast_vram_fraction: float | str = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float | str = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float | str = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | str = 0.0,
    ) -> io.NodeOutput:
        cache_key = _new_model_loader_cache_key(
            model_weights=model_weights,
            model_resources=model_resources,
            lora_name=lora_name,
            lora_strength=lora_strength,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
            device_map=device_map,
            max_memory=max_memory,
            vram_mode=vram_mode,
            fast_vram_fraction=fast_vram_fraction,
            fast_vram_headroom_gib=fast_vram_headroom_gib,
            fast_activation_reserve_gib=fast_activation_reserve_gib,
            fast_vram_budget_gib=fast_vram_budget_gib,
        )
        model = _get_or_load_local_model(cache_key)
        return io.NodeOutput(model, json.dumps(model.info, ensure_ascii=False))


class SenseNovaU1LocalLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaU1LocalLoader",
            display_name="SenseNova U1 Local Loader (Legacy)",
            category=f"{LOCAL_CATEGORY}/Legacy",
            inputs=[
                io.String.Input(
                    "model_path",
                    default="sensenova/SenseNova-U1-8B-MoT",
                    tooltip="HuggingFace model id, checkpoint directory, or standalone Safetensors/GGUF file.",
                ),
                io.String.Input(
                    "sensenova_u1_src",
                    default=default_source_path(),
                    tooltip="Optional SenseNova-U1 source checkout or src directory.",
                ),
                io.String.Input(
                    "device",
                    default=default_device(),
                    tooltip="Compute device, e.g. 'cuda', 'cuda:0', 'xpu', 'xpu:0', 'cpu'. Defaults to the best available accelerator.",
                ),
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
                        "  fast     — async prefetch, then retain generation weights within budget\n"
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
                io.String.Input(
                    "fast_vram_fraction",
                    default=str(DEFAULT_FAST_VRAM_FRACTION),
                    optional=True,
                    advanced=True,
                    tooltip="Automatic fast-mode VRAM fraction (0 < value <= 1). Blank uses the default.",
                ),
                io.String.Input(
                    "fast_vram_headroom_gib",
                    default=str(DEFAULT_FAST_VRAM_HEADROOM_GIB),
                    optional=True,
                    advanced=True,
                    tooltip="Reusable VRAM headroom in GiB. Blank uses the default.",
                ),
                io.String.Input(
                    "fast_activation_reserve_gib",
                    default=str(DEFAULT_FAST_ACTIVATION_RESERVE_GIB),
                    optional=True,
                    advanced=True,
                    tooltip="Activation reserve in GiB. Blank uses the default.",
                ),
                io.String.Input(
                    "fast_vram_budget_gib",
                    default="0.0",
                    optional=True,
                    advanced=True,
                    tooltip="Absolute fast-mode VRAM budget in GiB. Blank or 0 uses fast_vram_fraction.",
                ),
                io.Combo.Input(
                    "local_model",
                    options=_list_sensenova_model_options(),
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional model directory, .safetensors/.sft, or .gguf from "
                        "`<comfyui>/models/sensenova/`. "
                        "When selected, it overrides model_path. Restart ComfyUI to refresh this list."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=_list_lora_options(),
                    default="",
                    optional=True,
                    tooltip="Optional SenseNova LoRA from `<comfyui>/models/loras/`.",
                ),
                io.Float.Input(
                    "lora_strength",
                    default=1.0,
                    min=-4.0,
                    max=4.0,
                    step=0.05,
                    optional=True,
                    tooltip="LoRA merge strength. 1.0 uses the adapter's authored scale.",
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
        fast_vram_fraction: float | str = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float | str = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float | str = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | str = 0.0,
        local_model: str = "",
        lora_name: str = "",
        lora_strength: float = 1.0,
    ) -> str:
        fast_vram_fraction, fast_vram_headroom_gib, fast_activation_reserve_gib, fast_vram_budget_gib = (
            _normalize_fast_settings(
                fast_vram_fraction,
                fast_vram_headroom_gib,
                fast_activation_reserve_gib,
                fast_vram_budget_gib,
            )
        )
        resolved_model_path = _resolve_model_path(model_path, local_model)
        key = _make_local_model_cache_key(
            model_path=resolved_model_path,
            model_resources="",
            sensenova_u1_src=sensenova_u1_src.strip(),
            device=device.strip(),
            dtype=dtype,
            attn_backend=attn_backend,
            device_map=device_map,
            max_memory=max_memory.strip(),
            vram_mode=vram_mode,
            fast_vram_fraction=fast_vram_fraction,
            fast_vram_headroom_gib=fast_vram_headroom_gib,
            fast_activation_reserve_gib=fast_activation_reserve_gib,
            fast_vram_budget_gib=fast_vram_budget_gib,
            resolved_gguf=_resolve_gguf_choice(gguf_checkpoint.strip()),
            lora_signature=_lora_signature(lora_name),
            lora_strength=float(lora_strength),
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
        fast_vram_fraction: float | str = DEFAULT_FAST_VRAM_FRACTION,
        fast_vram_headroom_gib: float | str = DEFAULT_FAST_VRAM_HEADROOM_GIB,
        fast_activation_reserve_gib: float | str = DEFAULT_FAST_ACTIVATION_RESERVE_GIB,
        fast_vram_budget_gib: float | str = 0.0,
        local_model: str = "",
        lora_name: str = "",
        lora_strength: float = 1.0,
    ) -> io.NodeOutput:
        fast_vram_fraction, fast_vram_headroom_gib, fast_activation_reserve_gib, fast_vram_budget_gib = (
            _normalize_fast_settings(
                fast_vram_fraction,
                fast_vram_headroom_gib,
                fast_activation_reserve_gib,
                fast_vram_budget_gib,
            )
        )
        resolved_gguf = _resolve_gguf_choice(gguf_checkpoint.strip())
        resolved_model_path = _resolve_model_path(model_path, local_model)
        cache_key = _make_local_model_cache_key(
            model_path=resolved_model_path,
            model_resources="",
            sensenova_u1_src=sensenova_u1_src.strip(),
            device=device.strip(),
            dtype=dtype,
            attn_backend=attn_backend,
            device_map=device_map,
            max_memory=max_memory.strip(),
            vram_mode=vram_mode,
            fast_vram_fraction=fast_vram_fraction,
            fast_vram_headroom_gib=fast_vram_headroom_gib,
            fast_activation_reserve_gib=fast_activation_reserve_gib,
            fast_vram_budget_gib=fast_vram_budget_gib,
            resolved_gguf=resolved_gguf,
            lora_signature=_lora_signature(lora_name),
            lora_strength=float(lora_strength),
        )
        if cache_key not in _LOCAL_MODEL_CACHE:
            model = _load_local_model(cache_key)
        else:
            model = _get_or_load_local_model(cache_key)
        return io.NodeOutput(model, json.dumps(model.info, ensure_ascii=False))


def _load_local_model(cache_key: tuple) -> SenseNovaU1LocalModel:
    (
        model_path,
        model_resources,
        sensenova_u1_src,
        device,
        dtype,
        attn_backend,
        device_map,
        max_memory,
        vram_mode,
        fast_vram_fraction,
        fast_vram_headroom_gib,
        fast_activation_reserve_gib,
        fast_vram_budget_gib,
        resolved_gguf,
        lora_signature,
        lora_strength,
    ) = cache_key

    _evict_model_cache()
    if resolved_gguf:
        LOGGER.info(
            "SenseNova U1 loader: loading %s with GGUF checkpoint %s",
            model_path,
            resolved_gguf,
        )
    else:
        LOGGER.info("SenseNova U1 loader: loading model from %s", model_path)

    model = SenseNovaU1LocalModel(
        model_path=model_path,
        model_resources=model_resources,
        sensenova_u1_src=sensenova_u1_src,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        device_map=device_map,
        max_memory=max_memory,
        gguf_checkpoint=resolved_gguf,
        vram_mode=vram_mode,
        fast_vram_fraction=fast_vram_fraction,
        fast_vram_headroom_gib=fast_vram_headroom_gib,
        fast_activation_reserve_gib=fast_activation_reserve_gib,
        fast_vram_budget_gib=fast_vram_budget_gib,
        lora_path=lora_signature[0],
        lora_strength=lora_strength,
    )
    setattr(model, _LOCAL_MODEL_CACHE_KEY_ATTR, cache_key)
    _LOCAL_MODEL_CACHE[cache_key] = model
    return model


def _get_or_load_local_model(cache_key: tuple) -> SenseNovaU1LocalModel:
    cached = _LOCAL_MODEL_CACHE.get(cache_key)
    if cached is not None and hasattr(cached, "model") and hasattr(cached, "tokenizer"):
        LOGGER.info("SenseNova U1 loader: reusing cached model for %s", cache_key[0])
        return cached
    return _load_local_model(cache_key)


def _ensure_local_model_loaded(model: SenseNovaU1LocalModel) -> SenseNovaU1LocalModel:
    """Resolve a possibly stale ComfyUI output to the live local model.

    ComfyUI can retain a loader output after this module's single-entry cache
    has evicted its weights. The lightweight object still carries its cache
    key, allowing inference nodes to reacquire the correct model instead of
    dereferencing the deleted ``model``/``tokenizer`` attributes.
    """
    cache_key = getattr(model, _LOCAL_MODEL_CACHE_KEY_ATTR, None)
    if cache_key is None:
        if hasattr(model, "model") and hasattr(model, "tokenizer"):
            return model
        raise RuntimeError(
            "SenseNova U1 received an evicted legacy model handle. "
            "Run the Local Loader node again so it can attach cache metadata."
        )

    cached = _LOCAL_MODEL_CACHE.get(cache_key)
    if cached is not None and hasattr(cached, "model") and hasattr(cached, "tokenizer"):
        return cached

    LOGGER.info(
        "SenseNova U1 loader: restoring model %s referenced by a stale ComfyUI cache entry.",
        cache_key[0],
    )
    return _load_local_model(cache_key)


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
        u1_model = _ensure_local_model_loaded(u1_model)
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
                io.Autogrow.Input(
                    "reference_images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[
                            "image2",
                            "image3",
                            "image4",
                            "image5",
                            "image6",
                            "image7",
                            "image8",
                            "image9",
                            "image10",
                        ],
                        min=0,
                    ),
                    tooltip="Optional ordered reference images for editing.",
                ),
                io.String.Input("prompt", multiline=True, default=""),
                EditOutputSizeIO.Input(
                    "output_size",
                    optional=True,
                    tooltip=(
                        "Optional size policy. When disconnected, defaults to Auto · 4MP. "
                        "Connect SenseNova U1 Edit Output Size to select another policy."
                    ),
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
        reference_images: io.Autogrow.Type,
        prompt: str,
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
        output_size: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        u1_model = _ensure_local_model_loaded(u1_model)
        output_width, output_height, target_pixels = _resolve_edit_output_size(output_size)
        result = u1_model.edit_image(
            prompt=prompt,
            input_images=[image, *reference_images.values()],
            width=output_width,
            height=output_height,
            target_pixels=target_pixels,
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
        u1_model = _ensure_local_model_loaded(u1_model)
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
            SenseNovaU1LoraSelector,
            SenseNovaU1EditOutputSize,
            SenseNovaU1ModelLoader,
            SenseNovaU1LocalLoader,
            SenseNovaU1LocalTextToImage,
            SenseNovaU1LocalImageEdit,
            SenseNovaU1LocalInterleave,
            SenseNovaInterleavePreview,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()
