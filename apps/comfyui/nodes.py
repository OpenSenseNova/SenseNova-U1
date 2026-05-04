from __future__ import annotations

import json
import logging

try:
    from .image_utils import (
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
    )
    from .prompt_utils import load_prompt_template
    from .sensenova_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        VISION_MODELS,
        SenseNovaClient,
    )
except ImportError:  # pragma: no cover - supports direct imports during tests
    from image_utils import (
        comfy_image_info,
        comfy_image_to_png_data_url,
        image_bytes_to_comfy_image,
    )
    from prompt_utils import load_prompt_template
    from sensenova_client import (
        CHAT_MODELS,
        IMAGE_MODELS,
        IMAGE_SIZE_OPTIONS,
        VISION_MODELS,
        SenseNovaClient,
    )

CATEGORY = "SenseNova"
VISION_SYSTEM_PROMPT = "You are a careful vision assistant. Describe only visible details."
BUILDER_PROMPT_TEMPLATE = "builder_prompt.txt"
LOGGER = logging.getLogger(__name__)


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


NODE_CLASS_MAPPINGS = {
    "SenseNovaChat": SenseNovaChat,
    "SenseNovaImageGenerate": SenseNovaImageGenerate,
    "SenseNovaPromptBuilder": SenseNovaPromptBuilder,
    "SenseNovaVisionURL": SenseNovaVisionURL,
    "SenseNovaVisionImage": SenseNovaVisionImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SenseNovaChat": "SenseNova Chat",
    "SenseNovaImageGenerate": "SenseNova Image Generate",
    "SenseNovaPromptBuilder": "SenseNova Prompt Builder",
    "SenseNovaVisionURL": "SenseNova Vision URL",
    "SenseNovaVisionImage": "SenseNova Vision Image",
}
