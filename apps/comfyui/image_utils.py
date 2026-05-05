from __future__ import annotations

import base64
import binascii
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
from PIL import Image

MAX_IMAGE_BYTES = 50 * 1024 * 1024


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def strip_data_url(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def is_image_data_url(value: str) -> bool:
    return value.startswith("data:image/") and ";base64," in value


def is_supported_vision_image_url(value: str) -> bool:
    return is_http_url(value) or is_image_data_url(value)


def decode_base64_image(value: str) -> Image.Image:
    try:
        data = base64.b64decode(strip_data_url(value), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Image response contains invalid base64 data.") from exc
    return image_from_bytes(data)


def image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise RuntimeError("Image response is larger than the 50MB safety limit.")

    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise RuntimeError("Image response could not be decoded by Pillow.") from exc


def pil_to_comfy_image(image: Image.Image):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required by ComfyUI to output IMAGE tensors.") from exc

    rgb_image = image.convert("RGB")
    array = np.array(rgb_image, dtype=np.float32, copy=True) / 255.0
    array = np.ascontiguousarray(array)
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.contiguous().float()


def image_bytes_to_comfy_image(data: bytes):
    return pil_to_comfy_image(image_from_bytes(data))


def comfy_image_info(image) -> str:
    shape = tuple(image.shape) if hasattr(image, "shape") else "<unknown>"
    dtype = getattr(image, "dtype", "<unknown>")
    device = getattr(image, "device", "<unknown>")
    is_contiguous = image.is_contiguous() if hasattr(image, "is_contiguous") else "<unknown>"

    try:
        min_value = float(image.min())
        max_value = float(image.max())
        value_range = f"{min_value:.6f}..{max_value:.6f}"
    except Exception:
        value_range = "<unknown>"

    return f"shape={shape}; dtype={dtype}; device={device}; contiguous={is_contiguous}; range={value_range}"


def comfy_image_to_pil(image) -> Image.Image:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] < 1:
            raise RuntimeError("ComfyUI IMAGE batch is empty.")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise RuntimeError("ComfyUI IMAGE must have shape [B,H,W,C] or [H,W,C].")

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def comfy_batch_to_pil_images(image) -> list[Image.Image]:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] not in {3, 4}:
        raise RuntimeError("ComfyUI IMAGE batch must have shape [B,H,W,C].")

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return [Image.fromarray(item).convert("RGB") for item in array]


def pil_to_png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def comfy_image_to_png_data_url(image) -> str:
    return pil_to_png_data_url(comfy_image_to_pil(image))
