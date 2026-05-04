from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

try:
    from .config import SenseNovaConfig, load_config
    from .image_utils import MAX_IMAGE_BYTES, is_http_url, is_supported_vision_image_url
except ImportError:  # pragma: no cover - supports direct test imports
    from config import SenseNovaConfig, load_config
    from image_utils import MAX_IMAGE_BYTES, is_http_url, is_supported_vision_image_url

CHAT_MODELS = ("sensenova-6.7-flash-lite", "deepseek-v4")
VISION_MODELS = ("sensenova-6.7-flash-lite",)
IMAGE_MODELS = ("sensenova-u1-fast",)
IMAGE_SIZES = (
    "2752x1536",
    "1536x2752",
    "2048x2048",
    "2496x1664",
    "1664x2496",
    "2368x1760",
    "1760x2368",
    "2272x1824",
    "1824x2272",
    "3072x1376",
    "1344x3136",
)
IMAGE_SIZE_OPTIONS = (
    "2752x1536|16:9",
    "1536x2752|9:16",
    "2048x2048|1:1",
    "2496x1664|3:2",
    "1664x2496|2:3",
    "2368x1760|4:3",
    "1760x2368|3:4",
    "2272x1824|5:4",
    "1824x2272|4:5",
    "3072x1376|21:9",
    "1344x3136|9:21",
)


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ImageGenerationResult:
    image_base64: str
    image_url: str
    image_bytes: bytes
    raw: dict[str, Any]


class SenseNovaClient:
    def __init__(self, config: SenseNovaConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> SenseNovaClient:
        return cls(load_config())

    def chat(
        self,
        *,
        text: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> ChatResult:
        if model not in CHAT_MODELS:
            raise RuntimeError(f"Unsupported chat model: {model}")
        if not text.strip():
            raise RuntimeError("Chat text cannot be empty.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        raw = self._post_json("/chat/completions", payload, timeout=timeout)
        return ChatResult(text=_extract_chat_text(raw), usage=raw.get("usage", {}), raw=raw)

    def vision_chat(
        self,
        *,
        image_url: str,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
    ) -> ChatResult:
        if model not in VISION_MODELS:
            raise RuntimeError(f"Unsupported vision model: {model}")
        if not prompt.strip():
            raise RuntimeError("Vision prompt cannot be empty.")
        if not is_supported_vision_image_url(image_url):
            raise RuntimeError("Vision image URL must be http(s) or a base64 image data URL.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        raw = self._post_json("/chat/completions", payload, timeout=timeout)
        return ChatResult(text=_extract_chat_text(raw), usage=raw.get("usage", {}), raw=raw)

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size: str,
        timeout: int,
    ) -> ImageGenerationResult:
        if model not in IMAGE_MODELS:
            raise RuntimeError(f"Unsupported image model: {model}")
        normalized_size = normalize_image_size(size)
        if normalized_size not in IMAGE_SIZES:
            raise RuntimeError(f"Unsupported image size: {size}")
        if not prompt.strip():
            raise RuntimeError("Image prompt cannot be empty.")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": normalized_size,
            "n": 1,
        }
        raw = self._post_json("/images/generations", payload, timeout=timeout)
        image_base64, image_url = _extract_image_payload(raw)

        image_bytes = b""
        if image_base64:
            import base64

            try:
                from .image_utils import strip_data_url
            except ImportError:  # pragma: no cover - supports direct test imports
                from image_utils import strip_data_url

            image_bytes = base64.b64decode(strip_data_url(image_base64), validate=True)
        elif image_url:
            image_bytes = self.download_image(image_url, timeout=timeout)
        else:
            raise RuntimeError("Image response did not contain b64_json, base64, or url.")

        return ImageGenerationResult(
            image_base64=image_base64,
            image_url=image_url,
            image_bytes=image_bytes,
            raw=raw,
        )

    def download_image(self, url: str, *, timeout: int) -> bytes:
        if not is_http_url(url):
            raise RuntimeError("Image URL must use http or https.")

        try:
            with (
                httpx.Client(timeout=timeout, follow_redirects=True) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise RuntimeError("Downloaded image is larger than 50MB.")
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise RuntimeError(f"Image download failed with HTTP {status_code}.") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Image download failed: {exc.__class__.__name__}.") from exc

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2**attempt)
                    last_error = exc
                    continue
                raise RuntimeError(_format_api_error(exc.response, self.config.api_key)) from exc
            except httpx.HTTPError as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    last_error = exc
                    continue
                raise RuntimeError(f"SenseNova request failed: {exc.__class__.__name__}.") from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError("SenseNova response was not valid JSON.") from exc

        raise RuntimeError(f"SenseNova request failed: {last_error.__class__.__name__}.")


def _extract_chat_text(raw: dict[str, Any]) -> str:
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Chat response did not contain choices[0].message.content.") from exc


def normalize_image_size(size: str) -> str:
    return size.split("|", 1)[0].strip()


def _extract_image_payload(raw: dict[str, Any]) -> tuple[str, str]:
    try:
        first = raw["data"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Image response did not contain data[0].") from exc

    if not isinstance(first, dict):
        raise RuntimeError("Image response data[0] was not an object.")

    image_base64 = first.get("b64_json") or first.get("base64") or first.get("image_base64") or ""
    image_url = first.get("url") or ""
    return str(image_base64), str(image_url)


def _format_api_error(response: httpx.Response, api_key: str = "") -> str:
    message = ""
    try:
        body = response.json()
        message = body.get("error", {}).get("message") or body.get("message") or ""
    except Exception:
        message = response.text[:500]

    if message:
        return f"SenseNova API error HTTP {response.status_code}: {_redact(message, api_key)}"
    return f"SenseNova API error HTTP {response.status_code}."


def _redact(value: str, api_key: str = "") -> str:
    redacted = value.replace("Bearer ", "Bearer [REDACTED] ")
    if api_key:
        redacted = redacted.replace(api_key, "[REDACTED]")
    return redacted
