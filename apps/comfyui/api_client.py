from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    from .config import SenseNovaConfig, load_config
    from .image_utils import MAX_IMAGE_BYTES, is_http_url, is_supported_vision_image_url
except ImportError:  # pragma: no cover - supports direct test imports
    from config import SenseNovaConfig, load_config
    from image_utils import MAX_IMAGE_BYTES, is_http_url, is_supported_vision_image_url

CHAT_MODELS = (
    "sensenova-6.7-flash-lite",
    "deepseek-v4-flash",
    "glm-5.2",
    "sensenova-6.8-flash-lite",
)
MULTIMODAL_CHAT_MODELS = (
    "sensenova-6.7-flash-lite",
    "sensenova-6.8-flash-lite",
)
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

_API_REQUEST_WORKER_PATH = Path(__file__).with_name("api_request_worker.py")
_API_REQUEST_POLL_INTERVAL = 0.1
_API_WORKER_TERMINATE_TIMEOUT = 5.0
_RETRYABLE_POST_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_POST_TRANSPORT_ERRORS = {"ConnectError", "ConnectTimeout"}


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


@dataclass(frozen=True)
class ModelCatalog:
    chat_models: tuple[str, ...]
    multimodal_chat_models: tuple[str, ...]
    image_models: tuple[str, ...]


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
        image_urls: list[str] | None = None,
    ) -> ChatResult:
        if not model.strip():
            raise RuntimeError("Chat model cannot be empty.")
        if not text.strip():
            raise RuntimeError("Chat text cannot be empty.")

        user_content: str | list[dict[str, Any]] = text
        if image_urls:
            for image_url in image_urls:
                if not is_supported_vision_image_url(image_url):
                    raise RuntimeError("Chat image URLs must use http(s) or base64 image data URLs.")
            user_content = [
                {"type": "text", "text": text},
                *({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls),
            ]

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
        if not model.strip():
            raise RuntimeError("Image model cannot be empty.")
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

    def list_models(self, *, timeout: int = 5) -> ModelCatalog:
        raw = self._get_json("/models", timeout=timeout)
        return _extract_model_catalog(raw)

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

        for attempt in range(3):
            result = _post_json_interruptibly(url, headers, payload, timeout=timeout)
            result_kind = result.get("kind")
            if result_kind == "transport_error":
                error_type = str(result.get("error_type") or "HTTPError")
                if error_type in _RETRYABLE_POST_TRANSPORT_ERRORS and attempt < 2:
                    _sleep_interruptibly(2**attempt)
                    continue
                raise RuntimeError(f"SenseNova request failed: {error_type}.")
            if result_kind != "response":
                error_type = str(result.get("error_type") or "WorkerError")
                raise RuntimeError(f"SenseNova API request worker failed: {error_type}.")

            status_code = result.get("status_code")
            if not isinstance(status_code, int):
                raise RuntimeError("SenseNova request worker returned an invalid HTTP status.")
            if status_code in _RETRYABLE_POST_STATUS_CODES and attempt < 2:
                _sleep_interruptibly(2**attempt)
                continue

            response_body = result.get("body")
            if result.get("json_valid"):
                response = httpx.Response(status_code, json=response_body)
            else:
                response = httpx.Response(status_code, text=str(result.get("body_text") or ""))
            if not 200 <= status_code < 300:
                raise RuntimeError(_format_api_error(response, self.config.api_key))
            if not result.get("json_valid"):
                raise RuntimeError("SenseNova response was not valid JSON.")
            return response_body

        raise RuntimeError("SenseNova request failed after retries.")

    def _get_json(self, path: str, *, timeout: int) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(_format_api_error(exc.response, self.config.api_key)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"SenseNova request failed: {exc.__class__.__name__}.") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("SenseNova response was not valid JSON.") from exc


def _extract_chat_text(raw: dict[str, Any]) -> str:
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Chat response did not contain choices[0].message.content.") from exc


def _extract_model_catalog(raw: dict[str, Any]) -> ModelCatalog:
    data = raw.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Model response did not contain a data list.")

    chat_models: list[str] = []
    multimodal_chat_models: list[str] = []
    image_models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not (model_id := model_id.strip()):
            continue
        input_modalities = item.get("input_modalities")
        output_modalities = item.get("output_modalities")
        if not isinstance(output_modalities, list):
            continue
        if "image" in output_modalities and model_id not in image_models:
            image_models.append(model_id)
        if not isinstance(input_modalities, list):
            continue
        if "text" in input_modalities and "text" in output_modalities and model_id not in chat_models:
            chat_models.append(model_id)
        if "image" in input_modalities and "text" in output_modalities and model_id not in multimodal_chat_models:
            multimodal_chat_models.append(model_id)
    return ModelCatalog(
        chat_models=tuple(chat_models),
        multimodal_chat_models=tuple(multimodal_chat_models),
        image_models=tuple(image_models),
    )


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


def _throw_if_comfyui_interrupted() -> None:
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()


def _terminate_api_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_API_WORKER_TERMINATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _sleep_interruptibly(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        _throw_if_comfyui_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_API_REQUEST_POLL_INTERVAL, remaining))


def _post_json_interruptibly(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    _throw_if_comfyui_interrupted()
    with tempfile.TemporaryDirectory(prefix="sensenova-api-request-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        result_path = Path(temp_dir) / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                    "timeout": {
                        "connect": 10.0,
                        "read": float(timeout),
                        "write": 30.0,
                        "pool": 10.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        _throw_if_comfyui_interrupted()
        process = subprocess.Popen([sys.executable, str(_API_REQUEST_WORKER_PATH), str(request_path), str(result_path)])
        try:
            while process.poll() is None:
                _throw_if_comfyui_interrupted()
                time.sleep(_API_REQUEST_POLL_INTERVAL)
            _throw_if_comfyui_interrupted()
        except BaseException:
            _terminate_api_worker(process)
            raise

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"SenseNova API request worker exited with code {process.returncode} without a valid result."
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError("SenseNova API request worker returned an invalid result.")
        return result
