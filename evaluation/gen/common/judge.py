from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


def _encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    raw = path.read_bytes()
    return f"data:image/{suffix};base64,{base64.b64encode(raw).decode('utf-8')}"


class JudgeClient:
    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 240,
        max_retries: int = 6,
        retry_backoff_base: float = 2.0,
        retry_backoff_cap: float = 60.0,
    ) -> None:
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("JUDGE_MODEL")
        if not self.api_base or not self.model:
            raise ValueError("Judge API is not configured. Set JUDGE_API_BASE and JUDGE_MODEL.")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_cap = retry_backoff_cap

    def _request_chat_completion(
        self,
        *,
        image_path: Path,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ],
            }
        )
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                return str(text)
            except HTTPError as exc:
                should_retry = exc.code == 429 or 500 <= exc.code < 600
                if not should_retry or attempt >= self.max_retries:
                    raise
            except URLError:
                if attempt >= self.max_retries:
                    raise
            except TimeoutError:
                if attempt >= self.max_retries:
                    raise
            delay = min(self.retry_backoff_cap, self.retry_backoff_base**attempt)
            time.sleep(delay)
        raise RuntimeError("Judge request failed after retries.")

    def judge_image_text(
        self,
        *,
        image_path: Path,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        return self._request_chat_completion(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
