from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://token.sensenova.cn/v1"
API_KEY_ENV = "SN_API_KEY"
BASE_URL_ENV = "SN_BASE_URL"


@dataclass(frozen=True)
class SenseNovaConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL


def load_config(*, load_env_file: bool = True) -> SenseNovaConfig:
    if load_env_file:
        load_dotenv()

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {API_KEY_ENV}. Set it in your environment or in a local .env file.")

    base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url:
        base_url = DEFAULT_BASE_URL

    return SenseNovaConfig(api_key=api_key, base_url=base_url)
