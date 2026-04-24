from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value}") from exc


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of 1/0/true/false/yes/no/on/off, got: {value}"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _top_p_value(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


BASE_URL = os.environ.get("LIGHTLLM_BASE_URL", "http://10.119.22.239:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
MODEL = os.environ.get("LIGHTLLM_MODEL", "neo_chat")
DEFAULT_API_BACKEND = os.environ.get("OPENING_API_BACKEND", "generate")
GENERATE_URLS = os.environ.get(
    "LIGHTLLM_GENERATE_URLS",
    ",".join(
        [
            "http://10.119.24.4:8000/generate",
            "http://10.119.23.156:8000/generate",
            "http://10.119.23.23:8000/generate",
            "http://10.119.17.20:8000/generate",
        ]
    ),
)
DEFAULT_OUTPUT_DIR = Path(os.environ.get("LIGHTLLM_IMAGE_OUT", "./neo_chat_images"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_META_PATH = Path(
    os.environ.get(
        "OPENING_META_PATH",
        "./OpenING-benchmark",
    )
)
DEFAULT_DATA_FILE_NAME = os.environ.get("OPENING_DATA_FILE_NAME", "test_data.jsonl")
DEFAULT_SAVE_DIR = os.environ.get(
    "OPENING_SAVE_DIR",
    "./opening_results",
)
DEFAULT_ENABLE_THINKING = _env_flag("OPENING_ENABLE_THINKING", True)
DEFAULT_GENERATE_PLACEHOLDER_IMAGES = _env_flag("OPENING_GENERATE_PLACEHOLDER_IMAGES", True)
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = _env_int("OPENING_MAX_TOKENS", 4096)
DEFAULT_REQUEST_TIMEOUT = _env_int("OPENING_REQUEST_TIMEOUT", 600)
DEFAULT_GENERATE_MAX_RETRIES = _env_int("OPENING_GENERATE_MAX_RETRIES", 20)
DEFAULT_SEED = _env_int("OPENING_SEED", 200)
DEFAULT_STREAM_SEED = _env_int("OPENING_STREAM_SEED", 500)
DEFAULT_IMAGE_ASPECT_RATIO = os.environ.get("OPENING_IMAGE_ASPECT_RATIO", "16:9")
DEFAULT_IMAGE_SIZE = os.environ.get("OPENING_IMAGE_SIZE", "1K")
DEFAULT_IMAGE_TYPE = os.environ.get("OPENING_IMAGE_TYPE", "jpeg")
DEFAULT_IMAGE_WIDTH = _env_int("OPENING_IMAGE_WIDTH", 1920)
DEFAULT_IMAGE_HEIGHT = _env_int("OPENING_IMAGE_HEIGHT", 1088)
DEFAULT_DEBUG_PREVIEW_CHARS = _env_int("OPENING_DEBUG_PREVIEW_CHARS", 200)
DEFAULT_NUM_SHARDS = _env_int("OPENING_NUM_SHARDS", 1)
DEFAULT_SHARD_INDEX = _env_int("OPENING_SHARD_INDEX", 0)
DEFAULT_RETRY_SHORT_OUTPUTS = _env_int("OPENING_RETRY_SHORT_OUTPUTS", 2)
DEFAULT_PARALLEL_REQUESTS = _env_int("OPENING_PARALLEL_REQUESTS", 4)
DEFAULT_OPENING_STEP_PROMPT_STYLE = os.environ.get("OPENING_STEP_PROMPT_STYLE", "none")
MAX_FAILURE_DETAILS = 20

SYSTEM = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> "
    "and </think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without "
    "reasoning. Do not use tags like <image1>, <image2>; present any images naturally "
    "alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. The "
    "answer may include text, images, or both. Match the user's language in both "
    "reasoning and the final answer."
)
GENERATE_SYSTEM_PROMPT = (
    "Reason step by step and place the thought process within the "
    "<think></think> tags, and provide the final conclusion at the end."
)
GENERATE_SESSION_LOCAL = threading.local()
PLACEHOLDER_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a8dcAAAAASUVORK5CYII="
)


def _chat_url() -> str:
    return f"{BASE_URL.rstrip('/')}/chat/completions"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _generate_session() -> requests.Session:
    session = getattr(GENERATE_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        GENERATE_SESSION_LOCAL.session = session
    return session


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def local_image_to_base64(path: str | Path) -> str:
    path = Path(path)
    with path.open("rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{guess_mime_type(path)};base64,{base64_image}"


def detect_data_url_extension(data_url: str) -> str:
    header = data_url[:64].lower()
    if "image/png" in header:
        return "png"
    if "image/webp" in header:
        return "webp"
    if "image/gif" in header:
        return "gif"
    return "jpg"


def extract_base64_payload(data_url: str) -> str:
    if not data_url.startswith("data:"):
        raise ValueError(f"unsupported image url: {data_url[:80]}...")
    parts = data_url.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid data url: {data_url[:80]}...")
    return parts[1]


def save_data_url_to_file(data_url: str, path: Path) -> None:
    match = re.match(
        r"data:image/(?P<subtype>[\w+.-]+);base64,(?P<b64>.+)",
        data_url,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"unsupported data url prefix: {data_url[:80]}...")

    raw = base64.b64decode(match.group("b64"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    print(f"saved: {path} ({len(raw)} bytes)")


def build_message_image_specs(message: dict[str, Any], prefix: str) -> list[tuple[str, str]]:
    image_specs: list[tuple[str, str]] = []
    for index, item in enumerate(message.get("images") or []):
        if not isinstance(item, dict):
            continue
        data_url = (item.get("image_url") or {}).get("url")
        if not data_url:
            continue
        ext = detect_data_url_extension(data_url)
        image_specs.append((f"{prefix}_{index}.{ext}", data_url))
    return image_specs


def save_images_from_openai_message(
    message: dict[str, Any],
    prefix: str,
    output_dir: str | Path | None = None,
) -> list[str]:
    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    image_names: list[str] = []
    for image_name, data_url in build_message_image_specs(message, prefix):
        save_data_url_to_file(data_url, target_dir / image_name)
        image_names.append(image_name)
    return image_names


def extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def strip_think_block(text: str) -> str:
    raw = text or ""
    matches = re.findall(r"<think>.*?</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    return raw.strip()


def count_image_markers(text: str) -> int:
    if not text:
        return 0
    markers = re.findall(
        r"<(?:img|image)(?:_?\d+)?>|</(?:img|image)(?:_?\d+)?>",
        text,
        flags=re.IGNORECASE,
    )
    return sum(1 for marker in markers if not marker.startswith("</"))


def preview_text_for_log(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def build_image_config(args: argparse.Namespace) -> dict[str, Any]:
    image_config = {
        "aspect_ratio": args.image_aspect_ratio,
        "image_size": args.image_size,
        "image_type": args.image_type,
        "width": args.image_width,
        "height": args.image_height,
        "resolution": f"{args.image_width}x{args.image_height}",
    }
    if args.seed is not None:
        image_config["seed"] = args.seed
    return image_config


def run_nonstream_chat(
    messages: list[dict[str, Any]],
    modalities: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": messages,
        "modalities": modalities,
        "stream": False,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "image_config": build_image_config(args),
    }
    response = requests.post(
        _chat_url(),
        headers=_headers(),
        json=payload,
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    return response.json()


def run_stream_chat(
    messages: list[dict[str, Any]],
    modalities: list[str],
    args: argparse.Namespace,
    *,
    print_content: bool = False,
    output_dir: str | Path | None = None,
    prefix: str = "stream",
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": messages,
        "modalities": modalities,
        "stream": True,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "image_config": build_image_config(args),
        "seed": args.stream_seed,
    }
    response = requests.post(
        _chat_url(),
        headers=_headers(),
        json=payload,
        stream=True,
        timeout=args.request_timeout,
    )
    response.raise_for_status()

    target_dir = Path(output_dir) if output_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    text_parts: list[str] = []
    image_specs: list[tuple[str, str]] = []
    image_entries: list[dict[str, Any]] = []
    image_index = 0

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data = decoded[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        content = delta.get("content")
        if content:
            text_parts.append(content)
            if print_content:
                print(content, end="", flush=True)

        for item in delta.get("images") or []:
            data_url = (item.get("image_url") or {}).get("url")
            if not data_url or not data_url.startswith("data:"):
                continue
            ext = detect_data_url_extension(data_url)
            image_name = f"{prefix}_{image_index}.{ext}"
            image_specs.append((image_name, data_url))
            image_entries.append({"image_url": {"url": data_url}})
            if target_dir is not None:
                save_data_url_to_file(data_url, target_dir / image_name)
            image_index += 1

    return {
        "message": {
            "content": "".join(text_parts),
            "images": image_entries,
        },
        "image_specs": image_specs,
    }


def parse_generate_urls(raw_urls: str) -> list[str]:
    return [url.strip() for url in raw_urls.split(",") if url.strip()]


def rotate_generate_urls(urls: list[str], offset: int) -> list[str]:
    if not urls:
        return []
    start = offset % len(urls)
    return urls[start:] + urls[:start]


def summarize_failures(failures: list[str], limit: int = MAX_FAILURE_DETAILS) -> None:
    if not failures:
        return
    print(f"\nFailure summary: total={len(failures)}")
    for detail in failures[:limit]:
        print(f"- {detail}")
    remaining = len(failures) - limit
    if remaining > 0:
        print(f"... and {remaining} more failures")


def convert_message_content_to_generate_prompt(
    content: Any,
    image_payloads: list[str],
) -> str:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return str(content).strip() if content is not None else ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = (item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif item_type == "image_url":
            data_url = (item.get("image_url") or {}).get("url")
            if not data_url:
                continue
            image_payloads.append(extract_base64_payload(data_url))
            parts.append("<img></img>")
    return "\n".join(parts).strip()


def build_generate_query_from_messages(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    system_texts: list[str] = []
    user_parts: list[str] = []
    image_payloads: list[str] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_text = extract_message_text(message).strip()
            if system_text:
                system_texts.append(system_text)
            continue
        if role != "user":
            continue
        prompt = convert_message_content_to_generate_prompt(content, image_payloads)
        if prompt:
            user_parts.append(prompt)

    system_prompt = "\n\n".join(part for part in system_texts if part)
    if not system_prompt:
        system_prompt = GENERATE_SYSTEM_PROMPT

    user_prompt = "\n\n".join(part for part in user_parts if part).strip()
    if not user_prompt:
        user_prompt = "Please answer the task."

    return (
        f"<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        image_payloads,
    )


def build_generate_payload(
    query: str,
    image_payloads: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_config = build_image_config(args)
    parameters: dict[str, Any] = {
        "max_new_tokens": args.max_tokens,
        "stop_sequences": [" <|endoftext|>", " <|im_start|>", " <|im_end|>"],
        "add_output_think_tokens": args.enable_thinking,
        "image_aspect_ratio": args.image_aspect_ratio,
        "image_size": args.image_size,
        "image_type": args.image_type,
        "width": args.image_width,
        "height": args.image_height,
        "resolution": f"{args.image_width}x{args.image_height}",
    }
    if args.temperature > 0:
        parameters["do_sample"] = True
        parameters["temperature"] = args.temperature
        parameters["top_p"] = args.top_p
    else:
        parameters["do_sample"] = False

    return {
        "inputs": query,
        "parameters": parameters,
        "image_config": image_config,
        "multimodal_params": {
            "images": [{"type": "base64", "data": image_b64} for image_b64 in image_payloads],
        },
    }


def extract_generate_text(response_json: Any) -> str:
    if isinstance(response_json, dict) and "generated_text" in response_json:
        generated = response_json["generated_text"]
        if isinstance(generated, str):
            return generated
        if isinstance(generated, list):
            parts: list[str] = []
            for item in generated:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", json.dumps(item, ensure_ascii=False)))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(generated)

    if isinstance(response_json, list) and response_json:
        first = response_json[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("generated_text", json.dumps(first, ensure_ascii=False)))
        return str(first)

    if isinstance(response_json, dict):
        return json.dumps(response_json, ensure_ascii=False)
    return str(response_json)


def run_generate_chat(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    generate_urls: list[str] | None = None,
) -> dict[str, Any]:
    query, image_payloads = build_generate_query_from_messages(messages)
    payload = build_generate_payload(query, image_payloads, args)
    wait_time = 3.0
    urls = list(generate_urls) if generate_urls is not None else parse_generate_urls(args.generate_urls)
    if not urls:
        raise ValueError("generate backend requires at least one --generate_urls endpoint")

    last_error = "unknown error"
    for attempt in range(args.generate_max_retries):
        if generate_urls is None:
            url = random.choice(urls)
        else:
            url = urls[attempt % len(urls)]
        try:
            response = _generate_session().post(url, json=payload, timeout=args.request_timeout)
            if response.status_code == 200:
                try:
                    response_json = response.json()
                except ValueError as exc:
                    last_error = f"invalid JSON from {url}: {exc}; body={response.text[:300]}"
                    continue
                return {
                    "message": {
                        "content": extract_generate_text(response_json),
                        "images": [],
                    },
                    "image_specs": [],
                }
            if response.status_code == 400:
                raise RuntimeError(f"generate backend fatal 400: {response.text[:300]}")
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.exceptions.Timeout:
            last_error = f"timeout calling {url}"
        except requests.exceptions.RequestException as error:
            last_error = f"request error calling {url}: {error}"

        if attempt + 1 < args.generate_max_retries:
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 60)

    raise RuntimeError(f"generate backend failed after {args.generate_max_retries} retries: {last_error}")


def run_interleave_chat(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    print_content: bool = False,
    output_dir: str | Path | None = None,
    prefix: str = "stream",
    generate_urls_override: list[str] | None = None,
) -> dict[str, Any]:
    if args.api_backend == "generate":
        result = run_generate_chat(messages, args, generate_urls=generate_urls_override)
        cleaned_text = strip_think_block(extract_message_text(result["message"]))
        result["message"]["content"] = cleaned_text
        if args.generate_placeholder_images:
            image_count = count_image_markers(cleaned_text)
            result["image_specs"] = [
                (f"{prefix}_{index}.png", PLACEHOLDER_PNG_DATA_URL)
                for index in range(image_count)
            ]
            result["message"]["images"] = [
                {"image_url": {"url": PLACEHOLDER_PNG_DATA_URL}}
                for _ in range(image_count)
            ]
        if print_content:
            print(extract_message_text(result["message"]), end="", flush=True)
        return result

    return run_stream_chat(
        messages,
        ["text", "image"],
        args,
        print_content=print_content,
        output_dir=output_dir,
        prefix=prefix,
    )


def chat_t2i(user_text: str, args: argparse.Namespace) -> None:
    data = run_nonstream_chat(
        [{"role": "user", "content": user_text}],
        ["image"],
        args,
    )
    print("--- non-stream raw (truncated) ---")
    print(json.dumps(data, ensure_ascii=False)[:2000])
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    print("--- assistant content (含 <image> 占位) ---")
    print(extract_message_text(message))
    save_images_from_openai_message(message, prefix="nonstream", output_dir=args.output_dir)


def chat_it2i(user_text: str, image_path: str, args: argparse.Namespace) -> None:
    data = run_nonstream_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": local_image_to_base64(image_path)}},
                ],
            }
        ],
        ["image"],
        args,
    )
    print("--- non-stream raw (truncated) ---")
    print(json.dumps(data, ensure_ascii=False)[:2000])
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    print("--- assistant content (含 <image> 占位) ---")
    print(extract_message_text(message))
    save_images_from_openai_message(message, prefix="nonstream", output_dir=args.output_dir)


def chat_interleave(user_text: str, image_path: str, args: argparse.Namespace) -> None:
    data = run_nonstream_chat(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": local_image_to_base64(image_path)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        ["text", "image"],
        args,
    )
    print("--- non-stream raw (truncated) ---")
    print(json.dumps(data, ensure_ascii=False)[:2000])
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    print("--- assistant content (含 <image> 占位) ---")
    print(extract_message_text(message))
    save_images_from_openai_message(message, prefix="nonstream", output_dir=args.output_dir)


def chat_stream_interleave(user_text: str, args: argparse.Namespace) -> None:
    stream_result = run_interleave_chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ],
        args,
        print_content=True,
        output_dir=args.output_dir,
        prefix="stream",
    )
    if args.debug_response_preview:
        print()
        print(
            f"[STREAM DEBUG] response_preview="
            f"{preview_text_for_log(extract_message_text(stream_result['message']), args.debug_preview_chars)!r} "
            f"images={len(stream_result['image_specs'])}"
        )


def clean_opening_text(text: str) -> str:
    cleaned = (text or "").replace("<BEGIN>", "")
    cleaned = re.sub(r"</?img_?\d+>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?image_?\d+>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<IMG>", "").replace("</IMG>", "")
    cleaned = cleaned.replace("<image>", "").replace("<IMAGE>", "")
    return cleaned.strip()


def build_opening_prompt_prefix(gt_out_step: int, step_prompt_style: str) -> str:
    if step_prompt_style == "can_be":
        return f"The number of generated text-image pairs can be {gt_out_step}: "
    if step_prompt_style == "must_exact":
        return (
            f"The number of generated text-image pairs must be exactly {gt_out_step}. "
            f"Please generate exactly {gt_out_step} interleaved text-image pairs: "
        )
    return ""


def resolve_data_path(meta_path: str | Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return candidate

    meta_root = Path(meta_path)
    normalized_rel_path = rel_path[2:] if rel_path.startswith("./") else rel_path
    candidate_paths = [
        meta_root / rel_path,
        meta_root / normalized_rel_path,
        meta_root / "OpenING" / rel_path,
        meta_root / "OpenING" / normalized_rel_path,
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return candidate_paths[0]


def load_opening_data(data_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, list[Any]]]]:
    real_data_list: list[dict[str, Any]] = []
    io_data_list: list[dict[str, list[Any]]] = []
    with open(data_path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid OpenING JSONL at {data_path}:{line_no}: {exc}") from exc
            real_data_list.append(record)
            io_data_list.append(parse_opening_record(record))
    return real_data_list, io_data_list


def parse_opening_record(record: dict[str, Any]) -> dict[str, list[Any]]:
    input_text: list[str] = []
    input_image: list[str | None] = []
    output_text: list[str] = []
    output_image: list[str | None] = []

    for step in record["conversations"][0]["input"]:
        input_text.append((step.get("text") or "").strip())
        input_image.append(step.get("image"))

    for step in record["conversations"][1]["output"]:
        output_text.append((step.get("text") or "").strip())
        output_image.append(step.get("image"))

    return {
        "input_text": input_text,
        "input_image": input_image,
        "output_text": output_text,
        "output_image": output_image,
    }


def build_opening_user_content(
    io_data: dict[str, list[Any]],
    meta_path: str | Path,
    gt_out_step: int,
    step_prompt_style: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    prefix = build_opening_prompt_prefix(gt_out_step, step_prompt_style)
    missing_images: list[str] = []

    for index, raw_text in enumerate(io_data["input_text"]):
        cleaned = clean_opening_text(raw_text)
        if index == 0 and prefix:
            cleaned = prefix + cleaned if cleaned else prefix.strip()
        if cleaned:
            content.append({"type": "text", "text": cleaned})

        image_path = io_data["input_image"][index] if index < len(io_data["input_image"]) else None
        resolved_path = resolve_data_path(meta_path, image_path)
        if resolved_path and resolved_path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": local_image_to_base64(resolved_path)},
                }
            )
        elif image_path:
            missing_images.append(str(resolved_path))

    if missing_images:
        raise FileNotFoundError("missing input images: " + ", ".join(missing_images))

    if not content:
        content.append({"type": "text", "text": prefix.strip() or "Please answer the task."})
    return content


def split_generated_text(text: str) -> list[str]:
    text = strip_think_block((text or "").replace("**", "").strip())
    if not text:
        return [""]

    delimiter_pattern = r"<IMG>|<image>|<image_?\d+>|</image_?\d+>|<IMG_?\d+>|</IMG_?\d+>"
    parts = [
        part.strip()
        for part in re.split(delimiter_pattern, text, flags=re.IGNORECASE)
        if part.strip()
    ]
    return parts if parts else [text]


def normalize_output_steps(
    text_steps: list[str],
    image_names: list[str],
    gt_out_step: int,
) -> tuple[list[str], list[str | None]]:
    item_count = min(gt_out_step, max(len(text_steps), len(image_names), 1))
    normalized_text: list[str] = []
    normalized_images: list[str | None] = []

    for index in range(item_count):
        normalized_text.append(text_steps[index] if index < len(text_steps) else "")
        normalized_images.append(image_names[index] if index < len(image_names) else None)

    return normalized_text, normalized_images


def output_is_complete(
    output_text: list[str],
    output_images: list[str | None],
    gt_output_steps: list[dict[str, Any]],
) -> bool:
    required_steps = len(gt_output_steps)
    required_images = sum(1 for step in gt_output_steps if step.get("image"))
    if len(output_text) < required_steps:
        return False
    available_images = sum(1 for image_name in output_images[:required_steps] if image_name is not None)
    return available_images >= required_images


def atomic_save_json(data: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_name(f"{json_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)
    os.replace(tmp_path, json_path)


def save_opening_results(
    output_dir: str | Path,
    real_data_item: dict[str, Any],
    generated_text_list: list[str],
    image_out_list: list[str | None],
) -> None:
    output_dir = Path(output_dir)
    data_uid = real_data_item["total_uid"]
    json_path = output_dir / f"{data_uid}.json"

    saved_json = copy.deepcopy(real_data_item)
    if "conversations" in saved_json and len(saved_json["conversations"]) > 1:
        saved_json["conversations"][1]["output"] = []

    for index in range(max(len(generated_text_list), len(image_out_list))):
        output_item = {
            "text": generated_text_list[index].strip() if index < len(generated_text_list) else "",
            "image": image_out_list[index] if index < len(image_out_list) else None,
        }
        saved_json["conversations"][1]["output"].append(output_item)

    atomic_save_json(saved_json, json_path)


def run_opening_item(
    sample_index: int,
    real_data: dict[str, Any],
    io_data: dict[str, list[Any]],
    args: argparse.Namespace,
    output_dir: Path,
    base_generate_urls: list[str],
) -> tuple[str, str]:
    data_uid = real_data["total_uid"]
    gt_output_steps = real_data["conversations"][1]["output"]
    gt_out_step = len(gt_output_steps)
    user_content = build_opening_user_content(
        io_data,
        args.meta_path,
        gt_out_step,
        args.opening_step_prompt_style,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]
    if args.debug_request_preview:
        preview_items = []
        for item in user_content:
            if item.get("type") == "text":
                preview_items.append(
                    {
                        "type": "text",
                        "text": preview_text_for_log(item.get("text", ""), args.debug_preview_chars),
                    }
                )
            elif item.get("type") == "image_url":
                preview_items.append({"type": "image_url", "image_url": {"url": "<base64_image_omitted>"}})
            else:
                preview_items.append({"type": item.get("type", "unknown")})
        print(f"[OPENING DEBUG] UID={data_uid} request_preview={json.dumps(preview_items, ensure_ascii=False)}")

    best_output_text: list[str] = []
    best_output_images: list[str | None] = []
    best_image_specs: list[tuple[str, str]] = []
    best_score = -1
    max_attempts = max(1, args.retry_short_outputs + 1)
    generate_urls_override = (
        rotate_generate_urls(base_generate_urls, sample_index) if args.api_backend == "generate" else None
    )

    try:
        print(f"UID: {data_uid}, gt_out_step: {gt_out_step}")
        for attempt in range(max_attempts):
            stream_result = run_interleave_chat(
                messages,
                args,
                print_content=False,
                output_dir=None,
                prefix=f"{data_uid}-o",
                generate_urls_override=generate_urls_override,
            )
            message = stream_result["message"]
            raw_message_text = extract_message_text(message)
            text_steps = split_generated_text(raw_message_text)
            image_specs = list(stream_result["image_specs"])
            if args.api_backend == "generate" and args.generate_placeholder_images:
                required_images = sum(1 for step in gt_output_steps if step.get("image"))
                while len(image_specs) < required_images:
                    image_specs.append(
                        (
                            f"{data_uid}-o_{len(image_specs)}.png",
                            PLACEHOLDER_PNG_DATA_URL,
                        )
                    )
            if args.debug_response_preview:
                print(
                    f"[OPENING DEBUG] UID={data_uid} attempt={attempt + 1}/{max_attempts} "
                    f"response_preview={preview_text_for_log(raw_message_text, args.debug_preview_chars)!r} "
                    f"text_steps={len(text_steps)} images={len(image_specs)}"
                )
            image_names = [name for name, _ in image_specs[:gt_out_step]]
            candidate_text, candidate_images = normalize_output_steps(
                text_steps,
                image_names,
                gt_out_step,
            )
            candidate_score = min(len(text_steps), gt_out_step) + min(len(image_specs), gt_out_step)

            if candidate_score > best_score:
                best_output_text = candidate_text
                best_output_images = candidate_images
                best_image_specs = image_specs[:gt_out_step]
                best_score = candidate_score

            if output_is_complete(candidate_text, candidate_images, gt_output_steps):
                break

            if attempt + 1 < max_attempts:
                print(
                    f"UID: {data_uid} attempt {attempt + 1}/{max_attempts} "
                    f"输出不足: text={len(text_steps)}, images={len(image_specs)}, "
                    f"expected={gt_out_step}; retry"
                )

        for image_name, data_url in best_image_specs:
            save_data_url_to_file(data_url, output_dir / image_name)

        save_opening_results(output_dir, real_data, best_output_text, best_output_images)
        print(f"Processed and saved results for UID: {data_uid}")
        return ("processed", data_uid)
    except Exception as error:
        print(f"错误: UID {data_uid} 推理失败: {error}")
        return ("failed", data_uid)


def resolve_parallel_requests(args: argparse.Namespace) -> int:
    if args.parallel_requests is not None:
        return max(1, args.parallel_requests)
    if args.api_backend == "generate":
        return max(1, len(parse_generate_urls(args.generate_urls)))
    return 1


def run_opening_generation(args: argparse.Namespace) -> None:
    data_path = Path(args.meta_path) / args.data_file_name
    real_data_list, io_data_list = load_opening_data(data_path)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sharded_items: list[tuple[int, dict[str, Any], dict[str, list[Any]]]] = []
    for sample_index, (real_data, io_data) in enumerate(zip(real_data_list, io_data_list)):
        if sample_index % args.num_shards != args.shard_index:
            continue
        sharded_items.append((sample_index, real_data, io_data))

    if args.limit is not None:
        sharded_items = sharded_items[: args.limit]

    base_generate_urls = parse_generate_urls(args.generate_urls) if args.api_backend == "generate" else []
    parallel_requests = resolve_parallel_requests(args)

    print(f"OpenING 数据路径: {data_path}")
    print(f"处理数据样本总数: {len(real_data_list)}")
    print(
        f"分片: shard_index={args.shard_index}, num_shards={args.num_shards}, "
        f"当前分片样本数={len(sharded_items)}"
    )
    print(f"输出目录: {output_dir}")
    print(f"并发请求数: {parallel_requests}")
    if args.api_backend == "generate":
        print(f"generate endpoints: {len(base_generate_urls)}")

    pending_items: list[tuple[int, dict[str, Any], dict[str, list[Any]]]] = []
    skipped_count = 0
    failures: list[str] = []
    for sample_index, real_data, io_data in sharded_items:
        data_uid = real_data["total_uid"]
        json_path = output_dir / f"{data_uid}.json"
        if json_path.exists() and not args.overwrite:
            print(f"跳过 UID {data_uid}: 结果已存在")
            skipped_count += 1
            continue
        pending_items.append((sample_index, real_data, io_data))

    processed_count = 0
    failed_count = 0

    # OpenING 样本彼此独立，并发处理可以把多个 generate endpoint 同时利用起来。
    if parallel_requests == 1 or len(pending_items) <= 1:
        for sample_index, real_data, io_data in pending_items:
            status, _ = run_opening_item(
                sample_index,
                real_data,
                io_data,
                args,
                output_dir,
                base_generate_urls,
            )
            if status == "processed":
                processed_count += 1
            else:
                failed_count += 1
                failures.append(f"UID {real_data['total_uid']}: failed")
    else:
        with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
            futures = [
                executor.submit(
                    run_opening_item,
                    sample_index,
                    real_data,
                    io_data,
                    args,
                    output_dir,
                    base_generate_urls,
                )
                for sample_index, real_data, io_data in pending_items
            ]
            for future in as_completed(futures):
                try:
                    status, _ = future.result()
                except Exception as error:
                    failed_count += 1
                    detail = f"并发任务异常: {error}"
                    failures.append(detail)
                    print(f"错误: {detail}")
                    continue
                if status == "processed":
                    processed_count += 1
                else:
                    failed_count += 1
                    failures.append("并发任务返回 failed")

    print(
        f"完成统计: processed={processed_count}, failed={failed_count}, skipped={skipped_count}, "
        f"pending={len(pending_items)}"
    )
    summarize_failures(failures)
    if failed_count > 0:
        raise RuntimeError(f"OpenING failed for {failed_count} item(s)")


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("--image_width and --image_height must be > 0")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, --num_shards)")
    if args.parallel_requests is not None and args.parallel_requests <= 0:
        raise ValueError("--parallel_requests must be > 0")
    if args.retry_short_outputs < 0:
        raise ValueError("--retry_short_outputs must be >= 0")
    if args.debug_preview_chars <= 0:
        raise ValueError("--debug_preview_chars must be > 0")
    if args.api_backend == "generate":
        if not parse_generate_urls(args.generate_urls):
            raise ValueError("--generate_urls must contain at least one valid endpoint")
        if args.mode in {"t2i", "it2i", "interleave"}:
            raise ValueError("--api_backend generate 目前仅支持 --mode stream_interleave 或 --mode opening")
    if args.mode == "opening":
        data_path = Path(args.meta_path) / args.data_file_name
        if not data_path.is_file():
            raise FileNotFoundError(f"OpenING data file not found: {data_path}")
    if args.mode in {"it2i", "interleave"}:
        if not args.image_path:
            raise ValueError(f"--mode {args.mode} 需要指定 --image_path")
        if not Path(args.image_path).is_file():
            raise FileNotFoundError(f"input image not found: {args.image_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference API client with OpenING benchmark support.")
    parser.add_argument(
        "--mode",
        choices=["t2i", "it2i", "interleave", "stream_interleave", "opening"],
        default="stream_interleave",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--user_text", default="教我怎么腌制腊肉")
    parser.add_argument("--image_path", default=None)
    parser.set_defaults(enable_thinking=DEFAULT_ENABLE_THINKING)
    parser.add_argument("--enable_thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--disable_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--temperature", type=_non_negative_float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=_top_p_value, default=DEFAULT_TOP_P)
    parser.add_argument("--max_tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--request_timeout", type=_positive_int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--api_backend", choices=["openai", "generate"], default=DEFAULT_API_BACKEND)
    parser.add_argument("--generate_urls", default=GENERATE_URLS)
    parser.add_argument("--generate_max_retries", type=_positive_int, default=DEFAULT_GENERATE_MAX_RETRIES)
    parser.set_defaults(generate_placeholder_images=DEFAULT_GENERATE_PLACEHOLDER_IMAGES)
    parser.add_argument(
        "--generate-placeholder-images",
        dest="generate_placeholder_images",
        action="store_true",
        help="Generate 1x1 placeholder images when the generate backend returns image markers without image payloads.",
    )
    parser.add_argument(
        "--disable-generate-placeholder-images",
        dest="generate_placeholder_images",
        action="store_false",
        help="Disable placeholder image synthesis.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stream_seed", type=int, default=DEFAULT_STREAM_SEED)
    parser.add_argument("--image_aspect_ratio", default=DEFAULT_IMAGE_ASPECT_RATIO)
    parser.add_argument("--image_size", default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image_type", default=DEFAULT_IMAGE_TYPE)
    parser.add_argument("--image_width", type=_positive_int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image_height", type=_positive_int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--debug_request_preview", action="store_true")
    parser.add_argument("--debug_response_preview", action="store_true")
    parser.add_argument("--debug_preview_chars", type=_positive_int, default=DEFAULT_DEBUG_PREVIEW_CHARS)

    parser.add_argument("--meta_path", default=str(DEFAULT_META_PATH))
    parser.add_argument("--data_file_name", default=DEFAULT_DATA_FILE_NAME)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--num_shards", type=_positive_int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--shard_index", type=_non_negative_int, default=DEFAULT_SHARD_INDEX)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--parallel_requests", type=_positive_int, default=DEFAULT_PARALLEL_REQUESTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry_short_outputs", type=_non_negative_int, default=DEFAULT_RETRY_SHORT_OUTPUTS)
    parser.add_argument(
        "--opening_step_prompt_style",
        choices=["none", "can_be", "must_exact"],
        default=DEFAULT_OPENING_STEP_PROMPT_STYLE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.mode == "opening":
        run_opening_generation(args)
        return 0

    if args.mode == "t2i":
        chat_t2i(args.user_text, args)
        return 0

    if args.mode == "it2i":
        chat_it2i(args.user_text, args.image_path, args)
        return 0

    if args.mode == "interleave":
        chat_interleave(args.user_text, args.image_path, args)
        return 0

    chat_stream_interleave(args.user_text, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
