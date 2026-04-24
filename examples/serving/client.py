from __future__ import annotations

import argparse
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://0.0.0.0:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_MODEL = "sensenova-u1"

INTERLEAVE_SYSTEM_PROMPT = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. "
    "Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in both reasoning "
    "and the final answer."
)

GENERATION_SYSTEM_PROMPT = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)
"""
    _aspect_ratio_to_resolution: ClassVar[dict] = {
        "1:1": {"1K": (1024, 1024), "1.5K": (1536, 1536), "2K": (2048, 2048)},
        "16:9": {"1.5K": (2048, 1152), "2K": (2720, 1536)},
        "9:16": {"1.5K": (1152, 2048), "2K": (1536, 2720)},
        "3:2": {"1.5K": (1888, 1248), "2K": (2496, 1664)},
        "2:3": {"1.5K": (1248, 1888), "2K": (1664, 2496)},
        "4:3": {"1.5K": (1760, 1312), "2K": (2368, 1760)},
        "3:4": {"1.5K": (1312, 1760), "2K": (1760, 2368)},
        "1:2": {"1.5K": (1088, 2144), "2K": (1440, 2880)},
        "2:1": {"1.5K": (2144, 1088), "2K": (2880, 1440)},
        "1:3": {"1.5K": (864, 2592), "2K": (1152, 3456)},
        "3:1": {"1.5K": (2592, 864), "2K": (3456, 1152)},
    }
"""
IMAGE_CONFIG_DEFAULT = {
    "aspect_ratio": "16:9",
    "image_size": "2K",
    "image_type": "jpeg",
    "seed": 42,
    # If set to True, the generated image will have the same resolution as the input image.
    # If set to False, the resolution of the generated image will be determined by the image_size and aspect_ratio.
    "dynamic_resolution": True,
    # if you want to determine the resolution of the generated image by yourself, set the height and width.
    # the default value is -1.
    "height": -1,
    "width": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible API test client for LightLLM + LightX2V.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["t2i", "it2i", "interleave", "vqa"],
        help="Test mode. If omitted, the script asks interactively.",
    )
    parser.add_argument("--prompt", required=True, help="User prompt. If omitted, ask interactively.")
    parser.add_argument(
        "--image_path",
        default=None,
        help="Input image path for it2i / interleave.",
    )
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        default="./api_test_outputs",
        help="Directory to save generated images and raw responses.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass chat_template_kwargs.enable_thinking to backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=IMAGE_CONFIG_DEFAULT["seed"],
        help="Sampling seed for image config / streaming request.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=IMAGE_CONFIG_DEFAULT["aspect_ratio"],
        help="Aspect ratio for generated image (e.g. 16:9, 1:1).",
    )
    parser.add_argument(
        "--image-size",
        default=IMAGE_CONFIG_DEFAULT["image_size"],
        help="Image size preset for generation (e.g. 1.5K, 2K).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=IMAGE_CONFIG_DEFAULT["height"],
        help="Manual image height. Use with --width; keep -1 for auto resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=IMAGE_CONFIG_DEFAULT["width"],
        help="Manual image width. Use with --height; keep -1 for auto resolution.",
    )
    return parser.parse_args()


def build_image_config(args: argparse.Namespace) -> dict[str, Any]:
    image_config = {
        **IMAGE_CONFIG_DEFAULT,
        "aspect_ratio": args.aspect_ratio,
        "image_size": args.image_size,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
    }
    if args.height > 0 and args.width > 0:
        image_config["dynamic_resolution"] = False
    return image_config


def local_image_to_data_url(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def save_data_url_to_file(data_url: str, path: Path) -> None:
    matched = re.match(r"data:image/(?P<subtype>[\w+.-]+);base64,(?P<b64>.+)", data_url, re.DOTALL)
    if not matched:
        raise ValueError(f"unsupported data url prefix: {data_url[:80]}...")
    raw = base64.b64decode(matched.group("b64"))
    path.write_bytes(raw)
    print(f"[saved] {path} ({len(raw)} bytes)")


def save_images_from_message(message: dict[str, Any], out_dir: Path, prefix: str) -> None:
    images = message.get("images") or []
    for idx, item in enumerate(images):
        if not isinstance(item, dict):
            continue
        image_url = (item.get("image_url") or {}).get("url", "")
        if not image_url.startswith("data:image/"):
            continue
        ext = "png"
        if image_url.startswith("data:image/jpeg") or image_url.startswith("data:image/jpg"):
            ext = "jpg"
        elif image_url.startswith("data:image/webp"):
            ext = "webp"
        save_data_url_to_file(image_url, out_dir / f"{prefix}_{idx}.{ext}")


def build_client(base_url: str, api_key: str) -> tuple[str, dict[str, str]]:
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return chat_url, headers


def run_t2i(args: argparse.Namespace, out_dir: Path, timestamp: str) -> None:
    chat_url, headers = build_client(args.url, args.api_key)
    image_config = build_image_config(args)
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}, {"role": "user", "content": args.prompt}],
        "modalities": ["image"],
        "stream": False,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "image_config": image_config,
    }
    response = requests.post(chat_url, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    raw_path = out_dir / f"{timestamp}_t2i_response.json"
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {raw_path}")

    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    print("\n--- assistant content ---")
    print(message.get("content", ""))
    save_images_from_message(message, out_dir=out_dir, prefix=f"{timestamp}_t2i")


def run_it2i(args: argparse.Namespace, out_dir: Path, timestamp: str) -> None:
    chat_url, headers = build_client(args.url, args.api_key)
    assert args.image_path is not None, "image_path is required"
    image_config = build_image_config(args)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": local_image_to_data_url(args.image)}},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ],
        "modalities": ["image"],
        "stream": False,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "image_config": image_config,
    }
    response = requests.post(chat_url, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    raw_path = out_dir / f"{timestamp}_it2i_response.json"
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {raw_path}")

    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    print("\n--- assistant content ---")
    print(message.get("content", ""))
    save_images_from_message(message, out_dir=out_dir, prefix=f"{timestamp}_it2i")


def run_interleave_stream(args: argparse.Namespace, out_dir: Path, timestamp: str) -> None:
    chat_url, headers = build_client(args.url, args.api_key)
    image_config = build_image_config(args)
    content = []
    if args.image_path:
        content.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(args.image)}})
    content.append({"type": "text", "text": args.prompt})
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": INTERLEAVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": content,
            },
        ],
        "modalities": ["text", "image"],
        "stream": True,
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "image_config": image_config,
        "seed": args.seed,
    }

    response = requests.post(chat_url, headers=headers, json=payload, stream=True, timeout=600)
    response.raise_for_status()

    text_chunks: list[str] = []
    image_idx = 0
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        body = decoded[6:]
        if body.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            text_chunks.append(content)
            print(content, end="", flush=True)

        for image_item in delta.get("images") or []:
            image_url = (image_item.get("image_url") or {}).get("url", "")
            if image_url.startswith("data:image/"):
                out_file = out_dir / f"{timestamp}_interleave_stream_{image_idx}.png"
                save_data_url_to_file(image_url, out_file)
                image_idx += 1

    print("\n\n--- stream complete ---")
    final_text = "".join(text_chunks)
    text_path = out_dir / f"{timestamp}_interleave_stream.txt"
    text_path.write_text(final_text, encoding="utf-8")
    print(f"[saved] {text_path}")


def run_vqa(args: argparse.Namespace, out_dir: Path, timestamp: str) -> None:
    chat_url, headers = build_client(args.url, args.api_key)
    content = []
    if args.image_path:
        content.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(args.image_path)}})
    content.append({"type": "text", "text": args.prompt})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
    }
    response = requests.post(chat_url, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    print("\n--- assistant content ---")
    print(message.get("content", ""))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[config] mode={args.mode}, model={args.model}, url={args.url}")
    if args.image_path is not None:
        print(f"[config] input image_path={args.image_path}")
    print(f"[config] output_dir={out_dir.resolve()}")

    if args.mode == "t2i":
        run_t2i(args, out_dir=out_dir, timestamp=timestamp)
    elif args.mode == "it2i":
        run_it2i(args, out_dir=out_dir, timestamp=timestamp)
    elif args.mode == "interleave":
        run_interleave_stream(args, out_dir=out_dir, timestamp=timestamp)
    elif args.mode == "vqa":
        run_vqa(args, out_dir=out_dir, timestamp=timestamp)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
