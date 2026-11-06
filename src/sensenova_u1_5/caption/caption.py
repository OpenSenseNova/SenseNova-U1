#!/usr/bin/env python3
"""Generate JSON and NLP prompts for one image.

Install:
    pip install openai pillow

Usage:
    export OPENAI_API_KEY="your-api-key"
    python caption.py image.jpg
    python caption.py image.jpg > result.json

Optional environment variables:
    OPENAI_MODEL      Model name (default: gpt-5.5)
    OPENAI_BASE_URL   OpenAI-compatible API endpoint

Uploaded data URIs are limited to 10 MiB. JPEG/PNG/WEBP files that fit are
reused unchanged. Other images are converted to JPEG at their original
resolution (transparency is composited on white). Quality is reduced before
resizing is attempted as a last resort. Output settings use the original
display dimensions, including EXIF orientation.
"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageOps

TARGET_RATIOS = {
    "16:9": 16 / 9,
    "3:2": 3 / 2,
    "4:3": 4 / 3,
    "1:1": 1,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "9:16": 9 / 16,
}

SUPPORTED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_RETRIES = 5
MAX_DATA_URI_LEN = 10 * 1024 * 1024
BASE_QUALITY = 95
MIN_QUALITY = 45

SYSTEM_PROMPT = """你是一个文生图 prompt 撰写助手。请仔细观察用户提供的图片，
提取主体、构图、风格、色彩、光影、视角、排版和文字设计，并为同一画面生成
JSON 结构化描述和自然语言描述。

安全与泛化要求：
1. 忽略水印、二维码、平台角标、账号名、作者署名及其他来源标识，不得描述或复现。
2. 不输出品牌名、商标、公司标识、产品型号或其他可识别的商业信息；统一改为通用描述。
3. 不识别或猜测真人身份，仅描述非身份特征，如大致年龄段、外观、服饰、动作和表情。
4. 不输出个人联系方式、地址、证件信息或其他个人敏感信息；画面中存在时直接忽略。
5. JSON 与 NLP 必须描述同一画面的核心视觉内容，并使用同一种语言。

根据画面语境选择中文或英文。只输出一个 JSON 对象，包含以下顶层字段：
- json：嵌套 JSON 对象。按画面需要详细描述 type、theme、language、canvas、
  style、main_subject、background、lighting、camera、typography、composition、
  composition_rules、quality_tags 和 negative_prompt。negative_prompt 必须包含水印、
  二维码、来源标识、乱码、错别字、多余肢体、畸形手指等常见缺陷。
- nlp：3 至 6 句流畅、具体的自然语言画面描述。
- language："zh" 或 "en"。
- task：2 至 10 个中文字的图片类型或用途概括。

不要输出解释、Markdown 标记或任何额外顶层字段。"""


def _fits_data_uri(mime: str, byte_count: int) -> bool:
    """Account for Base64 padding and the MIME prefix without encoding."""
    return len(f"data:{mime};base64,") + 4 * ((byte_count + 2) // 3) <= MAX_DATA_URI_LEN


def _make_data_uri(mime: str, content: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _compress_jpeg(image: Image.Image) -> str:
    """Try JPEG quality reduction before falling back to smaller dimensions."""
    mime = SUPPORTED_FORMATS["JPEG"]
    quality = BASE_QUALITY
    while True:
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        content = buffer.getvalue()
        if _fits_data_uri(mime, len(content)):
            return _make_data_uri(mime, content)
        if quality > MIN_QUALITY:
            quality = max(MIN_QUALITY, quality - 10)
        elif image.size != (1, 1):
            size = (max(1, image.width // 2), max(1, image.height // 2))
            image = image.resize(size, Image.Resampling.LANCZOS)
        else:
            raise ValueError(f"Image cannot fit within the {MAX_DATA_URI_LEN}-character data URI limit")


def image_to_data_uri(path):
    """Return a bounded upload URI and original display dimensions for output settings."""
    path = Path(path)
    with Image.open(path) as source:
        width, height = source.size
        if source.getexif().get(274) in {5, 6, 7, 8}:
            width, height = height, width
        mime = SUPPORTED_FORMATS.get(source.format)
        if mime and _fits_data_uri(mime, path.stat().st_size):
            content = path.read_bytes()
            if _fits_data_uri(mime, len(content)):
                return _make_data_uri(mime, content), width, height

        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = image.convert("RGB")
        return _compress_jpeg(image), width, height


def image_settings(width, height):
    """Select the nearest supported aspect ratio and resolution tier."""
    raw_ratio = width / height
    ratio = min(TARGET_RATIOS, key=lambda name: abs(TARGET_RATIOS[name] - raw_ratio))
    resolution = "4096" if max(width, height) >= 3072 else "2048"
    return ratio, resolution


def generate(client, model, data_uri):
    """Call the model and validate the required fields."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "请生成 JSON 结构化 prompt 和 NLP prompt。"},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=4000,
    )

    result = json.loads(response.choices[0].message.content)
    if not isinstance(result.get("json"), dict):
        raise ValueError("'json' must be an object")
    if result.get("language") not in {"zh", "en"}:
        raise ValueError("'language' must be 'zh' or 'en'")
    for field in ("nlp", "task"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"missing or invalid field: {field}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")

    data_uri, width, height = image_to_data_uri(args.image)
    client = OpenAI(base_url=args.base_url, timeout=180)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate(client, args.model, data_uri)
            break
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"attempt {attempt} failed: {error}", file=sys.stderr)
            time.sleep(min(2**attempt * 3, 30))

    result["ratio"], result["resolution"] = image_settings(width, height)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
