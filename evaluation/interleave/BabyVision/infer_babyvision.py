import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import regex
import requests
from tqdm import tqdm


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value}") from exc


DEFAULT_GENERATE_URLS = os.environ.get(
    "BABYVISION_GENERATE_URLS",
    "http://127.0.0.1:8000/generate",
)

MODEL_NAME = os.environ.get("BABYVISION_MODEL_NAME", "local-model")
SUPPORTED_MODELS = [MODEL_NAME]

DEFAULT_DATA_PATH = os.environ.get(
    "BABYVISION_DATA_PATH",
    "./babyvision_data/meta_data.jsonl",
)
DEFAULT_IMAGE_ROOT = os.environ.get(
    "BABYVISION_IMAGE_ROOT",
    "./babyvision_data",
)
DEFAULT_OUTPUT_DIR = os.environ.get("BABYVISION_OUTPUT_DIR", "babyvision_results")
DEFAULT_WORKERS = _env_int("BABYVISION_WORKERS", 32)
DEFAULT_MAX_RETRIES = _env_int("BABYVISION_MAX_RETRIES", 3)
DEFAULT_BACKEND_MAX_RETRIES = _env_int("BABYVISION_BACKEND_MAX_RETRIES", 20)
DEFAULT_REQUEST_TIMEOUT = _env_int("BABYVISION_REQUEST_TIMEOUT", 600)
MAX_FAILURE_DETAILS = 20

DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_STOP_SEQUENCES = [" <|endoftext|>", " <|im_start|>", " <|im_end|>"]
DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MIN_PIXELS = 262144
DEFAULT_MAX_PIXELS = 16777216
GENERATION_CONFIG = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "do_sample": DEFAULT_DO_SAMPLE,
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": DEFAULT_TOP_P,
    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    "min_pixels": DEFAULT_MIN_PIXELS,
    "max_pixels": DEFAULT_MAX_PIXELS,
}

SYSTEM_PROMPT = (
    "Reason step by step and place the thought process within the "
    "<think></think> tags, and provide the final conclusion at the end."
)

# 全局 Session 复用 TCP 连接
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _top_p_value(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return value


def parse_generate_urls(raw_urls):
    return [url.strip() for url in str(raw_urls).split(",") if url.strip()]


def summarize_failures(failures, limit=MAX_FAILURE_DETAILS):
    if not failures:
        return
    print(f"\nFailure summary: total={len(failures)}")
    for detail in failures[:limit]:
        print(f"- {detail}")
    remaining = len(failures) - limit
    if remaining > 0:
        print(f"... and {remaining} more failures")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal inference against the /generate endpoints.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model name recorded in outputs and used for output filenames. Default: {MODEL_NAME}.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to BabyVision meta_data.jsonl. Default: {DEFAULT_DATA_PATH}.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Root directory for BabyVision images. Default: {DEFAULT_IMAGE_ROOT}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for BabyVision outputs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker threads. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--max-retries",
        type=_positive_int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries per sample. Default: {DEFAULT_MAX_RETRIES}.",
    )
    parser.add_argument(
        "--generate-urls",
        type=str,
        default=DEFAULT_GENERATE_URLS,
        help="Comma-separated /generate endpoints for BabyVision inference.",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"HTTP request timeout in seconds. Default: {DEFAULT_REQUEST_TIMEOUT}.",
    )
    parser.add_argument(
        "--backend-max-retries",
        type=_positive_int,
        default=DEFAULT_BACKEND_MAX_RETRIES,
        help=f"Maximum retries for one backend request. Default: {DEFAULT_BACKEND_MAX_RETRIES}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature value passed to the backend. Default: {DEFAULT_TEMPERATURE}.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=_top_p_value,
        default=DEFAULT_TOP_P,
        help=f"Top-p value in (0, 1] passed to the backend. Default: {DEFAULT_TOP_P}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=_positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum number of generated tokens. Default: {DEFAULT_MAX_NEW_TOKENS}.",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=_positive_float,
        default=DEFAULT_REPETITION_PENALTY,
        help=f"Repetition penalty (> 0). Default: {DEFAULT_REPETITION_PENALTY}.",
    )
    parser.add_argument(
        "--min-pixels",
        dest="min_pixels",
        type=_positive_int,
        default=DEFAULT_MIN_PIXELS,
        help=f"Minimum image pixels passed to the backend. Default: {DEFAULT_MIN_PIXELS}.",
    )
    parser.add_argument(
        "--max-pixels",
        dest="max_pixels",
        type=_positive_int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Maximum image pixels passed to the backend. Default: {DEFAULT_MAX_PIXELS}.",
    )
    parser.set_defaults(do_sample=DEFAULT_DO_SAMPLE)
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help=f"Enable sampling. Default: {DEFAULT_DO_SAMPLE}.",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling.",
    )
    return parser.parse_args()


def configure_generation(args):
    if args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels must be <= --max-pixels")

    global SUPPORTED_MODELS
    SUPPORTED_MODELS = [args.model_name]

    GENERATION_CONFIG["max_new_tokens"] = args.max_new_tokens
    GENERATION_CONFIG["do_sample"] = args.do_sample
    GENERATION_CONFIG["temperature"] = args.temperature
    GENERATION_CONFIG["top_p"] = args.top_p
    GENERATION_CONFIG["repetition_penalty"] = args.repetition_penalty
    GENERATION_CONFIG["min_pixels"] = args.min_pixels
    GENERATION_CONFIG["max_pixels"] = args.max_pixels

    print(
        "Generation config:",
        json.dumps(
            {
                "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
                "do_sample": GENERATION_CONFIG["do_sample"],
                "model_name": SUPPORTED_MODELS[0],
                "temperature": GENERATION_CONFIG["temperature"],
                "top_p": GENERATION_CONFIG["top_p"],
                "repetition_penalty": GENERATION_CONFIG["repetition_penalty"],
                "min_pixels": GENERATION_CONFIG["min_pixels"],
                "max_pixels": GENERATION_CONFIG["max_pixels"],
            },
            ensure_ascii=False,
        ),
    )


def build_generation_parameters():
    parameters = {
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG["do_sample"],
        "stop_sequences": DEFAULT_STOP_SEQUENCES,
        "add_output_think_tokens": False,
    }
    if GENERATION_CONFIG["temperature"] is not None:
        parameters["temperature"] = GENERATION_CONFIG["temperature"]
    if GENERATION_CONFIG["top_p"] is not None:
        parameters["top_p"] = GENERATION_CONFIG["top_p"]
    if GENERATION_CONFIG["repetition_penalty"] is not None:
        parameters["repetition_penalty"] = GENERATION_CONFIG["repetition_penalty"]
    return parameters


def _safe_model_name(model):
    """将模型名中的 / 替换为 --，用于文件名。如 Qwen/Qwen2.5-VL-72B → Qwen--Qwen2.5-VL-72B"""
    return model.replace("/", "--")


def extract_boxed_answer(text):
    """
    Extract the final answer from model output.

    Priority:
    1. The last <answer>...</answer> block.
    2. A trailing non-empty line after the last </think>.
    3. The last non-empty line of the whole response.
    """
    if text is None:
        return None

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = regex.findall(pattern, text, flags=regex.DOTALL | regex.IGNORECASE)
    if matches:
        return matches[-1].strip()

    def _clean_candidate(candidate):
        if candidate is None:
            return None
        candidate = str(candidate).strip()
        if not candidate:
            return None
        candidate = regex.sub(r"^```[\w-]*\s*", "", candidate)
        candidate = regex.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()
        return candidate or None

    think_splits = regex.split(r"</think>", text, flags=regex.IGNORECASE)
    if len(think_splits) > 1:
        tail = think_splits[-1].strip()
        if tail:
            tail_lines = [line.strip() for line in tail.splitlines() if line.strip()]
            if tail_lines:
                return _clean_candidate(tail_lines[-1])

    tail_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if tail_lines:
        return _clean_candidate(tail_lines[-1])

    return None


def format_choices(choices):
    """Format multiple choice options as (A), (B), (C), etc."""
    if len(choices) == 0:
        return ""
    formatted = ""
    for idx, choice in enumerate(choices):
        formatted += f"({chr(65 + idx)}) {choice}\n"
    return formatted.strip()


def build_query(prompt, num_images=1, system_prompt=None):
    """构建 itvl chat template 格式的 query"""
    img_tags = "<img></img>\n" * num_images
    system_block = ""
    if system_prompt:
        system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    return f"{system_block}<|im_start|>user\n{img_tags}{prompt}<|im_end|>\n<|im_start|>assistant\n"


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=32, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def dynamic_preprocess_native_resolution(
    image, size_factor=32, min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS
):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def encode_image_base64(img_path, min_pixels, max_pixels):
    try:
        from PIL import Image
    except ImportError:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    with Image.open(img_path) as image:
        original_format = (image.format or "").upper()
        image = image.convert("RGB")
        image = dynamic_preprocess_native_resolution(
            image,
            size_factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        buffer = io.BytesIO()
        if original_format in {"JPEG", "JPG"}:
            image.save(buffer, format="JPEG", quality=95)
        else:
            image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_payload(query, image_paths):
    """构建 /generate", 接口的请求 payload"""
    images = []
    for img_path in image_paths:
        b64 = encode_image_base64(
            img_path,
            min_pixels=GENERATION_CONFIG["min_pixels"],
            max_pixels=GENERATION_CONFIG["max_pixels"],
        )
        images.append({"type": "base64", "data": b64})

    return {
        "inputs": query,
        "parameters": build_generation_parameters(),
        "multimodal_params": {
            "images": images,
        },
    }


def call_vllm_with_retry(payload, urls, request_timeout, max_retries=20):
    """调用 /generate", 接口，带重试，自动切换节点"""
    wait_time = 3
    for attempt in range(max_retries):
        url = random.choice(urls)
        try:
            response = _session.post(url, json=payload, timeout=request_timeout)
            if response.status_code == 200:
                try:
                    resp_json = response.json()
                except ValueError as exc:
                    body_preview = response.text[:300]
                    return False, f"Invalid JSON response from {url}: {exc}; body={body_preview}"
                text = None
                if isinstance(resp_json, dict) and "generated_text" in resp_json:
                    gt = resp_json["generated_text"]
                    if isinstance(gt, str):
                        text = gt
                    elif isinstance(gt, list):
                        parts = []
                        for item in gt:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict):
                                parts.append(item.get("text", json.dumps(item, ensure_ascii=False)))
                            else:
                                parts.append(str(item))
                        text = "".join(parts)
                    else:
                        text = str(gt)
                elif isinstance(resp_json, dict):
                    tqdm.write(f"[vLLM] Unknown response keys: {list(resp_json.keys())}")
                    text = json.dumps(resp_json, ensure_ascii=False)
                elif isinstance(resp_json, list) and len(resp_json) > 0:
                    first = resp_json[0]
                    if isinstance(first, str):
                        text = first
                    elif isinstance(first, dict):
                        text = first.get("generated_text", json.dumps(first, ensure_ascii=False))
                    else:
                        text = str(first)
                else:
                    text = str(resp_json)
                return True, text
            elif response.status_code == 400:
                return False, f"Fatal 400: {response.text[:300]}"
            else:
                tqdm.write(f"[vLLM] HTTP {response.status_code}: {response.text[:100]}... retrying in {wait_time}s")
        except requests.exceptions.Timeout:
            tqdm.write(f"[vLLM] Timeout. Retrying in {wait_time}s...")
        except requests.exceptions.ConnectionError:
            tqdm.write(f"[vLLM] Connection Error to {url}. Retrying in {wait_time}s...")
        except requests.exceptions.RequestException as e:
            tqdm.write(f"[vLLM] Request Error: {e}. Retrying in {wait_time}s...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 1.5, 60)
    return False, f"Max retries ({max_retries}) exceeded"


def validate_args(args):
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"BabyVision data file not found: {args.data_path}")
    if not os.path.isdir(args.image_root):
        raise FileNotFoundError(f"BabyVision image root not found: {args.image_root}")
    if not parse_generate_urls(args.generate_urls):
        raise ValueError("--generate-urls must contain at least one valid endpoint")


def test_babyvison(args):
    """
    BabyVision benchmark 评测：加载 meta_data.jsonl，
    构建问题（根据 ansType 处理 blank/choice），调用 /generate API，
    提取 <answer>...</answer> 中的答案，保存结果。
    支持断点续传，每个模型独立输出 JSONL。
    """
    data_path = args.data_path
    image_root = args.image_root
    output_dir = args.output_dir
    workers = args.workers
    sample_max_retries = args.max_retries
    backend_max_retries = args.backend_max_retries
    request_timeout = args.request_timeout
    generate_urls = parse_generate_urls(args.generate_urls)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading BabyVision data from {data_path} ...")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Total items: {len(data)}")

    total_success_count = 0
    total_fail_count = 0
    failures = []

    for model in SUPPORTED_MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")

        output_path = os.path.join(output_dir, f"babyvision_{_safe_model_name(model)}.jsonl")

        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            processed_ids.add(json.loads(line)["taskId"])
                        except Exception as exc:
                            detail = (
                                f"[resume] model={model} output={output_path} line={line_no}: invalid JSONL row: {exc}"
                            )
                            failures.append(detail)
                            raise RuntimeError(detail) from exc
            print(f"Resuming... {len(processed_ids)} items already processed.")

        pending = [item for item in data if item["taskId"] not in processed_ids]
        if not pending:
            print(f"[SKIP] {model}: all done")
            continue
        print(f"Pending: {len(pending)} items")

        def process_item(item, _model=model):
            img_path = os.path.join(image_root, item["image"])
            if not os.path.exists(img_path):
                return None, f"[img] taskId={item['taskId']}: File Not Found: {img_path}"

            if item["ansType"] == "blank":
                question = item["question"]
                answer = item["blankAns"]
            else:
                question = item["question"] + "\nChoices:\n" + format_choices(item["options"])
                choice_ans = str(item["choiceAns"]).strip()
                if choice_ans.isdigit():
                    idx = int(choice_ans)
                    if 1 <= idx <= 26:
                        answer = chr(64 + idx)
                    else:
                        answer = chr(65 + idx)
                else:
                    answer = choice_ans

            question = question + "\nPut your final answer inside <answer></answer>."
            # + "\nThink about the question and give your final answer in <answer>Answer</answer> format."

            query = build_query(question, 1, system_prompt=None)
            payload = build_payload(query, [img_path])

            last_err = ""
            for attempt in range(sample_max_retries):
                try:
                    success, result = call_vllm_with_retry(
                        payload,
                        urls=generate_urls,
                        request_timeout=request_timeout,
                        max_retries=backend_max_retries,
                    )
                except Exception as exc:
                    last_err = str(exc)
                    time.sleep(2**attempt)
                    continue

                if success:
                    model_response = result
                    extracted_answer = extract_boxed_answer(model_response)
                    return {
                        "taskId": item["taskId"],
                        "type": item["type"],
                        "subtype": item["subtype"],
                        "ansType": item["ansType"],
                        "question": question,
                        "answer": answer,
                        "model": _model,
                        "model_response": model_response,
                        "extracted_answer": extracted_answer,
                        "reasoning": "",
                    }, ""

                last_err = str(result)
                time.sleep(2**attempt)

            return None, (
                f"[api] taskId={item['taskId']}: Failed after {sample_max_retries} retries, last_err={last_err}"
            )

        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_item, item): item["taskId"] for item in pending}
            with open(output_path, "a", encoding="utf-8") as f_out:
                pbar = tqdm(as_completed(futures), total=len(pending), desc=model)
                for future in pbar:
                    try:
                        record, err = future.result()
                    except Exception as exc:
                        record, err = None, f"[thread] {exc}"

                    if record:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        success_count += 1
                    else:
                        fail_count += 1
                        failures.append(err)
                        tqdm.write(err)

                    pbar.set_description(f"{model} ✓{success_count} ✗{fail_count}")

        print(f"Finished {model}: success={success_count}, fail={fail_count}")
        total_success_count += success_count
        total_fail_count += fail_count

    print(f"\nAll models done!")
    summarize_failures(failures)
    return {
        "success_count": total_success_count,
        "fail_count": total_fail_count,
        "failures": failures,
    }


def test_babyvision():
    """Alias for the correctly spelled BabyVision entrypoint."""
    args = parse_args()
    validate_args(args)
    configure_generation(args)
    return test_babyvison(args)


def main():
    args = parse_args()
    validate_args(args)
    configure_generation(args)
    result = test_babyvison(args)
    return 1 if result["fail_count"] else 0


if __name__ == "__main__":
    # test_realunify()
    # test_unimmmu()
    # test_tir()
    # test_mira()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
