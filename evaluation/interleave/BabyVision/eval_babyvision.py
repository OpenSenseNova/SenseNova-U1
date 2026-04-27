import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

DEFAULT_API_KEY = os.getenv("BABYVISION_JUDGE_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", ""))
DEFAULT_ENDPOINT = os.getenv("BABYVISION_JUDGE_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
DEFAULT_API_VERSION = os.getenv(
    "BABYVISION_JUDGE_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
)
DEFAULT_MODEL = os.getenv("BABYVISION_JUDGE_MODEL", "gpt-4.1")
REQUEST_TIMEOUT = 60


PROMPT_TEMPLATE = """You are an answer extraction and judging assistant.

Given a question, a model response, and the ground truth answer, do two things:
1. Extract the model's final answer from the response. Focus on the last explicit conclusion/final answer.
2. Judge whether the extracted answer is semantically equivalent to the ground truth answer.

Rules:
- Match the format of the ground truth as closely as possible.
- If the model never gives a final answer, use null for extracted_answer.
- Return only valid JSON.

Question: {question}

Model Response: {model_response}

Ground Truth: {ground_truth}

Reply in exactly this JSON schema:
{{"extracted_answer": "<string-or-null>", "is_correct": true}}
or
{{"extracted_answer": "<string-or-null>", "is_correct": false}}"""


JUDGE_ONLY_PROMPT_TEMPLATE = """You are an answer judging assistant.

Given a question, an extracted answer, and the ground truth answer, judge whether the extracted answer is semantically equivalent to the ground truth answer.

Rules:
- Match the format of the ground truth as closely as possible.
- If the extracted answer is empty or null, return false.
- Return only valid JSON.

Question: {question}

Extracted Answer: {extracted_answer}

Ground Truth: {ground_truth}

Reply in exactly this JSON schema:
{{"is_correct": true}}
or
{{"is_correct": false}}"""


RULE_PATTERNS = [
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL),
    re.compile(r"(?:final answer|correct answer)\s*[:：]\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:so|thus|therefore)\s+the answer is\s*[:：]?\s*(.+)", re.IGNORECASE),
    re.compile(
        r"(?:the answer|answer|correct answer)\s+is\s*[:：]?\s*(.+)",
        re.IGNORECASE,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Use LLM to extract answers and judge BabyVision outputs.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output", default=None, help="Output JSON/JSONL file, default: <input>_eval.<ext>")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Judge API key")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Judge endpoint")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="Judge API version")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model name")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--retries", type=int, default=3, help="Retries per item")
    parser.add_argument(
        "--extractor",
        choices=["llm", "rule_then_llm", "rule_only"],
        default="rule_then_llm",
        help="How to extract answers before judging",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if extracted_answer/LLMJudgeResult already exist"
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Only judge existing extracted_answer values; never re-extract from model_response",
    )
    return parser.parse_args()


def extract_json_block(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group())


def infer_output_path(input_path):
    if input_path.endswith(".jsonl"):
        return input_path[:-6] + "_eval.jsonl"
    if input_path.endswith(".json"):
        return input_path[:-5] + "_eval.json"
    return input_path + "_eval.json"


def is_azure_endpoint(endpoint):
    endpoint = (endpoint or "").strip().lower()
    return "/openai" in endpoint or ".openai.azure.com" in endpoint


def build_request(endpoint, api_key, api_version, model_name, prompt):
    endpoint = endpoint.rstrip("/")
    headers = {"Content-Type": "application/json"}
    params = {}

    if is_azure_endpoint(endpoint):
        if "/openai/deployments/" in endpoint:
            url = f"{endpoint}/chat/completions"
        elif endpoint.endswith("/openai"):
            url = f"{endpoint}/deployments/{model_name}/chat/completions"
        else:
            url = f"{endpoint}/openai/deployments/{model_name}/chat/completions"
        headers["api-key"] = api_key
        params["api-version"] = api_version
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        }
    else:
        if not endpoint.endswith("/v1"):
            endpoint = f"{endpoint}/v1"
        url = f"{endpoint}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        }

    return url, headers, params, body


def post_json(url, headers, params, body, timeout):
    if params:
        url = f"{url}?{urllib_parse.urlencode(params)}"
    data = json.dumps(body).encode("utf-8")
    request = urllib_request.Request(url, data=data, headers=headers, method="POST")
    with urllib_request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def response_text(resp_json):
    if not isinstance(resp_json, dict):
        raise TypeError(f"Unexpected response type from judge API: {type(resp_json).__name__}")

    choices = resp_json.get("choices")
    if not choices:
        raise ValueError(f"No choices found in judge response: {resp_json}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text_parts.append(item["text"])
        if text_parts:
            return "".join(text_parts).strip()

    raise ValueError(f"Unsupported judge response content: {resp_json}")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of entries in {path}")
    return data


def save_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")
        else:
            json.dump(data, f, ensure_ascii=False, indent=4)


def get_field(entry, *names):
    for name in names:
        if name in entry:
            return entry[name]
    return None


def normalize_optional_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def clean_extracted_answer(value):
    value = normalize_optional_text(value)
    if value is None:
        return None
    value = re.sub(r"</?answer>", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:final answer|answer)\s*[:：]\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip('"').strip("'").strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def rule_extract_answer(model_response):
    model_response = model_response or ""
    for pattern in RULE_PATTERNS:
        matches = pattern.findall(model_response)
        if not matches:
            continue
        candidate = matches[-1]
        if isinstance(candidate, tuple):
            candidate = next((item for item in candidate if item), "")
        candidate = clean_extracted_answer(candidate)
        if candidate and candidate.lower() not in {"given", "none", "null", "unknown"}:
            return candidate
    return None


def process_one(
    endpoint, api_key, api_version, model_name, retries, idx, entry, extractor="rule_then_llm", judge_only=False
):
    question = get_field(entry, "question", "Question") or ""
    ground_truth = get_field(entry, "answer", "GroundTruth") or ""
    model_response = get_field(entry, "model_response", "ModelResult") or ""
    existing_extract = normalize_optional_text(get_field(entry, "extracted_answer", "ExtractedAnswer"))

    if judge_only:
        prompt = JUDGE_ONLY_PROMPT_TEMPLATE.format(
            question=question,
            extracted_answer=existing_extract,
            ground_truth=ground_truth,
        )
        use_judge_only_prompt = True
    else:
        rule_extract = None
        if extractor != "llm":
            rule_extract = rule_extract_answer(model_response)
        if rule_extract is not None:
            prompt = JUDGE_ONLY_PROMPT_TEMPLATE.format(
                question=question,
                extracted_answer=rule_extract,
                ground_truth=ground_truth,
            )
            use_judge_only_prompt = True
        elif extractor == "rule_only":
            return idx, None, False
        else:
            prompt = PROMPT_TEMPLATE.format(
                question=question,
                model_response=model_response,
                ground_truth=ground_truth,
            )
            use_judge_only_prompt = False

    for attempt in range(retries):
        try:
            url, headers, params, body = build_request(endpoint, api_key, api_version, model_name, prompt)
            resp_json = post_json(url, headers, params, body, REQUEST_TIMEOUT)
            payload = extract_json_block(response_text(resp_json))
            if judge_only:
                extracted_answer = existing_extract
            elif use_judge_only_prompt:
                extracted_answer = rule_extract
            else:
                extracted_answer = clean_extracted_answer(payload.get("extracted_answer"))
            return idx, extracted_answer, bool(payload.get("is_correct", False))
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed after {retries} retries: {exc}") from exc
            message = str(exc).lower()
            if isinstance(exc, urllib_error.HTTPError):
                message = f"{message} {exc.code}"
            if "429" in message or "too many requests" in message or "too_many_requests" in message:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(1.5**attempt)


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing judge API key. Use --api-key or BABYVISION_JUDGE_API_KEY / AZURE_OPENAI_API_KEY.")
    if not args.endpoint:
        raise ValueError("Missing judge endpoint. Use --endpoint or BABYVISION_JUDGE_ENDPOINT / AZURE_OPENAI_ENDPOINT.")

    output_path = args.output
    if output_path is None:
        output_path = infer_output_path(args.input)

    data = load_data(args.input)

    tasks = []
    judge_only_count = 0
    extract_and_judge_count = 0
    skipped_without_extract = 0
    for idx, entry in enumerate(data):
        already_has_extract = (
            normalize_optional_text(get_field(entry, "extracted_answer", "ExtractedAnswer")) is not None
        )
        already_has_judge = get_field(entry, "LLMJudgeResult") is not None
        if args.judge_only:
            if already_has_extract and (args.force or not already_has_judge):
                tasks.append((idx, entry, True))
                judge_only_count += 1
            elif not already_has_extract:
                skipped_without_extract += 1
        elif args.force:
            tasks.append((idx, entry, False))
            extract_and_judge_count += 1
        elif already_has_extract and not already_has_judge:
            tasks.append((idx, entry, True))
            judge_only_count += 1
        elif not (already_has_extract and already_has_judge):
            tasks.append((idx, entry, False))
            extract_and_judge_count += 1

    print(f"Loaded {len(data)} entries")
    print(f"To process: {len(tasks)}")
    print(f"Judge only: {judge_only_count}")
    print(f"Extract + judge: {extract_and_judge_count}")
    print(f"Skipped without extracted_answer: {skipped_without_extract}")
    print(f"Extractor: {args.extractor}")
    print(f"Output: {output_path}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one,
                args.endpoint,
                args.api_key,
                args.api_version,
                args.model,
                args.retries,
                idx,
                entry,
                args.extractor,
                judge_only,
            ): (idx, judge_only)
            for idx, entry, judge_only in tasks
        }
        done = 0
        for future in as_completed(futures):
            idx, judge_only = futures[future]
            try:
                _, extracted_answer, is_correct = future.result()
                data[idx]["extracted_answer"] = extracted_answer
                data[idx]["LLMJudgeResult"] = is_correct
            except Exception as exc:
                print(f"[ERROR] idx={idx} taskId={data[idx].get('taskId', data[idx].get('Id'))}: {exc}")
                if not judge_only:
                    data[idx]["extracted_answer"] = None
                data[idx]["LLMJudgeResult"] = False
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"Processed {done}/{len(tasks)}")

    total = len(data)
    extracted_nonempty = sum(
        1 for item in data if get_field(item, "extracted_answer", "ExtractedAnswer") not in (None, "")
    )
    correct = sum(1 for item in data if item.get("LLMJudgeResult") is True)

    print(f"Extracted answers: {extracted_nonempty}/{total} = {100 * extracted_nonempty / max(total, 1):.2f}%")
    print(f"Correct: {correct}/{total} = {100 * correct / max(total, 1):.2f}%")

    save_data(output_path, data)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
