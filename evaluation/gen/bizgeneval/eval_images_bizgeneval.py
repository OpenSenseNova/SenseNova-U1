from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

from evaluation.gen.bizgeneval.eval_prompt import EVAL_GENERATION_PROMPTS as EVAL_PROMPTS
from evaluation.gen.common.judge import JudgeClient

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "test.jsonl"
ERROR_ALPHA = 0.1

# Reference:
#   BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation
#   https://arxiv.org/abs/2603.25732


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BizGenEval images with Gemini/OpenAI-compatible judge.")
    parser.add_argument("--image-dir", required=True, help="Directory containing generated BizGenEval images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-item results and summary.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="BizGenEval JSONL path.")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return None


def _strip_json_fence(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json_safe(text: str) -> dict[str, Any]:
    content = _strip_json_fence(text)
    try:
        return json.loads(content)
    except Exception:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        extracted = match.group(0)
        try:
            return json.loads(extracted)
        except Exception:
            pass
        normalized = re.sub(r"\bTrue\b", "true", extracted)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        try:
            return json.loads(normalized)
        except Exception:
            pass
    raise ValueError("failed to parse judge JSON")


def _extract_results_only(raw_text: str, n_questions: int) -> dict[str, Any] | None:
    if not isinstance(raw_text, str) or n_questions <= 0:
        return None
    matches = re.findall(
        r'"result"\s*:\s*(true|false|True|False|"true"|"false"|1|0)',
        raw_text,
        flags=re.DOTALL,
    )
    if len(matches) < n_questions:
        return None
    parsed: dict[str, Any] = {}
    for idx, match in enumerate(matches[:n_questions], start=1):
        val = _to_bool(str(match).strip().strip('"'))
        if val is None:
            return None
        parsed[str(idx)] = {"result": val}
    return parsed


def _format_checklist(questions: list[str]) -> str:
    return "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))


def _render_prompt(user_template: str, questions: list[str]) -> str:
    kwargs = {"checklist": _format_checklist(questions)}
    if "{expected_count}" in user_template:
        kwargs["expected_count"] = len(questions)
    if "{required_keys}" in user_template:
        kwargs["required_keys"] = ", ".join(str(i) for i in range(1, len(questions) + 1))
    return user_template.format(**kwargs)


def _compute_item_scores(
    item: dict[str, Any],
    meta_info: dict[str, dict[str, Any]],
    error_alpha: float,
    qidxs_key: str | None = None,
) -> dict[str, float | int]:
    questions = item.get("questions", [])
    qidxs = list(item.get(qidxs_key, []) or []) if qidxs_key else list(range(1, len(questions) + 1))
    if not qidxs:
        return {"accuracy": 0.0, "error_score": 0.0, "errors": 0, "n_questions": 0}
    errors = 0
    for qidx in qidxs:
        if meta_info.get(str(qidx), {}).get("result") is not True:
            errors += 1
    n_questions = len(qidxs)
    accuracy = (n_questions - errors) / n_questions
    error_score = max(0.0, 1.0 - error_alpha * errors)
    return {
        "accuracy": accuracy,
        "error_score": error_score,
        "errors": errors,
        "n_questions": n_questions,
    }


def _aggregate(records: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(record)
    summary: dict[str, dict[str, float | int]] = {}
    for key, items in sorted(grouped.items()):
        summary[key] = {
            "count": len(items),
            "accuracy": mean([float(item["accuracy"]) for item in items]) if items else 0.0,
            "error_score": mean([float(item["error_score"]) for item in items]) if items else 0.0,
        }
    return summary


def _load_items(data_path: Path) -> list[dict[str, Any]]:
    items = []
    with data_path.open(encoding="utf-8") as f:
        for prompt_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_prompt_id"] = prompt_id
            items.append(item)
    return items


def _resolve_image_path(image_dir: Path, prompt_id: int) -> Path | None:
    direct = image_dir / f"{prompt_id:04d}.png"
    if direct.exists():
        return direct
    repeat0 = image_dir / f"{prompt_id:04d}_0.png"
    if repeat0.exists():
        return repeat0
    return None


def _is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta_info") or {}
        n_questions = int(data.get("n_questions", 0))
        return (
            data.get("accuracy") is not None
            and len(meta) == n_questions
            and all(isinstance(v, dict) and "result" in v for v in meta.values())
        )
    except Exception:
        return False


def _record_error(prompt_id: int, exc: Exception) -> None:
    print(f"[warn] prompt_id={prompt_id} failed: {type(exc).__name__}: {exc}")


def eval_one(
    item: dict[str, Any],
    *,
    image_dir: Path,
    items_dir: Path,
    client: JudgeClient,
    error_alpha: float,
    force_rerun: bool,
    write_lock: threading.Lock,
) -> dict[str, Any] | None:
    prompt_id = int(item["_prompt_id"])
    dataset_id = item.get("id")
    image_path = _resolve_image_path(image_dir, prompt_id)
    if image_path is None:
        print(f"[warn] missing image for prompt_id={prompt_id}")
        return None

    cache_path = items_dir / f"{prompt_id:04d}.json"
    if not force_rerun and _is_complete(cache_path):
        with cache_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        cached["_cached"] = True
        return cached

    eval_tag = str(item.get("eval_tag") or item.get("dimension") or "").strip()
    if eval_tag not in EVAL_PROMPTS:
        print(f"[warn] skip prompt_id={prompt_id}: unsupported eval_tag={eval_tag!r}")
        return None
    questions = list(item.get("questions") or [])
    if not questions:
        print(f"[warn] skip prompt_id={prompt_id}: empty questions")
        return None

    system_prompt, user_template = EVAL_PROMPTS[eval_tag]
    raw_output = client.judge_image_text(
        image_path=image_path,
        system_prompt=system_prompt,
        user_prompt=_render_prompt(user_template, questions),
    ).strip()

    try:
        parsed = _parse_json_safe(raw_output)
    except Exception:
        parsed = _extract_results_only(raw_output, len(questions)) or {}

    meta_info: dict[str, dict[str, Any]] = {}
    for idx, question in enumerate(questions, start=1):
        rec = parsed.get(str(idx)) if isinstance(parsed, dict) else None
        if not isinstance(rec, dict):
            meta_info[str(idx)] = {
                "result": False,
                "raw_description": question,
                "reason": "missing_from_output",
            }
            continue
        val = _to_bool(rec.get("result"))
        meta_info[str(idx)] = {
            "result": bool(val) if isinstance(val, bool) else False,
            "raw_description": question,
            "reason": rec.get("reason", ""),
        }

    overall = _compute_item_scores(item, meta_info, error_alpha)
    easy = _compute_item_scores(item, meta_info, error_alpha * 2, "easy_qidxs")
    hard = _compute_item_scores(item, meta_info, error_alpha * 2, "hard_qidxs")
    result = {
        "prompt_id": prompt_id,
        "dataset_id": dataset_id,
        "domain": item.get("domain", ""),
        "dimension": item.get("dimension", ""),
        "eval_tag": eval_tag,
        "prompt": item.get("prompt", ""),
        "image_path": str(image_path),
        "n_questions": len(questions),
        "accuracy": overall["accuracy"],
        "error_score": overall["error_score"],
        "easy_accuracy": easy["accuracy"],
        "easy_error_score": easy["error_score"],
        "hard_accuracy": hard["accuracy"],
        "hard_error_score": hard["error_score"],
        "meta_info": meta_info,
        "raw_model_response": raw_output,
    }
    with write_lock:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = _parse_args()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    items_dir = output_dir / "items"
    output_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)

    client = JudgeClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.judge_model,
        timeout=args.timeout,
    )
    items = _load_items(Path(args.data_path).resolve())
    print(
        f"[bizgeneval] items={len(items)} concurrency={args.concurrency} "
        f"judge_model={client.model} image_dir={image_dir}"
    )

    write_lock = threading.Lock()

    tasks = list(items)
    results: list[dict[str, Any] | None] = []

    def _result_status(result: dict[str, Any] | None) -> str:
        if result is None:
            return "skipped"
        if result.get("_cached"):
            return "cached"
        return "done"

    max_workers = max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                eval_one,
                item,
                image_dir=image_dir,
                items_dir=items_dir,
                client=client,
                error_alpha=ERROR_ALPHA,
                force_rerun=args.force_rerun,
                write_lock=write_lock,
            ): item
            for item in tasks
        }
        total = len(futures)
        progress = tqdm(total=total, desc="bizgeneval eval", dynamic_ncols=True) if tqdm else None
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    _record_error(int(item["_prompt_id"]), exc)
                    result = None
                results.append(result)
                status = _result_status(result)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(f"prompt_id={item['_prompt_id']} {status}")
                else:
                    print(f"[{done}/{total}] prompt_id={item['_prompt_id']} {status}")
        finally:
            if progress is not None:
                progress.close()

    valid_results = [result for result in results if result]
    if not valid_results:
        raise RuntimeError("BizGenEval produced no valid evaluation results.")

    final_records = [
        {
            "prompt_id": int(result["prompt_id"]),
            "dataset_id": result["dataset_id"],
            "domain": result["domain"],
            "dimension": result["dimension"],
            "accuracy": float(result["accuracy"]),
            "error_score": float(result["error_score"]),
            "easy_accuracy": float(result["easy_accuracy"]),
            "easy_error_score": float(result["easy_error_score"]),
            "hard_accuracy": float(result["hard_accuracy"]),
            "hard_error_score": float(result["hard_error_score"]),
            "n_questions": int(result["n_questions"]),
        }
        for result in sorted(valid_results, key=lambda item: int(item["prompt_id"]))
    ]

    by_domain = _aggregate(final_records, "domain")
    by_dimension = _aggregate(final_records, "dimension")
    overall_accuracy = mean([record["accuracy"] for record in final_records]) if final_records else 0.0
    overall_error_score = mean([record["error_score"] for record in final_records]) if final_records else 0.0
    easy_accuracy = mean([record["easy_accuracy"] for record in final_records]) if final_records else 0.0
    easy_error_score = mean([record["easy_error_score"] for record in final_records]) if final_records else 0.0
    hard_accuracy = mean([record["hard_accuracy"] for record in final_records]) if final_records else 0.0
    hard_error_score = mean([record["hard_error_score"] for record in final_records]) if final_records else 0.0

    summary = {
        "benchmark": "bizgeneval",
        "data_path": str(Path(args.data_path).resolve()),
        "eval_provider": "judge_client",
        "judge_model": client.model,
        "error_alpha": ERROR_ALPHA,
        "easy_hard_error_alpha": ERROR_ALPHA * 2,
        "items": len(final_records),
        "overall_accuracy": overall_accuracy,
        "overall_error_score": overall_error_score,
        "easy_accuracy": easy_accuracy,
        "easy_error_score": easy_error_score,
        "hard_accuracy": hard_accuracy,
        "hard_error_score": hard_error_score,
        "by_domain": by_domain,
        "by_dimension": by_dimension,
        "records": final_records,
    }

    summary_path = output_dir / "bizgeneval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[bizgeneval] overall_error_score={overall_error_score:.4f} "
        f"overall_accuracy={overall_accuracy:.4f} "
        f"easy_error_score={easy_error_score:.4f} "
        f"hard_error_score={hard_error_score:.4f}"
    )
    print(f"[bizgeneval] summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
