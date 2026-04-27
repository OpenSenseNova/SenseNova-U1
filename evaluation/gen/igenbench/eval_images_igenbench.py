from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from evaluation.gen.common.judge import JudgeClient

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


# Reference:
#   IGenBench: Benchmarking the Reliability of Text-to-Infographic Generation
#   https://arxiv.org/abs/2601.04498


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IGenBench images with direct image-question judging.")
    parser.add_argument("--image-dir", required=True, help="Directory containing generated IGenBench images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-item judgments and summary.")
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR), help="IGenBench directory with per-item JSON files."
    )
    parser.add_argument("--gen-model-name", default="local_model", help="Generation model tag stored in judgments.")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible judge API base URL.")
    parser.add_argument("--api-key", required=True, help="OpenAI-compatible judge API key.")
    parser.add_argument("--judge-model", required=True, help="Judge model name.")
    parser.add_argument("--eval-timeout", type=int, default=240)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--force-rerun", action="store_true", help="Ignore saved judgments and rerun from source items."
    )
    return parser.parse_args()


def _iter_items(data_dir: Path, max_items: int | None) -> list[tuple[int, Path]]:
    items = []
    for path in data_dir.glob("*.json"):
        try:
            items.append((int(path.stem), path))
        except ValueError:
            continue
    items.sort(key=lambda item: item[0])
    if max_items is not None:
        items = items[:max_items]
    return items


def _resolve_image_path(image_dir: Path, prompt_id: int) -> Path | None:
    direct = image_dir / f"{prompt_id:04d}.png"
    if direct.exists():
        return direct
    return None


def _saved_item_path(output_dir: Path, item_id: str) -> Path:
    return output_dir / "items" / f"{int(item_id):04d}.json"


def _count_done_questions(item_data: dict[str, Any], *, gen_model: str, eval_model: str) -> tuple[int, int]:
    total = 0
    done = 0
    for entry in item_data.get("evaluation", []) or []:
        total += 1
        for judgment in entry.get("judgments", []) or []:
            if judgment.get("gen_model") == gen_model and judgment.get("eval_model") == eval_model:
                done += 1
                break
    return done, total


def _score_item(item_data: dict[str, Any], *, gen_model: str, eval_model: str) -> dict[str, Any]:
    total = 0
    correct = 0
    q_by_type: dict[str, tuple[int, int]] = {}
    for eval_entry in item_data.get("evaluation", []) or []:
        for judgment in eval_entry.get("judgments", []) or []:
            if judgment.get("gen_model") == gen_model and judgment.get("eval_model") == eval_model:
                q_type = str(eval_entry.get("question_type") or "unknown").strip()
                ok = int(str(judgment.get("answer") or "").strip() == "1")
                total += 1
                correct += ok
                prev_correct, prev_total = q_by_type.get(q_type, (0, 0))
                q_by_type[q_type] = (prev_correct + ok, prev_total + 1)
                break
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "strict": total > 0 and correct == total,
        "q_by_type": q_by_type,
    }


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
    if not match:
        raise ValueError("judge response does not contain JSON")
    extracted = match.group(0)
    try:
        return json.loads(extracted)
    except Exception:
        normalized = re.sub(r"\bTrue\b", "true", extracted)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        return json.loads(normalized)


def _to_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and value in {0, 1}:
        return str(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "pass", "supported"}:
            return "1"
        if text in {"0", "false", "no", "fail", "unsupported"}:
            return "0"
    raise ValueError(f"unsupported judge answer: {value!r}")


def _parse_judgment(raw_text: str) -> tuple[str, str]:
    try:
        parsed = _parse_json_safe(raw_text)
        answer_value = parsed.get("answer", parsed.get("result", parsed.get("score")))
        return _to_answer(answer_value), str(parsed.get("analysis", parsed.get("reason", "")))
    except Exception:
        match = re.search(
            r'"(?:answer|result|score)"\s*:\s*(true|false|True|False|"true"|"false"|1|0)',
            raw_text,
            flags=re.DOTALL,
        )
        if match:
            return _to_answer(match.group(1).strip('"')), ""
    raise ValueError("failed to parse judge answer")


def build_factual_qa_judgment_prompt(question: str) -> str:
    return f"""
You are a strict factual evaluator.

Your task:
Inspect the infographic image (provided separately) and answer the binary factual question below.

Rules:
- Answer **1** ONLY if the requirement is clearly satisfied in the image.
- Answer **0** if the requirement is NOT satisfied, unclear, ambiguous, partially met, or cannot be confirmed.
- No partial credit. Ambiguity = 0.
- Base your judgment ONLY on visible evidence in the infographic.
- Even if the image is empty, blank, corrupted, unreadable, or clearly incorrect, you MUST still output a valid JSON object following the required format. In such cases, the answer should be 0.

-------------------------------------
FACTUAL QUESTION:
{question}
-------------------------------------

**Output Format (JSON ONLY)**:
```json
{{
  "analysis": "<your reasoning based strictly on what is visible>",
  "answer": "<0 or 1>"
}}
```
The response must contain only valid JSON.
"""


def _has_matching_judgment(entry: dict[str, Any], *, gen_model: str, eval_model: str) -> bool:
    for judgment in entry.get("judgments", []) or []:
        if judgment.get("gen_model") == gen_model and judgment.get("eval_model") == eval_model:
            return True
    return False


def _remove_matching_judgment(entry: dict[str, Any], *, gen_model: str, eval_model: str) -> None:
    entry["judgments"] = [
        judgment
        for judgment in entry.get("judgments", []) or []
        if not (judgment.get("gen_model") == gen_model and judgment.get("eval_model") == eval_model)
    ]


def _record_error(prompt_id: int, exc: Exception) -> None:
    print(f"[warn] prompt_id={prompt_id} failed: {type(exc).__name__}: {exc}")


def eval_one(
    task: tuple[int, Path, Path],
    *,
    output_dir: Path,
    force_rerun: bool,
    gen_model_name: str,
    judge_model: str,
    client: JudgeClient,
    write_lock: threading.Lock,
) -> dict[str, Any] | None:
    prompt_id, json_path, image_path = task
    saved_path = _saved_item_path(output_dir, json_path.stem)
    load_path = json_path if force_rerun or not saved_path.exists() else saved_path
    item_data = json.loads(load_path.read_text(encoding="utf-8"))

    for eval_entry in item_data.get("evaluation", []) or []:
        if not force_rerun and _has_matching_judgment(
            eval_entry,
            gen_model=gen_model_name,
            eval_model=judge_model,
        ):
            continue

        question = str(eval_entry.get("question") or "").strip()
        if not question:
            continue

        _remove_matching_judgment(
            eval_entry,
            gen_model=gen_model_name,
            eval_model=judge_model,
        )
        raw_output = client.judge_image_text(
            image_path=image_path,
            system_prompt=None,
            user_prompt=build_factual_qa_judgment_prompt(question),
        ).strip()
        answer, reason = _parse_judgment(raw_output)
        eval_entry.setdefault("judgments", []).append(
            {
                "gen_model": gen_model_name,
                "eval_model": judge_model,
                "answer": answer,
                "reason": reason,
                "raw_output": raw_output,
            }
        )

        with write_lock:
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_text(
                json.dumps(item_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(
        json.dumps(item_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scores = _score_item(item_data, gen_model=gen_model_name, eval_model=judge_model)
    return {
        "prompt_id": prompt_id,
        "dataset_id": json_path.stem,
        "chart_type": item_data.get("chart_type"),
        "image_path": str(image_path),
        **scores,
    }


def main() -> None:
    args = _parse_args()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    judge_model = args.judge_model
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "items").mkdir(parents=True, exist_ok=True)

    client = JudgeClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=judge_model,
        timeout=args.eval_timeout,
    )

    items = _iter_items(data_dir, args.max_items)
    tasks = []
    missing_images = 0
    pending_questions = 0
    total_questions = 0

    for prompt_id, json_path in items:
        image_path = _resolve_image_path(image_dir, prompt_id)
        if image_path is None:
            print(f"[warn] skip prompt_id={prompt_id}: missing image")
            missing_images += 1
            continue

        saved_path = _saved_item_path(output_dir, json_path.stem)
        load_path = json_path if args.force_rerun or not saved_path.exists() else saved_path
        item_data = json.loads(load_path.read_text(encoding="utf-8"))
        done_q, total_q = _count_done_questions(
            item_data,
            gen_model=args.gen_model_name,
            eval_model=judge_model,
        )
        pending_questions += max(0, total_q - done_q)
        total_questions += total_q
        tasks.append((prompt_id, json_path, image_path))

    print(
        f"[igenbench] tasks={len(tasks)} missing_images={missing_images} "
        f"total_questions={total_questions} pending_questions={pending_questions} "
        f"concurrency={args.concurrency} judge_model={judge_model}"
    )

    write_lock = threading.Lock()

    results: list[dict[str, Any] | None] = []
    max_workers = max(1, args.concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                eval_one,
                task,
                output_dir=output_dir,
                force_rerun=args.force_rerun,
                gen_model_name=args.gen_model_name,
                judge_model=judge_model,
                client=client,
                write_lock=write_lock,
            ): task
            for task in tasks
        }
        progress = tqdm(total=len(futures), desc="igenbench eval", dynamic_ncols=True) if tqdm else None
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                prompt_id = futures[future][0]
                try:
                    result = future.result()
                except Exception as exc:
                    _record_error(prompt_id, exc)
                    traceback.print_exc()
                    result = None
                results.append(result)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(f"prompt_id={prompt_id}")
                else:
                    print(f"[{done}/{len(futures)}] prompt_id={prompt_id}")
        finally:
            if progress is not None:
                progress.close()

    valid_results = [result for result in results if result and result["total"] > 0]
    if not valid_results:
        raise RuntimeError("IGenBench produced no valid evaluation results.")

    q_total_by_type: dict[str, int] = defaultdict(int)
    q_correct_by_type: dict[str, int] = defaultdict(int)
    correct_questions = 0
    scored_questions = 0
    for result in valid_results:
        correct_questions += int(result["correct"])
        scored_questions += int(result["total"])
        for q_type, counts in result["q_by_type"].items():
            correct, total = counts
            q_correct_by_type[q_type] += correct
            q_total_by_type[q_type] += total

    records = [
        {
            "prompt_id": int(result["prompt_id"]),
            "dataset_id": result["dataset_id"],
            "chart_type": result["chart_type"],
            "image_path": result["image_path"],
            "accuracy": float(result["accuracy"]),
            "strict": bool(result["strict"]),
            "correct": int(result["correct"]),
            "total": int(result["total"]),
        }
        for result in sorted(valid_results, key=lambda item: int(item["prompt_id"]))
    ]
    question_type_acc = {
        q_type: q_correct_by_type[q_type] / q_total_by_type[q_type]
        for q_type in sorted(q_total_by_type)
        if q_total_by_type[q_type] > 0
    }
    summary = {
        "benchmark": "igenbench",
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "judge_model": judge_model,
        "gen_model_name": args.gen_model_name,
        "items": len(records),
        "skipped_items": len(items) - len(records),
        "missing_images": missing_images,
        "score": mean([record["accuracy"] for record in records]) if records else 0.0,
        "q_acc": correct_questions / scored_questions if scored_questions else 0.0,
        "i_acc_strict": mean([1.0 if record["strict"] else 0.0 for record in records]) if records else 0.0,
        "correct_questions": correct_questions,
        "total_questions": scored_questions,
        "question_type_acc": question_type_acc,
        "question_type_counts": dict(q_total_by_type),
        "records": records,
    }

    summary_path = output_dir / "igenbench_summary.json"
    records_path = output_dir / "igenbench_records.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[igenbench] score={summary['score']:.4f} q_acc={summary['q_acc']:.4f} "
        f"strict={summary['i_acc_strict']:.4f} "
        f"({correct_questions}/{scored_questions} correct, {len(records)} items)"
    )
    print(f"[igenbench] summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
