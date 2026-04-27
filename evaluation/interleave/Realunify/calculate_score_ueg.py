#!/usr/bin/env python3
"""
Score RealUnify UEG inference results using Gemini judge.

Input: JSON (or JSONL) file with fields per sample:
  - generated_image (str or list[str])
  - task_type (one of 6 UEG task types)
  - question_list (list of {question, answer})

Output (saved next to input):
  - <input>_scored.json   full per-sample scores + judge outputs

NOTE: This script requires a GeminiAPI class for judge model calls.
      You need to provide your own implementation or API wrapper.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import traceback
from copy import deepcopy

from tqdm import tqdm

# You need to provide your own GeminiAPI implementation
# from your_api_module import GeminiAPI

# ================= Constants =================

TASK_TYPES = [
    "world_knowledge",
    "commonsense_reasoning",
    "logical_reasoning",
    "mathematical_reasoning",
    "scientific_reasoning",
    "code_to_image",
]


# ================= IO =================


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ================= Scoring =================


def adapt_generated_image(item):
    """Normalize generated_image to a single string path."""
    adapted = deepcopy(item)
    gi = adapted.get("generated_image")
    if isinstance(gi, list):
        adapted["generated_image"] = gi[-1] if gi else None
    return adapted


def process_one(item_0, gemini_api):
    """Score a single UEG sample via Gemini judge."""
    item = adapt_generated_image(item_0)
    judge_outputs = []

    try:
        score = 1
        img = item.get("generated_image")

        if not img or not os.path.exists(img):
            item["score"] = 0
            item["judge_outputs"] = [{"error": "Image not found or empty"}]
            return item

        for qa in item.get("question_list", []):
            origin_q = qa["question"]
            gt_answer = qa["answer"]

            prompt = (
                "Please answer the following question based on the image:\n"
                f"Question: {origin_q}\n\n"
                "You should only reply yes or no, and do not provide any other extra content."
            )

            try:
                response_text = gemini_api.generate_text(prompt, image_paths=[img], temperature=0.2)
            except Exception:
                traceback.print_exc()
                response_text = ""

            raw_response = response_text
            response_lower = response_text.lower().strip()
            parsed = ""
            if "yes" in response_lower:
                parsed = "yes"
            if "no" in response_lower:
                parsed = "no"

            judge_outputs.append(
                {
                    "question": origin_q,
                    "gt_answer": gt_answer,
                    "judge_raw_response": raw_response,
                    "judge_parsed_answer": parsed,
                    "correct": parsed == gt_answer,
                }
            )

            if parsed != gt_answer:
                score = 0
                break

        item["score"] = score
        item["judge_outputs"] = judge_outputs
        return item

    except Exception as e:
        print(f"Error: {repr(e)}")
        traceback.print_exc()
        return None


def evaluate_accuracy(data):
    """Print and return per-task + overall accuracy."""
    task_correct = {t: 0 for t in TASK_TYPES}
    task_total = {t: 0 for t in TASK_TYPES}

    for item in data:
        if item is None:
            continue
        tt = item.get("task_type")
        if tt not in TASK_TYPES:
            continue
        task_total[tt] += 1
        if item.get("score") == 1:
            task_correct[tt] += 1

    print("=" * 70)
    print("Realunify_UEG Evaluation Results")
    print("=" * 70)
    for t in TASK_TYPES:
        total = task_total[t]
        correct = task_correct[t]
        acc = correct / total if total > 0 else 0.0
        print(f"  {t:<25} {correct:>4}/{total:<4}  = {acc:.4f}")

    total_correct = sum(task_correct.values())
    total_samples = sum(task_total.values())
    total_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print("-" * 70)
    print(f"  {'Overall':<25} {total_correct:>4}/{total_samples:<4}  = {total_acc:.4f}")
    print("=" * 70)

    return {
        "per_task": {
            t: {
                "correct": task_correct[t],
                "total": task_total[t],
                "accuracy": task_correct[t] / task_total[t] if task_total[t] > 0 else 0.0,
            }
            for t in TASK_TYPES
        },
        "total": {"correct": total_correct, "total": total_samples, "accuracy": total_acc},
    }


# ================= Main =================


def calculate_score(input_file, num_workers=16):
    """Main scoring entry point."""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return None

    print(f"Loading: {input_file}")
    if input_file.endswith(".jsonl"):
        data = load_jsonl(input_file)
    else:
        data = load_json(input_file)
    print(f"Scoring {len(data)} samples with {num_workers} workers...")

    # NOTE: You need to provide your own GeminiAPI implementation
    # gemini_api = GeminiAPI()
    raise NotImplementedError(
        "GeminiAPI is not included. Please provide your own implementation "
        "of a Gemini judge API wrapper with a generate_text(prompt, image_paths, temperature) method."
    )

    scored = []
    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_one, item, gemini_api): item for item in data}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Scoring"):
            result = fut.result()
            if result is not None:
                scored.append(result)

    # Save scored results
    scored_path = input_file.rsplit(".", 1)[0] + "_scored.json"
    save_json(scored, scored_path)
    print(f"Saved: {scored_path}")

    # Evaluate
    results = evaluate_accuracy(scored)
    return results


def main():
    parser = argparse.ArgumentParser(description="Score RealUnify UEG results")
    parser.add_argument("--input_file", type=str, required=True, help="Path to ueg_results.json (or .jsonl)")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    calculate_score(args.input_file, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
