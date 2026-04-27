#!/usr/bin/env python3
"""
RealUnify Benchmark Score Calculator

Calculates accuracy for multiple-choice QA tasks.
Compatible with output from inference_realunify.py
"""

import argparse
import json
import os
import re
from collections import defaultdict


def extract_answer_from_response(response):
    """
    Extract answer (A/B/C/D) from model response.

    Priority:
    1. <answer>...</answer> tags
    2. First occurrence of A-D letter
    """
    if not response:
        return ""

    # 1. Try matching <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    candidate = ""
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = response.strip()

    # 2. Extract A, B, C, D from candidate
    choice_match = re.search(r"[A-D]", candidate, re.IGNORECASE)

    if choice_match:
        return choice_match.group(0).upper()

    return ""


def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json_or_jsonl(file_path):
    """Universal loader for JSON or JSONL"""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def evaluate_json_data(data_list):
    """
    Calculate accuracy by task type.

    Args:
        data_list: List of result items

    Returns:
        dict: { task_type: {'correct': int, 'total': int, 'acc': float} }
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in data_list:
        # Get model response (compatible with different field names)
        raw_response = item.get("model_response", item.get("response", ""))

        # Handle list type response
        if isinstance(raw_response, list) and len(raw_response) > 0:
            raw_response = raw_response[0]

        # Extract predicted answer
        pred_answer = extract_answer_from_response(str(raw_response))

        # Get ground truth answer
        gt_answer = item.get("answer", "").strip().upper()
        task_type = item.get("task_type", "unknown")

        # Statistics
        stats[task_type]["total"] += 1
        if pred_answer == gt_answer:
            stats[task_type]["correct"] += 1

    # Calculate accuracy
    final_results = {}
    for task, metrics in stats.items():
        total = metrics["total"]
        correct = metrics["correct"]
        acc = (correct / total) if total > 0 else 0.0
        final_results[task] = {"correct": correct, "total": total, "acc": acc}

    return final_results


def print_report(results):
    """Print formatted evaluation report"""
    print("\n" + "=" * 60)
    print(f"{'Task Type':<40} | {'Acc':<10} | {'Count'}")
    print("-" * 60)

    total_correct = 0
    total_count = 0

    sorted_tasks = sorted(results.keys())

    for task in sorted_tasks:
        metrics = results[task]
        acc_percent = metrics["acc"] * 100
        print(f"{task:<40} | {acc_percent:>6.2f}%    | ({metrics['correct']}/{metrics['total']})")

        total_correct += metrics["correct"]
        total_count += metrics["total"]

    print("-" * 60)
    overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0.0
    print(f"{'Overall':<40} | {overall_acc:>6.2f}%    | ({total_correct}/{total_count})")
    print("=" * 60 + "\n")

    return {"overall": {"correct": total_correct, "total": total_count, "acc": overall_acc / 100}, "by_task": results}


def save_results(results, output_path):
    """Save results to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate RealUnify Benchmark Score")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input JSONL file (inference results)"
    )
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the score results (optional)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    print(f"Loading data from: {args.input_file}")

    try:
        # Load data
        data = load_json_or_jsonl(args.input_file)
        print(f"Loaded {len(data)} samples")

        # Calculate scores
        results = evaluate_json_data(data)

        # Print report
        final_results = print_report(results)

        # Save results if output path specified
        if args.output_file:
            save_results(final_results, args.output_file)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
