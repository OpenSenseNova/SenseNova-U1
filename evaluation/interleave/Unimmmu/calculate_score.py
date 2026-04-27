#!/usr/bin/env python3
"""
Unimmmu Benchmark Score Calculator

Wraps the evaluation logic from the image_text_agent benchmark repository.
Supports both direct and tool_call modes.

Usage:
    python calculate_score.py --input_file results.jsonl --output_dir ./scores
"""

import argparse
import json
import os
import sys


def load_json_or_jsonl(file_path):
    """Universal loader for JSON or JSONL files"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
    return data


def main():
    parser = argparse.ArgumentParser(description="Unimmmu Benchmark Score Calculator")
    parser.add_argument("--input_file", type=str, required=True, help="Path to inference results JSONL")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for scores (default: same as input)"
    )
    parser.add_argument(
        "--use_tools", action="store_true", help="Use tool_call mode for geometry evaluation (default: direct)"
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=None,
        help="Path to the image_text_agent benchmark repo (containing evaluation/)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input_file))

    print(f"Loading data from: {args.input_file}")
    data = load_json_or_jsonl(args.input_file)
    print(f"Loaded {len(data)} samples")

    # Print task distribution
    task_counts = {}
    for item in data:
        t = item.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"Task distribution: {task_counts}")

    # Import scorer from benchmark repo
    benchmark_path = args.benchmark_path
    if benchmark_path is None:
        print("Error: --benchmark_path is required. Please provide the path to the image_text_agent benchmark repo.")
        sys.exit(1)

    if benchmark_path not in sys.path:
        sys.path.insert(0, benchmark_path)

    try:
        from evaluation.all_benchmark.calculate_score.unimmmu import calculate_score

        # Call Unimmmu score calculation
        # use_tools=False for direct mode (Unimmmu_direct)
        calculate_score(input_data=data, use_tools=args.use_tools, output_dir=args.output_dir)

    except ImportError as e:
        print(f"Error: Could not import scorer from benchmark repo: {e}")
        print(f"Benchmark path: {benchmark_path}")
        print("Make sure the benchmark repository is available at the expected path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
