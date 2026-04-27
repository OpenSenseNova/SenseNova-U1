#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge Unimmmu shard jsonl files")
    parser.add_argument("--data_path", required=True, help="Original Unimmmu jsonl for ordering and coverage check")
    parser.add_argument("--shard_dir", required=True, help="Directory containing unimmmu_results_shard_*.jsonl")
    parser.add_argument("--output_file", required=True, help="Merged output jsonl path")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "unimmmu_results_shard_*.jsonl")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {args.shard_dir}")

    order = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            order.append(item.get("hash_uid"))
    order_map: Dict[str, int] = {uid: i for i, uid in enumerate(order) if uid is not None}

    merged: Dict[str, dict] = {}
    duplicates = 0
    for path in shard_paths:
        for item in load_jsonl(path):
            uid = item.get("hash_uid")
            if uid is None:
                continue
            if uid in merged:
                duplicates += 1
                continue
            merged[uid] = item

    ordered_items = sorted(merged.values(), key=lambda x: order_map.get(x.get("hash_uid"), 10**18))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in ordered_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    missing = [uid for uid in order if uid is not None and uid not in merged]
    print(f"Shard files: {len(shard_paths)}")
    print(f"Merged samples: {len(ordered_items)}")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Missing samples: {len(missing)}")
    if missing:
        print("First missing hash_uids:", missing[:20])
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()
