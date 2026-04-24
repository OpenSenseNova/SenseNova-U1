from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EN_DATA_PATH = DEFAULT_DATA_DIR / "text_prompts.jsonl"
DEFAULT_ZH_DATA_PATH = DEFAULT_DATA_DIR / "text_prompts_zh.jsonl"


# Reference:
#   X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image
#   Generative Models Great Again
#   https://arxiv.org/abs/2507.22058


def clean_and_remove_hallucinations(text: str) -> str:
    keywords_list = ["addCriterion", "No text recognized."]
    s = text
    for keyword in keywords_list:
        s = s.replace(keyword, "").replace(f"\n{keyword}", "").replace(f"{keyword}\n", "")
    return s


class ImageEvaluator:
    def __init__(self, device: int | str, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
        self.device = device

        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        self.qwen_model.eval()
        self.qwen_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.qwen_prompt = (
            "Recognize the text in the image, only reply with the text content, "
            "but avoid repeating previously mentioned content. "
            "If no text is recognized, please reply with 'No text recognized'."
        )

    def qwen_ocr(self, image: str) -> str:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.qwen_prompt},
                ],
            }
        ]
        texts = self.qwen_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(message)
        inputs = self.qwen_processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        outputs = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return clean_and_remove_hallucinations(outputs[0])

    def evaluate(self, data_chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eval_results = []
        with torch.no_grad():
            for data in tqdm(data_chunk):
                ocr_results = self.qwen_ocr(data["image"])
                eval_results.append(
                    {
                        "image": data["image"],
                        "prompt_id": data["prompt_id"],
                        "prompt": data["prompt"],
                        "category": data.get("category", ""),
                        "length": data.get("length", ""),
                        "text_length": data.get("text_length"),
                        "ocr_gt": data["text"],
                        "ocr_results": ocr_results,
                    }
                )
        return eval_results


def split_list(x: list[Any], n: int) -> list[list[Any]]:
    k, m = divmod(len(x), n)
    return [x[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def prompt_file_for_mode(mode: str) -> Path:
    return DEFAULT_EN_DATA_PATH if mode == "en" else DEFAULT_ZH_DATA_PATH


def load_prompts(prompt_file: Path) -> dict[int, dict[str, Any]]:
    with prompt_file.open(encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    return {int(p["prompt_id"]): p for p in prompts}


def collect_data(sample_dir: str, prompt_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    data = []
    for image_file in sorted(glob.glob(f"{sample_dir}/*.png")):
        fname = os.path.basename(image_file)
        prompt_id = int(fname.split("_")[0] if "_" in fname else fname.split(".")[0])
        info = prompt_map[prompt_id]
        data.append(
            {
                "image": image_file,
                "prompt_id": prompt_id,
                "prompt": info["prompt"],
                "text": info["text"],
                "category": info.get("category", ""),
                "length": info.get("length", ""),
                "text_length": info.get("text_length"),
            }
        )
    return data


def preprocess_string(s: str, mode: str = "en") -> str:
    cleaned = re.sub(
        r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]",
        "",
        s,
    )
    if mode == "en":
        return re.sub(r"\s+", " ", cleaned).strip().lower()
    pattern = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]")
    return "".join(pattern.findall(s)).strip()


def counter2list(counter: Counter) -> list[str]:
    return [item for item, count in counter.items() for _ in range(count)]


def calculate_match_count(text_gt: str, ocr_str: str, mode: str = "en") -> tuple[int, int]:
    if mode == "en":
        gt_tokens = text_gt.split()
        ocr_tokens = ocr_str.split()
        return len(counter2list(Counter(gt_tokens) & Counter(ocr_tokens))), len(gt_tokens)

    return len(counter2list(Counter(text_gt) & Counter(ocr_str))), len(text_gt)


def score_row(row: dict[str, Any], mode: str) -> dict[str, Any]:
    ocr_gt_raw = row["ocr_gt"]
    if isinstance(ocr_gt_raw, list):
        ocr_gt_raw = " ".join(str(x) for x in ocr_gt_raw)
    ocr_gt = preprocess_string(str(ocr_gt_raw), mode)
    ocr_results = preprocess_string(str(row.get("ocr_results", "")), mode)
    matched, total_gt = calculate_match_count(ocr_gt, ocr_results, mode)
    scored = dict(row)
    scored["ocr_gt"] = ocr_gt_raw
    scored["matched"] = matched
    scored["total_gt"] = total_gt
    scored["score"] = matched / total_gt if total_gt else 0.0
    return scored


def aggregate_score(rows: list[dict[str, Any]]) -> float:
    matched = sum(int(row["matched"]) for row in rows)
    total_gt = sum(int(row["total_gt"]) for row in rows)
    return matched / total_gt if total_gt else 0.0


def aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return {
        name: {
            "count": len(items),
            "score": aggregate_score(items),
            "matched": sum(int(item["matched"]) for item in items),
            "total_gt": sum(int(item["total_gt"]) for item in items),
        }
        for name, items in sorted(grouped.items())
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(output_dir: Path, mode: str, prompt_file: Path) -> None:
    rows = []
    for chunk_file in sorted(output_dir.glob("results_chunk*.jsonl")):
        rows.extend(load_jsonl(chunk_file))
    rows = [score_row(row, mode) for row in sorted(rows, key=lambda x: int(x["prompt_id"]))]

    write_jsonl(output_dir / "results.jsonl", rows)

    score = aggregate_score(rows)
    summary = {
        "benchmark": f"longtext_{mode}",
        "prompt_file": str(prompt_file),
        "items": len(rows),
        "score": score,
        "matched": sum(int(row["matched"]) for row in rows),
        "total_gt": sum(int(row["total_gt"]) for row in rows),
        "by_category": aggregate_by(rows, "category"),
        "by_length": aggregate_by(rows, "length"),
    }
    with (output_dir / "longtext_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (output_dir / "scores.txt").open("w", encoding="utf-8") as f:
        f.write(f"Text Score: {score:.4f}\n")

    print(f"[longtext] Text Score: {score:.4f}")
    print(f"[longtext] results -> {output_dir / 'results.jsonl'}")
    print(f"[longtext] summary -> {output_dir / 'longtext_summary.json'}")


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    device: int | str = "cuda" if torch.cuda.is_available() else "cpu"

    seed = args.global_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = Path(args.prompt_file) if args.prompt_file else prompt_file_for_mode(args.mode)
    prompt_map = load_prompts(prompt_file)
    data = collect_data(args.sample_dir, prompt_map)

    evaluator = ImageEvaluator(device)
    print(f"=============Evaluate {len(data)} images on device {device}=============")
    results = evaluator.evaluate(data)
    write_jsonl(output_dir / "results_chunk0.jsonl", results)
    write_summary(output_dir, args.mode, prompt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", "--image-dir", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True)
    parser.add_argument("--mode", "--lang", type=str, choices=["en", "zh"], default="en")
    parser.add_argument("--prompt_file", "--data-path", type=str, default=None)
    parser.add_argument("--global_seed", type=int, default=42)
    main(parser.parse_args())
