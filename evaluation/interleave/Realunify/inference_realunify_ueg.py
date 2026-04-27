#!/usr/bin/env python3
"""
NEO Model Inference for RealUnify UEG Benchmark

Three inference modes:
1. understand_t2i: Refine prompt via understanding, then t2i generation
2. interleave: Direct interleave generation with think (preprocessed prompt)
3. t2i: Direct t2i generation (preprocessed prompt)

Output: InternVL JSONL + JSON (compatible with external scoring script)
"""

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

# Import from inference_realunify.py
_REALUNIFY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REALUNIFY_DIR)
from inference_realunify import (
    INTERLEAVE_SYSTEM_PROMPT,
    NEOInferenceEngine,
    _build_think_kwargs,
    extract_final_answer,
    load_completed_ids,
    load_image_native,
    parse_square_image_size,
    resolve_output_path,
    save_result,
    set_random_seeds,
    setup_distributed,
)

# ============================================================================
# Constants
# ============================================================================

UEG_DATA_PATH = "<DATA_ROOT>/RealUnify/UEG_step.json"


# ============================================================================
# Engine Extension
# ============================================================================


class UEGInferenceEngine(NEOInferenceEngine):
    """Extended engine with t2i and text-only chat capabilities."""

    def chat_t2i(
        self,
        prompt: str,
        image_size: Tuple[int, int] = (1024, 1024),
        cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
        cfg_interval: Tuple[float, float] = (0.1, 1.0),
        cfg_norm: str = "none",
        num_steps: int = 50,
    ) -> Image.Image:
        """Text -> Image generation via model.t2i_generate()."""
        output = self.model.t2i_generate(
            self.tokenizer,
            prompt,
            image_size=image_size,
            cfg_scale=cfg_scale,
            cfg_norm=cfg_norm,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            batch_size=1,
        )
        return self._tensor_to_pil(output)

    def chat_text(self, prompt: str, max_new_tokens: int = 512, do_sample: bool = False) -> str:
        """Text-only chat (no image input). Used for prompt refinement.

        Creates a small dummy image so the vision model runs without error,
        but does NOT include <image> in the prompt, so visual features are
        not injected into the text sequence.
        """
        import tempfile

        # Create a small dummy image (32x32 white)
        dummy_img = Image.new("RGB", (32, 32), (255, 255, 255))
        dummy_path = os.path.join(tempfile.gettempdir(), "_ueg_dummy.png")
        dummy_img.save(dummy_path)

        device = next(self.model.parameters()).device
        pixel_values, grid_hw = load_image_native(
            dummy_path,
            patch_size=16,
            downsample_ratio=0.5,
            min_pixels=256,
            max_pixels=4096,
            upscale=False,
            device=device,
        )

        generation_config = dict(do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=None, num_beams=1)
        # No <image> tag in prompt -- vision features computed but not used
        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            grid_hw=grid_hw,
            question=prompt,
            generation_config=generation_config,
        )
        return response


# ============================================================================
# Prompt Preprocessing
# ============================================================================


def extract_raw_prompt(new_prompt: str) -> str:
    """Extract raw generation prompt from UEG new_prompt.

    Removes "Here is the prompt for image generation:" prefix
    and everything after the first "\\n\\n".
    """
    prefix = "Here is the prompt for image generation:"
    text = new_prompt
    idx = text.find(prefix)
    if idx >= 0:
        text = text[idx + len(prefix) :]
    nn_idx = text.find("\n\n")
    if nn_idx >= 0:
        text = text[:nn_idx]
    return text.strip()


# ============================================================================
# Data Loading
# ============================================================================


def load_ueg_data(data_path: str) -> List[Dict]:
    """Load UEG benchmark data from JSON."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Add hash_uid from index (for resume compatibility)
    for item in data:
        item["hash_uid"] = str(item["index"])
    return data


# ============================================================================
# Processing Functions
# ============================================================================


def process_ueg_understand_t2i(
    engine: UEGInferenceEngine,
    item: Dict,
    output_dir: str,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm: str,
    num_steps: int,
    image_size: Tuple[int, int] = (1024, 1024),
) -> Optional[Dict]:
    """Mode 1: Understanding + T2I two-step pipeline.

    Step 1: Refine prompt via text-only chat (new_prompt -> refined prompt)
    Step 2: Generate image from refined prompt via t2i
    """
    hash_uid = item.get("hash_uid", "unknown")
    task_type = item.get("task_type", "unknown")

    try:
        images_dir = os.path.join(output_dir, "images", task_type)
        os.makedirs(images_dir, exist_ok=True)

        # Step 1: Understanding - refine prompt
        refined_prompt = engine.chat_text(item["new_prompt"])

        # Step 2: T2I generation
        generated_image = engine.chat_t2i(
            prompt=refined_prompt,
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm=cfg_norm,
            num_steps=num_steps,
        )

        # Save image
        img_filename = f"{hash_uid}_gen.png"
        img_path = os.path.join(images_dir, img_filename)
        generated_image.save(img_path)
        abs_img_path = os.path.abspath(img_path)

        result = {
            "index": item.get("index", -1),
            "image": [abs_img_path],
            "conversations": [
                {"from": "human", "value": item["new_prompt"]},
                {"from": "gpt", "value": refined_prompt},
                {"from": "human", "value": refined_prompt},
                {"from": "gpt", "value": "<image>"},
            ],
            "generated_images": [abs_img_path],
            "generated_image": abs_img_path,
            "task_type": task_type,
            "question_list": item.get("question_list", []),
            "hash_uid": hash_uid,
            "inference_mode": "understand_t2i",
            "new_prompt": item.get("new_prompt", ""),
            "model_response": refined_prompt,
            "mid_output": refined_prompt + "\n\n<image>",
        }
        return result

    except Exception as e:
        print(f"Error processing {hash_uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_ueg_interleave(
    engine: UEGInferenceEngine,
    item: Dict,
    output_dir: str,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm: str,
    num_steps: int,
    timestep_shift: float = 3.0,
    image_size: Tuple[int, int] = (1024, 1024),
) -> Optional[Dict]:
    """Mode 2: Direct interleave generation with think.

    Preprocesses new_prompt to extract raw prompt, then uses interleave_gen.
    """
    hash_uid = item.get("hash_uid", "unknown")
    task_type = item.get("task_type", "unknown")

    try:
        images_dir = os.path.join(output_dir, "images", task_type)
        os.makedirs(images_dir, exist_ok=True)

        raw_prompt = extract_raw_prompt(item["new_prompt"])

        generated_text, generated_images = engine.chat_interleave(
            prompt=raw_prompt,
            input_images=None,
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            num_steps=num_steps,
            system_message=INTERLEAVE_SYSTEM_PROMPT,
        )

        # Save generated images
        generated_image_paths = []
        for idx, img in enumerate(generated_images):
            img_filename = f"{hash_uid}_gen_{idx}.png"
            img_path = os.path.join(images_dir, img_filename)
            img.save(img_path)
            generated_image_paths.append(os.path.abspath(img_path))

        final_answer = extract_final_answer(generated_text)

        # For scoring: use last generated image
        last_img = generated_image_paths[-1] if generated_image_paths else ""

        result = {
            "index": item.get("index", -1),
            "image": generated_image_paths,
            "conversations": [
                {"from": "system", "value": INTERLEAVE_SYSTEM_PROMPT},
                {"from": "human", "value": raw_prompt},
                {"from": "gpt", "value": generated_text},
            ],
            "generated_images": generated_image_paths,
            "generated_image": last_img,
            "task_type": task_type,
            "question_list": item.get("question_list", []),
            "hash_uid": hash_uid,
            "inference_mode": "interleave",
            "new_prompt": item.get("new_prompt", ""),
            "model_response": final_answer,
            "mid_output": generated_text,
            "full_generated_text": generated_text,
        }
        return result

    except Exception as e:
        print(f"Error processing {hash_uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_ueg_t2i(
    engine: UEGInferenceEngine,
    item: Dict,
    output_dir: str,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm: str,
    num_steps: int,
    image_size: Tuple[int, int] = (1024, 1024),
) -> Optional[Dict]:
    """Mode 3: Direct t2i generation with preprocessed prompt."""
    hash_uid = item.get("hash_uid", "unknown")
    task_type = item.get("task_type", "unknown")

    try:
        images_dir = os.path.join(output_dir, "images", task_type)
        os.makedirs(images_dir, exist_ok=True)

        raw_prompt = extract_raw_prompt(item["new_prompt"])

        generated_image = engine.chat_t2i(
            prompt=raw_prompt,
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm=cfg_norm,
            num_steps=num_steps,
        )

        img_filename = f"{hash_uid}_gen.png"
        img_path = os.path.join(images_dir, img_filename)
        generated_image.save(img_path)
        abs_img_path = os.path.abspath(img_path)

        result = {
            "index": item.get("index", -1),
            "image": [abs_img_path],
            "conversations": [{"from": "human", "value": raw_prompt}, {"from": "gpt", "value": "<image>"}],
            "generated_images": [abs_img_path],
            "generated_image": abs_img_path,
            "task_type": task_type,
            "question_list": item.get("question_list", []),
            "hash_uid": hash_uid,
            "inference_mode": "t2i",
            "new_prompt": item.get("new_prompt", ""),
            "model_response": "<image>",
            "mid_output": "<image>",
        }
        return result

    except Exception as e:
        print(f"Error processing {hash_uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NEO Model Inference for RealUnify UEG Benchmark")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=UEG_DATA_PATH)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_mode", type=str, required=True, choices=["understand_t2i", "interleave", "t2i"])
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--img_cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0], metavar=("START", "END"))
    parser.add_argument("--cfg_norm", type=str, default="none", choices=["none", "global", "channel"])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--target_image_size",
        type=parse_square_image_size,
        default=None,
        help="Force square image size (e.g. 1024). Default: 1024",
    )
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=None)

    args = parser.parse_args()
    args.cfg_interval = tuple(args.cfg_interval)

    # Default image size
    image_size_val = args.target_image_size if args.target_image_size else 1024
    image_size = (image_size_val, image_size_val)

    set_random_seeds(args.seed)

    # Distributed setup
    if args.device_map is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError("device_map mode must be run as a single process.")
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"

    max_memory = None
    if args.device_map is not None and args.max_memory_per_gpu_gb is not None:
        max_memory = {gpu_idx: f"{args.max_memory_per_gpu_gb}GiB" for gpu_idx in range(torch.cuda.device_count())}

    # Load model
    engine = UEGInferenceEngine(
        args.model_path,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
    )

    # Load data
    data = load_ueg_data(args.data_path)
    print(f"Loaded {len(data)} UEG samples from {args.data_path}")

    # Output paths
    output_jsonl = resolve_output_path(args.output_dir, "ueg_results.jsonl")
    output_json = os.path.join(args.output_dir, "ueg_results.json")

    # Resume
    if args.resume:
        completed = load_completed_ids(output_jsonl)
        data = [item for item in data if item.get("hash_uid") not in completed]
        print(f"Resume mode: {len(completed)} completed, {len(data)} remaining")

    # Limit
    if args.limit:
        data = data[: args.limit]
        print(f"Limited to {len(data)} samples")

    # Distributed sharding
    if world_size > 1:
        data = data[rank::world_size]
        print(f"Rank {rank}/{world_size}: processing {len(data)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    # Select process function
    process_fn_map = {
        "understand_t2i": process_ueg_understand_t2i,
        "interleave": process_ueg_interleave,
        "t2i": process_ueg_t2i,
    }
    process_fn = process_fn_map[args.inference_mode]

    # Common kwargs
    common_kwargs = dict(
        engine=engine,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        img_cfg_scale=args.img_cfg_scale,
        cfg_interval=args.cfg_interval,
        cfg_norm=args.cfg_norm,
        num_steps=args.num_steps,
        image_size=image_size,
    )
    if args.inference_mode == "interleave":
        common_kwargs["timestep_shift"] = args.timestep_shift

    # Process
    success_count = 0
    fail_count = 0
    all_results = []

    pbar = tqdm(data, desc=f"UEG {args.inference_mode} (rank {rank})", disable=(rank != 0))
    for item in pbar:
        result = process_fn(item=item, **common_kwargs)

        if result:
            save_result(result, output_jsonl)
            all_results.append(result)
            success_count += 1
        else:
            fail_count += 1

        if args.inference_mode == "interleave":
            gc.collect()
            torch.cuda.empty_cache()

        pbar.set_postfix(success=success_count, fail=fail_count)

    # Save JSON for external scoring
    if rank == 0 or world_size == 1:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_results)} results to {output_json}")

    # Sync
    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: {success_count} success, {fail_count} failed")
    if rank == 0:
        print(f"\nOutput: {args.output_dir}")
        print(f"JSONL: {output_jsonl}")
        print(f"JSON: {output_json}")


if __name__ == "__main__":
    main()
