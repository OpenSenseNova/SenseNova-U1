#!/usr/bin/env python3
"""
NEO Model Inference for Unimmmu Benchmark (Unimmmu_direct)

Two modes:
1. i2t:        Multi-image understanding only (model.chat)
2. interleave: Multimodal reasoning with text+image generation (model.interleave_gen)

Output format: JSONL compatible with unimmmu.calculate_score
"""

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ============================================================================
# Constants (same as Realunify)
# ============================================================================

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

UNIMMMU_DATA_PATH = "<DATA_ROOT>/unimmmu/vqa/unimmmu_direct.jsonl"

INTERLEAVE_SYSTEM_PROMPT = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:

Think Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.

Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.

After the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""

# ============================================================================
# Utility Functions (reused from Realunify)
# ============================================================================


def set_random_seeds(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def parse_square_image_size(value):
    raw_value = str(value).strip().lower()
    if "x" in raw_value:
        width_str, height_str = raw_value.split("x", 1)
    elif "*" in raw_value:
        width_str, height_str = raw_value.split("*", 1)
    else:
        width_str = raw_value
        height_str = raw_value

    if not width_str.isdigit() or not height_str.isdigit():
        raise argparse.ArgumentTypeError(
            f"target_image_size must be a positive integer or square spec like 1024x1024, got {value!r}"
        )

    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"target_image_size must be positive, got {value!r}")
    if width != height:
        raise argparse.ArgumentTypeError(f"target_image_size must be square, got {value!r}")
    return width


def resolve_target_image_size(
    width: int, height: int, target_image_size: Optional[int], min_pixels: int, max_pixels: int
):
    if target_image_size is not None:
        return target_image_size, target_image_size
    resized_h, resized_w = smart_resize(height, width, factor=32, min_pixels=min_pixels, max_pixels=max_pixels)
    return resized_h, resized_w


def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 256 * 32 * 32, max_pixels: int = 16384 * 32 * 32
) -> tuple:
    """Smart resize from Qwen2.5-VL"""
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


def setup_distributed():
    """Initialize distributed environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


# ============================================================================
# Image Processing Functions (reused from Realunify)
# ============================================================================


def dynamic_preprocess_native_resolution(image, size_factor=32, min_pixels=65536, max_pixels=4194304, **kwargs):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image


def preprocess_pixel_values(pixel_values, patch_size=16):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]]).to(device=pixel_values.device)
    return flatten_pixel_values, grid_hw


def get_contrasting_background(image):
    """Get contrasting background color for RGBA images"""
    if image.mode != "RGBA":
        return None
    alpha = image.split()[3]
    if alpha.getextrema() == (255, 255):
        return None
    return (255, 255, 255)


def load_image_native(
    image_file, patch_size=16, downsample_ratio=0.5, min_pixels=65536, max_pixels=4194304, upscale=False, device="cuda"
):
    """Load and preprocess image for model.chat()"""
    image = Image.open(image_file)
    if image.mode == "RGBA":
        bg_color = get_contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    new_image = dynamic_preprocess_native_resolution(
        image, size_factor=int(patch_size // downsample_ratio), min_pixels=min_pixels, max_pixels=max_pixels
    )
    pixel_values, grid_hw = preprocess_pixel_values(transform(new_image).to(torch.float32), patch_size=patch_size)

    grid_hw = grid_hw.to(device)
    pixel_values = pixel_values.to(device).to(torch.bfloat16)

    return pixel_values, grid_hw


# ============================================================================
# NEO Inference Engine
# ============================================================================


def _build_think_kwargs(model_func, think_pattern):
    params = inspect.signature(model_func).parameters
    if "think_pattern" in params:
        return {"think_pattern": think_pattern}
    elif "think_mode" in params:
        return {"think_mode": think_pattern == "think"}
    return {}


class NEOInferenceEngine:
    """
    NEO model inference engine for Unimmmu benchmark

    Supports:
    - chat_understanding(): Single-image QA (Image + Text -> Text)
    - chat_understanding_multi(): Multi-image QA (Images + Text -> Text)
    - chat_interleave(): Multimodal reasoning (Images + Text -> Text + Images)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = None,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        self.device = device
        self.device_map = device_map

        print(f"Loading model from {model_path}...")
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if device_map is not None:
            model_kwargs["device_map"] = device_map
            model_kwargs["low_cpu_mem_usage"] = True
            model_kwargs["debug_key_match"] = True

            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded successfully!")

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD) -> torch.Tensor:
        """Denormalize tensor: x: [B,3,H,W] normalized -> [0,1] clamped"""
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert model output tensor to PIL Image"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        image = self._denorm(tensor.float())
        image = (image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(image[0])

    def _tensors_to_pils(self, tensors: List[torch.Tensor]) -> List[Image.Image]:
        """Convert list of tensors to list of PIL Images."""
        pil_images = []
        for t in tensors:
            image = self._denorm(t.float())
            image = (image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            pil_images.append(Image.fromarray(image[0]))
        return pil_images

    def chat_understanding(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
    ) -> str:
        """
        Single-image understanding/QA: Image + Text -> Text
        """
        pixel_values, grid_hw = load_image_native(
            image_path,
            patch_size=16,
            downsample_ratio=0.5,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            upscale=False,
            device=self.device,
        )

        if not question.startswith("<image>"):
            question = "<image>\n" + question

        generation_config = dict(do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=None, num_beams=1)

        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            grid_hw=grid_hw,
            question=question,
            generation_config=generation_config,
        )

        return response

    def chat_understanding_multi(
        self,
        image_paths: List[str],
        question: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
    ) -> str:
        """
        Multi-image understanding: Multiple Images + Text -> Text

        Loads each image separately, concatenates pixel_values and grid_hw,
        then calls model.chat() with the concatenated inputs.
        """
        all_pixel_values = []
        all_grid_hw = []

        for img_path in image_paths:
            pixel_values, grid_hw = load_image_native(
                img_path,
                patch_size=16,
                downsample_ratio=0.5,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                upscale=False,
                device=self.device,
            )
            all_pixel_values.append(pixel_values)
            all_grid_hw.append(grid_hw)

        # Concatenate all images along the first dimension
        pixel_values = torch.cat(all_pixel_values, dim=0)
        grid_hw = torch.cat(all_grid_hw, dim=0)

        generation_config = dict(do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=None, num_beams=1)

        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            grid_hw=grid_hw,
            question=question,
            generation_config=generation_config,
        )

        return response

    def chat_interleave(
        self,
        prompt: str,
        input_images: Optional[List[Image.Image]] = None,
        image_size: Tuple[int, int] = (512, 512),
        cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
        cfg_interval: Tuple[float, float] = (0.1, 1.0),
        cfg_norm: str = "none",
        timestep_shift: float = 1.0,
        num_steps: int = 50,
        system_message: str = "",
    ) -> Tuple[str, List[Image.Image]]:
        """
        Interleaved generation: predicts both text and images.

        Args:
            prompt: Input prompt with <image> tags for input images
            input_images: List of input PIL Images
            image_size: Output image size (W, H)
            cfg_scale: CFG scale for text guidance
            img_cfg_scale: CFG scale for image guidance
            timestep_shift: Timestep shift for generation
            num_steps: Number of generation steps
            system_message: System message for the model

        Returns:
            Tuple of (generated_text, list_of_generated_images)
        """
        # Clear any existing KV cache before generation
        if hasattr(self.model, "clear_kv_cache"):
            self.model.clear_kv_cache()

        text, images = self.model.interleave_gen(
            self.tokenizer,
            prompt,
            image_size=image_size,
            images=input_images,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_interval=cfg_interval,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            num_steps=num_steps,
            system_message=system_message,
            **_build_think_kwargs(self.model.interleave_gen, "think"),
        )

        # Convert tensors to PIL images and clear GPU memory
        pil_images = self._tensors_to_pils(images)

        # Explicit cleanup to prevent memory accumulation
        del images
        torch.cuda.empty_cache()

        return text, pil_images


# ============================================================================
# Unimmmu Data Loading & Parsing
# ============================================================================


def load_unimmmu_data(data_path: str = UNIMMMU_DATA_PATH) -> List[Dict]:
    """Load Unimmmu benchmark data from JSONL"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data


def parse_messages_to_prompt_and_images(item: Dict) -> Tuple[str, List[str]]:
    """
    Parse the 'messages' field to extract prompt text and image paths.

    The messages field has format:
    [{"role": "user", "content": [
        {"type": "text", "text": "prompt"},
        {"type": "image_url", "image_url": {"url": "path.png"}},
        ...
    ]}]

    Returns:
        (prompt_text, image_paths)
    """
    messages = item.get("messages", [])
    prompt_parts = []
    image_paths = []

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            prompt_parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        text = c.get("text", "")
                        if text:
                            prompt_parts.append(text)
                    elif c.get("type") == "image_url":
                        url = c.get("image_url", {}).get("url", "")
                        if url:
                            image_paths.append(url)

    prompt = prompt_parts[0] if prompt_parts else ""
    return prompt, image_paths


def load_completed_ids(output_path: str) -> set:
    """Load completed sample IDs for resume"""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "hash_uid" in item:
                        completed.add(item["hash_uid"])
                except:
                    continue
    return completed


def extract_final_answer(generated_text: str) -> str:
    """
    Extract final answer after </think> tag.
    Example: '<think>aaa<image></think>bbb' -> 'bbb'
    If no </think> tag, return the full text.
    """
    if "</think>" in generated_text:
        return generated_text.split("</think>")[-1].strip()
    return generated_text.strip()


def save_result(result: Dict, output_path: str):
    """Save result to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def resolve_output_path(output_dir: str, base_filename: str, shard_rank: Optional[int] = None) -> str:
    """Resolve output jsonl path. Sharded runs write into output_dir/shards/."""
    if shard_rank is None or shard_rank < 0:
        return os.path.join(output_dir, base_filename)

    stem, ext = os.path.splitext(base_filename)
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"{stem}_shard_{shard_rank:03d}{ext}")


# ============================================================================
# Processing Functions
# ============================================================================


def process_unimmmu_i2t(
    engine: NEOInferenceEngine,
    item: Dict,
    output_dir: str,
    min_pixels: int = 1024 * 1024,
    max_pixels: int = 2048 * 2048,
) -> Optional[Dict]:
    """
    Process a single Unimmmu sample with i2t mode (understanding only).

    No image generation -- only text response via model.chat().
    Supports 1-3 input images depending on task type.
    """
    hash_uid = item.get("hash_uid", "unknown")
    task = item.get("task", "unknown")

    try:
        # Parse messages to get prompt and image paths
        prompt, image_paths = parse_messages_to_prompt_and_images(item)

        if not image_paths:
            print(f"Warning: No images found for sample {hash_uid}")
            return None

        # Validate all image paths exist
        valid_paths = []
        for p in image_paths:
            if os.path.exists(p):
                valid_paths.append(p)
            else:
                print(f"Warning: Image not found: {p}")

        if not valid_paths:
            print(f"Warning: No valid images for sample {hash_uid}")
            return None

        # Format prompt with <image> tags for each image
        formatted_prompt = ""
        for _ in valid_paths:
            formatted_prompt += "<image>\n"
        formatted_prompt += prompt

        # Call multi-image understanding
        response = engine.chat_understanding_multi(
            image_paths=valid_paths,
            question=formatted_prompt,
            max_new_tokens=512,
            do_sample=False,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Build result -- preserve all original fields for scorer compatibility
        result = {
            "task": task,
            "id": item.get("id", ""),
            "hash_uid": hash_uid,
            "model_response": response,
            "gt_text": item.get("gt_text", ""),
            "reasoning_imgs": item.get("reasoning_imgs", []),
            "inference_mode": "i2t",
        }

        # Preserve geometry-specific fields for scorer
        if task == "geometry":
            result["task_type"] = item.get("task_type", "")
            result["reasoning_qs"] = item.get("reasoning_qs", "")
            result["auxiliary_qs"] = item.get("auxiliary_qs", "")
            result["gt_auxiliary_imgs"] = item.get("gt_auxiliary_imgs", [])

        return result

    except Exception as e:
        print(f"Error processing {hash_uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_unimmmu_interleave(
    engine: NEOInferenceEngine,
    item: Dict,
    output_dir: str,
    cfg_scale: float,
    img_cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm: str,
    num_steps: int,
    timestep_shift: float = 1.0,
    min_pixels: int = 1024 * 1024,
    max_pixels: int = 2048 * 2048,
    target_image_size: Optional[int] = None,
) -> Optional[Dict]:
    """
    Process a single Unimmmu sample with interleave mode.

    The model uses multimodal reasoning (thinking mode) to generate
    both text and images. System prompt is applied.
    """
    hash_uid = item.get("hash_uid", "unknown")
    task = item.get("task", "unknown")

    try:
        # Create output directory for generated images
        images_output_dir = os.path.join(output_dir, "images", task)
        os.makedirs(images_output_dir, exist_ok=True)

        # Parse messages to get prompt and image paths
        prompt, image_paths = parse_messages_to_prompt_and_images(item)

        if not image_paths:
            print(f"Warning: No images found for sample {hash_uid}")
            return None

        # Load and resize all input images
        input_images = []
        valid_paths = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            resized_h, resized_w = resolve_target_image_size(w, h, target_image_size, min_pixels, max_pixels)
            img = img.resize((resized_w, resized_h))
            input_images.append(img)
            valid_paths.append(img_path)

        if not input_images:
            print(f"Warning: No valid images for sample {hash_uid}")
            return None

        # Compute output image size from first input image (W, H format)
        image_size = (input_images[0].width, input_images[0].height)

        # Format prompt with <image> tags
        formatted_prompt = ""
        for _ in input_images:
            formatted_prompt += "<image>\n"
        formatted_prompt += prompt

        # Call interleave generation with system prompt
        generated_text, generated_images = engine.chat_interleave(
            prompt=formatted_prompt,
            input_images=input_images,
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
            img_path = os.path.join(images_output_dir, img_filename)
            img.save(img_path)
            generated_image_paths.append(os.path.abspath(img_path))

        # Extract final answer (after </think> tag)
        final_answer = extract_final_answer(generated_text)

        # Build result -- preserve all original fields for scorer compatibility
        result = {
            "task": task,
            "id": item.get("id", ""),
            "hash_uid": hash_uid,
            "model_response": final_answer,
            "full_generated_text": generated_text,
            "generated_images": generated_image_paths,
            "gt_text": item.get("gt_text", ""),
            "reasoning_imgs": item.get("reasoning_imgs", []),
            "inference_mode": "interleave",
        }

        # Preserve geometry-specific fields for scorer
        if task == "geometry":
            result["task_type"] = item.get("task_type", "")
            result["reasoning_qs"] = item.get("reasoning_qs", "")
            result["auxiliary_qs"] = item.get("auxiliary_qs", "")
            result["gt_auxiliary_imgs"] = item.get("gt_auxiliary_imgs", [])

        return result

    except Exception as e:
        print(f"Error processing {hash_uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NEO Model Inference for Unimmmu Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to NEO model")
    parser.add_argument("--data_path", type=str, default=UNIMMMU_DATA_PATH, help="Path to Unimmmu benchmark JSONL")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument(
        "--min_pixels", type=int, default=1024 * 1024, help="Minimum pixels for image resize (default: 1024*1024)"
    )
    parser.add_argument(
        "--max_pixels", type=int, default=2048 * 2048, help="Maximum pixels for image resize (default: 2048*2048)"
    )
    parser.add_argument(
        "--target_image_size",
        type=parse_square_image_size,
        default=None,
        help="Force square image size for generation/editing (e.g. 1024 or 1024x1024). If None, keep smart_resize strategy.",
    )
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale for text guidance")
    parser.add_argument("--img_cfg_scale", type=float, default=1.0, help="CFG scale for image guidance")
    parser.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("START", "END"),
        help="CFG interval as two floats: start end",
    )
    parser.add_argument(
        "--cfg_norm",
        type=str,
        default="none",
        choices=["none", "global", "channel"],
        help="CFG normalization mode (default: none)",
    )
    parser.add_argument("--num_steps", type=int, default=50, help="Number of generation steps")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Total number of logical shards for manual sharding"
    )
    parser.add_argument("--shard_rank", type=int, default=None, help="Current logical shard rank for manual sharding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="i2t",
        choices=["i2t", "interleave"],
        help="Inference mode: 'i2t' for understanding, 'interleave' for multimodal reasoning",
    )
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift for interleave generation")
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional HuggingFace device_map (e.g. 'auto'). Default keeps original single-device/DDP behavior",
    )
    parser.add_argument(
        "--max_memory_per_gpu_gb",
        type=int,
        default=None,
        help="Optional per-GPU max memory in GiB when using --device_map",
    )

    args = parser.parse_args()

    args.cfg_interval = tuple(args.cfg_interval)

    # Set random seed
    set_random_seeds(args.seed)

    # Initialize runtime mode
    if args.device_map is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError(
                "device_map mode must be run as a single process. Use python directly or torchrun --nproc_per_node=1."
            )
        world_size, rank = 1, 0
        device = None
    else:
        local_rank, world_size, rank = setup_distributed()
        device = f"cuda:{local_rank}"

    max_memory = None
    if args.device_map is not None and args.max_memory_per_gpu_gb is not None:
        max_memory = {gpu_idx: f"{args.max_memory_per_gpu_gb}GiB" for gpu_idx in range(torch.cuda.device_count())}

    # Load model
    engine = NEOInferenceEngine(
        args.model_path,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
    )

    # Load data
    data = load_unimmmu_data(args.data_path)
    if rank == 0:
        print(f"Loaded {len(data)} samples from {args.data_path}")
        # Print task distribution
        task_counts = {}
        for item in data:
            t = item.get("task", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1
        print(f"Task distribution: {task_counts}")

    # Output path
    output_filename = "unimmmu_results.jsonl"
    shard_rank_for_output = args.shard_rank if args.num_shards is not None else None
    output_path = resolve_output_path(args.output_dir, output_filename, shard_rank_for_output)

    # Resume: filter completed samples
    if args.resume:
        completed = load_completed_ids(output_path)
        data = [item for item in data if item.get("hash_uid") not in completed]
        if rank == 0:
            print(f"Resume mode: {len(completed)} completed, {len(data)} remaining")

    # Limit samples for testing
    if args.limit:
        data = data[: args.limit]
        if rank == 0:
            print(f"Limited to {len(data)} samples")

    # Logical/manual sharding takes precedence over torch.distributed sharding
    if args.num_shards is not None or args.shard_rank is not None:
        if args.num_shards is None or args.shard_rank is None:
            raise ValueError("--num_shards and --shard_rank must be provided together")
        if args.num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {args.num_shards}")
        if not (0 <= args.shard_rank < args.num_shards):
            raise ValueError(f"shard_rank must be in [0, num_shards), got {args.shard_rank}/{args.num_shards}")
        data = data[args.shard_rank :: args.num_shards]
        print(f"Logical shard {args.shard_rank}/{args.num_shards}: processing {len(data)} samples")
    elif world_size > 1:
        data = data[rank::world_size]
        print(f"Rank {rank}/{world_size}: processing {len(data)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process samples
    success_count = 0
    fail_count = 0

    pbar = tqdm(data, desc=f"Processing (rank {rank})", disable=(rank != 0))
    for item in pbar:
        if args.inference_mode == "interleave":
            result = process_unimmmu_interleave(
                engine=engine,
                item=item,
                output_dir=args.output_dir,
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                cfg_interval=args.cfg_interval,
                cfg_norm=args.cfg_norm,
                num_steps=args.num_steps,
                timestep_shift=args.timestep_shift,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                target_image_size=args.target_image_size,
            )
        else:
            result = process_unimmmu_i2t(
                engine=engine,
                item=item,
                output_dir=args.output_dir,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )

        if result:
            save_result(result, output_path)
            success_count += 1
        else:
            fail_count += 1

        # Force memory cleanup after each sample in interleave mode
        if args.inference_mode == "interleave":
            gc.collect()
            torch.cuda.empty_cache()

        pbar.set_postfix(success=success_count, fail=fail_count)

    # Synchronize all processes
    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: Completed {success_count} successful, {fail_count} failed")

    # Rank 0 summary
    if rank == 0:
        print("\n" + "=" * 50)
        print("Inference completed!")
        print(f"Output directory: {args.output_dir}")
        print(f"Results file: {output_path}")
        if args.num_shards is not None:
            print(f"Logical sharding: shard_rank={args.shard_rank}, num_shards={args.num_shards}")
        print("=" * 50)


if __name__ == "__main__":
    main()
