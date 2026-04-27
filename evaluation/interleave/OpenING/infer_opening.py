"""
交错生成推理脚本 - 支持 think/no think 模式
交错数据统一使用默认的system prompt。no think模式拼一个空的think tag

功能说明：
1. 读取输入的 JSON 配置文件，包含多个数据集的信息
2. 对每个数据集中的样本依次进行推理预测
3. 支持两种 think 模式：
   - think: 使用 <think></think> 块进行推理
   - no_think: 直接给出答案，不使用推理块
4. 保存推理结果到指定目录

使用方法：
python infer_opening.py --model_path <模型路径> --save_dir <保存目录> --step <步数> \
    --input_json_path <输入JSON路径> --think_mode think no_think

参考的超参： cfg = 4， shift = 3， cfg_interval = (0, 1) 
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import io
import json
import math
import os
import random
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, GenerationConfig

try:
    from aoss_client.client import Client
except ImportError:
    print("警告: aoss_client 未安装，S3 图片读取不可用")
    Client = None
    client = None
else:
    aoss_conf_path = os.getenv("AOSS_CONF_PATH")
    if not aoss_conf_path:
        print("提示: 未设置 AOSS_CONF_PATH，S3 图片读取不可用")
        client = None
    else:
        try:
            client = Client(aoss_conf_path)
        except Exception as exc:
            print(f"警告: 初始化 aoss_client 失败，S3 图片读取不可用: {exc}")
            client = None


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")


def read_image_from_aoss(fn):
    if client is None:
        raise RuntimeError("S3 图片读取不可用，请先安装 aoss_client 并设置 AOSS_CONF_PATH")
    img_value_str = client.get(fn)
    img = pil_loader(img_value_str)
    return img


def is_s3_path(path):
    return isinstance(path, str) and path.startswith("s3")


def load_rgb_image(path):
    if is_s3_path(path):
        return read_image_from_aoss(path)
    return Image.open(path).convert("RGB")


def get_image_size(path):
    if is_s3_path(path):
        image = read_image_from_aoss(path)
        try:
            return image.size
        finally:
            image.close()
    with Image.open(path) as image:
        return image.size


def set_random_seeds(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


# copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60
def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 256 * 32 * 32, max_pixels: int = 16384 * 32 * 32
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}"
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


def dynamic_preprocess_native_resolution(image, size_factor=32, min_pixels=4 * 32 * 32, max_pixels=16384 * 32 * 32):
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    # print(f'[debug/dynamic_preprocess_native_resolution] Resizing image from {width}x{height} to {resized_width}x{resized_height}')
    image = image.resize((resized_width, resized_height))

    return image


def read_jsonl(file):
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def get_default_total_dir():
    for dirname in ("OpenING-benchmark/OpenING", "OpenING-Benchmark/OpenING", "OpenING-benchmark", "OpenING-Benchmark"):
        candidate = os.path.join(".", dirname)
        if os.path.exists(candidate):
            return candidate
    return "./OpenING-benchmark"


DEFAULT_TOTAL_DIR = get_default_total_dir()
DEFAULT_OUTPUT_DIR = "./gen_outputs/opening_output"
DEFAULT_MAX_GENERATION_PIXELS = 2048 * 2048
DEFAULT_OOM_RETRY_MAX_PIXELS = 1024 * 1024

DEFAULT_INTERLEAVE_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""


def parse_and_load_json(content):
    input_text_list = []
    input_image_list = []
    output_text_list = []
    output_image_list = []

    for input_content in content["conversations"][0]["input"]:
        input_text_list.append(input_content["text"].strip())
        input_image_list.append(input_content["image"])

    for output_content in content["conversations"][1]["output"]:
        output_text_list.append(output_content["text"].strip())
        output_image_list.append(output_content["image"])

    return input_text_list, input_image_list, output_text_list, output_image_list


def load_opening_data(data_path):
    real_data_list = []
    io_data_list = []

    with open(data_path, encoding="utf-8") as file:
        for line in tqdm(file, desc="加载 OpenING 数据"):
            line = line.strip()
            if not line:
                continue
            content = json.loads(line)
            real_data_list.append(content)
            input_text, input_image, output_text, output_image = parse_and_load_json(content)
            io_data_list.append(
                {
                    "input_text": input_text,
                    "input_image": input_image,
                    "output_text": output_text,
                    "output_image": output_image,
                }
            )

    return real_data_list, io_data_list


def resolve_data_path(root, path):
    if not path:
        return None
    if os.path.isabs(path) or is_s3_path(path):
        return path
    resolved_path = os.path.normpath(os.path.join(root, path))
    if os.path.exists(resolved_path):
        return resolved_path

    nested_resolved_path = os.path.normpath(os.path.join(root, "OpenING", path))
    if os.path.exists(nested_resolved_path):
        return nested_resolved_path

    return resolved_path


def saved_opening_result_is_complete(json_path, output_dir, expected_steps):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        outputs = data.get("conversations", [{}, {"output": []}])[1].get("output", [])
        if len(outputs) < expected_steps:
            return False

        for index in range(expected_steps):
            output_item = outputs[index]
            if not str(output_item.get("text", "")).strip():
                return False

            image_name = output_item.get("image")
            if not image_name:
                return False

            image_path = (
                image_name if os.path.isabs(image_name) else os.path.join(output_dir, os.path.basename(image_name))
            )
            if not os.path.exists(image_path):
                return False

        return True
    except Exception:
        return False


def get_saved_ids(output_dir, expected_steps_by_uid=None):
    if not os.path.exists(output_dir):
        return set()

    saved_ids = set()
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            uid = os.path.splitext(file_name)[0]
            if expected_steps_by_uid is None:
                saved_ids.add(uid)
                continue

            expected_steps = expected_steps_by_uid.get(uid)
            json_path = os.path.join(output_dir, file_name)
            if expected_steps is not None and saved_opening_result_is_complete(json_path, output_dir, expected_steps):
                saved_ids.add(uid)
    return saved_ids


def load_uid_filter(uid_file):
    if not uid_file:
        return None

    uid_filter = set()
    with open(uid_file, "r", encoding="utf-8") as f:
        for line in f:
            uid = line.strip()
            if uid:
                uid_filter.add(uid)
    return uid_filter


def build_opening_prompt(input_text_list, gt_out_step, step_prompt_style="none"):
    prompt_parts = [text.replace("<BEGIN>", "").strip() for text in input_text_list]

    if step_prompt_style == "can_be":
        prefix = f"The number of generated text-image pairs can be {gt_out_step}: "
        if not prompt_parts:
            return prefix
        prompt_parts[0] = prefix + prompt_parts[0]
    elif step_prompt_style == "must_exact":
        prefix = (
            f"The number of generated text-image pairs must be exactly {gt_out_step}. "
            f"Please generate exactly {gt_out_step} interleaved text-image pairs: "
        )
        if not prompt_parts:
            return prefix
        prompt_parts[0] = prefix + prompt_parts[0]

    return "\n".join(prompt_parts)


def split_generated_text(text):
    text = (text or "").replace("**", "").strip()
    if not text:
        return [""]

    delimiter_pattern = r"<IMG>|<image>|<image_?\d+>|</image_?\d+>|<IMG_?\d+>|</IMG_?\d+>"
    parts = [part.strip() for part in re.split(delimiter_pattern, text, flags=re.IGNORECASE) if part.strip()]
    return parts if parts else [text]


def normalize_output_steps(text_steps, image_names, gt_out_step):
    item_count = min(gt_out_step, max(len(text_steps), len(image_names), 1))
    normalized_text = []
    normalized_images = []

    for index in range(item_count):
        if index < len(text_steps):
            normalized_text.append(text_steps[index])
        elif text_steps:
            normalized_text.append("")
        else:
            normalized_text.append("")

        if index < len(image_names):
            normalized_images.append(image_names[index])
        else:
            normalized_images.append(None)

    return normalized_text, normalized_images


def output_is_complete(output_text, output_images, gt_out_step):
    if len(output_text) < gt_out_step or len(output_images) < gt_out_step:
        return False
    return all(image_name is not None for image_name in output_images[:gt_out_step])


def atomic_save_json(data, json_path):
    tmp_path = f"{json_path}.tmp.{os.getpid()}"
    with open(tmp_path, mode="w", encoding="utf-8") as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)
    os.replace(tmp_path, json_path)


def get_lanczos_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def smart_resize_output_image(image, target_size):
    if target_size is None:
        return image

    target_w, target_h = target_size
    if image.size == (target_w, target_h):
        return image

    # Preserve all content by resizing with aspect ratio intact, then padding.
    resized = ImageOps.contain(
        image,
        (target_w, target_h),
        method=get_lanczos_resample(),
    )
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset_x = (target_w - resized.size[0]) // 2
    offset_y = (target_h - resized.size[1]) // 2
    canvas.paste(resized.convert("RGB"), (offset_x, offset_y))
    return canvas


def atomic_save_image(image, image_save_path, target_size=None):
    image = smart_resize_output_image(image, target_size)
    tmp_path = f"{image_save_path}.tmp.{os.getpid()}"
    image.save(tmp_path, format="JPEG")
    os.replace(tmp_path, image_save_path)


def save_opening_results(output_dir, real_data_item, generated_text_list, image_out_list):
    data_uid = real_data_item["total_uid"]
    json_path = os.path.join(output_dir, f"{data_uid}.json")

    saved_json = copy.deepcopy(real_data_item)
    if "conversations" in saved_json and len(saved_json["conversations"]) > 1:
        saved_json["conversations"][1]["output"] = []

    for index in range(max(len(generated_text_list), len(image_out_list))):
        output_item = {"text": generated_text_list[index].strip() if index < len(generated_text_list) else ""}
        output_item["image"] = image_out_list[index] if index < len(image_out_list) else None
        saved_json["conversations"][1]["output"].append(output_item)

    atomic_save_json(saved_json, json_path)


def load_opening_input_images(input_image_paths, meta_path):
    input_images = []
    for img_path in input_image_paths:
        if not img_path:
            continue
        resolved_path = resolve_data_path(meta_path, img_path)
        if not resolved_path:
            print(f"警告: 输入图像路径为空: {img_path}")
            continue
        try:
            if is_s3_path(resolved_path) or os.path.exists(resolved_path):
                input_images.append(load_rgb_image(resolved_path))
                continue
        except Exception as e:
            print(f"警告: 无法读取输入图像 {resolved_path}: {e}")
            continue
        if not is_s3_path(resolved_path):
            print(f"警告: 输入图像不存在: {resolved_path}")
    return input_images


def is_cuda_oom_error(error):
    message = str(error).lower()
    oom_error_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return (
        isinstance(error, oom_error_type)
        or "cuda out of memory" in message
        or "cublas_status_alloc_failed" in message
        or ("out of memory" in message and "cuda" in message)
    )


def clear_cuda_memory():
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def resize_generation_size(width, height, max_pixels):
    max_pixels = max(32 * 32, int(max_pixels))
    min_pixels = min(512 * 512, max_pixels)
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resized_w, resized_h


def resolve_fixed_generation_size(image_width=None, image_height=None, size_factor=32):
    if image_width is None and image_height is None:
        return None
    if image_width is None or image_height is None:
        raise ValueError("--image_width and --image_height must be provided together")

    image_width = int(image_width)
    image_height = int(image_height)
    if image_width < size_factor or image_height < size_factor:
        raise ValueError(f"--image_width and --image_height must be >= {size_factor}")

    aligned_width = max(size_factor, floor_by_factor(image_width, size_factor))
    aligned_height = max(size_factor, floor_by_factor(image_height, size_factor))

    if max(aligned_width, aligned_height) / min(aligned_width, aligned_height) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(aligned_width, aligned_height) / min(aligned_width, aligned_height)}"
        )

    if aligned_width != image_width or aligned_height != image_height:
        print(
            f"警告: 固定输出尺寸 {(image_width, image_height)} 不是 {size_factor} 的倍数，"
            f"将向下对齐为 {(aligned_width, aligned_height)}"
        )

    return aligned_width, aligned_height


def resolve_requested_output_size(image_width=None, image_height=None):
    if image_width is None and image_height is None:
        return None
    if image_width is None or image_height is None:
        raise ValueError("--image_width and --image_height must be provided together")

    image_width = int(image_width)
    image_height = int(image_height)
    if image_width < 1 or image_height < 1:
        raise ValueError("--image_width and --image_height must be >= 1")

    return image_width, image_height


def choose_opening_image_sizes(
    io_data,
    input_images,
    meta_path,
    gt_out_step,
    max_pixels=DEFAULT_MAX_GENERATION_PIXELS,
    fixed_image_size=None,
):
    if gt_out_step <= 0:
        return []

    if fixed_image_size is not None:
        return [fixed_image_size] * gt_out_step

    if input_images:
        ref_img = input_images[0]
        w, h = ref_img.size
        resized_w, resized_h = resize_generation_size(w, h, max_pixels)
        return [(resized_w, resized_h)] * gt_out_step

    for img_path in io_data.get("output_image", []):
        if not img_path:
            continue
        resolved_path = resolve_data_path(meta_path, img_path)
        if not resolved_path:
            continue
        try:
            if is_s3_path(resolved_path) or os.path.exists(resolved_path):
                w, h = get_image_size(resolved_path)
                resized_w, resized_h = resize_generation_size(w, h, max_pixels)
                return [(resized_w, resized_h)] * gt_out_step
        except Exception as e:
            print(f"警告: 无法读取 GT 图像 {resolved_path}: {e}")

    resized_w, resized_h = resize_generation_size(1024, 1024, max_pixels)
    return [(resized_w, resized_h)] * gt_out_step


def get_pattern_output_dir(save_dir, pattern, all_patterns):
    if len(all_patterns) == 1:
        return save_dir

    if save_dir.endswith("_output"):
        base_dir = save_dir[: -len("_output")]
    else:
        base_dir = save_dir
    return f"{base_dir}_{pattern}_output"


def load_system_message(args, mode):
    if args.system_message:
        return args.system_message

    if args.system_prompt_path:
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    return DEFAULT_INTERLEAVE_SYSTEM_MESSAGE


def make_grid(images, rows=2, cols=8):
    # find max width and height
    max_w = max(img.size[0] for img in images)
    max_h = max(img.size[1] for img in images)

    grid_w = cols * max_w
    grid_h = rows * max_h

    grid_img = Image.new("RGB", (grid_w, grid_h))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        # create a cell canvas
        cell = Image.new("RGB", (max_w, max_h))
        w, h = img.size

        # center the image in the cell
        x = (max_w - w) // 2
        y = (max_h - h) // 2
        cell.paste(img, (x, y))

        # paste the cell into the grid
        grid_img.paste(cell, (c * max_w, r * max_h))

    return grid_img


# NORM_MEAN = [0.485, 0.456, 0.406]
# NORM_STD  = [0.229, 0.224, 0.225]

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


class NEOT2IInferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def chat(
        self,
        prompt,
        input_images,
        cfg_scale=1.0,
        img_cfg_scale=1.0,
        timestep_shift=1.0,
        cfg_interval=(0, 1),
        image_size=(256, 256),
        num_steps=50,
        gt_text=None,
        gt_images=None,
        system_message="",
        max_new_tokens=4096,
        think_mode="no_think",
    ):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens)
        think_mode_enabled = think_mode is True or think_mode == "think"

        if gt_text is None:
            interleave_kwargs = {
                "image_size": image_size,
                "images": input_images,
                "cfg_scale": cfg_scale,
                "img_cfg_scale": img_cfg_scale,
                "timestep_shift": timestep_shift,
                "cfg_interval": cfg_interval,
                "num_steps": num_steps,
                "system_message": system_message,
                "generation_config": generation_config,
            }
            interleave_params = inspect.signature(self.model.interleave_gen).parameters
            if "think_pattern" in interleave_params:
                interleave_kwargs["think_pattern"] = "think" if think_mode_enabled else "no_think"
            elif "think_mode" in interleave_params:
                interleave_kwargs["think_mode"] = think_mode_enabled

            text, images = self.model.interleave_gen(
                self.tokenizer,
                prompt,
                **interleave_kwargs,
            )
        else:
            image_only_kwargs = {
                "image_size": image_size,
                "images": input_images,
                "cfg_scale": cfg_scale,
                "img_cfg_scale": img_cfg_scale,
                "timestep_shift": timestep_shift,
                "cfg_interval": cfg_interval,
                "num_steps": num_steps,
                "gt_text": gt_text,
                "gt_images": gt_images,
                "system_message": system_message,
            }
            image_only_params = inspect.signature(self.model.interleave_gen_image_only).parameters
            if "generation_config" in image_only_params:
                image_only_kwargs["generation_config"] = generation_config

            images = self.model.interleave_gen_image_only(
                self.tokenizer,
                prompt,
                **image_only_kwargs,
            )
            text = gt_text

        images = [self._denorm(image.float()) for image in images]
        images = [
            (image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8) for image in images
        ]
        images = [Image.fromarray(image[0]) for image in images]

        return text, images

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)


def run_opening_generation(args, engine, system_message):
    data_path = os.path.join(args.meta_path, args.data_file_name)
    real_data_list, io_data_list = load_opening_data(data_path)
    expected_steps_by_uid = {
        real_data["total_uid"]: len(io_data["output_text"]) for real_data, io_data in zip(real_data_list, io_data_list)
    }
    uid_filter = load_uid_filter(args.uid_file)
    sharded_items = [
        (sample_index, io_data)
        for sample_index, io_data in enumerate(io_data_list)
        if sample_index % args.num_shards == args.shard_index
        and (uid_filter is None or real_data_list[sample_index]["total_uid"] in uid_filter)
    ]
    if args.limit is not None:
        sharded_items = sharded_items[: args.limit]

    pattern_output_dirs = {}
    saved_ids_by_pattern = {}
    for pattern in args.think_modes:
        output_dir = get_pattern_output_dir(args.save_dir, pattern, args.think_modes)
        os.makedirs(output_dir, exist_ok=True)
        pattern_output_dirs[pattern] = output_dir
        saved_ids_by_pattern[pattern] = get_saved_ids(output_dir, expected_steps_by_uid)

    print(f"OpenING 数据路径: {data_path}")
    print(f"处理数据样本总数: {len(real_data_list)}")
    if uid_filter is not None:
        print(f"UID 过滤文件: {args.uid_file}, UID 数量={len(uid_filter)}")
    print(f"分片: shard_index={args.shard_index}, num_shards={args.num_shards}, 当前分片样本数={len(sharded_items)}")
    if args.fixed_generation_size is not None:
        print(
            f"固定生成尺寸: {args.fixed_generation_size}, "
            f"保存输出尺寸: {args.requested_output_size}, "
            f"oom_retry_max_pixels={args.oom_retry_max_pixels}"
        )
    else:
        print(
            f"生成图像像素上限: max_generation_pixels={args.max_generation_pixels}, "
            f"oom_retry_max_pixels={args.oom_retry_max_pixels}"
        )
    print(f"输出目录: {pattern_output_dirs}")

    for sample_index, io_data in tqdm(sharded_items, desc=f"OpenING 推理 shard {args.shard_index}/{args.num_shards}"):
        real_data = real_data_list[sample_index]
        data_uid = real_data["total_uid"]
        gt_out_step = len(io_data["output_text"])
        if gt_out_step == 0:
            print(f"警告: UID {data_uid} 无输出步骤，跳过")
            continue

        input_images = load_opening_input_images(io_data["input_image"], args.meta_path)
        image_size_list = choose_opening_image_sizes(
            io_data,
            input_images,
            args.meta_path,
            gt_out_step,
            max_pixels=args.max_generation_pixels,
            fixed_image_size=args.fixed_generation_size,
        )
        oom_retry_image_size_list = None
        if args.fixed_generation_size is not None:
            fixed_w, fixed_h = args.fixed_generation_size
            fixed_pixels = fixed_w * fixed_h
            if 0 < args.oom_retry_max_pixels < fixed_pixels:
                retry_w, retry_h = resize_generation_size(fixed_w, fixed_h, args.oom_retry_max_pixels)
                oom_retry_image_size_list = [(retry_w, retry_h)] * gt_out_step
        elif 0 < args.oom_retry_max_pixels < args.max_generation_pixels:
            oom_retry_image_size_list = choose_opening_image_sizes(
                io_data,
                input_images,
                args.meta_path,
                gt_out_step,
                max_pixels=args.oom_retry_max_pixels,
            )
        prompt = build_opening_prompt(
            io_data["input_text"],
            gt_out_step,
            step_prompt_style=args.opening_step_prompt_style,
        )

        for pattern in args.think_modes:
            output_dir = pattern_output_dirs[pattern]
            if not args.overwrite and data_uid in saved_ids_by_pattern[pattern]:
                continue

            try:
                print(f"UID: {data_uid}, think模式: {pattern}, gt_out_step: {gt_out_step}")
                best_output_text = []
                best_output_images = []
                best_images = []
                best_score = -1
                max_attempts = max(1, args.retry_short_outputs + 1)
                current_image_size_list = image_size_list
                attempt = 0

                while attempt < max_attempts:
                    try:
                        text, images = engine.chat(
                            prompt,
                            input_images=input_images,
                            cfg_scale=args.cfg_scale,
                            img_cfg_scale=args.img_cfg_scale,
                            timestep_shift=args.timestep_shift,
                            cfg_interval=tuple(args.cfg_interval),
                            image_size=current_image_size_list,
                            num_steps=args.num_steps,
                            gt_text=None,
                            gt_images=None,
                            system_message=system_message,
                            max_new_tokens=args.max_new_tokens,
                            think_mode=pattern,
                        )
                    except Exception as e:
                        if (
                            is_cuda_oom_error(e)
                            and oom_retry_image_size_list
                            and current_image_size_list != oom_retry_image_size_list
                        ):
                            print(
                                f"UID: {data_uid} CUDA OOM，清理显存并降输出分辨率重试: "
                                f"{current_image_size_list[:1]} -> {oom_retry_image_size_list[:1]}"
                            )
                            clear_cuda_memory()
                            current_image_size_list = oom_retry_image_size_list
                            continue
                        raise

                    candidate_image_names = [
                        f"{data_uid}-o-{image_index}.jpg" for image_index, _ in enumerate(images[:gt_out_step])
                    ]
                    text_steps = split_generated_text(text)
                    candidate_text, candidate_images = normalize_output_steps(
                        text_steps, candidate_image_names, gt_out_step
                    )
                    candidate_score = min(len(candidate_text), gt_out_step) + min(
                        len(candidate_image_names), gt_out_step
                    )

                    if candidate_score > best_score:
                        best_output_text = candidate_text
                        best_output_images = candidate_images
                        best_images = images[:gt_out_step]
                        best_score = candidate_score

                    if output_is_complete(candidate_text, candidate_images, gt_out_step):
                        break

                    attempt += 1
                    if attempt < max_attempts:
                        print(
                            f"UID: {data_uid} attempt {attempt}/{max_attempts} "
                            f"输出不足: text={len(candidate_text)}, images={len(candidate_image_names)}, expected={gt_out_step}; retry"
                        )
                        clear_cuda_memory()

                for image_index, image in enumerate(best_images):
                    image_name = f"{data_uid}-o-{image_index}.jpg"
                    image_save_path = os.path.join(output_dir, image_name)
                    atomic_save_image(image, image_save_path, target_size=args.requested_output_size)

                output_text, output_images = best_output_text, best_output_images
                save_opening_results(output_dir, real_data, output_text, output_images)
                saved_ids_by_pattern[pattern].add(data_uid)
                print(f"Processed and saved results for UID: {data_uid}, pattern: {pattern}")

            except Exception as e:
                if is_cuda_oom_error(e):
                    clear_cuda_memory()
                print(f"错误: UID {data_uid} 模式 {pattern} 推理失败: {e}")
                continue

        torch.cuda.empty_cache()


def run_annotation_config_generation(args, engine, system_message):
    if not args.input_json_path:
        raise ValueError("annotation_config 模式需要指定 --input_json_path")

    with open(args.input_json_path, "r") as f:
        input_json = json.load(f)

    print(f"处理数据样本总数: {len(input_json)}")
    output_dir = args.save_dir
    think_modes = args.think_modes
    print(f"think模式: {think_modes}")
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(
        output_dir, os.path.basename(args.input_json_path)[:-5] + f"_step{args.step}_result.json"
    )
    result_json = {}

    for annotation_name, info in tqdm(input_json.items(), desc="处理数据集"):
        print(f"\n处理数据集: {annotation_name}")
        result_json[annotation_name] = info

        annotation_path = info["annotation"]
        image_root = info.get("root", "")
        data_list = read_jsonl(annotation_path)

        image_save_dir = os.path.join(output_dir, f"{annotation_name}_images")
        os.makedirs(image_save_dir, exist_ok=True)

        pattern_files_write = {}
        pattern_files = {}
        for pattern in think_modes:
            output_jsonl_path = os.path.join(output_dir, f"{annotation_name}_step_{args.step}_pattern_{pattern}.jsonl")
            pattern_files[pattern] = output_jsonl_path
            pattern_files_write[pattern] = open(output_jsonl_path, "w")

        for sample_index, sample_data in enumerate(tqdm(data_list, desc=f"推理 {annotation_name}")):
            conversations = sample_data["conversations"]
            prompt = None
            gt_text = None

            for conv in conversations:
                if conv["from"] == "human":
                    prompt = conv["value"]
                elif conv["from"] == "gpt":
                    gt_text = conv["value"]
                    break

            if prompt is None or gt_text is None:
                print(f"警告: 样本 {sample_index} 缺少 prompt 或 gt_text，跳过")
                continue

            input_images = []
            input_image_paths = []
            image_count_in_prompt = prompt.count("<image>")
            print(f"image_count_in_prompt: {image_count_in_prompt}")
            if image_count_in_prompt > 0:
                if "image" not in sample_data or len(sample_data["image"]) < image_count_in_prompt:
                    print(f"警告: 样本 {sample_index} 输入图像不足，跳过")
                    continue

                for i in range(image_count_in_prompt):
                    img_path = sample_data["image"][i]
                    if image_root and not os.path.isabs(img_path) and not is_s3_path(img_path):
                        img_path = os.path.join(image_root, img_path)
                    if not is_s3_path(img_path) and not os.path.exists(img_path):
                        print(f"警告: 图像不存在: {img_path}，跳过")
                        break
                    input_images.append(load_rgb_image(img_path))
                    input_image_paths.append(img_path)

                if len(input_images) != image_count_in_prompt:
                    continue

            gt_image_count = gt_text.count("<image>")
            print(f"gt_image_count: {gt_image_count}")
            if gt_image_count == 0:
                print(f"警告: 样本 {sample_index} gt_text 中无 <image>，跳过")
                continue

            if args.fixed_generation_size is not None:
                image_size_list = [args.fixed_generation_size] * gt_image_count
                print(f"使用固定输出尺寸: {args.fixed_generation_size}")
            elif image_count_in_prompt > 0 and len(input_images) > 0:
                ref_img = input_images[0]
                w, h = ref_img.size
                resized_w, resized_h = resize_generation_size(w, h, args.max_generation_pixels)
                image_size_list = [(resized_w, resized_h)] * gt_image_count
                print(f"使用输入图片尺寸: {(resized_w, resized_h)}")
            else:
                sample_images = sample_data.get("image", [])
                if sample_images and len(sample_images) > image_count_in_prompt:
                    img_path = sample_images[image_count_in_prompt]
                    if image_root and not os.path.isabs(img_path) and not is_s3_path(img_path):
                        img_path = os.path.join(image_root, img_path)

                    if is_s3_path(img_path) or os.path.exists(img_path):
                        try:
                            w, h = get_image_size(img_path)
                            resized_w, resized_h = resize_generation_size(w, h, args.max_generation_pixels)
                            image_size_list = [(resized_w, resized_h)] * gt_image_count
                            print(f"使用GT图片尺寸: {(resized_w, resized_h)}")
                        except Exception as e:
                            print(f"警告: 无法读取图像 {img_path}: {e}，使用默认尺寸")
                            default_w, default_h = resize_generation_size(1024, 1024, args.max_generation_pixels)
                            image_size_list = [(default_w, default_h)] * gt_image_count
                    else:
                        default_w, default_h = resize_generation_size(1024, 1024, args.max_generation_pixels)
                        image_size_list = [(default_w, default_h)] * gt_image_count
                else:
                    default_w, default_h = resize_generation_size(1024, 1024, args.max_generation_pixels)
                    image_size_list = [(default_w, default_h)] * gt_image_count
                    print(f"使用默认尺寸: {(default_w, default_h)}")

            print(f"image size list: {image_size_list}")
            print(f"input image size: {len(input_images)}")

            for pattern in think_modes:
                try:
                    print(f"think模式:{pattern}")
                    text, images = engine.chat(
                        prompt,
                        input_images=input_images,
                        cfg_scale=args.cfg_scale,
                        img_cfg_scale=args.img_cfg_scale,
                        timestep_shift=args.timestep_shift,
                        cfg_interval=tuple(args.cfg_interval),
                        image_size=image_size_list,
                        num_steps=args.num_steps,
                        gt_text=None,
                        gt_images=None,
                        system_message=system_message,
                        max_new_tokens=args.max_new_tokens,
                        think_mode=pattern,
                    )

                    print(f"pattern: {pattern}, text: {text}")
                    save_images = []
                    for i, image in enumerate(images):
                        image_save_path = os.path.join(
                            image_save_dir, f"{annotation_name}_sample_{sample_index}_{pattern}_image_{i}.jpg"
                        )
                        save_images.append(image_save_path)
                        atomic_save_image(image, image_save_path, target_size=args.requested_output_size)

                    conversations_output = [
                        {"from": "system", "value": system_message},
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": text},
                    ]
                    all_images = input_image_paths + save_images
                    line = {"conversations": conversations_output, "image": all_images, "think_mode": pattern}
                    pattern_files_write[pattern].write(json.dumps(line, ensure_ascii=False) + "\n")
                    pattern_files_write[pattern].flush()

                except Exception as e:
                    print(f"错误: 样本 {sample_index} 模式 {pattern} 推理失败: {e}")
                    continue

        for pattern, fw in pattern_files_write.items():
            fw.close()

        for pattern in think_modes:
            save_info = copy.deepcopy(info)
            save_info["annotation"] = pattern_files[pattern]
            save_info["length"] = info["length"]
            result_json[os.path.basename(pattern_files[pattern])[:-6]] = save_info

    with open(output_json, "w") as fj:
        json.dump(result_json, fj, ensure_ascii=False, indent=2)

    print(f"\n推理完成！结果已保存到: {output_json}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="保存目录")
    parser.add_argument("--step", type=int, default=0, help="checkpoint 步数")
    parser.add_argument(
        "--input_json_path", type=str, default=None, help="输入 JSON 配置文件路径；提供时默认走 annotation_config 模式"
    )
    parser.add_argument(
        "--mode", type=str, default="auto", choices=["auto", "opening", "annotation_config"], help="推理数据格式"
    )
    parser.add_argument("--meta-path", type=str, default=DEFAULT_TOTAL_DIR, help="OpenING 数据集目录")
    parser.add_argument("--data-file-name", type=str, default="test_data.jsonl", help="OpenING JSONL 文件名")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--img_cfg_scale", type=float, default=1.0, help="Image CFG scale")
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Timestep shift")
    parser.add_argument(
        "--cfg_interval", type=float, nargs=2, default=[0, 1.0], help="CFG生效的时间步区间, e.g. --cfg_interval 0 1.0"
    )
    parser.add_argument("--num_steps", type=int, default=50, help="生成步数")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument(
        "--max_generation_pixels",
        type=int,
        default=DEFAULT_MAX_GENERATION_PIXELS,
        help="初始输出图片像素上限，默认约 2048x2048",
    )
    parser.add_argument(
        "--oom_retry_max_pixels",
        type=int,
        default=DEFAULT_OOM_RETRY_MAX_PIXELS,
        help="CUDA OOM 后降分辨率重试的像素上限；设为 0 可关闭",
    )
    parser.add_argument("--image_width", type=int, default=None, help="固定输出图片宽度；需与 --image_height 一起传入")
    parser.add_argument("--image_height", type=int, default=None, help="固定输出图片高度；需与 --image_width 一起传入")
    parser.add_argument("--system_message", type=str, default="", help="系统消息；优先级高于 --system_prompt_path")
    parser.add_argument("--system_prompt_path", type=str, default=None, help="系统 prompt 文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--think_mode",
        "--think_pattern",
        dest="think_modes",
        type=str,
        nargs="+",
        default=["think", "no_think"],
        help="think模式列表；--think_pattern 为兼容旧命令的别名",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已生成的 OpenING 样本")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="总分片数；多进程/多卡并行时设为总进程数",
    )
    parser.add_argument("--shard_index", type=int, default=int(os.getenv("RANK", "0")), help="当前分片编号，从 0 开始")
    parser.add_argument("--limit", type=int, default=None, help="只跑当前分片前 N 条样本，便于 smoke test")
    parser.add_argument("--uid_file", type=str, default=None, help="只处理该文件中列出的 total_uid，每行一个 UID")
    parser.add_argument(
        "--retry_short_outputs", type=int, default=0, help="如果生成步数少于 benchmark 期望步数，额外重试次数"
    )
    parser.add_argument(
        "--opening_step_prompt_style",
        type=str,
        default="can_be",
        choices=["none", "can_be", "must_exact"],
        help="OpenING prompt 前的步数提示风格；can_be 对齐 gpt-dalle_generation.py",
    )
    parser.add_argument(
        "--enforce_expected_steps_prompt",
        action="store_true",
        help="兼容旧参数：等价于 --opening_step_prompt_style must_exact",
    )
    args = parser.parse_args()
    if args.enforce_expected_steps_prompt:
        args.opening_step_prompt_style = "must_exact"
    return args


if __name__ == "__main__":
    args = parse_args()
    args.requested_output_size = resolve_requested_output_size(args.image_width, args.image_height)
    args.fixed_generation_size = resolve_fixed_generation_size(args.image_width, args.image_height)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.max_generation_pixels < 32 * 32:
        raise ValueError("--max_generation_pixels must be >= 1024")
    if args.oom_retry_max_pixels < 0:
        raise ValueError("--oom_retry_max_pixels must be >= 0")
    set_random_seeds(args.seed)

    mode = args.mode
    if mode == "auto":
        mode = "annotation_config" if args.input_json_path else "opening"

    system_message = load_system_message(args, mode)

    print(f"加载模型: {args.model_path}")
    print(f"运行模式: {mode}")
    print(f"think模式: {args.think_modes}")
    if args.fixed_generation_size is not None:
        print(f"固定生成尺寸配置: {args.fixed_generation_size}")
        print(f"保存输出尺寸配置: {args.requested_output_size}")
    engine = NEOT2IInferenceEngine(args.model_path)

    if mode == "opening":
        run_opening_generation(args, engine, system_message)
    else:
        run_annotation_config_generation(args, engine, system_message)
