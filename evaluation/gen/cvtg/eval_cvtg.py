import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_random_seeds(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


def parse_csv(raw_value, cast_fn=str):
    if raw_value is None:
        return None

    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast_fn(part))
    return values or None


def load_cvtg_samples(benchmark_root, subsets, areas, target_keys=None):
    data = []
    for subset in subsets:
        for area in areas:
            json_path = os.path.join(benchmark_root, subset, f"{area}_combined.json")
            with open(json_path, "r", encoding="utf-8") as file:
                subset_data = json.load(file)
            for key, prompt in subset_data.items():
                if target_keys is not None and key not in target_keys:
                    continue
                data.append(
                    {
                        "subset": subset,
                        "key": key,
                        "prompt": prompt,
                        "area": area,
                    }
                )
    return data


NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


class T2IInferenceEngine:
    def __init__(
        self,
        model_path,
        device="cuda",
        device_map=None,
        max_memory=None,
    ):
        self.device = device
        self.device_map = device_map

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map
            model_kwargs["low_cpu_mem_usage"] = True
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(
                self.device
            )

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def chat(
        self,
        prompt,
        cfg_scale=1.0,
        enable_timestep_shift=True,
        timestep_shift=1.0,
        image_size=(256, 256),
        num_steps=50,
    ):
        with torch.inference_mode():
            output = self.model.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                timestep_shift=timestep_shift,
                enable_timestep_shift=enable_timestep_shift,
                num_steps=num_steps,
                batch_size=1,
            )

        image = self._denorm(output)
        image = image.detach().to(device="cpu", dtype=torch.float32)
        image = (
            (image.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        grid_image = Image.fromarray(image[0])

        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return grid_image

    def _denorm(self, x: torch.Tensor, mean=NORM_MEAN, std=NORM_STD):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local checkpoint path or HF model id.",
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help="CVTG-2K benchmark root (containing CVTG/ and CVTG-Style/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--image_size", "--resolution", dest="image_size", type=int, default=2048
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--timestep_shift", type=float, default=1.0)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_memory_per_gpu_gb", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--subsets", type=str, default="CVTG,CVTG-Style")
    parser.add_argument("--areas", type=str, default="2,3,4,5")
    parser.add_argument("--target_keys", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )

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

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    image_size = (args.image_size, args.image_size)
    enable_timestep_shift = args.timestep_shift >= 1.0

    max_memory = None
    if args.device_map is not None and args.max_memory_per_gpu_gb is not None:
        max_memory = {
            gpu_idx: f"{args.max_memory_per_gpu_gb}GiB"
            for gpu_idx in range(torch.cuda.device_count())
        }

    engine = T2IInferenceEngine(
        args.model_path,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
    )

    subsets = parse_csv(args.subsets, str) or ["CVTG", "CVTG-Style"]
    areas = parse_csv(args.areas, int) or [2, 3, 4, 5]
    raw_target_keys = parse_csv(args.target_keys, str)
    target_keys = set(raw_target_keys) if raw_target_keys is not None else None
    data = load_cvtg_samples(
        args.benchmark_root, subsets, areas, target_keys=target_keys
    )
    if not data:
        raise ValueError(
            "No CVTG samples were found. Check benchmark_root or filtering arguments."
        )

    if target_keys is not None:
        print(f"Filtered to target keys: {sorted(target_keys)} => {len(data)} samples")

    set_random_seeds(args.seed)
    random.shuffle(data)
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_data = data[global_shard_rank::total_shards]
    print(
        f"Processing shard {global_shard_rank + 1}/{total_shards} "
        f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
        f"with {len(rank_data)} samples"
    )

    for sample in tqdm(rank_data, disable=rank != 0):
        prompt = sample["prompt"]
        subset = sample["subset"]
        key = sample["key"]
        area = sample["area"]

        cur_output_folder = os.path.join(output_path, subset, str(area))
        output_file = os.path.join(cur_output_folder, f"{key}.png")
        if os.path.exists(output_file):
            continue

        grid_image = engine.chat(
            prompt,
            cfg_scale=args.cfg_scale,
            enable_timestep_shift=enable_timestep_shift,
            timestep_shift=args.timestep_shift,
            image_size=image_size,
            num_steps=args.num_steps,
        )

        os.makedirs(cur_output_folder, exist_ok=True)
        grid_image.save(output_file)

    print(f"Rank {rank} has finished processing {len(rank_data)} examples")
    if dist.is_initialized():
        dist.barrier()
