import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

import sensenova_u1
from examples.t2i.inference import SenseNovaU1T2I, _warn_if_unsupported


def set_random_seeds(seed_value, rank=0):
    seed_value += rank
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
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}", flush=True)
    return local_rank, world_size, rank


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_jsonl_files(input_folder, specific_file=None):
    if specific_file is not None:
        return [os.path.join(input_folder, specific_file)]
    return sorted(
        os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(".jsonl")
    )


def load_all_samples(jsonl_files):
    all_samples = []
    file_sample_counts = []

    for jsonl_file in jsonl_files:
        data = read_jsonl(jsonl_file)
        file_sample_counts.append((os.path.basename(jsonl_file), len(data)))
        for sample_idx, item in enumerate(data):
            sample = dict(item)
            sample["_sample_idx"] = sample_idx
            sample["_source_file"] = os.path.basename(jsonl_file)
            all_samples.append(sample)

    return all_samples, file_sample_counts


def resolve_dtype(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def make_grid_image(images, output_grid_size=None):
    if len(images) == 1:
        return images[0]

    rows, cols = output_grid_size if output_grid_size is not None else (1, len(images))
    if rows * cols < len(images):
        raise ValueError(f"rows * cols must cover batch_size, got rows={rows}, cols={cols}, batch_size={len(images)}")

    pad = 2
    width, height = images[0].size
    grid = Image.new(
        "RGB",
        (cols * width + (cols - 1) * pad, rows * height + (rows - 1) * pad),
        color=(0, 0, 0),
    )
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        grid.paste(image, (col * (width + pad), row * (height + pad)))
    return grid


def main():
    parser = argparse.ArgumentParser(description="TIIF-Bench T2I generation.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Directory containing TIIF-Bench prompt JSONL files.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--save_size",
        type=int,
        default=None,
        help=(
            "If set, downsample each generated grid to this resolution per cell "
            "(LANCZOS) before writing it to disk. Useful for the "
            "'generate at 2048, evaluate at 1024' protocol. Defaults to "
            "--resolution (no resize)."
        ),
    )
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument(
        "--cfg_norm",
        type=str,
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
    )
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_backend", default="auto", choices=["auto", "flash", "sdpa"])
    parser.add_argument("--specific_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if not 0 <= args.shard_rank < args.num_shards:
        raise ValueError(
            f"shard_rank must be in [0, num_shards), got shard_rank={args.shard_rank}, num_shards={args.num_shards}"
        )
    if args.rows * args.cols < args.batch_size:
        raise ValueError(
            f"rows * cols must cover batch_size, got rows={args.rows}, cols={args.cols}, batch_size={args.batch_size}"
        )

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    set_random_seeds(args.seed, rank)
    image_size = (args.resolution, args.resolution)
    cfg_interval = tuple(args.cfg_interval)

    sensenova_u1.set_attn_backend(args.attn_backend)
    if rank == 0:
        print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")
        _warn_if_unsupported(*image_size)

    engine = SenseNovaU1T2I(
        args.model_path,
        device=device,
        dtype=resolve_dtype(args.dtype),
    )

    jsonl_files = get_jsonl_files(args.input_folder, args.specific_file)
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(args.model_path))

    if rank == 0:
        print(f"Total prompt files: {len(jsonl_files)}", flush=True)
        print(f"Output dir: {args.output_dir}", flush=True)

    all_samples, file_sample_counts = load_all_samples(jsonl_files)
    total_shards = world_size * args.num_shards
    global_shard_rank = rank * args.num_shards + args.shard_rank
    rank_data = all_samples[global_shard_rank::total_shards]

    if rank == 0:
        for file_name, sample_count in file_sample_counts:
            print(f"Loaded file: {file_name} | total samples: {sample_count}", flush=True)
        print(f"Total samples across all files: {len(all_samples)}", flush=True)
        print(
            f"Processing shard {global_shard_rank + 1}/{total_shards} "
            f"(ddp_rank={rank}/{world_size}, local_shard={args.shard_rank}/{args.num_shards}) "
            f"with {len(rank_data)} samples",
            flush=True,
        )

    for item in tqdm(rank_data, disable=rank != 0):
        sample_idx = item["_sample_idx"]
        data_type = item["type"]
        prompts = {
            "short_description": item["short_description"],
            "long_description": item["long_description"],
        }

        save_root = Path(args.output_dir) / data_type / model_name
        for prompt_type, prompt in prompts.items():
            save_dir = save_root / prompt_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_idx}.png"

            if save_path.exists():
                continue

            images = engine.generate(
                prompt,
                image_size=image_size,
                cfg_scale=args.cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            grid_image = make_grid_image(images, output_grid_size=(args.rows, args.cols))

            if args.save_size is not None and args.save_size != args.resolution:
                scale = args.save_size / args.resolution
                target_w = max(1, round(grid_image.width * scale))
                target_h = max(1, round(grid_image.height * scale))
                grid_image = grid_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

            grid_image.save(save_path)

    dist.barrier()
    if rank == 0:
        print("TIIF-Bench generation finished.")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
