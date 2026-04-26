import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

import sensenova_u1
from examples.t2i.inference import SenseNovaU1T2I, _warn_if_unsupported


def set_random_seeds(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
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
    parser.add_argument("--image_size", "--resolution", dest="image_size", type=int, default=2048)
    parser.add_argument(
        "--save_size",
        type=int,
        default=None,
        help=(
            "If set, downsample each generated image to this resolution "
            "(LANCZOS) before writing it to disk. Useful for the "
            "'generate at 2048, evaluate at 1024' protocol. Defaults to "
            "--image_size (no resize)."
        ),
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument(
        "--cfg_norm",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used by examples/t2i/inference.py::SenseNovaU1T2I.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help="Attention backend forwarded to sensenova_u1.set_attn_backend.",
    )
    parser.add_argument("--subsets", type=str, default="CVTG,CVTG-Style")
    parser.add_argument("--areas", type=str, default="2,3,4,5")
    parser.add_argument("--target_keys", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    image_size = (args.image_size, args.image_size)
    save_size = args.save_size if args.save_size and args.save_size != args.image_size else None
    cfg_interval = tuple(args.cfg_interval)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    engine = SenseNovaU1T2I(
        args.model_path,
        device=args.device,
        dtype=dtype,
    )

    subsets = parse_csv(args.subsets, str) or ["CVTG", "CVTG-Style"]
    areas = parse_csv(args.areas, int) or [2, 3, 4, 5]
    raw_target_keys = parse_csv(args.target_keys, str)
    target_keys = set(raw_target_keys) if raw_target_keys is not None else None
    data = load_cvtg_samples(args.benchmark_root, subsets, areas, target_keys=target_keys)
    if not data:
        raise ValueError("No CVTG samples were found. Check benchmark_root or filtering arguments.")

    if target_keys is not None:
        print(f"Filtered to target keys: {sorted(target_keys)} => {len(data)} samples")

    set_random_seeds(args.seed)
    print(f"Processing {len(data)} samples")
    _warn_if_unsupported(*image_size)

    for sample in tqdm(data):
        prompt = sample["prompt"]
        subset = sample["subset"]
        key = sample["key"]
        area = sample["area"]

        cur_output_folder = os.path.join(output_path, subset, str(area))
        output_file = os.path.join(cur_output_folder, f"{key}.png")
        if os.path.exists(output_file):
            continue

        grid_image = engine.generate(
            prompt,
            image_size=image_size,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            timestep_shift=args.timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=args.num_steps,
            batch_size=1,
            seed=args.seed,
        )[0]

        if save_size is not None:
            grid_image = grid_image.resize((save_size, save_size), Image.Resampling.LANCZOS)

        os.makedirs(cur_output_folder, exist_ok=True)
        grid_image.save(output_file)

    print(f"Finished processing {len(data)} examples")
