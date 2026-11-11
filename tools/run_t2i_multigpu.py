#!/usr/bin/env python3
"""Unofficial, version-pinned GPU acceptance wrapper, not a validated GPU recipe.
Runs the unchanged official SenseNovaU1T2I engine. No quantization/offload/LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

CODE_COMMIT = "94bf26c94ab7c30211dcc29466152d0302c7a262"
PROMPT = "A red ceramic mug on a wooden table, soft daylight, simple background."


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def validate_placement(device_map, expected_gpus):
    if not device_map:
        raise RuntimeError("Missing hf_device_map; do not claim multi-GPU placement.")
    normalized = set()
    for target in device_map.values():
        s = str(target)
        if s.isdigit():
            normalized.add(int(s))
        elif s.startswith("cuda:") and s[5:].isdigit():
            normalized.add(int(s[5:]))
        else:
            raise RuntimeError(f"Unexpected device {target!r}; CPU/disk/meta are not accepted in this profile.")
    if normalized != set(range(expected_gpus)):
        raise RuntimeError(f"Expected all {expected_gpus} GPUs, map actually uses {sorted(normalized)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gpus", type=int, required=True, choices=[2, 4])
    parser.add_argument(
        "--device-map", choices=["sequential", "auto", "balanced", "balanced_low_0"], default="balanced"
    )
    parser.add_argument("--max-memory", required=True, help="e.g. 0=18GiB,1=18GiB,cpu=0GiB")
    parser.add_argument(
        "--headroom-gib",
        type=float,
        default=4.0,
        help="preflight free-memory reserve, NOT a measured safe activation ceiling",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.headroom_gib) or args.headroom_gib < 0:
        parser.error("--headroom-gib must be finite and nonnegative")
    args.run_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"t2i-{args.gpus}gpu-", dir=args.run_root))
    print(f"RUN_DIR={run}", flush=True)
    result = {
        "status": "running",
        "stage": "start",
        "gpu_generation_and_file_checks_passed": False,
        "complete_acceptance_passed": False,
        "manual_visual_review": "not_done",
        "settings": vars(args),
        "command": sys.argv,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": PROMPT,
        "width": 2048,
        "height": 2048,
        "steps": 50,
        "seed": 42,
        "batch_size": 1,
        "cfg_scale": 4.0,
        "cfg_norm": "none",
        "timestep_shift": 3.0,
        "cfg_interval": [0.0, 1.0],
        "dtype": "bfloat16",
        "attention": "sdpa",
        "vram_mode": "full",
    }
    json_write(run / "result.json", result)
    start = time.perf_counter()
    torch = None

    def stage(name):
        result["stage"] = name
        json_write(run / "result.json", result)
        print(f"STAGE={name}", flush=True)

    try:
        stage("verify_code")
        actual = subprocess.check_output(["git", "-C", str(args.repo_dir), "rev-parse", "HEAD"], text=True).strip()
        if actual != CODE_COMMIT:
            raise RuntimeError(f"Code commit mismatch {actual}")
        dirty = subprocess.check_output(
            ["git", "-C", str(args.repo_dir), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        if dirty.strip():
            raise RuntimeError("Tracked official source is dirty")
        # Only read a local full-HF directory. Prior full hash verification remains required.
        if not (args.model_dir / "model.safetensors.index.json").is_file():
            raise RuntimeError("Full local HF checkpoint required")
        config_bytes = (args.model_dir / "config.json").read_bytes()
        result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        result["code_commit"] = actual
        sys.path.insert(0, str(args.repo_dir / "src"))
        import torch as torch_module

        torch = torch_module
        import accelerate
        import transformers
        from accelerate.utils.modeling import convert_file_size_to_int

        import sensenova_u1
        from sensenova_u1.utils.checkpoint_loading import parse_max_memory

        result["versions"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
        }
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Python 3.11 required for this profile")
        if result["versions"] != {
            "python": platform.python_version(),
            "torch": "2.8.0+cu128",
            "cuda_build": "12.8",
            "transformers": "5.15.0",
            "accelerate": "1.14.0",
        }:
            raise RuntimeError("Runtime differs from checked baseline; validate it as a separate profile.")
        stage("gpu_preflight")
        result["cuda_available"] = torch.cuda.is_available()
        result["visible_gpu_count"] = torch.cuda.device_count()
        result["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        if not result["cuda_available"] or result["visible_gpu_count"] != args.gpus:
            raise RuntimeError(
                f"Requires exactly {args.gpus} visible CUDA GPUs; observed {result['visible_gpu_count']}. No model load attempted."
            )
        result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
        budget = parse_max_memory(args.max_memory)
        if set(budget) != set(range(args.gpus)) | {"cpu"}:
            raise RuntimeError("Explicit memory budget required for every logical GPU plus cpu=0GiB.")
        numeric = {k: convert_file_size_to_int(v) for k, v in budget.items()}
        if numeric["cpu"] != 0 or any(numeric[i] <= 0 for i in range(args.gpus)):
            raise RuntimeError("GPU budgets must be positive and CPU budget zero for this profile.")
        result["gpu_preflight"] = []
        for i in range(args.gpus):
            with torch.cuda.device(i):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise RuntimeError(f"GPU {i}: native BF16 unsupported")
                q = torch.randn(1, 2, 16, 64, device=f"cuda:{i}", dtype=torch.bfloat16)
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                if not torch.isfinite(out).all().item():
                    raise RuntimeError(f"GPU {i}: nonfinite SDPA")
                mat = torch.randn(16, 16, device=f"cuda:{i}", dtype=torch.bfloat16)
                if not torch.isfinite(mat @ mat).all().item():
                    raise RuntimeError(f"GPU {i}: nonfinite BF16 matmul")
                del q, out, mat
                torch.cuda.synchronize(i)
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(i)
                if numeric[i] + args.headroom_gib * 2**30 > free:
                    raise RuntimeError(f"GPU {i}: budget+headroom exceeds currently free memory ({free} bytes)")
                result["gpu_preflight"].append(
                    {
                        "logical_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "free_bytes": free,
                        "total_bytes": total,
                        "budget_bytes": numeric[i],
                    }
                )
                torch.cuda.reset_peak_memory_stats(i)
        result["peer_access"] = [
            [True if i == j else torch.cuda.can_device_access_peer(i, j) for j in range(args.gpus)]
            for i in range(args.gpus)
        ]

        def sync_all():
            for i in range(args.gpus):
                torch.cuda.synchronize(i)

        def peaks():
            return [
                {
                    "logical_id": i,
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(i),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(i),
                }
                for i in range(args.gpus)
            ]

        stage("plan_placement")
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModel
        from transformers.integrations.accelerate import _get_device_map

        cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModel.from_config(cfg, dtype=torch.bfloat16)
        planned = _get_device_map(empty_model, args.device_map, dict(budget), None)
        json_write(run / "planned_device_map.json", planned)
        validate_placement(planned, args.gpus)
        del empty_model
        stage("load_model")
        spec = importlib.util.spec_from_file_location("u15_official_t2i", args.repo_dir / "examples/t2i/inference.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        sensenova_u1.set_attn_backend("sdpa")
        t = time.perf_counter()
        engine = entry.SenseNovaU1T2I(
            str(args.model_dir),
            device="cuda:0",
            dtype=torch.bfloat16,
            vram_mode="full",
            device_map=args.device_map,
            max_memory=args.max_memory,
        )
        sync_all()
        result["load_seconds"] = time.perf_counter() - t
        result["hf_device_map"] = getattr(engine.model, "hf_device_map", None)
        json_write(run / "hf_device_map.json", result["hf_device_map"])
        validate_placement(result["hf_device_map"], args.gpus)
        result["parameters_bytes_by_device"] = {}
        for p in engine.model.parameters():
            key = str(p.device)
            result["parameters_bytes_by_device"][key] = (
                result["parameters_bytes_by_device"].get(key, 0) + p.numel() * p.element_size()
            )
        if set(result["parameters_bytes_by_device"]) != {f"cuda:{i}" for i in range(args.gpus)}:
            raise RuntimeError(f"Unexpected actual parameter devices: {result['parameters_bytes_by_device']}")
        result["load_peak_memory"] = peaks()
        for i in range(args.gpus):
            torch.cuda.reset_peak_memory_stats(i)
        stage("generate")
        t = time.perf_counter()
        images = engine.generate(
            PROMPT,
            image_size=(2048, 2048),
            cfg_scale=4.0,
            cfg_norm="none",
            timestep_shift=3.0,
            cfg_interval=(0.0, 1.0),
            num_steps=50,
            batch_size=1,
            seed=42,
            think_mode=False,
        )
        sync_all()
        result["generate_and_to_pil_seconds"] = time.perf_counter() - t
        result["generate_peak_memory"] = peaks()
        stage("save_and_validate")
        if len(images) != 1:
            raise RuntimeError(f"Expected one image; got {len(images)}")
        output = run / "first.png"
        t = time.perf_counter()
        images[0].save(output)
        result["save_seconds"] = time.perf_counter() - t
        from PIL import Image

        with Image.open(output) as im:
            if im.format != "PNG" or im.size != (2048, 2048):
                raise RuntimeError(f"Output format/size mismatch: {im.format}/{im.size}")
            im.verify()
        with Image.open(output) as im:
            im.load()
        result["output"] = str(output)
        result["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        result["status"] = "automatic_checks_passed"
        result["gpu_generation_and_file_checks_passed"] = True
        result["exit_code"] = 0
        stage("complete_awaiting_manual_visual_review")
        return 0
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["exit_code"] = 1
        (run / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        result["wall_seconds"] = time.perf_counter() - start
        result["process_peak_rss_bytes_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        json_write(run / "result.json", result)
        print(f"FINAL_STATUS={result['status']}; RESULT={run / 'result.json'}", flush=True)


if __name__ == "__main__":

    def terminate(sig, frame):
        raise RuntimeError(f"Received signal {sig}")

    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
