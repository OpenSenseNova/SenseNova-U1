"""T2I validation for BF16, GGUF and v2 FP8/INT8 weights."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import torch
from quant_runtime import QuantLinear, load_quant

import sensenova_u1
from examples.t2i.inference import SenseNovaU1T2I


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--format", choices=["bf16", "gguf", "fp8", "int8-convrot"], required=True)
    p.add_argument("--prompt", default="A studio photograph of a red apple on a white table, soft daylight.")
    p.add_argument("--cases", help="Optional JSON list of id/prompt cases")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=4)
    p.add_argument("--timestep_shift", type=float, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 0:
        p.error("repeats must be positive and warmup nonnegative")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else [{"id": "00_apple", "prompt": a.prompt}]
    if any((out / (c["id"] + ".png")).exists() for c in cases):
        raise FileExistsError("Use a new output directory")
    sensenova_u1.set_attn_backend("flash")
    torch.cuda.synchronize()
    started = time.perf_counter()
    if a.format in ("fp8", "int8-convrot"):
        engine = SenseNovaU1T2I.__new__(SenseNovaU1T2I)
        engine.model, engine.tokenizer = load_quant(a.model_path, "cuda")
        engine.device, engine.vram_mode, engine.prefetch_count = "cuda", "full", 0
        engine.fast_vram_fraction, engine.fast_vram_headroom_gib = 1.0, 0.0
        engine.fast_activation_reserve_gib, engine.fast_vram_budget_gib = 0.0, None
        engine._last_think_text = ""
    else:
        engine = SenseNovaU1T2I(model_path=a.model_path, dtype=torch.bfloat16, device="cuda", vram_mode="full")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    # Fail before PNG conversion if the model produces invalid pixels.
    import examples.t2i.inference as base

    to_pil = base._to_pil

    def checked(tensor):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Nonfinite generated pixels")
        return to_pil(tensor)

    base._to_pil = checked
    summary = []
    for case in cases:
        times, peaks = [], []
        for repeat in range(a.warmup + a.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                images = engine.generate(
                    case["prompt"],
                    image_size=(a.width, a.height),
                    cfg_scale=a.cfg_scale,
                    cfg_norm="none",
                    cfg_interval=(0, 1),
                    timestep_shift=a.timestep_shift,
                    num_steps=a.num_steps,
                    seed=a.seed,
                    cfg_batching=True,
                )
            torch.cuda.synchronize()
            if repeat >= a.warmup:
                times.append(time.perf_counter() - start)
                peaks.append([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30])
        image_path = out / (case["id"] + ".png")
        images[0].save(image_path)
        quant = [m for m in engine.model.modules() if isinstance(m, QuantLinear)]
        if quant and (len(quant) != 588 or any(m.calls == 0 for m in quant)):
            raise RuntimeError("Quantized layer coverage failed")
        result = dict(
            case=case,
            format=a.format,
            model=a.model_path,
            output=str(image_path),
            parameters=vars(a),
            load_seconds=load_seconds,
            seconds=times,
            mean_seconds=sum(times) / len(times),
            peak_allocated_gib=max(p[0] for p in peaks),
            peak_reserved_gib=max(p[1] for p in peaks),
            quantized_layers=len(quant),
            quantized_calls=sum(m.calls for m in quant),
            kernel={"fp8": "torch._scaled_mm FP8xFP8", "int8-convrot": "torch._int_mm INT8xINT8"}.get(a.format),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            memory_policy="full model on GPU",
        )
        image_path.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        summary.append(result)
        print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
