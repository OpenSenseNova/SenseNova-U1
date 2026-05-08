from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import sensenova_u1
from sensenova_u1.models.neo_unify.utils import load_image_native
from sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    InferenceProfiler,
    add_offload_args,
    infer_input_device,
    load_model_and_tokenizer,
    make_offload_ctx,
    vram_mode_to_prefetch_count,
)


class SenseNovaU1VQA:
    """Thin wrapper for visual understanding / VQA inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gguf_checkpoint: str | None = None,
        device_map: str | None = None,
        max_memory: str | None = None,
        vram_mode: str = DEFAULT_VRAM_MODE,
    ) -> None:
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            gguf_checkpoint=gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.device = str(infer_input_device(self.model, fallback=device)) if device_map else device

    @torch.inference_mode()
    def answer(
        self,
        image,
        question: str,
        history: list | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[str, list]:
        pixel_values, grid_hw = load_image_native(image)
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
        grid_hw = grid_hw.to(self.device)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty

        with make_offload_ctx(self.model, self.prefetch_count, self.device) as offloaded:
            response, updated_history = offloaded.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=history,
                return_history=True,
                grid_hw=grid_hw,
            )
        return response, updated_history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual understanding / VQA inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image file.")
    src.add_argument(
        "--jsonl",
        help='JSONL file, one sample per line. Required fields: {"image": ..., "question": ...}. '
        'Optional: {"id": ...}.',
    )

    p.add_argument("--question", help="Question to ask about the image (used with --image).")
    p.add_argument("--output", default=None, help="Output file for single-image result (default: stdout).")
    p.add_argument("--output_dir", default="outputs", help="Output directory when using --jsonl.")

    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=None, help="Top-k sampling (default: None).")
    p.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (default: None).")

    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    add_offload_args(p)
    p.add_argument(
        "--gguf_checkpoint",
        default=None,
        help=(
            "Optional path to a .gguf quantized checkpoint. When set, the dequantizing "
            "diffusers GGUF Linear layer is used instead of safetensors weights. "
            "Requires the [gguf] extra (gguf>=0.10.0, diffusers>=0.30.0)."
        ),
    )
    p.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help=(
            "Attention kernel used by the Qwen3 layers. 'auto' picks flash-attn when importable and falls back to SDPA."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print timing and CUDA memory stats: model load time, average "
            "per-image inference time, peak GPU memory, and the same time "
            f"normalized per image token (patch size = {DEFAULT_IMAGE_PATCH_SIZE})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.image is not None and args.question is None:
        print("[error] --question is required when using --image", file=sys.stderr)
        sys.exit(1)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(
        enabled=args.profile,
        device=args.device,
        config={
            "vram_mode": args.vram_mode,
            "attn_backend": sensenova_u1.effective_attn_backend(),
            "dtype": args.dtype,
            "gguf": args.gguf_checkpoint,
        },
    )

    with profiler.time_load():
        engine = SenseNovaU1VQA(
            args.model_path,
            device=args.device,
            dtype=dtype,
            gguf_checkpoint=args.gguf_checkpoint,
            device_map=args.device_map,
            max_memory=args.max_memory,
            vram_mode=args.vram_mode,
        )

    if args.image is not None:
        # single image mode — image size used as proxy for profiler dimensions
        img_path = Path(args.image)
        with profiler.time_generate(width=1, height=1, batch=1):
            response, _ = engine.answer(
                img_path,
                args.question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(response)
            print(f"[saved] {out}")
        else:
            print(response)
        profiler.report()
        return

    # batch JSONL mode
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.jsonl) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_kw):  # type: ignore[no-redef]
            return x

    results = []
    for sample in tqdm(samples, desc="VQA"):
        img_path = Path(sample["image"])
        question = sample["question"]
        with profiler.time_generate(width=1, height=1, batch=1):
            response, _ = engine.answer(
                img_path,
                question,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        result = {"id": sample.get("id", ""), "image": str(img_path), "question": question, "answer": response}
        results.append(result)
        print(f"[{result['id'] or '?'}] {response[:80]}{'...' if len(response) > 80 else ''}")

    out_file = out_dir / "answers.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {out_file}")
    profiler.report()


if __name__ == "__main__":
    main()
