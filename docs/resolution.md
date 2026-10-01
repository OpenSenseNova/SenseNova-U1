# Resolution terminology and custom high-resolution inference

For **SenseNova-U1.5 (final)**, “native 4K” describes high-resolution training extending to **4096 × 4096 (16.78 MP)**, not 4 megapixels and not a default UHD output format. Native refers to direct pixel-space generation rather than an upscaling stage.

## Evidence

The [technical report](pdf/SenseNOVA_U1_5.pdf), Section 2.1 / Figure 3, uses a 4096² noise-scale reference. Section 3.2 / Table 2 specifies initial 256²–1024² generation pre-training, followed by 512²–4096² high-resolution training; later listed stages retain this range. Section 3.2, Stage 5, samples T2I target areas from {1024², 1536², 2048², 3072², 4096²}, with aspect ratios {1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 1:2, 2:1}, then rounds dimensions to multiples of 32. These are area levels, not a universal longest-side cap; exact shape frequencies are not provided.

The released [PT launcher](../training/shell/train_u1/U1.5_8B.sh) sets generation pixels to 256²–1024² and sequence length to 8192. It is an initial-phase example, not the full high-resolution recipe. Its [configuration](../training/configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_pt.py) reads the bounds from the environment. This example alone does not establish the final checkpoint's training distribution.

The [official model card](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) claims native 4K while showing a 2048² quick start. Model descriptions should distinguish training capability from inference defaults. The final report does not independently establish the same coverage for Preview or the original U1.

## Recommended presets versus custom output

The T2I example recommends approximately **4 MP total area** (3.98–4.19 MP): 2048 × 2048 for 1:1, 2720 × 1536 for 16:9, and a longest side of 3456 for 3:1. **There is no standard UHD or 4096-class bucket in that table.** ComfyUI's editing policy `Auto · 4MP` similarly targets 2048² output pixels; it is not the definition of the model's 4K training capability.

Custom `--width` and `--height` are accepted by the released T2I path, which constructs the pixel grid from the requested size. Both must be positive multiples of **32** for the released 16-pixel patch / 2× merge path. Exact **3840 × 2160 is not directly compatible**, since 2160 is not a multiple of 32. An aligned alternative is 3840 × 2176 (8.36 MP), followed by a separate crop if exact UHD is needed; this is not native exact-UHD output.

For a custom **4096 × 4096** experiment with final U1.5, start with the existing base-model defaults:

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1.5-8B-MoT \
  --prompt "A cinematic mountain lake at sunrise, realistic photography." \
  --width 4096 --height 4096 \
  --batch_size 1 --dtype bfloat16 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --seed 42 --profile --output output_4096.png
```

For aligned near-UHD, replace dimensions with `--width 3840 --height 2176`. These sampling values are a starting point, **not a separately validated 4K optimum**. With the matching 8-step LoRA, follow [the LoRA guide](base_vs_distill.md#sensenova-u15-recommended) and its CFG/step settings.

4096² is explicitly within the report's training resolution/area distribution. The near-UHD size is code-compatible, but the report does not establish that this exact area/shape was sampled. Code acceptance alone is not evidence of training coverage or guaranteed quality.

At 32 × 32 pixels per visual token, 2048² uses 4096 image tokens, aligned near-UHD uses 8160, and 4096² uses 16384. Activation/cache memory and attention compute grow substantially. No universal VRAM requirement or successful 4K GPU run is established here. Start with batch size 1 and `--profile`; consult [memory-efficient inference](../README.md#-memory-efficient-inference-gguf--vram-modes). Offloading/quantization reduces weight memory but does not eliminate activation costs or guarantee that large outputs fit. If generation runs out of memory, lower the area or use hardware with more memory.

## Wording for separately hosted model cards

Use **“Native high-resolution generation, with training extending to 4096 × 4096; reference inference defaults to approximately 4 MP presets”**, with a link to these size constraints. This repository change does not automatically update Hugging Face or ModelScope cards.
