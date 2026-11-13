# Examples

Reference inference scripts for SenseNova-U1. Every script here is intentionally
self-contained — on top of the `sensenova_u1` package itself it only pulls in
`torch`, `transformers`, `pillow`, `numpy` (and optionally `tqdm` /
`flash-attn`).

Each task lives in its own subfolder with a matching `data/` directory of
sample inputs:

```
examples/
├── README.md
├── t2i/                       # text-to-image
│   ├── inference.py
│   └── data/
│       ├── samples.jsonl
|       ├── samples_reasoning.jsonl 
│       └── samples_infographic.jsonl
├── editing/                   # image editing (it2i)
│   ├── inference.py
│   └── data/
│       ├── samples.jsonl
│       ├── samples_reasoning.jsonl
│       ├── images/
│       └── images_reasoning/
├── interleave/                # interleaved text+image gen  (runnable)
│   ├── inference.py
│   ├── run.sh
│   └── data/
│       ├── samples.jsonl
│       ├── samples_reasoning.jsonl
│       ├── images/
│       └── images_reasoning/
└── vqa/                       # visual understanding / VQA
    ├── inference.py
    └── data/
        ├── samples.jsonl
        └── images/
```

## Single-GPU memory modes

On a single GPU, start U1.5 with `--vram_mode balanced`; the CLI uses `full` when this option is omitted. All four reference scripts (t2i, interleave, editing, vqa) accept the memory modes below. `balanced` and `low` need substantial host RAM: process CPU RSS peaked around 52.2 GiB in the RTX 5090 U1.5 runs. Plan for 64 GB of system RAM as a starting point; see the [U1.5 memory and timing results](../docs/u1.5_inference.md) for the tested scope and GPU figures.

| `--vram_mode` | Behavior |
|---------------|----------|
| `full`        | Whole model on GPU, no offload. (Default.) |
| `fast`        | Async prefetch, then retain generation layers within the GPU memory budget. |
| `low`         | Synchronous per-layer CPU<->GPU swap to reduce weight VRAM use. |
| `balanced`    | Async prefetch to overlap data transfer with compute. Recommended starting mode for U1.5. |

Fast-mode residency controls are shared by all four scripts:

| Option | Default | Meaning |
|--------|---------|---------|
| `--fast_vram_fraction` | `0.90` | Automatic budget as a fraction of physical VRAM. |
| `--fast_vram_headroom_gib` | `2` | Reusable VRAM headroom reserved for later allocations. |
| `--fast_activation_reserve_gib` | `4` | Allowance for post-decoder activations. |
| `--fast_vram_budget_gib` | unset | Absolute budget overriding the fraction. |

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A cinematic mountain village at sunrise" \
  --vram_mode balanced \
  --output output.png
```

## Text-to-Image

Single prompt:

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A four-step pour-over coffee infographic on a light background, titled Pour-over Coffee" \
  --width 2720 --height 1536 \
  --vram_mode balanced \
  --batch_size 1 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output output.png \
  --profile
```

The [original infographic prompt](t2i/data/u1_infographic_prompt.txt) is kept as a sample input. Its release copy does not describe the current version.

Batched prompts from a JSONL file (each line must contain a `prompt`;
`width` / `height` / `seed` are optional):

```bash
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

See [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) for a tiny starter file. Run `python examples/t2i/inference.py --help` for the full flag list.

Infographic-focused V3 batched generation:

```bash
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
    --jsonl examples/t2i/data/samples_infographic_V2.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

The prompts in the existing [`t2i/data/samples_infographic_V2.jsonl`](./t2i/data/samples_infographic_V2.jsonl) are also suitable for V3. For model details, see [Infographic Model](../docs/u1_infographic_model.md).

Infographic-focused 8-step LoRA generation (V1.0 LoRA)

```bash
hf download sensenova/SenseNova-U1-8B-MoT-LoRAs \
  SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors \
  --local-dir ./models/SenseNova-U1-8B-MoT-LoRAs

python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT-Infographic \
    --lora_path ./models/SenseNova-U1-8B-MoT-LoRAs/SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors \
    --jsonl examples/t2i/data/samples_infographic.jsonl \
    --output_dir outputs/ \
    --cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 8 \
    --profile
```

The 8-step infographic LoRA above is released for `SenseNova-U1-8B-MoT-Infographic` and can also be used with `SenseNova-U1-8B-MoT-Infographic-V2` for faster generation, but may reduce aesthetics or cause blurry/repeated text. For best quality, use the full V2 checkpoint.

### T2I reasoning (think mode)

The model can run a **reasoning** phase before denoising: it autoregressively fills `<think>...</think>`, then generates the image.

Single prompt (image + reasoning text):

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A male peacock trying to attract a female" \
  --width 2048 --height 2048 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --seed 42 \
  --think \
  --print_think \
  --output outputs/peacock.png
```

This writes `outputs/peacock.think.txt` with the raw thinking tokens. Use `--think_output /path/to/reasoning.txt` to choose another path, or `--print_think` to echo it to stdout.

```bash
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples_reasoning.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --seed 42 \
    --think \
    --print_think \
    --profile
```

JSONL: set `"think": true` per sample, or pass `--think` for all samples.

### Supported resolution buckets

SenseNova-U1 is trained on ~2K-pixel resolution buckets. Passing arbitrary `--width` / `--height` is allowed but quality may degrade for untrained shapes.

| Aspect ratio | Width × Height |
| :----------- | :------------- |
| 1:1          | 2048 × 2048    |
| 16:9 / 9:16  | 2720 × 1536 / 1536 × 2720 |
| 3:2 / 2:3    | 2496 × 1664 / 1664 × 2496 |
| 4:3 / 3:4    | 2368 × 1760 / 1760 × 2368 |
| 2:1 / 1:2    | 2880 × 1440 / 1440 × 2880 |
| 3:1 / 1:3    | 3456 × 1152 / 1152 × 3456 |

### Prompt Enhancement for Infographics

`--enhance` calls an external model to expand the prompt before U1 generates the image. It requires an API key and adds one request. See [prompt enhancement](../docs/prompt_enhancement.md) for backend configuration.

```bash
export U1_ENHANCE_API_KEY="your-api-key"

python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "如何制作咖啡的教程" \
  --enhance --print_enhance \
  --output output.png
```

## Image Editing (it2i)

Single edit:

```bash
python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "Change the jacket of the person on the left to bright yellow." \
  --image examples/editing/data/images/1.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited.png \
  --profile --compare
```

Batched edits from a JSONL file (each line must contain a `prompt` and
`image` path; `seed` / `type` are optional; `image` can also be a list of
paths to pass multiple reference images; a per-sample `width` + `height` pair
overrides the CLI default for that line):

```bash
python examples/editing/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 \
    --profile --compare
```

### Infographic Editing (V3)

Use `SenseNova-U1-8B-MoT-Infographic-V3` for infographic editing:

```bash
python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --prompt "Change the main title to 'SenseNova-U1 V3' while preserving the remaining content and layout." \
  --image docs/assets/showcases/t2i_infographic/0004.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited_infographic.png \
  --profile --compare
```

Output resolution has two modes:

- **Auto (default)**: omit `--width / --height` — output tracks the first input via `smart_resize` (aspect ratio preserved, total pixels normalized to `--target_pixels` default `2048 * 2048`, H / W snapped to multiples of 32).
- **Explicit**: pass `--width W --height H` (both multiples of 32). 2048 × 2048 is a good general-purpose choice.

For best editing quality, provide high-resolution input/reference images when possible.
Use `--input_max_pixels auto` to keep up to two inputs at 2048 × 2048 and divide that total budget across more inputs.
This is a conservative default based on a two-reference 2048 × 2048 memory budget;
tune the cap for your GPU memory and quality needs, or pass an integer such as `1048576` for a 1024 × 1024 cap.
Resize-to-budget is enabled by default; pass `--no-do-resize` to skip this outer resize step.
The model's native preprocessing still applies its own input limits.
The script prints the per-image input cap before generation.

CFG defaults: `--cfg_scale 4.0` (text guidance), `--img_cfg_scale 1.0` (image CFG off by default). Run `python examples/editing/inference.py --help` for the full flag list.

See [`editing/data/samples.jsonl`](./editing/data/samples.jsonl) for a tiny starter file.

## Interleave

`examples/interleave/inference.py` drives `model.interleave_gen`, which produces
**interleaved text and images in a single response**. The model can emit a
`<think>...</think>` reasoning block that generates intermediate images, followed
by a concise final answer. See [`interleave/run.sh`](./interleave/run.sh) for a
three-mode launcher covering every usage pattern below.

**Output files:** every sample writes `<stem>.txt` (generated text) plus `<stem>_image_<i>.png` for each generated image; `--jsonl` mode also emits a `results.jsonl` manifest.

Use `--max_images 1` to cap a run at one generated image (default: 10), for example when comparing single-image inference times. A prompt can still return text without an image.

**Resolution:** when input images are provided via `--image` or the JSONL `image` field, the output resolution follows the first input image (snapped to 32-aligned buckets via `smart_resize`), overriding `--resolution` / `--width` / `--height`.

### 1) Single sample, text prompt only
```bash
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

### 2) Single sample, text prompt + input image

```bash
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "<image>\n图文交错生成小猫游览故宫的场景" \
  --image examples/interleave/data/images/image0.jpg \
  --output_dir outputs/interleave/text_image \
  --stem demo_text_image
```

### 3) Batched samples from JSONL

Each line is one sample:

```json
{"prompt": "..."}
{"prompt": "...", "image": ["a.jpg"]}
```

```bash
python examples/interleave/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/interleave/data/samples.jsonl \
    --resolution "16:9" \
    --output_dir outputs/interleave/jsonl
```

See [`interleave/data/samples.jsonl`](./interleave/data/samples.jsonl) for a
two-sample starter (one text-only, one image-conditioned).

## Visual Understanding (VQA)

Single image, with sampling enabled:

```bash
python examples/vqa/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image examples/vqa/data/images/menu.jpg \
  --question "My friend and I are dining together tonight. Looking at this menu, can you recommend a good combination of dishes for 2 people? We want a balanced meal — a mix of mains and maybe a starter or dessert. Budget-conscious but want to try the highlights." \
  --output outputs/menu_answer.txt \
  --max_new_tokens 8192 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.05 \
  --profile
```

Omit `--do_sample` (and the sampling flags) for deterministic greedy decoding.

Batched questions from a JSONL file (each line must contain `image` and `question`; `id` is optional):

```bash
python examples/vqa/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/vqa/data/samples.jsonl \
    --output_dir outputs/vqa/ \
    --max_new_tokens 8192 \
    --do_sample \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --repetition_penalty 1.05 \
    --profile
```

Results are written to `outputs/vqa/answers.jsonl`, one JSON object per line with `id`, `image`, `question`, and `answer` fields.

See [`vqa/data/samples.jsonl`](./vqa/data/samples.jsonl) for a starter file.

### Generation parameters

| Flag | Default | Description |
| :--- | :------ | :---------- |
| `--max_new_tokens` | 1024 | Maximum response length |
| `--do_sample` | off (greedy) | Enable sampling |
| `--temperature` | 0.7 | Sampling temperature (used with `--do_sample`) |
| `--top_p` | 0.9 | Nucleus sampling threshold (used with `--do_sample`) |
| `--top_k` | None | Top-k sampling (used with `--do_sample`) |
| `--repetition_penalty` | None | Repetition penalty |

Run `python examples/vqa/inference.py --help` for the full flag list.
