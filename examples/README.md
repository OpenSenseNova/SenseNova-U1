# Examples

For ComfyUI, see the [official SenseNova-U1.6 templates](comfyui/README.md).

Reference inference scripts for SenseNova-U1 and its releases. The
[U1.5 model guide](../docs/u1.5_model.md) contains the shortest commands for
the current U1.5 checkpoint; this page keeps the full CLI reference and older
U1 or infographic examples. Every script here is intentionally self-contained
— on top of the `sensenova_u1` package itself it only pulls in
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
  --bucket 4m --aspect_ratio 16:9 \
  --vram_mode balanced \
  --batch_size 1 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output output.png \
  --profile
```

The [original infographic prompt](t2i/data/u1_infographic_prompt.txt) is kept as a sample input. Its release copy does not describe the current version.

Batched prompts from a JSONL file. Each line needs a `prompt` and can set a `seed`.
A `width` and `height` pair overrides the CLI output size for that sample.

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
  --bucket 4m --aspect_ratio 1:1 \
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

In JSONL mode, `--think` sets the run default. A sample's `think` field overrides it.

### Supported resolution buckets

Each script prints the resolved size before generating a sample, in the form `[resolution] sample=1 --width 1536 --height 1536`. Sample numbers start at 1. Copy both flags to reuse that output size. Both values must be positive multiples of 32, and the pair overrides `--bucket` and `--aspect_ratio`.

Select an output size with `--bucket` and `--aspect_ratio`. The CLI accepts pixel budgets from `1m` through `16m`, including fractional values such as `2.5m` and `12.5m`. Each `m` represents 1024 × 1024 pixels. Start with `4m`. For the aspect ratio, use two positive integers in `W:H` format and a ratio from `1:3` through `3:1`, such as `21:9` or `5:4`. When you supply just one flag, the other defaults to `4m` or `9:16`. The scripts calculate dimensions from the chosen budget and aspect ratio, then round both axes down to multiples of 32. Use a checkpoint that supports the requested output size. The table shows some example sizes.

| Aspect ratio | 1M width × height | 4M width × height | 16M width × height |
| :----------- | :---------------- | :---------------- | :----------------- |
| 1:1          | 1024 × 1024             | 2048 × 2048             | 4096 × 4096             |
| 16:9 / 9:16  | 1344 × 768 / 768 × 1344 | 2720 × 1536 / 1536 × 2720 | 5440 × 3072 / 3072 × 5440 |
| 3:2 / 2:3    | 1248 × 832 / 832 × 1248 | 2496 × 1664 / 1664 × 2496 | 4992 × 3328 / 3328 × 4992 |
| 4:3 / 3:4    | 1152 × 864 / 864 × 1152 | 2336 × 1760 / 1760 × 2336 | 4704 × 3520 / 3520 × 4704 |
| 2:1 / 1:2    | 1440 × 704 / 704 × 1440 | 2880 × 1440 / 1440 × 2880 | 5792 × 2880 / 2880 × 5792 |
| 3:1 / 1:3    | 1760 × 576 / 576 × 1760 | 3520 × 1152 / 1152 × 3520 | 7072 × 2336 / 2336 × 7072 |

Text-to-image defaults to `4m` at `9:16` (1536 × 2720). For example, `--bucket 8m --aspect_ratio 1:1` selects 2880 × 2880. In JSONL mode, the flags set the size for samples without their own dimensions.

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
  --bucket 4m --aspect_ratio 1:1 \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited.png \
  --profile --compare
```

Batched edits from a JSONL file. Each line needs a `prompt` and an `image` path or list of reference paths. It can also have a `type` label for the output filename. Use `--seed` to set the seed for the run.

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

The editing script accepts an exact size, a pixel budget and aspect ratio, or a size derived from the input.

- Pass `--width` and `--height` together to reuse an exact output size.
- Pass `--bucket 4m --aspect_ratio 1:1` for 2048 × 2048 or select another combination from the [resolution table](#supported-resolution-buckets).
- For auto sizing, `smart_resize` uses the aspect ratio of the first input after any input resizing. It targets `--target_pixels` (default `2048 * 2048`) and aligns the output dimensions to multiples of 32.

A JSONL sample can override the CLI selection with both `width` and `height`, each a multiple of 32.

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
by a concise final answer. Set `MODEL_PATH` to a local checkpoint path or Hugging Face model ID, then run `bash examples/interleave/run.sh` to launch the examples below. See [`interleave/run.sh`](./interleave/run.sh).

Each sample writes its prompt and generated text to `<stem>.txt`, generated images to `<stem>_image_<i>.png`, and copies of any input images to `<stem>_input_<i>.png`. JSONL mode also writes a `results.jsonl` manifest.

Use `--max_images 1` to cap a run at one generated image (default: 10), for example when comparing single-image inference times. A prompt can still return text without an image.

Use `--width` and `--height` for an exact output size, or use `--bucket` and `--aspect_ratio` to calculate it. With no CLI size options, a sample with input images uses the first image's aspect ratio. The script resizes as needed for an area of 262,144 to 2,097,152 pixels and aligns both dimensions to multiples of 32. Text-only samples fall back to `4m` at `9:16` (1536 × 2720).

### 1) Single sample, text prompt only
```bash
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --bucket 4m --aspect_ratio 16:9 \
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

Each line is one sample. A `width` and `height` pair overrides the default or CLI size for text-only samples and for image-conditioned samples with a CLI size selection. Use `--seed` to set the seed for the run.

```json
{"prompt": "..."}
{"prompt": "...", "image": ["a.jpg"]}
```

```bash
python examples/interleave/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/interleave/data/samples.jsonl \
    --bucket 4m --aspect_ratio 16:9 \
    --output_dir outputs/interleave/jsonl
```

See [`interleave/data/samples.jsonl`](./interleave/data/samples.jsonl) for a batch with both text-only and image-conditioned samples.

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
