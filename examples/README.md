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
│       └── samples_infographic.jsonl
├── editing/                   # image editing (it2i)
│   ├── inference.py
│   ├── resize_inputs.py       # offline pre-resize helper (recommended)
│   └── data/
│       ├── samples.jsonl
│       └── images/
├── interleave/                # interleaved text+image gen  (runnable)
│   ├── inference.py
│   ├── run.sh
│   └── data/
│       ├── sample.jsonl
│       └── images/
└── vqa/                       # visual understanding / VQA
    ├── inference.py
    └── data/
        ├── questions.jsonl
        └── images/
```

## Text-to-Image

Single prompt:

```bash
python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
  --prompt "一个咖啡店门口有一个黑板，上面写着日日新咖啡，2元一杯，旁边有个霓虹灯，写着商汤科技，旁边有个海报，海报上面是一只小浣熊，海报下方写着SenseNova newbee。" \
  --width 2048 --height 2048 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output out.png \
  --profile
```

Batched prompts from a JSONL file (each line must contain a `prompt`;
`width` / `height` / `seed` are optional):

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-Mini-Beta \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

See [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) for a tiny starter file. Run `python examples/t2i/inference.py --help` for the full flag list.

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

Short prompts — especially for **infographic** generation — can be enhanced by a strong LLM before inference, which noticeably lifts information density, typography fidelity, and layout adherence. Enable with `--enhance`:

```bash
# export U1_ENHANCE_API_KEY=sk-...                # required
# defaults target Gemini 3.1 Pro via its OpenAI-compatible endpoint;
# override any of these to point at SenseNova / Claude / Kimi 2.5 etc.:
# export U1_ENHANCE_BACKEND=chat_completions   # or 'anthropic'
# export U1_ENHANCE_ENDPOINT=https://...chat/completions
# export U1_ENHANCE_MODEL=gemini-3.1-pro

python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
  --prompt "如何制作咖啡的教程" \
  --enhance --print_enhance \
  --output output.png
```

See [`docs/prompt_enhancement.md`](../docs/prompt_enhancement.md) for full details.

## Image Editing (it2i)

> 💡 **Pre-resize your inputs for best results.**
> Before running inference, down-/up-sample each source image **offline**
> so that `width * height ≈ 2048 * 2048` (aspect ratio preserved)
> — use [`editing/resize_inputs.py`](./editing/resize_inputs.py):
>
> ```bash
> python examples/editing/resize_inputs.py \
>   --src examples/editing/data/images \
>   --dst examples/editing/data/images_2048
> ```
>
> Then point `--image` / the JSONL manifest at the resized folder. The
> examples below assume you have already done this.

Single edit:

```bash
python examples/editing/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
  --prompt "Change the animal's fur color to a darker shade." \
  --image examples/editing/data/images/1.jpg \
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
    --model_path SenseNova/SenseNova-U1-Mini-Beta \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 \
    --profile --compare
```

Output resolution has two modes:

- **Auto (default)**: omit `--width / --height` — output tracks the first input via `smart_resize` (aspect ratio preserved, total pixels normalized to `--target_pixels` default `2048 * 2048`, H / W snapped to multiples of 32).
- **Explicit**: pass `--width W --height H` (both multiples of 32). 2048 × 2048 is a good general-purpose choice.

CFG defaults: `--cfg_scale 4.0` (text guidance), `--img_cfg_scale 1.0` (image CFG off by default). Run `python examples/editing/inference.py --help` for the full flag list.


## Interleave

`examples/interleave/inference.py` drives `model.interleave_gen`, which produces
**interleaved text and images in a single response**. The model can emit a
`<think>...</think>` reasoning block that generates intermediate images, followed
by a concise final answer. See [`interleave/run.sh`](./interleave/run.sh) for a
three-mode launcher covering every usage pattern below.

**Output files:** every sample writes `<stem>.txt` (generated text) plus `<stem>_image_<i>.png` for each generated image; `--jsonl` mode also emits a `results.jsonl` manifest.

**Resolution:** when input images are provided via `--image` or the JSONL `image` field, the output resolution follows the first input image (snapped to 32-aligned buckets via `smart_resize`), overriding `--resolution` / `--width` / `--height`.

### 1) Single sample, text prompt only
```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

### 2) Single sample, text prompt + input image

```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
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
    --model_path SenseNova/SenseNova-U1-Mini-Beta \
    --jsonl examples/interleave/data/sample.jsonl \
    --image_root examples/interleave/data/images\
    --resolution "16:9" \
    --output_dir outputs/interleave/jsonl
```

See [`interleave/data/sample.jsonl`](./interleave/data/sample.jsonl) for a
two-sample starter (one text-only, one image-conditioned).

## Visual Understanding (VQA)

Single image, with sampling enabled:

```bash
python examples/vqa/inference.py \
  --model_path SenseNova/SenseNova-U1-Mini-Beta \
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
    --model_path SenseNova/SenseNova-U1-Mini-Beta \
    --jsonl examples/vqa/data/questions.jsonl \
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

See [`vqa/data/questions.jsonl`](./vqa/data/questions.jsonl) for a starter file.

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
