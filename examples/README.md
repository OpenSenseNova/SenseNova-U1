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
│   ├── resize_inputs.py       # offline pre-resize helper (recommended)
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

## Text-to-Image

Single prompt:

```bash
python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "这张信息图的标题是“SenseNova-U1”，采用现代极简科技矩阵风格。整体布局为水平三列网格结构，背景是带有极浅银灰色细密点阵的哑光纯白高级纸张纹理，画面长宽比为16:9。\n\n排版采用严谨的视觉层级：主标题使用粗体无衬线黑体字，正文使用清晰的现代等宽字体。配色方案极其克制，以纯白色为底，深炭黑为主视觉文字和边框，浅石板灰用于背景色块和次要信息区分，图标采用精致的银灰色线框绘制。\n\n在画面正上方居中位置，使用醒目的深炭黑粗体字排布着大标题“SenseNova-U1”。标题正下方是浅石板灰色的等宽字体副标题“新一代端到端统一多模态大模型家族”。\n\n画面主体分为左、中、右三个相等的垂直信息区块，区块之间通过充足的负空间进行物理隔离。\n\n左侧区块的主题是概述。顶部有一个银灰色线框绘制的、由放大镜和齿轮交织的图标，旁边是粗体小标题“Overview”。该区块内从上到下垂直排列着三个要点：第一个要点旁边是一个代表文档与照片重叠的极简图标，紧跟着文字“多模态模型家族，统一文本/图像理解和生成”。向下是由两个相连的同心圆组成的架构图标，配有文字“基于NEO-Unify架构（端到端统一理解和生成）”。最下方是一个带有斜线划掉的眼睛和漏斗形状的图标，明确指示文本“无需视觉编码器(VE)和变分自编码器(VAE)”。\n\n中间区块展示模型矩阵。顶部是一个包含两个分支节点的树状网络图标，旁边是粗体小标题“两个模型规格”。区块内分为上下两个包裹在浅石板灰色极细边框内的卡片。上方的卡片内画着一个代表高密度的实心几何立方体图标，大字标注“SenseNova-U1-8B-MoT”，下方是等宽字体说明“8B MoT 密集主干模型”。下方的卡片内画着一个带有闪电符号的网状发光大脑图标，大字标注“SenseNova-U1-A3B-MoT”，下方是等宽字体说明“A3B MoT 混合专家（MoE）主干模型”。在这两个独立卡片的正下方，左侧放置一个笑脸轮廓图标搭配文字“将在HF等平台公开”，右侧放置一个带有折角的书面报告图标搭配文字“将发布技术报告”。\n\n右侧区块呈现核心优势。顶部是一个代表巅峰的上升阶梯折线图图标，旁边是粗体小标题“Highlights”。该区块内部垂直分布着四个带有浅石板灰底色的长方形色块，每个色块内部左侧对应一个具体的图标，右侧为文字。第一个色块内是一个无缝相连的莫比乌斯环图标，配文“原生统一架构，无VE和VAE”。第二个色块内是一个顶端带有星星的奖杯图标，配文“单一统一模型在理解和生成任务上均达到SOTA性能”。第三个色块内是代表文本行与拍立得照片交替穿插的图标，配文“强大的原生交错推理能力（模型原生生成图像进行推理）”。最后一个色块内是一个被切分出一小块的硬币与详细饼状图结合的图标，配文“能生成复杂信息图表，性价比出色”。" \
  --width 2720 --height 1536 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output output.png \
  --profile
```

Batched prompts from a JSONL file (each line must contain a `prompt`;
`width` / `height` / `seed` are optional):

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

See [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) for a tiny starter file. Run `python examples/t2i/inference.py --help` for the full flag list.

Infographic-focused batched generation:

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples_infographic.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

See [`t2i/data/samples_infographic.jsonl`](./t2i/data/samples_infographic.jsonl) to reproduce the infographic showcases.

### T2I reasoning (think mode)

The model can run a **reasoning** phase before denoising: it autoregressively fills `<think>...</think>`, then generates the image.

Single prompt (image + reasoning text):

```bash
python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
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
    --model_path SenseNova/SenseNova-U1-8B-MoT \
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

Short prompts — especially for **infographic** generation — can be enhanced by a strong LLM before inference, which noticeably lifts information density, typography fidelity, and layout adherence. Enable with `--enhance`:

```bash
# export U1_ENHANCE_API_KEY=sk-...                # required
# defaults target Gemini 3.1 Pro via its OpenAI-compatible endpoint;
# override any of these to point at SenseNova / Claude / Kimi 2.5 etc.:
# export U1_ENHANCE_BACKEND=chat_completions   # or 'anthropic'
# export U1_ENHANCE_ENDPOINT=https://...chat/completions
# export U1_ENHANCE_MODEL=gemini-3.1-pro

python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "如何制作咖啡的教程" \
  --enhance --print_enhance \
  --output output.png
```

See [`docs/prompt_enhancement.md`](../docs/prompt_enhancement.md) for full details.

## Image Editing (it2i)

Single edit:

```bash
python examples/editing/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
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
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 \
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

**Resolution:** when input images are provided via `--image` or the JSONL `image` field, the output resolution follows the first input image (snapped to 32-aligned buckets via `smart_resize`), overriding `--resolution` / `--width` / `--height`.

### 1) Single sample, text prompt only
```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

### 2) Single sample, text prompt + input image

```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
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
    --model_path SenseNova/SenseNova-U1-8B-MoT \
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
  --model_path SenseNova/SenseNova-U1-8B-MoT \
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
    --model_path SenseNova/SenseNova-U1-8B-MoT \
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
