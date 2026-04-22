# SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-Unify Architecture

<p align="center">
  <strong>English</strong> | <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://unify.light-ai.top/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20SenseNova_U1-Demo-Green" alt="SenseNova-U1 Demo"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <img src="docs/assets/teaser.png" alt="SenseNova-U1" width="720">
</p>

## Overview

Recent large vision–language models (VLMs) remain fundamentally constrained by a persistent dichotomy: understanding and generation are treated as distinct problems, leading to fragmented architectures, cascaded pipelines, and misaligned representation spaces. We argue that this divide is not merely an engineering artifact, but a structural limitation that hinders the emergence of native multimodal intelligence.
Hence, we introduce **SenseNova-U1**, a native unified multimodal paradigm built upon the **NEO-Unify** model, in which understanding and generation evolve as synergistic views of a single underlying process.
The key pillars are:
(i) A near-lossless visual interface that preserves both semantic richness and pixel fidelity without pre-trained vision encoders (VEs) and variational autoencoders (VAEs);
(ii) An end-to-end framework that operates directly on native inputs (i.e., pixels and text), showing impressive expressivity and generalization over modular counterparts;
(iii) A native Mixture-of-Transformers (MoT) architecture that supports modality-agnostic reasoning with minimal intrinsic conflict and high data-scaling efficiency.
We launch two native unified variants, **SenseNova-U1-Mini** and **SenseNova-U1-Flash**, built on dense (8B) and mixture-of-expert (30B-A3B) understanding baselines, respectively. Designed from first principles, they rival top-tier understanding-only VLMs across text understanding, vision–language perception, knowledge reasoning, agentic decision-making, and spatial intelligence. Meanwhile, they deliver strong semantic consistency and visual fidelity, excelling in conventional or knowledge-intensive any-to-image (X2I) synthesis, complex text-rich infographic generation, and interleaved vision–language generation, with or without think patterns. Beyond performance, we provide a comprehensive analysis of model design, data preprocessing, pre-/post-training, and inference strategies to support community research.
Last but not least, preliminary evidence displays that our models extend beyond perception and generation, performing strongly in vision–language–action (VLA) and world model (WM) scenarios. This points toward a broader roadmap where models do not translate between modalities, but think-and-act across them natively. Multimodal AI is no longer about connecting disparate systems. It is about building one that was never divided, and trusting the necessary capabilities to emerge from within.

## 📣 News

- `[TBD]` Initial release of SenseNova-U1 (code, weights, and technical report).

## 🦁 Model Zoo

<!-- TODO: fill in the table once weights are released -->

| Model | Params | HF Weights |
| :---- | :------- | :--------- |
| SenseNova-U1-Mini | 16B | [🤗 link (TBD)](#) |
| SenseNova-U1-Flash | 38BA3B | [🤗 link (TBD)](#) |

## 🎨 Showcases

A quick tour below; see [`docs/showcases.md`](./docs/showcases.md) for
additional editing and visual understanding samples.

### Text-to-Image (Infographics)

| | | |
| :---: | :---: | :---: |
| [<img width="300" alt="t2i landscape 0001" src="./docs/assets/showcases/t2i_infographic/0001_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0001_2720x1536.webp) | [<img width="300" alt="t2i landscape 0002" src="./docs/assets/showcases/t2i_infographic/0002_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0002_2720x1536.webp) | [<img width="300" alt="t2i landscape 0003" src="./docs/assets/showcases/t2i_infographic/0003_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0003_2720x1536.webp) |
| [<img width="300" alt="t2i square 0004" src="./docs/assets/showcases/t2i_infographic/0004_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0004_2048x2048.webp) | [<img width="300" alt="t2i square 0005" src="./docs/assets/showcases/t2i_infographic/0005_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0005_2048x2048.webp) | [<img width="300" alt="t2i square 0006" src="./docs/assets/showcases/t2i_infographic/0006_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0006_2048x2048.webp) |
| [<img width="200" alt="t2i portrait 0007" src="./docs/assets/showcases/t2i_infographic/0007_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0007_1536x2720.webp) | [<img width="200" alt="t2i portrait 0008" src="./docs/assets/showcases/t2i_infographic/0008_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0008_1536x2720.webp) | [<img width="200" alt="t2i portrait 0009" src="./docs/assets/showcases/t2i_infographic/0009_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0009_1536x2720.webp) |

### Image Editing

| | | |
| :---: | :---: | :---: |
| [<img alt="editing sample 0001" src="./docs/assets/showcases/editing/0001_2048x2048_compare.webp">](./docs/assets/showcases/editing/0001_2048x2048_compare.webp) | [<img alt="editing sample 0002" src="./docs/assets/showcases/editing/0002_2048x2048_compare.webp">](./docs/assets/showcases/editing/0002_2048x2048_compare.webp) | [<img alt="editing sample 0003" src="./docs/assets/showcases/editing/0003_2048x2048_compare.webp">](./docs/assets/showcases/editing/0003_2048x2048_compare.webp) |

> 📸 **More editing samples:** see [Image Editing gallery](./docs/showcases.md#image-editing).

### Interleaved Generation

| |
| :---: |
| [<img alt="interleave case 02" src="./docs/assets/showcases/interleave/case_02.webp">](./docs/assets/showcases/interleave/case_02.webp) |
| [<img alt="interleave case 03" src="./docs/assets/showcases/interleave/case_03.webp">](./docs/assets/showcases/interleave/case_03.webp) |

> 📸 **More interleaved samples:** see [Interleaved Generation gallery](./docs/showcases.md#interleaved-generation).

### Visual Understanding

| |
| :---: |
| [<img width="600" alt="vqa agentic case" src="./docs/assets/showcases/vqa/agentic_case.webp">](./docs/assets/showcases/vqa/agentic_case.webp) |
| [<img width="600" alt="vqa general cases" src="./docs/assets/showcases/vqa/general_case.webp">](./docs/assets/showcases/vqa/general_case.webp) |

## 📊 Benchmarks

> TODO: Add Benchmark Chart

Evaluation scripts and benchmark reproduction guides will be added in `evaluation/`.


## 🛠️ Quick Start

### Use with SenseNova-Skills (zero-config, recommended)

The easiest way to try SenseNova-U1 is through our companion repository **[SenseNova-Skills](https://github.com/OpenSenseNova/SenseNova-Skills)**, which ships SenseNova-U1 as a ready-to-use skill.

Refer to the [SenseNova-Skills README](https://github.com/OpenSenseNova/SenseNova-Skills) for installation and usage details.


### Run with LightLLM + LightX2V

To efficiently serve a unified model that jointly handles understanding and generation, we co-design a dedicated inference stack on top of **[LightLLM](https://github.com/ModelTC/lightllm)** and **[LightX2V](https://github.com/ModelTC/lightx2v)**, featuring:

- **Disaggregated serving & transfer design** — understanding and generation workloads are served on separate engines with a low-overhead KV / feature transfer channel.
- **Understanding-side optimizations** — tailored kernels, scheduling, and KV management for the VLM path.
- **Generation-side optimizations** — Kernel fusion, CFG parallelism, Ulysses parallelism, and improved memory management for KV cache.

We observe competitive end-to-end latency and throughput across understanding, generation, and interleaved workloads.

> 📖 **Full design, benchmarking protocol, and performance numbers:** see [`docs/inference_infrastructure.md`](./docs/inference_infrastructure.md).


TBA: run with lightx2v


### Run with transformers

We recommend [**uv**](https://docs.astral.sh/uv/) to manage the Python environment.

> uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

### 1. Clone the repository

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

### 2. Install dependencies with uv

```bash
uv sync
source .venv/bin/activate
```

The `sensenova_u1` package is installed in
editable mode, so the canonical [NEO-Unify model](src/sensenova_u1/models/neo_unify/) is automatically registered with `transformers.Auto*` at import time.

> **Older NVIDIA drivers:** the default index is CUDA 12.8. If your driver
> does not support cu128, change `[tool.uv.sources]` / `[[tool.uv.index]]`
> in `pyproject.toml` to e.g. `https://download.pytorch.org/whl/cu126` (and
> adjust the pinned torch / torchvision versions accordingly) before
> running `uv sync`.

#### Optional: flash-attn

`flash-attn` is declared as an optional extra;
without it the model transparently falls back to torch SDPA;
once flash-attn is importable the runtime picks it automatically (`--attn_backend auto`).

```bash
# (a) Build from source via PyPI
uv sync --extra flash

# (b) Install a prebuilt CUDA wheel matching your torch + Python
uv pip install /path/to/flash_attn-2.8.3+cu12torch28cxx11abitrue-cp311-cp311-*.whl
```

#### Visual Understanding

[`examples/vqa/inference.py`](./examples/vqa/inference.py) is a minimal visual question answering (VQA) inference script for SenseNova-U1.

**Single image mode:**

```bash
python examples/vqa/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --image examples/vqa/data/images/menu.jpg \
  --question "My friend and I are dining together tonight. Looking at this menu, can you recommend a good combination of dishes for 2 people? We want a balanced meal — a mix of mains and maybe a starter or dessert. Budget-conscious but want to try the highlights." \
  --output outputs/answer.txt \
  --max_new_tokens 8192 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.05 \
  --profile
```

Omit `--do_sample` (and the sampling flags) for deterministic greedy decoding.

**Batched inference with JSONL:**

For batched inference, pass a JSONL file via `--jsonl` (see [`examples/vqa/data/questions.jsonl`](./examples/vqa/data/questions.jsonl)). Each line requires `{"image": ..., "question": ...}` and optionally `{"id": ...}`:

```bash
python examples/vqa/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
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

**Generation parameters:**

- `--max_new_tokens` — maximum response length (default: 1024)
- `--do_sample` — enable sampling (default: greedy decoding)
- `--temperature` — sampling temperature (default: 0.7, used when `--do_sample`)
- `--top_p` — nucleus sampling threshold (default: 0.9, used when `--do_sample`)
- `--top_k` — top-k sampling (default: None, used when `--do_sample`)
- `--repetition_penalty` — repetition penalty (default: None)

Run `python examples/vqa/inference.py --help` for the full flag list.

#### Visual Generation

##### Text-to-Image

[`examples/t2i/inference.py`](./examples/t2i/inference.py) is a minimal text-to-image inference script for SenseNova-U1.

By default the model renders at **2048 × 2048** (1:1). You can override with `--width` / `--height`. SenseNova-U1 is trained on a set of resolution buckets (~2K total pixels) covering the following aspect ratios:

| Aspect ratio | Width × Height |
| :----------- | :------------- |
| 1:1          | 2048 × 2048    |
| 16:9 / 9:16  | 2720 × 1536 / 1536 × 2720 |
| 3:2 / 2:3    | 2496 × 1664 / 1664 × 2496 |
| 4:3 / 3:4    | 2368 × 1760 / 1760 × 2368 |
| 2:1 / 1:2    | 2880 × 1440 / 1440 × 2880 |
| 3:1 / 1:3    | 3456 × 1152 / 1152 × 3456 |

The script accepts arbitrary `--width` / `--height` and only emits a warning when they fall outside this table; quality may degrade for untrained shapes.

```bash
python examples/t2i/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "一个咖啡店门口有一个黑板，上面写着日日新咖啡，2元一杯，旁边有个霓虹灯，写着商汤科技，旁边有个海报，海报上面是一只小浣熊，海报下方写着SenseNova newbee。" \
  --width 2048 \
  --height 2048 \
  --cfg_scale 4.0 \
  --cfg_norm none \
  --timestep_shift 3.0 \
  --num_steps 50 \
  --output output.png \
  --profile
```

Run `python examples/t2i/inference.py --help` for the full flag list.


For batched inference, pass a JSONL file via `--jsonl` (see [`examples/t2i/data/samples.jsonl`](./examples/t2i/data/samples.jsonl)). Each line is `{"prompt": ...}` and optionally `{"width": W, "height": H, "seed": S}`:

```bash
python examples/t2i/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --profile
```

##### Prompt Enhancement for Infographics Generation

Short user prompts — especially for **infographic** generation — can be enhanced by a strong LLM before T2I inference,
which noticeably lifts information density, typography fidelity, and layout adherence.
Flip it on with `--enhance`:

```bash
# export U1_ENHANCE_API_KEY=sk-...                # required
# defaults target Gemini 3.1 Pro via its OpenAI-compatible endpoint;
# override any of these to point at SenseNova / Claude / Kimi 2.5 etc.:
# export U1_ENHANCE_BACKEND=chat_completions   # or 'anthropic'
# export U1_ENHANCE_ENDPOINT=https://...chat/completions
# export U1_ENHANCE_MODEL=gemini-3.1-pro

python examples/t2i/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "如何制作咖啡的教程" \
  --enhance \
  --print_enhance \
  --output output.png
```

Refer to [`docs/prompt_enhancement.md`](./docs/prompt_enhancement.md) for more details.

##### Image Editing

[`examples/editing/inference.py`](./examples/editing/inference.py) demonstrates the image editing capability of SenseNova-U1.

Output resolution is derived via `smart_resize` on the first input image — aspect ratio preserved, total pixels normalized to `--target_pixels` (default `2048 * 2048`). Pass `--width W --height H` (both multiples of 32) to override.

> 💡 **Best practice — pre-resize inputs offline.**
> For best quality, down-/up-sample each source image **offline**
> so its total pixels match `--target_pixels` (aspect ratio preserved) before running inference.
> A reference helper is provided at [`examples/editing/resize_inputs.py`](./examples/editing/resize_inputs.py):
>
> ```bash
> python examples/editing/resize_inputs.py \
>   --src examples/editing/data/images \
>   --dst examples/editing/data/images_2048
> ```
>
> Then point `--image` / the JSONL manifest at the resized folder.

Single edit:

```bash
python examples/editing/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "Change the animal's fur color to a darker shade." \
  --image examples/editing/data/images/1.jpg \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --cfg_norm none \
  --timestep_shift 3.0 \
  --num_steps 50 \
  --output output_edited.png \
  --profile --compare
```

For batched inference, pass a JSONL file via `--jsonl` (see
[`examples/editing/data/samples.jsonl`](./examples/editing/data/samples.jsonl)).
Each line is `{"prompt": ..., "image": ...}` where `image` can be a single
path or a list of paths for multi-reference editing; `width` + `height`,
`seed`, and `type` are optional. A per-sample `width` + `height` pair
overrides the CLI default for that line:

```bash
python examples/editing/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 \
    --img_cfg_scale 1.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \    
    --profile --compare
```

Run `python examples/editing/inference.py --help` for the full flag list.


#### Interleaved Generation

[`examples/interleave/inference.py`](./examples/interleave/inference.py) drives `model.interleave_gen` — the model emits **interleaved text and generated images in a single response**, optionally preceded by a `<think></think>` block whose intermediate images guide the final answer. See [`examples/interleave/run.sh`](./examples/interleave/run.sh) for a three-mode launcher and [`examples/README.md#interleave`](./examples/README.md#interleave) for the full walkthrough.

When input images are provided (either via `--image` or a JSONL sample's `image` field), the output resolution follows the first input image (snapped to 32-aligned buckets via `smart_resize`), overriding `--resolution` / `--width` / `--height`.

Every sample writes `<stem>.txt` (generated text) plus `<stem>_image_<i>.png` for each generated image; `--jsonl` mode also emits a `results.jsonl` manifest.

Single prompt, text only:

```bash
python examples/interleave/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

Single prompt with an input image — each `<image>` placeholder binds to one `--image` path, in order (repeatable):

```bash
python examples/interleave/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "<image>\n图文交错生成小猫游览故宫的场景" \
  --image examples/interleave/data/images/image0.jpg \
  --output_dir outputs/interleave/text_image \
  --stem demo_text_image
```

Batched inference from a JSONL file. Each line is `{"prompt": ...}` and optionally `{"image": [...], "width": W, "height": H, "seed": S, "think_mode": bool}`. Relative `image` paths resolve against `--image_root`:

```bash
python examples/interleave/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/interleave/data/sample.jsonl \
    --image_root examples/interleave/data/images \
    --resolution "16:9" \
    --output_dir outputs/interleave/jsonl
```

Run `python examples/interleave/inference.py --help` for the full flag list.


## 🛠️ Development

To catch lint / formatting issues locally before they fail CI, install the
pre-commit hook once after cloning:

```bash
uv pip install pre-commit   # or: pip install pre-commit
pre-commit install
pre-commit run --all-files  # optional: check the whole repo now
```


## 🖊️ Citation

<!-- TODO: fill in once the paper is released -->
```bibtex

```

## ⚖️ License

This project is released under the [Apache 2.0 License](./LICENSE).