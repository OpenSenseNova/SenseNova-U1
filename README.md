# SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-Unify Architecture

<p align="center">
  <strong>English</strong> | <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/assets/teaser.png" alt="SenseNova-U1" width="720">
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://unify.light-ai.top/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20SenseNova_U1-Demo-Green" alt="SenseNova-U1 Demo"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
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

## News

- `[TBD]` Initial release of SenseNova-U1 (code, weights, and technical report).

## Model Zoo

<!-- TODO: fill in the table once weights are released -->

| Model | Params | HF Weights |
| :---- | :------- | :--------- |
| SenseNova-U1-Mini | 16B | [🤗 link (TBD)](#) |
| SenseNova-U1-Flash | 38BA3B | [🤗 link (TBD)](#) |

## 🎨 Showcases

### Infographics Generation

### Interleaved Generation


## 🛠️ Quick Start

### Use with SenseNova-Skills (zero-config, recommended)

The easiest way to try SenseNova-U1 is through our companion repository **[SenseNova-Skills](https://github.com/OpenSenseNova/SenseNova-Skills)**, which ships SenseNova-U1 as a ready-to-use skill. No extra environment setup, dependency pinning, or model-loading boilerplate is required — install the skills package and call into SenseNova-U1 directly.

Refer to the [SenseNova-Skills README](https://github.com/OpenSenseNova/SenseNova-Skills) for installation and usage details.

### Run with transformers + diffusers

We recommend [**uv**](https://docs.astral.sh/uv/) to manage the Python environment.

> uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

### 1. Clone the repository

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

### 2. Install dependencies with uv

```bash
# Pick the CUDA extra that matches your system, e.g. cu121 / cu124 / cu126 / cu128
uv sync --extra cu124
source .venv/bin/activate
```

#### Visual Understanding

```bash
TBA
```

#### Visual Generation

Text-to-Image

```bash
TBA
```

#### Interleaved Generation

```bash
TBA
```

### Run with LightLLM + LightX2V

To efficiently serve a unified model that jointly handles understanding and generation, we co-design a dedicated inference stack on top of **[LightLLM](https://github.com/ModelTC/lightllm)** and **[LightX2V](https://github.com/ModelTC/lightx2v)**, featuring:

- **Disaggregated serving & transfer design** — understanding and generation workloads are served on separate engines with a low-overhead KV / feature transfer channel.
- **Understanding-side optimizations** — tailored kernels, scheduling, and KV management for the VLM path.
- **Generation-side optimizations** — step / sampler / cache optimizations for the X2I generation path.

We observe competitive end-to-end latency and throughput across understanding, generation, and interleaved workloads.

> 📖 **Full design, benchmarking protocol, and performance numbers:** see [`docs/inference_infrastructure.md`](./docs/inference_infrastructure.md).


TBA: run with lightx2v

## Evaluation

<!-- TODO: link to evaluation guide once available -->
Evaluation scripts and benchmark reproduction guides will be released in `evaluation/`.


## 🖊️ Citation

<!-- TODO: fill in once the paper is released -->
```bibtex

```

## License

This project is released under the [Apache 2.0 License](./LICENSE).