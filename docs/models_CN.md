# SenseNova-U 模型目录

<p align="center"><a href="./models.md">English</a> | <strong>简体中文</strong></p>

[SenseNova-U 项目首页](../README_CN.md)介绍模型家族；本目录集中列出已发布权重。表中的型号名称在有指南时链接到对应指南，末列统一提供权重入口。新任务可从 [U1.5 指南](u1.5_model_CN.md)开始。8 步 V2 LoRA 是该基座的适配器，不能替代基座权重。

## 官方模型权重

状态列区分有仓库指南的发布版本和仅作索引的权重。

| 模型 | 作用 | 仓库状态 | 权重 |
| --- | --- | --- | --- |
| [SenseNova-U1.5-8B-MoT](u1.5_model_CN.md) | U1.5 基座，支持图像生成、编辑、图文交错和视觉理解 | 指南 + 案例；首页主推 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) |
| SenseNova-U1.5-8B-MoT-TailSteer | U1.5 版本，改善多轮图像编辑稳定性；用法见模型卡 | 仅模型卡 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-TailSteer) |
| SenseNova-U1.5-8B-MoT-SFT | U1.5 监督微调阶段权重 | 仅模型卡 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT) |
| [SenseNova-U1.5-8B-MoT-Preview](u1.5_preview_CN.md) | Preview 权重，侧重原生 4K 生成和图像编辑；与 U1.5 正式版权重分开 | 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview) |
| [SenseNova-U1-8B-MoT](u1_model_CN.md) | U1 8B 版本，支持视觉理解和图像生成 | 共用 U1 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-8B-MoT-SFT | U1 8B 监督微调阶段权重 | 仅模型卡 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-SFT) |
| [SenseNova-U1-A3B-MoT](u1_model_CN.md) | U1 混合专家版本，支持理解和生成 | 共用 U1 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) |
| SenseNova-U1-A3B-MoT-SFT | U1 A3B 监督微调阶段权重 | 仅模型卡 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT-SFT) |
| SenseNova-U1-8B-MoT-Interleaved | 支持多页图文交错生成，改善叙事和风格一致性 | 目录索引 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved) |
| [SenseNova-U1-8B-MoT-Infographic-V3](u1_infographic_model_CN.md) | U1 信息图生成和编辑 | 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V3) |
| [SenseNova-U1-8B-MoT-Infographic-V2](u1_infographic_model_CN.md) | 信息图生成权重，改善密集小字和复杂版式 | 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2) |
| [SenseNova-U1-8B-MoT-Infographic](u1_infographic_model_CN.md) | 首个信息图专用权重，用于文字丰富的视觉内容 | 指南 + 案例 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic) |

## 官方 LoRA 适配器

这些 LoRA 文件需要搭配对应基座权重，不能单独运行。

| 适配器 | 对应基座 | 用途 | 权重 |
| --- | --- | --- | --- |
| [SenseNova-U1.5-8B-MoT-LoRA-8step-V2](u1.5_model_CN.md#8-步-v2-lora) | U1.5 | 当前 8 步适配器；相比首版改善色彩平衡与对比度 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors) |
| SenseNova-U1.5-8B-MoT-LoRA-8step | U1.5 | 该权重的首个 8 步适配器 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors) |
| SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0 | SenseNova-U1-8B-MoT-Infographic | 信息图 8 步生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors) |
| SenseNova-U1-8B-MoT-LoRA-8step-V1.0 | U1-8B-MoT | 原版 U1 的 8 步生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors) |

架构细节见 [U1 技术报告](pdf/SenseNOVA_U1.pdf)和[参数说明](parameter_breakdown_CN.md)。

## U1.6 仓库转换产物

仓库中的[量化指南](../tools/quantization/README_CN.md)可以基于匹配的 U1.6
源模型生成以下转换产物。这些文件是面向运行时的转换结果，不属于新增的官方
模型权重。每个格式目录都会包含模型配置、tokenizer 文件、转换报告和
`manifest.json`，用于确认产物身份和转换范围。

| 格式 | 文件 | 运行时 | 放置和依赖说明 |
| --- | --- | --- | --- |
| Native BF16 | `SenseNova-U1.6-BF16.safetensors` | SenseNova-U1 Python 示例 | 标准 BF16 基线。 |
| Native FP8 W8A8 | `SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors` | SenseNova-U1 Python 示例 | 使用原生 FP8 runtime，需要 CUDA 全设备驻留。 |
| Native INT8 ConvRot W8A8 | `SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors` | SenseNova-U1 Python 示例 | 使用原生 INT8 runtime，需要 CUDA 全设备驻留。 |
| GGUF Q4/Q6/Q8 | `SenseNova-U1.6-8B-MoT-Q4_0.gguf`、`Q6_K`、`Q8_0` | SenseNova-U1 GGUF loader | 六个配置和 tokenizer 文件需要与 GGUF 文件放在同一目录。外部 GGUF 引擎需要单独支持 U1.6 模型结构。 |

Native FP8 和 INT8 在参考示例中使用 `--model_format fp8` 或
`--model_format int8-convrot`、`--dtype bfloat16`、`--device cuda` 和
`--vram_mode full`。GGUF 使用 `--model_format gguf` 和相同的 BF16 计算 dtype。
具体产物的哈希和验证范围以对应格式目录的 `manifest.json` 与转换报告为准。
ComfyUI checkpoint 转换遵循官方 ComfyUI 集成的发布流程。

## 社区模型与工具

以下权重和集成由表中作者维护，与上方官方权重分开列出。使用前请阅读对应项目的说明，核对转换权重所基于的 U1、U1.5 Preview 或 U1.5。

### 转换权重

| 维护者 | 资源 | 适用范围 |
| --- | --- | --- |
| [realrebelai](https://huggingface.co/realrebelai) | [U1.5 GGUF 转换权重](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs/tree/main) | 基于 U1.5 权重转换的社区 GGUF 文件。 |
| [smthem](https://huggingface.co/smthem) | [U1.5 Preview Q8 GGUF](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf/blob/main/SenseNova-U1.5-8B-MoT-Preview-Q8.gguf)及 [U1 GGUF 版本](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf/tree/main) | 对应 Preview 和 U1，下载前需核对具体文件名。 |
| [joyfox](https://huggingface.co/joyfox) | [U1.5 裁剪 BF16、FP8 与 INT8 权重](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8) | 供下方社区 ComfyUI 节点使用；裁剪版权重支持图像生成和编辑，不支持 VQA 或图文交错文字输出。 |

### ComfyUI 与其他集成

| 维护者 | 资源 | 适用范围 |
| --- | --- | --- |
| [starsFriday](https://github.com/starsFriday) | [ComfyUI-SenseNova](https://github.com/starsFriday/ComfyUI-SenseNova) | 搭配 joyfox 权重使用的 U1.5 文生图、图像编辑社区节点和工作流。 |
| [leejet](https://github.com/leejet) | [stable-diffusion.cpp 指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sensenova_u1.md) | U1.5 文生图集成。 |
| [ggml-org](https://github.com/ggml-org) | [llama.cpp SenseNova-U1 支持](https://github.com/ggml-org/llama.cpp/pull/28919) | 文本与图像理解集成；具体支持的模型和格式以该项目说明为准。 |

U1.5 和 U1.6 的 [ComfyUI 原生工作流](../apps/comfyui/README.md#native-workflows)需要 ComfyUI 0.46.0 或更高版本。[OpenSenseNova ComfyUI 仓库](https://github.com/OpenSenseNova/ComfyUI-SenseNova-U1)是项目维护的[自定义节点](../apps/comfyui/README.md#nodes)的发布镜像。
