# SenseNova-U 模型目录

<p align="center"><a href="./models.md">English</a> | <strong>简体中文</strong></p>

[SenseNova-U 项目首页](../README_CN.md)介绍模型家族；本目录集中列出已发布权重。表中的型号名称在有指南时链接到对应指南，末列统一提供权重入口。新任务可从 [U1.5 指南](u1.5_model_CN.md)开始。8 步 V2 LoRA 是该基座的适配器，不能替代基座权重。

## 官方模型权重

| 模型 | 定位 | 权重 |
| --- | --- | --- |
| [SenseNova-U1.5-8B-MoT](u1.5_model_CN.md) | U1.5 基座，支持图像生成、编辑、图文交错输出与视觉理解 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) |
| SenseNova-U1.5-8B-MoT-TailSteer | U1.5 多轮图像编辑版本；用法以模型卡为准 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-TailSteer) |
| SenseNova-U1.5-8B-MoT-SFT | U1.5 监督微调阶段的权重 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT) |
| [SenseNova-U1.5-8B-MoT-Preview](u1.5_preview_CN.md) | 预览版；重点展示原生 4K 生成与图像编辑，权重与 U1.5 不同 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview) |
| [SenseNova-U1-8B-MoT](u1_model_CN.md) | U1 8B 模型，支持视觉理解与图像生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-8B-MoT-SFT | U1 8B 监督微调阶段的权重 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-SFT) |
| [SenseNova-U1-A3B-MoT](u1_model_CN.md) | U1 混合专家版本，支持视觉理解与图像生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) |
| SenseNova-U1-A3B-MoT-SFT | U1 A3B 监督微调阶段的权重 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT-SFT) |
| SenseNova-U1-8B-MoT-Interleaved | 面向多页内容的图文交错生成，强调叙事与风格一致性 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved) |
| [SenseNova-U1-8B-MoT-Infographic-V3](u1_infographic_model_CN.md) | U1 信息图生成与编辑 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V3) |
| [SenseNova-U1-8B-MoT-Infographic-V2](u1_infographic_model_CN.md) | 以信息图生成为主，改进密集小字与复杂版式 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2) |
| [SenseNova-U1-8B-MoT-Infographic](u1_infographic_model_CN.md) | 首个信息图专项版本，面向文字密集的视觉内容 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic) |

## 官方 LoRA 适配器

这些 LoRA 文件需要搭配对应基座权重，不能单独运行。

| 适配器 | 对应基座 | 用途 | 权重 |
| --- | --- | --- | --- |
| [SenseNova-U1.5-8B-MoT-LoRA-8step-V2](u1.5_model_CN.md#8-步-v2-lora) | U1.5 | 当前 8 步适配器；相比首版改善色彩平衡与对比度 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors) |
| SenseNova-U1.5-8B-MoT-LoRA-8step | U1.5 | 该权重的首个 8 步适配器 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors) |
| SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0 | SenseNova-U1-8B-MoT-Infographic | 信息图 8 步生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors) |
| SenseNova-U1-8B-MoT-LoRA-8step-V1.0 | U1-8B-MoT | 原版 U1 的 8 步生成 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors) |

架构细节见 [U1 技术报告](pdf/SenseNOVA_U1.pdf)和[参数说明](parameter_breakdown_CN.md)。

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

项目维护的 [ComfyUI 节点](../apps/comfyui/README.md)另有使用说明和工作流；[OpenSenseNova ComfyUI 仓库](https://github.com/OpenSenseNova/ComfyUI-SenseNova-U1)是其发布镜像。
