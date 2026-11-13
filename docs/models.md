# SenseNova-U model catalog

<p align="center"><strong>English</strong> | <a href="./models_CN.md">简体中文</a></p>

The [SenseNova-U homepage](../README.md) introduces the model family. This catalog lists the released weights. Model names link to their guides where available; the last column links to the weights. For new work, start with the [U1.5 guide](u1.5_model.md). Its 8-step V2 LoRA is an adapter for that base checkpoint, not a replacement for the base weights.

## Official checkpoints

| Model | Role | Weights |
| --- | --- | --- |
| [SenseNova-U1.5-8B-MoT](u1.5_model.md) | U1.5 base for image generation, editing, interleaved output, and visual understanding | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) |
| SenseNova-U1.5-8B-MoT-TailSteer | U1.5 variant for more stable multi-turn image editing; see its model card for usage | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-TailSteer) |
| SenseNova-U1.5-8B-MoT-SFT | U1.5 checkpoint from the supervised fine-tuning stage | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT) |
| [SenseNova-U1.5-8B-MoT-Preview](u1.5_preview.md) | Preview checkpoint focused on native 4K generation and image editing; separate weights from U1.5 | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview) |
| [SenseNova-U1-8B-MoT](u1_model.md) | U1 8B release for visual understanding and image generation | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-8B-MoT-SFT | U1 8B checkpoint from the supervised fine-tuning stage | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-SFT) |
| [SenseNova-U1-A3B-MoT](u1_model.md) | U1 mixture-of-experts variant for understanding and generation | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) |
| SenseNova-U1-A3B-MoT-SFT | U1 A3B checkpoint from the supervised fine-tuning stage | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT-SFT) |
| SenseNova-U1-8B-MoT-Interleaved | Multi-page text-image generation with narrative and style consistency | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved) |
| [SenseNova-U1-8B-MoT-Infographic-V3](u1_infographic_model.md) | U1 infographic generation and editing | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V3) |
| [SenseNova-U1-8B-MoT-Infographic-V2](u1_infographic_model.md) | Generation-focused infographic checkpoint with denser small text and more complex layouts | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2) |
| [SenseNova-U1-8B-MoT-Infographic](u1_infographic_model.md) | First infographic-specialized checkpoint for text-rich visual content | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic) |

## Official adapters

These LoRA files require their matching base checkpoint; they are not standalone models.

| Adapter | Base checkpoint | Purpose | Weights |
| --- | --- | --- | --- |
| [SenseNova-U1.5-8B-MoT-LoRA-8step-V2](u1.5_model.md#8-step-v2-lora) | U1.5 | Current 8-step adapter; improved color balance and contrast over the first version | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors) |
| SenseNova-U1.5-8B-MoT-LoRA-8step | U1.5 | First 8-step adapter for this checkpoint | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors) |
| SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0 | SenseNova-U1-8B-MoT-Infographic | 8-step infographic generation | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors) |
| SenseNova-U1-8B-MoT-LoRA-8step-V1.0 | U1-8B-MoT | 8-step generation with the original U1 base | [Hugging Face](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors) |

For architecture context, see the [U1 technical report](pdf/SenseNOVA_U1.pdf) and [parameter breakdown](parameter_breakdown.md).

## Community models and tools

The following weights and integrations are published by their named maintainers. They are separate from the official checkpoints above; follow each project's instructions and check that the conversion matches the U1, U1.5 Preview, or U1.5 base you intend to use.

### Converted weights

| Maintainer | Resource | Scope |
| --- | --- | --- |
| [realrebelai](https://huggingface.co/realrebelai) | [U1.5 GGUF conversions](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs/tree/main) | Community GGUF weights derived from the U1.5 checkpoint. |
| [smthem](https://huggingface.co/smthem) | [U1.5 Preview Q8 GGUF](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf/blob/main/SenseNova-U1.5-8B-MoT-Preview-Q8.gguf) and [U1 GGUF variants](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf/tree/main) | Preview and U1 conversions; check the exact filename before downloading. |
| [joyfox](https://huggingface.co/joyfox) | [U1.5 pruned BF16, FP8, and INT8 weights](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8) | Repacked for the community ComfyUI nodes below; these pruned weights support image generation and editing, but not VQA or interleaved text output. |

### ComfyUI and other integrations

| Maintainer | Resource | Scope |
| --- | --- | --- |
| [starsFriday](https://github.com/starsFriday) | [ComfyUI-SenseNova](https://github.com/starsFriday/ComfyUI-SenseNova) | Community nodes and workflows for U1.5 text-to-image and image editing with the joyfox weights. |
| [leejet](https://github.com/leejet) | [stable-diffusion.cpp guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sensenova_u1.md) | U1.5 text-to-image integration. |
| [ggml-org](https://github.com/ggml-org) | [llama.cpp SenseNova-U1 support](https://github.com/ggml-org/llama.cpp/pull/28919) | Text and image understanding integration; check the linked project for supported models and formats. |

The project-maintained [ComfyUI nodes](../apps/comfyui/README.md) have their own instructions and workflows. The [OpenSenseNova ComfyUI repository](https://github.com/OpenSenseNova/ComfyUI-SenseNova-U1) is their publish mirror.
