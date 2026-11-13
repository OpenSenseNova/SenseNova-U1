# SenseNova-U1

<p align="center"><strong>English</strong> | <a href="./u1_model_CN.md">简体中文</a></p>

U1 is the original SenseNova-U release for visual understanding and image generation. The [U1 paper](https://arxiv.org/abs/2605.12500) describes its NEO-unify architecture and the 8B and A3B model variants. The later [U1.5 checkpoint](u1.5_model.md) has its own weights and guide.

## Checkpoints

| Checkpoint | Role |
| --- | --- |
| [U1 8B](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) | General U1 checkpoint used in the command below and the reference examples. |
| [U1 A3B](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) | Mixture-of-experts variant for understanding and generation. |
| [U1 Interleaved](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved) | Specialized checkpoint for multi-page text-image generation. |

The [model catalog](models.md) also lists SFT checkpoints and U1 LoRA adapters. Match each adapter with its stated base checkpoint.

## Run U1

Set up the shared Python environment with the [installation guide](installation.md), then run the U1 8B reference script from the repository root:

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A cinematic mountain village at sunrise" \
  --vram_mode balanced \
  --output output.png
```

The [examples](../examples/README.md) cover image editing, visual understanding, and interleaved generation, along with device placement and CLI options. Use the matching checkpoint for any specialized task.

## Examples

The [U1 showcase](showcases.md) collects generated images, edits, and interleaved responses. Its prompts and sample inputs are linked from the gallery; the [infographic model series](u1_infographic_model.md) has separate weights and examples. [Prompt enhancement](prompt_enhancement.md) is available as an optional preparation step.

## Evaluations

The [U1 technical report](pdf/SenseNOVA_U1.pdf) provides the evaluation details. The published figures cover [image generation](assets/benchmarks/generation.webp), [visual understanding](assets/benchmarks/understanding.webp), and [interleaved generation](assets/benchmarks/interleaved.webp). The [parameter breakdown](parameter_breakdown.md) explains the model's understanding and generation pathways.
