# SenseNova-U1 Infographic Model Series

<p align="center"><strong>English</strong> | <a href="./u1_infographic_model_CN.md">简体中文</a></p>

This guide covers the U1 infographic V1–V3 checkpoints. V3 handles both infographic generation and editing; V2 focuses on text-to-image. See the [model catalog](models.md) for all official weights and the [showcase](u1_infographic_showcases.md) for outputs.

## Model Overview

| Release | Main use |
| --- | --- |
| [SenseNova-U1-8B-MoT-Infographic-V3](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V3) | Infographic generation and edits to text, content, and overall layout. |
| [SenseNova-U1-8B-MoT-Infographic-V2](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2) | Text-to-image infographics, including dense small text and complex layouts. |
| [SenseNova-U1-8B-MoT-Infographic](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic) | The first infographic checkpoint, mainly for text-to-image. |

V2 and V3 were trained independently from the same U1 base for different tasks. The [release notes and full evaluation tables](evaluations/u1_infographic_release.md) preserve the training details and published comparisons.

## Run Infographic V3

Install the shared Python environment using the [installation guide](installation.md). From the repository root, use the V3 checkpoint for generation and editing; the command below reads prompts from the V2 sample file.

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --jsonl examples/t2i/data/samples_infographic_V2.jsonl \
  --output_dir outputs/infographic_v3 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --profile

python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --prompt "Change the main title to 'SenseNova-U1 V3' while preserving the remaining content and layout." \
  --image docs/assets/showcases/t2i_infographic/0004.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited_infographic.png --profile --compare
```

The [examples](../examples/README.md) explain the remaining CLI options and the V1 8-step LoRA. For V2 or V1 generation, use the same text-to-image script with `--model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V2` or `--model_path sensenova/SenseNova-U1-8B-MoT-Infographic`, respectively. The editing command above is for V3.

## Benchmark Results

In the published evaluations, V3 scored 50.23 on Qwen-Image-Bench Overall Total (V2: 48.00), 5.89 on WeEdit Average, and 7.89 on GEdit-Bench EN_G_O. V2 scored 71.4 / 18.3 on IGenBench Q-ACC / I-ACC, versus V3's 66.3 / 13.2. Compare both for text-to-image work; use V3 when editing is needed. See the [release tables](evaluations/u1_infographic_release.md) for the per-model scores.

## Visual examples

The [V3 infographic showcase](u1_infographic_showcases.md) shows generation and editing examples. The [earlier gallery](u1_infographic_archive.md) retains V2 and original-model examples and version comparisons.
