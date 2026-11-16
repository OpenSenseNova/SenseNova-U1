# SenseNova-U1 信息图模型系列

<p align="center"><a href="./u1_infographic_model.md">English</a> | <strong>简体中文</strong></p>

这里介绍 U1 信息图 V1–V3 的权重和用法。V3 同时支持信息图生成与编辑；V2 主要面向文生图。完整官方权重见[模型目录](models_CN.md)，生成与编辑案例见[效果展示](u1_infographic_showcases_CN.md)。

## 模型概览

| 版本 | 适用任务 |
| --- | --- |
| [SenseNova-U1-8B-MoT-Infographic-V3](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V3) | 信息图生成，以及局部文字、内容和整体版式编辑。 |
| [SenseNova-U1-8B-MoT-Infographic-V2](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2) | 侧重信息图文生图，尤其是密集小字和复杂排版。 |
| [SenseNova-U1-8B-MoT-Infographic](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic) | 首个信息图专项权重，主要用于文生图。 |

V2 和 V3 从同一 U1 基座独立训练，目标不同。[版本说明与完整评测](evaluations/u1_infographic_release_CN.md)保留了各版的训练说明和发布对比表。

## 运行信息图 V3

先按[安装指南](installation_CN.md)配置共用的 Python 环境。从仓库根目录使用 V3 权重进行生成和编辑；下方命令使用 V2 样本文件中的提示词。

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --jsonl examples/t2i/data/samples_infographic_V2.jsonl \
  --output_dir outputs/infographic_v3 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --profile

python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --prompt "将信息图的主标题修改为‘SenseNova-U1 V3’，保持其他内容和布局不变。" \
  --image docs/assets/showcases/t2i_infographic/0004.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited_infographic.png --profile --compare
```

其他命令参数和 V1 的 8 步 LoRA 用法见[任务示例](../examples/README_CN.md)。运行 V2 或 V1 文生图时，可在同一个文生图脚本中分别指定 `--model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V2` 或 `--model_path sensenova/SenseNova-U1-8B-MoT-Infographic`。上面的编辑命令面向 V3。

## 评测结果

发布评测中，V3 的 Qwen-Image-Bench Overall Total 为 50.23（V2 为 48.00），WeEdit Average 为 5.89，GEdit-Bench EN_G_O 为 7.89。V2 的 IGenBench Q-ACC / I-ACC 为 71.4 / 18.3，高于 V3 的 66.3 / 13.2。文生图任务可比较两版；需要编辑时使用 V3。逐项结果见[版本评测资料](evaluations/u1_infographic_release_CN.md)。

## 效果展示

[V3 信息图案例](u1_infographic_showcases_CN.md)展示生成与编辑效果；V2、首版模型及版本对比见[早期图集](u1_infographic_archive_CN.md)，另有[首版模型的扩展图集](u1_infographic_archive.md#more-examples-from-the-first-infographic-model)（英文）。
