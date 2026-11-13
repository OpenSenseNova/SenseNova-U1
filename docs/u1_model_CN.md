# SenseNova-U1

<p align="center"><a href="./u1_model.md">English</a> | <strong>简体中文</strong></p>

U1 是 SenseNova-U 最初发布的视觉理解与图像生成模型。[U1 论文](https://arxiv.org/abs/2605.12500)介绍 NEO-unify 架构以及 8B、A3B 两种模型规格。后续的 [U1.5](u1.5_model_CN.md)使用独立权重，另有对应指南。

## 模型权重

| 权重 | 定位 |
| --- | --- |
| [U1 8B](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) | 通用 U1 权重，下方命令和任务示例使用该模型。 |
| [U1 A3B](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) | 面向理解与生成的混合专家版本。 |
| [U1 图文交错](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved) | 面向多页图文生成的专项权重。 |

[模型目录](models_CN.md)还列出 SFT 权重与 U1 LoRA 适配器；使用适配器时须核对对应的基座权重。

## 运行 U1

先按[安装指南](installation_CN.md)配置共用的 Python 环境，再从仓库根目录运行 U1 8B 参考脚本：

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A cinematic mountain village at sunrise" \
  --vram_mode balanced \
  --output output.png
```

[任务示例](../examples/README_CN.md)还介绍图像编辑、视觉理解和图文交错生成，以及设备放置与命令参数。专项任务请使用相应权重。

## 效果案例

[U1 效果展示](showcases_CN.md)收录生成、编辑和图文交错案例，图集也链接了提示词与样本输入。[信息图模型系列](u1_infographic_model_CN.md)使用独立权重，另有对应案例。需要补充输入时，可选用[提示词增强](prompt_enhancement_CN.md)。

## 评测

[U1 技术报告](pdf/SenseNOVA_U1.pdf)说明评测细节。已发布图表覆盖[图像生成](assets/benchmarks/generation.webp)、[视觉理解](assets/benchmarks/understanding.webp)与[图文交错生成](assets/benchmarks/interleaved.webp)。[参数说明](parameter_breakdown_CN.md)解释模型理解和生成路径。
