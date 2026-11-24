# SenseNova-U 系列：从第一性原理出发基于 NEO-unify的原生统一范式

<p align="center"><a href="./README.md">English</a> | <strong>简体中文</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.11929"><img src="https://img.shields.io/badge/arXiv-2609.11929-b31b1b.svg" alt="SenseNova-U1.5 论文"></a>
  <a href="https://arxiv.org/abs/2605.12500"><img src="https://img.shields.io/badge/arXiv-2605.12500-b31b1b.svg" alt="SenseNova-U1 论文"></a>
  <a href="https://huggingface.co/collections/sensenova/sensenova-u15"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-U1.5-yellow" alt="Hugging Face 上的 SenseNova-U1.5"></a>
  <a href="https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT"><img src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-%E6%A8%A1%E5%9E%8B-purple" alt="ModelScope 上的 SenseNova-U1.5"></a>
  <a href="https://huggingface.co/collections/sensenova/sensenova-u1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-U1-yellow" alt="Hugging Face 上的 SenseNova-U1"></a>
  <a href="https://huggingface.co/blog/sensenova/neo-unify"><img src="https://img.shields.io/badge/Architecture-NEO--unify-2459B8" alt="NEO-unify"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 许可证"></a>
  <a href="https://unify.light-ai.top/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20SenseNova_U-Demo-Green" alt="SenseNova-U 在线体验"></a>
  <a href="https://discord.com/invite/BuTXPHmQub"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center"><img src="docs/assets/teaserU1.5.png" alt="SenseNova-U1.5 原生统一多模态架构" width="100%"></p>

## 模型概览

SenseNova-U 是基于 [NEO-unify](https://huggingface.co/blog/sensenova/neo-unify) 的原生多模态模型家族。同一个模型可以理解图像、生成或编辑图像，也可以在回答中交替输出文字和图片。[U1](https://arxiv.org/abs/2605.12500) 首次实现这一方向；[U1.5](https://arxiv.org/abs/2609.11929) 进一步采用 8B-MoT 模型、无需独立视觉编码器和 VAE 的视觉路径，以及具有空间连续性的图像重建方法。**U1.6** 在 U1.5 的基础上，专注提升美学、创意表现与图文交错的原生自修正能力。

**U1.6** 的新增能力如下：

- **美学表现提升**：重点提升了人像、海报与插画创作中呈现细腻质感、多元创意和潮流灵感。在用户主观评测中，综合美感较 U1.5 提升 10%。
- **原生自修正**：U1.6 生成信息图后，会对照提示词自主检查并修正文字错误，同时保留富有表现力的字体设计和版式。
- **更高的文字准确性**：在 UltraTextBench 的高文字密度信息图任务上，自修正较单次生成平均相对提升 7%。

**U1.5** 重点改善了四类实际任务：

- **图像生成**：支持最高原生 4K 分辨率，改进构图、局部细节和材质表现。
- **文字与版式**：改善海报、信息图等结构化图像中的中英文文字。
- **图像编辑**：支持局部修改和多图参考，更好地保留主体身份与未修改区域。
- **视觉理解与图文交错**：支持图像问答，以及文字和图片交替出现的内容生成。

本仓库提供 U1、U1.5 和 U1.6 的推理示例、训练代码及文档。[U1.6 指南](docs/u1.6_model_CN.md)介绍当前权重、工作流和本地运行方法；[U1.6 效果展示](docs/u1.6_showcases_CN.md)收录文生图、图像编辑和自修正案例。

## 近期更新

- **2026-11-24**：发布 [SenseNova-U1.6](docs/u1.6_model_CN.md)。SenseNova-U1.6 是兼具美学创造力与原生自修正能力的统一多模态模型，让图像创作兼顾美感与准确性。
- **2026-11-04**：发布 [U1.5 TailSteer](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-TailSteer) 完整权重，改善连续多轮图像编辑的稳定性。使用现有编辑脚本，将模型路径指向该权重即可。
- **2026-09-24**：发布 [U1.5 8 步 V2 LoRA](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors)。相比第一版 8 步适配器，V2 改善了色彩平衡与对比度，并减轻过锐化；使用时需搭配 U1.5 基座。
- **2026-09-11**：发布 [U1.5 技术报告](docs/pdf/SenseNOVA_U1_5.pdf)，介绍模型架构、训练方法和评测结果。
- **2026-08-20**：发布 [U1.5](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)及[首版 8 步 LoRA](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors)。相比 Preview，U1.5 在指令遵循、文字与版式、原生 4K 生成和图像编辑上进一步改进。
[三个月以前的更新](docs/news_CN.md)

## U1.6 权重

| 权重 | 内容 |
| --- | --- |
| [SenseNova-U1.6-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.6-8B-MoT/tree/main) | 原版 BF16 分片权重，也是快速开始使用的默认模型 |
| [SenseNova-U1.6-8B-MoT-LoRAs](https://huggingface.co/sensenova/SenseNova-U1.6-8B-MoT-LoRAs/tree/main) | 搭配 U1.6 基座的 4 步和 8 步 LoRA，提供 Python 和 ComfyUI 文件 |
| [SenseNova-U1.6-8B-MoT-Native](https://huggingface.co/sensenova/SenseNova-U1.6-8B-MoT-Native/tree/main) | 合并版 BF16、FP8 W8A8 和 INT8 ConvRot W8A8 转换权重 |
| [SenseNova-U1.6-8B-MoT-GGUF](https://huggingface.co/sensenova/SenseNova-U1.6-8B-MoT-GGUF/tree/main) | Q4_0、Q6_K 和 Q8_0 转换权重 |

## U1.5 权重

其他发布版本和社区集成见[模型目录](docs/models_CN.md)。

| 权重 | 类型 | 适用任务 | 使用说明 |
| --- | --- | --- | --- |
| [SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) | 完整权重 | 文生图、图像编辑、图文交错、视觉问答 | [指南](docs/u1.5_model_CN.md) · [案例](docs/u1.5_showcases_CN.md) |
| [SenseNova-U1.5-8B-MoT-LoRA-8step-V2](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/main/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors) | LoRA 适配器；需搭配 U1.5 | 8 步文生图、图像编辑、图文交错 | [LoRA 用法](docs/u1.5_model_CN.md#8-步-v2-lora) |

## 快速开始

建议用尽量空闲的 Linux 单卡 CUDA 机器，按 **24 GB 显存、64 GB 系统内存**规划，另为 U1.6 权重预留约 **35 GB 磁盘空间**。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)；下面的命令使用 Python 3.11 和 `balanced` 显存模式。

~~~bash
git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
uv sync --locked --python 3.11
source .venv/bin/activate

hf download sensenova/SenseNova-U1.6-8B-MoT \
  --local-dir ./models/SenseNova-U1.6-8B-MoT

python examples/t2i/inference.py \
  --model_path ./models/SenseNova-U1.6-8B-MoT \
  --prompt "A cinematic mountain lake at sunrise, realistic photography." \
  --width 2048 --height 2048 --num_steps 50 \
  --vram_mode balanced --attn_backend sdpa \
  --output output.png
~~~

`balanced` 会占用系统内存。RTX 5090 测试中的进程 CPU RSS 峰值约 52.2 GiB，因此上面按 64 GB 系统内存规划。

环境配置见[安装指南](docs/installation_CN.md)，其他任务的命令见[任务示例](examples/README_CN.md)。8 步 V2 LoRA 的命令见[U1.5 模型指南](docs/u1.5_model_CN.md#8-步-v2-lora)；[U1.5 显存与 8 步耗时](docs/u1.5_inference_CN.md)记录了 RTX 5090 和 H100 分配器限额测试。

## 评测 (U1.6)

<p align="center"><a href="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/0957bb9307c6c788412ba9395c43d19db92dc8c4/u1.6/benchmark/U1.6-Benchmark-zh.png"><img src="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/0957bb9307c6c788412ba9395c43d19db92dc8c4/u1.6/benchmark/U1.6-Benchmark-zh.png" alt="U1.6 评测结果" width="90%"></a></p>

## 评测（U1.5）

这些图表是 U1.5 权重的发布评测记录，与 [U1.5 推理记录](docs/u1.5_inference_CN.md)中的 RTX 5090 和 H100 运行数据分开；[详细结果](docs/assets/benchmarks/u1.5_combined.webp)列出发布评测的范围。

<p align="center"><a href="docs/assets/benchmarks/u1.5_radial.webp"><img src="docs/assets/benchmarks/u1.5_radial.webp" alt="U1.5 在图像生成与编辑任务上的评测概览" width="90%"></a></p>

<p align="center"><img src="docs/assets/u15_perform_vs_speed_6bench.webp" alt="U1.5 与其他模型的六项图像生成评测平均分及 2K 图像生成耗时" width="90%"></p>

## 图像样例（U1.6）

### 文生图

<p align="center"><a href="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/fea160e6c6bc5c8819ca0e37a26c3dfd937dfdc6/u1.6/t2i/SenseNova-U1.6-T2I.webp"><img src="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/fea160e6c6bc5c8819ca0e37a26c3dfd937dfdc6/u1.6/t2i/SenseNova-U1.6-T2I.webp" alt="SenseNova-U1.6 文生图样例" width="100%"></a></p>

### 自修正

<p align="center"><a href="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/fea160e6c6bc5c8819ca0e37a26c3dfd937dfdc6/u1.6/self-correction/SenseNova-U1.6-SC.gif"><img src="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/891fa45eead98e014227138e2de4d3148bc5565c/u1.6/self-correction/SenseNova-U1.6-SC-preview.webp" alt="SenseNova-U1.6 自修正前后对比" width="100%"></a></p>

<p align="center"><sub>图中结果使用 U1.6 基座权重。<a href="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/fea160e6c6bc5c8819ca0e37a26c3dfd937dfdc6/u1.6/t2i/SenseNova-U1.6-T2I-HD.png">原尺寸拼图（约 47.5 MiB）</a>和<a href="https://huggingface.co/datasets/sensenova/SenseNova-U-showcases/resolve/fea160e6c6bc5c8819ca0e37a26c3dfd937dfdc6/u1.6/self-correction/SenseNova-U1.6-SC-HD.gif">完整高清自修正 GIF（约 36 MiB）</a>另行提供。更多结果、提示词、参考图和模型详情请见 <a href="docs/u1.6_showcases_CN.md">U1.6 效果展示</a>。</sub></p>

**已知问题：** U1.6 生成人像时，偶尔会出现肢体比例失调或姿势不自然的问题。

## 文档

完整的权重列表和社区版本见[模型目录](docs/models_CN.md)。

### 模型指南与案例

| 版本 | 适用范围 | 指南 | 案例 |
| --- | --- | --- | --- |
| U1.6 | 文生图、图像编辑和原生自修正 | [指南](docs/u1.6_model_CN.md) | [案例](docs/u1.6_showcases_CN.md) |
| U1.5 | 图像生成、编辑、图文交错和视觉理解 | [指南](docs/u1.5_model_CN.md) | [案例](docs/u1.5_showcases_CN.md) |
| U1.5 Preview | 原生 4K 生成和图像编辑 | [指南](docs/u1.5_preview_CN.md) | [案例](docs/u1.5_preview_showcases_CN.md) |
| U1 信息图 V1–V3 | 信息图生成和编辑 | [指南](docs/u1_infographic_model_CN.md) | [案例](docs/u1_infographic_showcases_CN.md) |
| U1 | 初代 8B、A3B 和图文交错权重 | [指南](docs/u1_model_CN.md) | [案例](docs/showcases_CN.md) |

### 运行与扩展

本地推理先看[安装指南](docs/installation_CN.md)和[任务示例](examples/README_CN.md)。[U1.5 显存与 8 步耗时](docs/u1.5_inference_CN.md)记录 RTX 5090 和 H100 分配器限额测试；8 步 V2 LoRA 命令见[模型指南](docs/u1.5_model_CN.md#8-步-v2-lora)。常见问题见 [FAQ](docs/FAQ_CN.md)。

- **U1 服务部署：**[LightLLM + LightX2V 部署步骤](docs/deployment_CN.md)。使用独立的 Docker 环境。
- **训练：**[训练代码与环境](training/README.md)。
- **提示词增强：**可选的 [U1.5](docs/u1.5_best_practices_CN.md)和 [U1](docs/prompt_enhancement_CN.md) 流程。

## 参与贡献

提交记录见 [GitHub 贡献者页面](https://github.com/OpenSenseNova/SenseNova-U1/graphs/contributors)。发现问题可提 [Issue](https://github.com/OpenSenseNova/SenseNova-U1/issues)；代码、案例和文档改动可提 [PR](https://github.com/OpenSenseNova/SenseNova-U1/pulls)。新增官方权重时，请参考[模型文档流程](docs/contributing/model_release_CN.md)。

## 社区交流

欢迎在 [Discord](https://discord.com/invite/BuTXPHmQub) 交流，或通过 [SenseNova 在线体验](https://unify.light-ai.top/)试用模型。加入飞书交流群，请打开[原尺寸二维码](docs/assets/feishu.png)，再用飞书扫码。

<p align="center"><a href="docs/assets/feishu.png"><img src="docs/assets/feishu.png" alt="SenseNova-U 飞书交流群二维码；点击查看原尺寸图片" width="200"></a></p>

## 论文、引用与许可证

[U1.5 论文](https://arxiv.org/abs/2609.11929)、[U1.5 技术报告](docs/pdf/SenseNOVA_U1_5.pdf)和 [U1 论文](https://arxiv.org/abs/2605.12500)提供技术细节。

~~~bibtex
@misc{sensenova2026neounify,
  title   = {NEO-unify: Building Native Multimodal Unified Models End to End},
  author  = {SenseNova},
  url     = {https://huggingface.co/blog/sensenova/neo-unify},
  year    = {2026}
}

@article{sensenova2026sensenovau1,
  title   = {SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture},
  author  = {Diao, Haiwen and Wu, Penghao and Deng, Hanming and Wang, Jiahao and Bai, Shihao and Wu, Silei and Fan, Weichen and Ye, Wenjie and Tong, Wenwen and Fan, Xiangyu and others},
  journal = {arXiv preprint arXiv:2605.12500},
  year    = {2026}
}

@article{sensenova2026sensenovau1.5,
  title   = {SenseNova-U1.5: Towards Native Unified Visual Intelligence},
  author  = {Diao, Haiwen and Wang, Jiahao and Ding, Chenjing and Deng, Hanming and Chen, Jiangnan and Zhang, Ruixi and Wang, Ruohui and Tong, Wenwen and Fan, Xiangyu and Wang, Yubo and others},
  journal = {arXiv preprint arXiv:2609.11929},
  year    = {2026}
}
~~~

仓库代码使用 [Apache 2.0 许可证](LICENSE)。模型权重的使用条件请以对应模型卡的许可证为准。
