# SenseNova-U1 信息图 V1–V3

<p align="center"><a href="./u1_infographic_release.md">English</a> | <strong>简体中文</strong></p>

以下是 V1–V3 发布时的训练说明和对比表。权重和运行命令见[信息图模型指南](../u1_infographic_model_CN.md)。

## 各版本的训练说明

### SenseNova-U1-8B-MoT-Infographic-V3

**SenseNova-U1-8B-MoT-Infographic-V2** 与 **SenseNova-U1-8B-MoT-Infographic-V3** 均从同一基础模型出发独立训练，并面向不同目标进行重点优化。V2 主要聚焦信息图文生图；V3 则面向信息图生成与编辑一体化场景，保留文生图（T2I）能力不下降的同时，并重点增强图像编辑（IT2I）能力。

为兼顾生成与编辑能力，V3 从 MT 阶段重新启动训练，构建多样的信息图编辑任务数据，并按合理比例联合训练 T2I 与图像编辑任务，随后进入 SFT 和 RL 阶段。

- **模型性能：** Qwen-Image-Bench Overall Total 为 50.23，相比 V2 的 48.00 提升 2.23；WeEdit Average 为 5.89，GEdit-Bench 综合得分为 7.89，均在本次评测的开源模型中排名第一。
- **生成质量：** V3 在保留信息图 T2I 能力的同时新增 IT2I 能力，既可通过标记目标区域定点修改，也可仅使用自然语言指令进行编辑；支持局部文字修改、局部内容插入/删除/替换、全局风格编辑和布局编辑，同时尽可能保持未编辑区域不变。

### SenseNova-U1-8B-MoT-Infographic-V2

V2 是上一版本的升级版本。在 MT/SFT 阶段，我们引入更多高质量合成与真实数据，重新平衡信息图样本内部及其与其他数据之间的分布，重点优化小字渲染、复杂密集排版和高美观度生成。在 RL 阶段，我们引入 DPO 以提升整体视觉质量；在 GRPO 阶段继续优化 reward recipe，减少非预期黑底或过暗背景。

- **模型性能：** BizGenEval hard/easy 从 46.6 / 65.4 提升到 50.3 / 67.9（+3.7 / +2.5），IGenBench Q-ACC/I-ACC 从 69.5 / 17.0 提升到 71.4 / 18.3（+1.9 / +1.3）；OneIG(EN/ZH) 为 55.4 / 53.5，与上一版基本持平。
- **生成质量：** 小字边缘更加锐利，高信息密度版面组织更稳定，海报、看板和报告类信息图更加精致，并修复了背景变黑问题。

### SenseNova-U1-8B-MoT-Infographic

首个信息图专项版本在 **SenseNova-U1-8B-MoT** 基础上延长 MT 训练，并在 MT/SFT 阶段调整理解与生成任务的数据配比；在 RL 阶段进一步优化 reward recipe，减少生成信息图中的非预期黑色背景。

- **模型性能：** BizGenEval hard/easy 从 39.8 / 61.1 提升到 46.6 / 65.4（+6.8 / +4.3），IGenBench Q-ACC/I-ACC 从 51.3 / 4.2 提升到 69.5 / 17.0（+18.2 / +12.8），同时保持稳健的视觉理解能力。
- **生成质量：** 支持 100+ 种风格与布局的复杂信息图，提升视觉美观度和文字渲染能力，可处理 arXiv 风格页面等高密度小字场景。

## 评测结果

### WeEdit（图像编辑）

| 模型 | Instruction Adherence&nbsp;↑ | Text Clarity&nbsp;↑ | Background Preservation&nbsp;↑ | Average&nbsp;↑ |
| :--- | ---: | ---: | ---: | ---: |
| ***闭源模型*** | | | | |
| Nano-Banana-Pro | **8.58** | **9.10** | **8.85** | **8.84** |
| Seedream 4.5 | 6.29 | 7.66 | 6.38 | 6.78 |
| Gemini-2.5-Flash-Image | 3.92 | 7.14 | 7.80 | 6.29 |
| Qwen-Image-2.0 | 5.04 | 6.08 | 5.68 | 5.60 |
| ***开源模型*** | | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | **5.67** | 5.94 | 6.06 | **5.89** |
| FireRed-Image-Edit | 4.15 | **6.33** | **7.14** | 5.87 |
| HY-Image-3-Instruct | 4.16 | 5.99 | 7.03 | 5.73 |
| Qwen-Image-Edit-2509 | 3.49 | 5.84 | 6.80 | 5.38 |
| LongCat-Image-Edit | 2.78 | 4.36 | 7.03 | 4.72 |
| Qwen-Image-Edit-2511 | 3.18 | 3.93 | 4.63 | 3.91 |
| BAGEL | 1.97 | 4.01 | 5.75 | 3.91 |

<sub>注：越高越好；粗体为组内最高分，Average 为三项指标的算术平均值。V3 的 Average 在本次评测的开源模型中排名第一。</sub>

### GEdit-Bench（图像编辑）

| 模型 | EN_G_SC&nbsp;↑ | EN_G_PQ&nbsp;↑ | EN_G_O&nbsp;↑ |
| :--- | ---: | ---: | ---: |
| ***闭源模型*** | | | |
| Qwen-Image-2.0 | **9.02** | 8.02 | **8.37** |
| UniWorld-V2 | 8.39 | 8.02 | 7.83 |
| Seedream 4.5 | 8.27 | 8.17 | 7.82 |
| Nano-Banana-Pro | 8.10 | 8.34 | 7.74 |
| Seedream 4.0 | 8.14 | 8.12 | 7.70 |
| Nano-Banana | 7.40 | **8.45** | 7.29 |
| ***开源模型*** | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | **8.65** | 7.58 | **7.89** |
| LongCat-Image-Edit | 8.18 | **8.00** | 7.64 |
| Qwen-Image-Edit-2511 | 8.00 | 7.86 | 7.56 |
| Qwen-Image-Edit-2509 | 8.15 | 7.86 | 7.54 |
| Step1X-Edit | 7.66 | 7.35 | 6.97 |
| BAGEL | 7.36 | 6.83 | 6.52 |
| OmniGen2 | 7.16 | 6.77 | 6.41 |
| UniWorld-v1 | 4.93 | 7.43 | 4.85 |
| AnyEdit | 3.18 | 5.82 | 3.21 |

<sub>注：越高越好；粗体为组内最高分。V3 在开源模型中综合表现最优。</sub>

### Qwen-Image-Bench（文生图）

| 模型 | Quality&nbsp;↑ | Aesthetics&nbsp;↑ | Alignment&nbsp;↑ | Real-world Fidelity&nbsp;↑ | Creative Generation&nbsp;↑ | Overall Total&nbsp;↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-Image-2 | **59.09** | **68.48** | **65.78** | **59.40** | **75.34** | **64.58** |
| Nano-Banana-Pro | 55.30 | 61.38 | 60.30 | 55.91 | 64.54 | 57.84 |
| Qwen-Image-2.0 | 55.16 | 60.36 | 57.86 | 53.06 | 63.59 | 57.84 |
| HunyuanImage-3.0 | 50.76 | 54.66 | 53.16 | 45.33 | 48.33 | 50.81 |
| **SenseNova-U1-8B-MoT-Infographic-V3** | 49.53 | 52.82 | 52.00 | 44.32 | 48.34 | 50.23 |
| SenseNova-U1-8B-MoT-Infographic-V2 | 48.15 | 49.60 | 50.08 | 43.68 | 44.65 | 48.00 |
| SenseNova-U1-8B-MoT-Infographic | 47.12 | 48.15 | 48.90 | 43.35 | 45.40 | 47.11 |
| HiDream-O1 | 44.20 | 45.35 | 43.74 | 40.28 | 36.24 | 42.84 |

<sub>注：越高越好；各列最高分加粗，V3 模型名加粗。V3 的 Overall Total 为 50.23，相比 V2 的 48.00 提升 2.23。</sub>

### 历史信息图生成评测

| 模型 | BizGenEval Avg. (hard / easy)&nbsp;↑ | IGenBench Q-ACC&nbsp;↑ | IGenBench I-ACC&nbsp;↑ | OneIG(EN)&nbsp;↑ | OneIG(ZH)&nbsp;↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Commercial Models*** | | | | | |
| Nano-Banana-Pro | 76.7 / 93.7 | 90.6 | 48.8 | 58.1 | 56.8 |
| Nano-Banana-2.0 | 68.5 / 92.5 | 85.6 | 34.4 | 54.0 | 54.9 |
| GPT-Image-1.5 | 35.9 / 81.6 | 55.0 | 12.0 | - | - |
| Qwen-Image-2.0 | 45.5 / 65.8 | 50.0 | 3.0 | 54.1 | 50.9 |
| Seedream-4.5 | 30.1 / 66.2 | 61.0 | 6.0 | 56.4 | 55.0 |
| ***Open-source Models*** | | | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | 49.5 / **68.5** | 66.3 | 13.2 | 54.6 | 52.4 |
| **SenseNova-U1-8B-MoT-Infographic-V2** | **50.3** / 67.9 | **71.4** | **18.3** | 55.4 | 53.5 |
| **SenseNova-U1-8B-MoT-Infographic** | 46.6 / 65.4 | 69.5 | 17.0 | **55.6** | 53.3 |
| **SenseNova-U1-8B-MoT** | 39.8 / 61.1 | 51.3 | 4.2 | 54.5 | 53.8 |
| Z-Image | 8.2 / 43.8 | 30.0 | 1.0 | 54.6 | 53.5 |
| Qwen-Image-2512 | 6.3 / 41.0 | 32.2 | 1.0 | 53.0 | 51.5 |
| Qwen-Image | 2.8 / 23.8 | 36.0 | 0.0 | 53.9 | 54.8 |
| Bagel | 2.0 / 3.7 | 4.9 | 0.0 | 36.1 | 37.0 |

<sub>注：V3 为当前发布版本，固定列于开源模型首行。IGenBench 指标按百分制展示；其余商业模型与开源模型分别按 BizGenEval hard/easy、IGenBench Q-ACC/I-ACC 四项平均分排序。OneIG 仅作为通用生成能力参考。</sub>

V2 与 V3 从同一基础模型出发独立训练，并采用不同的任务与数据配比。V3 联合训练信息图文生图和图像编辑任务，图表相关训练数据占比略低于 V2，因此其 IGenBench 分数低于以信息图生成为主要目标的 V2。与此同时，V3 在 BizGenEval 上与 V2 基本持平，在覆盖更广泛文生图能力的 Qwen-Image-Bench 上由 48.00 提升至 50.23，并在信息图编辑能力上有明显提升。
