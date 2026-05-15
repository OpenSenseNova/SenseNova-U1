# SenseNova-U1-8B-MoT-Infographic 📊

**SenseNova-U1-8B-MoT-Infographic** 是在 U1-8B-MoT 模型基础上延长了 MT 阶段训练，并在 MT 与 SFT 阶段调整了理解和生成任务中的数据配比产生的模型。在 RL 阶段，我们进一步优化了 reward recipe，以减少生成信息图中非预期黑色背景的出现。

- **模型性能：** 相比基础版 **SenseNova-U1-8B-MoT** 模型，BizGenEval hard/easy 从 **39.8 / 61.1** 提升至 **46.6 / 65.4**（**+6.8 / +4.3 points**），IGenBench Q-ACC/I-ACC 从 **51.3 / 4.2** 提升至 **69.5 / 17.0**（**+18.2 / +12.8 points**），同时保持稳健的视觉理解能力，无明显退化。
- **生成质量：** 模型能够生成涵盖 100+ 种风格与布局的复杂信息图，具备更优的视觉美观度与文字渲染能力 —— 甚至能够渲染如 arXiv 风格页面等高密度小字。

## Benchmark Highlights

| Model | BizGenEval Avg. (hard / easy) ↑ | IGenBench Q-ACC  ↑ | IGenBench I-ACC ↑ | OneIG(EN) ↑ | OneIG(ZH) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Commercial Models*** ||
| Nano-Banana-Pro | 76.7 / 93.7 | 90.6 | 48.8 | 58.1 | 56.8 |
| Nano-Banana-2.0 | 68.5 / 92.5 | 85.6 | 34.4 | 54.0 | 54.9 |
| GPT-Image-1.5 | 35.9 / 81.6 | 55.0 | 12.0 | - | - |
| Qwen-Image-2.0 | 45.5 / 65.8 | 50.0 | 3.0 | 54.1 | 50.9 |
| Seedream-4.5 | 30.1 / 66.2 | 61.0 | 6.0 | 56.4 | 55.0 |
| ***Open-source Models*** ||
| **SenseNova-U1-8B-MoT-Infographic** | **46.6 / 65.4** | **69.5** | **17.0** | **55.6** | **53.3** |
| **SenseNova-U1-8B-MoT** | 39.8 / 61.1 | 51.3 | 4.2 | 54.5 | 53.8 |
| Z-Image | 8.2 / 43.8 | 30.0 | 1.0 | 54.6 | 53.5 |
| Qwen-Image-2512 | 6.3 / 41.0 | 32.2 | 1.0 | 53.0 | 51.5 |
| Qwen-Image | 2.8 / 23.8 | 36.0 | 0.0 | 53.9 | 54.8 |
| Bagel | 2.0 / 3.7 | 4.9 | 0.0 | 36.1 | 37.0 |

<sub>IGenBench 分数以百分制展示。Commercial 与 open-source 组内模型按照 BizGenEval hard、BizGenEval easy、IGenBench Q-ACC、IGenBench I-ACC 四项算术平均值排序。OneIG 作为通用生成能力参考。完整分项结果建议放在 Hugging Face model card 中。</sub>

## 案例展示

> ✨ **想了解模型的实际效果？欢迎前往 👉 [ 🖼️ 信息图案例展示 ](./u1_infographic_showcases.md) 👈 浏览 100 个生成样例！**
