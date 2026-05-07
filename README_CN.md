# SenseNova-U1：基于 NEO-Unify 架构统一多模态理解与生成

<p align="center">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Coming-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/sensenova/sensenova-u1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://modelscope.cn/collections/SenseNova/SenseNova-U1"><img src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-模型-purple" alt="ModelScope-模型"></a>
  <a href="https://unify.light-ai.top/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20SenseNova_U1-Demo-Green" alt="SenseNova-U1 Demo"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://discord.gg/cxkwXWjp"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <img src="docs/assets/teaser.webp" alt="SenseNova-U1" width="900">
</p>

<p align="center">
  <img src="docs/assets/teaser_2.webp" alt="visualization" width="900">
</p>

## 🌟 概述

🚀 **SenseNova U1** 是全新一代原生多模态模型系列，在单一架构中统一了多模态理解、推理与生成。
它代表着多模态 AI 的根本性范式转变：**从模态集成走向真正的统一**。SenseNova U1 不再依赖适配器在不同模态之间进行翻译，而是以原生方式跨语言与视觉进行思考与行动。

视觉理解与生成的统一开启了巨大的可能性。SenseNova U1 立足于**数据驱动学习阶段**（如 ChatGPT），并指向下一阶段——**智能体学习阶段**（如 OpenClaw），以原生多模态的方式进行学习、思考和行动。

<p align="center">
  <img src="docs/assets/teaser_1.webp" alt="radar plot" width="900">
</p>

#### 🏗️ *核心支柱：*

SenseNova U1 的核心是 **[NEO-Unify](https://huggingface.co/blog/sensenova/neo-unify)** —— 一个为多模态 AI 而设计、从第一性原理出发的全新架构：*它彻底摒弃了视觉编码器（VE）与变分自编码器（VAE），因为像素与文字信息在本质上是深度相关的。* 其主要特性如下：

- 🔗 端到端地将语言与视觉信息建模为统一整体。
- 🖼️ 在保留语义丰富度的同时，维持像素级的视觉保真度。
- 🧠 通过原生 MoT 实现跨模态推理，效率高、冲突少。

#### ✨ *能力突破：*

基于这一全新的核心架构，SenseNova U1 在多模态学习中展现出卓越的效率：

<p align="center">
  <img src="docs/assets/perform_vs_speed_5bench.webp" width="48%" />
  <img src="docs/assets/perform_vs_speed_infobench.webp" width="48%" />
</p>

<p align="center">
  <sub>
    左图：在 OneIG（EN、ZH）、LongText（EN、ZH）、CVTG、BizGenEval（Easy、Hard）与 IGenBench 上的生成延迟与平均性能对比。<br>
    右图：在信息图基准（BizGenEval（Easy、Hard）、IGenBench）上的生成延迟与平均性能对比。
  </sub>
</p>

- 🏆 **理解与生成均达到开源 SoTA**：SenseNova U1 在统一多模态理解与生成上树立了新的标杆，在多种理解、推理与生成基准上均达到开源模型中最先进的水平，比肩商用大模型。

- 📖 **原生图文交错生成**：SenseNova U1 可以用单一模型在单次生成流程中连贯产出图文交错内容，支持生活指南、旅行日记等既需要清晰表达又富有叙事性与表现力的场景，把复杂信息浓缩为直观的图示。

- 📰 **高密度信息呈现**：SenseNova U1 在高密度视觉信息表达上展现出强大能力，能够生成结构丰富、排版复杂的内容，适用于知识图解、海报、PPT、漫画、简历等多种信息密集型场景。

#### 🌍 *不止于多模态：*

- 🤖 视觉-语言-动作（VLA）
- 🌐 世界建模（WM）

## 🦁 模型库

在本次发布中，我们开源了 SenseNova U1 Lite 系列，共两个规格：

- SenseNova U1-8B-MoT — 密集主干网络
- SenseNova U1-A3B-MoT — MoE 主干网络


| 模型 | 参数量 | HF 权重 |
| :---- | :------- | :--------- |
| SenseNova-U1-8B-MoT-SFT | 8B MoT | [🤗 链接](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-SFT) |
| SenseNova-U1-8B-MoT | 8B MoT | [🤗 链接](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-8B-MoT-LoRA-8step-V1.0 | 0.4B | [🤗 链接](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors) |
| SenseNova-U1-A3B-MoT-SFT | A3B MoT | 🤗 链接 |
| SenseNova-U1-A3B-MoT | A3B MoT | 🤗 链接 |

其中 **SFT 模型**（*×32 下采样比例*）经过四个阶段训练：(1) *理解预热*，(2) *生成预训练*，(3) *统一中期训练*，(4) *统一监督微调*。**最终模型**是在基座模型之上进行了一轮 T2I 强化学习（RL）训练后得到的版本。

目前这些模型在规模上相对紧凑，但已在多种任务上展现出强劲性能，与商用模型相当且具备出色的性价比。未来还将推出规模更大的版本，进一步提升能力。

> 💡 `SenseNova-U1-8B-MoT` 中的 `8B-MoT` 指的是 ~8B 理解参数**与** ~8B 生成参数。可参阅 [模型参数分解](docs/parameter_breakdown_CN.md) 查看详细的分组明细。


## 📣 最新动态

- `[2026.05.06]` 发布[SenseNova-U1-8B-MoT-LoRA-8step-V1.0](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs/blob/main/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors). 请查看[推理示例脚本](docs/base_vs_distill.md#run-base-and-distilled-model).

- `[2026.04.30]` 发布8步推理模型的预览版 [SenseNova-U1-8B-MoT-8step-preview](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-8step-preview). 在大多数情况下，该模型的图像生成质量与基础模型非常接近 (查看 [效果对比和存在的问题](docs/base_vs_distill.md))。要测试该模型，可以参考[推理脚本](examples/README.md), 但需替换如下参数: ```--cfg_scale 1.0 --num_steps 8``` .

- `[2026.04.27]` 首发 [SenseNova-U1-8B-MoT-SFT](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-SFT) 与 [SenseNova-U1-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) 模型权重。

- `[2026.04.27]` 首发 SenseNova-U1 的[推理代码](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/README_CN.md)。

## 📋 后续计划

- [ ] SenseNova-U1 训练代码

- [ ] SenseNova-U1 最终版权重与技术报告

## 🎨 效果展示

<details>
<summary>🖼️ 文生图（通用）</summary>

| | | |
| :---: | :---: | :---: |
| [<img width="300" alt="t2i general dense face hd 07" src="./docs/assets/showcases/t2i_general/16_9_dense_face_hd_07.webp">](./docs/assets/showcases/t2i_general/16_9_dense_face_hd_07.webp) | [<img width="300" alt="t2i general dense text rendering 18" src="./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_18.webp">](./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_18.webp) | [<img width="300" alt="t2i general dense text rendering 12" src="./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_12.webp">](./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_12.webp) |
| [<img width="260" alt="t2i general face hd 13" src="./docs/assets/showcases/t2i_general/1_1_face_hd_13.webp">](./docs/assets/showcases/t2i_general/1_1_face_hd_13.webp) | [<img width="260" alt="t2i general face hd 17" src="./docs/assets/showcases/t2i_general/1_1_face_hd_17.webp">](./docs/assets/showcases/t2i_general/1_1_face_hd_17.webp) | [<img width="260" alt="t2i general face hd 07" src="./docs/assets/showcases/t2i_general/1_1_dense_artistic_10.webp">](./docs/assets/showcases/t2i_general/1_1_dense_artistic_10.webp) |
| [<img width="260" alt="t2i general landscape 06" src="./docs/assets/showcases/t2i_general/1_1_landscape_06.webp">](./docs/assets/showcases/t2i_general/1_1_landscape_06.webp) | [<img width="260" alt="t2i general dense landscape 12" src="./docs/assets/showcases/t2i_general/1_1_dense_landscape_12.webp">](./docs/assets/showcases/t2i_general/1_1_dense_landscape_12.webp) | [<img width="260" alt="t2i general landscape 07" src="./docs/assets/showcases/t2i_general/1_1_landscape_07.webp">](./docs/assets/showcases/t2i_general/1_1_landscape_07.webp) |
| [<img width="200" alt="t2i general portrait artistic 02 a" src="./docs/assets/showcases/t2i_general/9_16_dense_face_hd_10.webp">](./docs/assets/showcases/t2i_general/9_16_dense_face_hd_10.webp) | [<img width="200" alt="t2i general portrait artistic 02 b" src="./docs/assets/showcases/t2i_general/9_16_human_pose_11.webp">](./docs/assets/showcases/t2i_general/9_16_human_pose_11.webp) | [<img width="200" alt="t2i general portrait artistic 07" src="./docs/assets/showcases/t2i_general/9_16_artistic_07.webp">](./docs/assets/showcases/t2i_general/9_16_artistic_07.webp) |
| [<img width="200" alt="t2i general portrait text rendering 02" src="./docs/assets/showcases/t2i_general/9_16_sensenova_u1_31.webp">](./docs/assets/showcases/t2i_general/9_16_sensenova_u1_31.webp) | [<img width="200" alt="t2i general portrait dense landscape 05" src="./docs/assets/showcases/t2i_general/9_16_dense_landscape_05.webp">](./docs/assets/showcases/t2i_general/9_16_dense_landscape_05.webp) | [<img width="200" alt="t2i general portrait dense artistic 11" src="./docs/assets/showcases/t2i_general/9_16_dense_artistic_11.webp">](./docs/assets/showcases/t2i_general/9_16_dense_artistic_11.webp) |

</details>

<details>
<summary>🖼️ 文生图（推理）</summary>

<table>
  <tr>
    <th style="width: 20%">原始文本</th>
    <th style="width: 50%">推理过程</th>
    <th style="width: 30%">生成图像</th>
  </tr>
  <tr>
    <td style="vertical-align: top;">A male peacock trying to attract a female</td>
    <td><div style="max-height: 200px; overflow-y: auto;">1. <b>Instruction Understanding:</b> The core subject is a single male peacock. The action involves the male peacock trying to attract a female. 2. <b>Reasoning Process:</b> Male peacocks perform this ritual by fanning their tail feathers into a massive, colorful circular display. 3. <b>Establish the frame:</b> The frame captures the male peacock in a full-body shot to show the entire display. The angle is eye-level to emphasize the height of the tail feathers. 4. <b>Set the lighting and color:</b> Natural lighting illuminates the colorful circular display with soft shadows. The palette features vibrant blues and greens on the tail feathers. 5. <b>Lock the style:</b> The style is a realistic photograph with sharp focus on the male peacock. The finish is natural without artistic filters. 6. <b>Explicit Prompt:</b> A vibrant male peacock with his tail feathers fully fanned out in a wide, iridescent display of blues and greens, standing in a garden.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/t2i_reasoning/4_reasoning.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">A small piece of dry wood and a dense iron block are in a transparent water tank</td>
    <td><div style="max-height: 200px; overflow-y: auto;">1. <b>Instruction Understanding:</b> The core subjects are a small piece of dry wood and a dense iron block positioned within a transparent water tank. The wood is floating on the surface while the iron block is submerged at the bottom. 2. <b>Reasoning Process:</b> The wood is less dense than water so it will float, while the iron is denser and will sink to the bottom. 3. <b>Establish the frame:</b> The composition is a vertical medium shot centering the rectangular tank within the frame. The camera angle is eye-level to clearly display the water line and the submerged base. Focus is sharp across the entire depth of the tank to ensure both materials are distinct. 4. <b>Build the environment:</b> The scene is contained entirely within the clear glass walls of the water tank. The water fills the majority of the volume, providing a medium for the floating wood and sunken iron block. The background remains out of focus to keep attention on the tank's interior. 5. <b>Set the lighting and color:</b> Soft natural light illuminates the scene from the left, creating gentle reflections on the water surface. The color palette features the brown grain of the wood contrasting against the dark grey metallic finish of the iron. Shadows are soft and diffused through the liquid. 6. <b>Explicit Prompt:</b> A realistic photo of a transparent water tank showing a piece of wood floating on the surface and an iron block resting at the bottom.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/t2i_reasoning/5_reasoning.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">A typical dish from the country where Naples is located</td>
    <td><div style="max-height: 200px; overflow-y: auto;">1. <b>Instruction Understanding:</b> The core subject is a Neapolitan pizza presented as a typical dish. The context specifies Naples, Italy, as the country of origin for this food item. 2. <b>Reasoning Process:</b> Naples is in Italy, and a classic dish is a Neapolitan pizza. 3. <b>Establish the frame:</b> The Neapolitan pizza is captured in a close-up shot that fills the central frame. The angle is slightly elevated to show the round form of the dish clearly. 4. <b>Set the lighting and color:</b> Soft lighting illuminates the surface of the Neapolitan pizza to reveal texture. Warm tones dominate the color palette, emphasizing the baked nature of the dish. 5. <b>Lock the style:</b> The image utilizes a realistic photographic style with sharp focus on the main subject. The finish appears natural and appetizing, suitable for food documentation. 6. <b>Explicit Prompt:</b> A delicious Neapolitan pizza with a soft, charred crust, tomato sauce, and fresh mozzarella, served on a rustic wooden table, realistic food photography.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/t2i_reasoning/2_reasoning.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">A solution of calcium carbonate reacting with acetic acid</td>
    <td><div style="max-height: 200px; overflow-y: auto;">1. <b>Instruction Understanding:</b> The core subject is a solution of calcium carbonate and acetic acid. The prompt specifies the reacting state of the chemical mixture. 2. <b>Reasoning Process:</b> The reaction produces carbon dioxide gas, which would be visible as a steady stream of bubbles rising through the liquid. 3. <b>Establish the frame:</b> The camera frames the solution closely to capture the details of the reaction. The composition centers on the liquid where the gas is visible. 4. <b>Set the lighting and color:</b> The liquid appears clear, allowing the white bubbles to stand out distinctly. The lighting is bright and even to illuminate the stream of gas. 5. <b>Lock the style:</b> The image maintains a realistic photographic style suitable for scientific observation. The focus is sharp on the reacting solution and bubbles. 6. <b>Explicit Prompt:</b> A test tube filled with a clear liquid and a rapid, effervescent stream of carbon dioxide bubbles rising to the surface, laboratory experiment.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/t2i_reasoning/7_reasoning.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
</table>

</details>

<details>
<summary>🖼️ 文生图（信息图）</summary>

<table align="center">
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0004.webp"><img width="300" alt="t2i landscape 0001" src="./docs/assets/showcases/t2i_infographic/0004.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0012.webp"><img width="300" alt="t2i landscape 0002" src="./docs/assets/showcases/t2i_infographic/0012.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0005.webp"><img width="300" alt="t2i landscape 0003" src="./docs/assets/showcases/t2i_infographic/0005.webp"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0018.webp"><img width="300" alt="t2i landscape 0004" src="./docs/assets/showcases/t2i_infographic/0018.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0024.webp"><img width="300" alt="t2i landscape 0005" src="./docs/assets/showcases/t2i_infographic/0024.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0019.webp"><img width="300" alt="t2i landscape 0006" src="./docs/assets/showcases/t2i_infographic/0019.webp"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0006.webp"><img width="300" alt="t2i landscape 0007" src="./docs/assets/showcases/t2i_infographic/0006.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0015.webp"><img width="300" alt="t2i landscape 0008" src="./docs/assets/showcases/t2i_infographic/0015.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0025.webp"><img width="300" alt="t2i landscape 0009" src="./docs/assets/showcases/t2i_infographic/0025.webp"></a></td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0000.webp"><img width="220" alt="t2i landscape 0010" src="./docs/assets/showcases/t2i_infographic/0000.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0003.webp"><img width="220" alt="t2i landscape 0011" src="./docs/assets/showcases/t2i_infographic/0003.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0001.webp"><img width="220" alt="t2i landscape 0012" src="./docs/assets/showcases/t2i_infographic/0001.webp"></a></td>
      <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0022.webp"><img width="220" alt="t2i landscape 0012" src="./docs/assets/showcases/t2i_infographic/0022.webp"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0016.webp"><img width="220" alt="t2i image 0022" src="./docs/assets/showcases/t2i_infographic/0016.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0010.webp"><img width="220" alt="t2i image 0020" src="./docs/assets/showcases/t2i_infographic/0010.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0007.webp"><img width="220" alt="t2i image 0021" src="./docs/assets/showcases/t2i_infographic/0007.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0021.webp"><img width="220" alt="t2i image 0023" src="./docs/assets/showcases/t2i_infographic/0021.webp"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0014.webp"><img width="220" alt="t2i image 0024" src="./docs/assets/showcases/t2i_infographic/0014.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0028.webp"><img width="220" alt="t2i image 0025" src="./docs/assets/showcases/t2i_infographic/0028.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0033.webp"><img width="220" alt="t2i image 0026" src="./docs/assets/showcases/t2i_infographic/0033.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0002.webp"><img width="220" alt="t2i image 0027" src="./docs/assets/showcases/t2i_infographic/0002.webp"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0031.webp"><img width="230" alt="t2i image 0028" src="./docs/assets/showcases/t2i_infographic/0031.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0030.webp"><img width="230" alt="t2i image 0029" src="./docs/assets/showcases/t2i_infographic/0030.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0032.webp"><img width="230" alt="t2i image 0030" src="./docs/assets/showcases/t2i_infographic/0032.webp"></a></td>
    <td align="center"><a href="./docs/assets/showcases/t2i_infographic/0029.webp"><img width="230" alt="t2i image 0031" src="./docs/assets/showcases/t2i_infographic/0029.webp"></a></td>
  </tr>
</table>

</details>

> 📸 **更多生成样例：** 参见 [文生图样例集](./docs/showcases_CN.md#文生图)。


<details>
<summary>✏️ 图像编辑（通用）</summary>

| | |
| :---: | :---: |
| <div align="center"><a href="./examples/editing/data/images/1.webp"><img width="150" alt="editing input 1" src="./examples/editing/data/images/1.webp"></a> <a href="./docs/assets/showcases/editing/1_out.webp"><img width="150" alt="editing output 1" src="./docs/assets/showcases/editing/1_out.webp"></a><br><sub>Change the jacket of the person on the left to bright yellow.</sub></div> | <div align="center"><a href="./examples/editing/data/images/3.webp"><img width="150" alt="editing input 3" src="./examples/editing/data/images/3.webp"></a> <a href="./docs/assets/showcases/editing/3_out.webp"><img width="150" alt="editing output 3" src="./docs/assets/showcases/editing/3_out.webp"></a><br><sub>在小狗头上放一个花环，并且把图片变为吉卜力风格。</sub></div> |
| <div align="center"><a href="./examples/editing/data/images/2.webp"><img width="150" alt="editing input 2" src="./examples/editing/data/images/2.webp"></a> <a href="./docs/assets/showcases/editing/2_out.webp"><img width="150" alt="editing output 2" src="./docs/assets/showcases/editing/2_out.webp"></a><br><sub>Make the person in the image smile.</sub></div> | <div align="center"><a href="./examples/editing/data/images/4.webp"><img width="150" alt="editing input 4" src="./examples/editing/data/images/4.webp"></a> <a href="./docs/assets/showcases/editing/4_out.webp"><img width="150" alt="editing output 4" src="./docs/assets/showcases/editing/4_out.webp"></a><br><sub>Add a bouquet of flowers.</sub></div> |
| <div align="center"><a href="./examples/editing/data/images/8.webp"><img width="150" alt="editing input 8" src="./examples/editing/data/images/8.webp"></a> <a href="./docs/assets/showcases/editing/8_out.webp"><img width="150" alt="editing output 8" src="./docs/assets/showcases/editing/8_out.webp"></a><br><sub>Replace the man with a woman.</sub></div> | <div align="center"><a href="./examples/editing/data/images/6.webp"><img width="150" alt="editing input 6" src="./examples/editing/data/images/6.webp"></a> <a href="./docs/assets/showcases/editing/6_out.webp"><img width="150" alt="editing output 6" src="./docs/assets/showcases/editing/6_out.webp"></a><br><sub>Replace the text "WARFIGHTER" to "BATTLEFIELD" in the bold orange-red font.</sub></div> |

</details>


<details>
<summary>✏️ 图像编辑（推理）</summary>

<table>
  <tr>
    <th style="width: 20%">编辑指令</th>
    <th style="width: 30%">原始图像</th>
    <th style="width: 20%">推理过程</th>
    <th style="width: 30%">编辑结果</th>
  </tr>
  <tr>
    <td style="vertical-align: top;">Draw what it will look like one hour later.</td>
    <td style="vertical-align: top;"><img src="./examples/editing/data/images_reasoning/034_temporal_reasoning_draw_what_it_will_look_like.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
    <td><div style="max-height: 200px; overflow-y: auto;">
    1. <b>Source Image Analysis:</b> The source image shows a glass cup of hot tea with steeping tea leaves, and the water appears relatively clear. 2. <b>Instruction Understanding:</b> The instruction asks for the appearance one hour later. 3. <b>Reasoning Process:</b> Over time, tannins and pigments leach out, making the tea noticeably darker and more uniformly colored, and the leaves may look more swollen and darker from soaking. 4. <b>Expected Visual Changes:</b> The expected visible result is a deeper amber-to-brown tea color and more fully saturated liquid. 5. <b>Elements to Preserve:</b> The glass cup, scattered leaves around it, background, and camera angle should remain unchanged. 6. <b>Explicit Edit Prompt:</b> Edit the tea liquid to a much darker, more saturated amber-brown color as if fully steeped, and make the tea leaves look slightly darker and more swollen, while keeping the glass cup, surrounding leaves, background, and framing unchanged.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/editing/034_temporal_reasoning_draw_what_it_will_look_like_result.jpeg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">Draw what it will look like immediately after someone stands up from sitting on it for a long time.</td>
    <td style="vertical-align: top;"><img src="./examples/editing/data/images_reasoning/036_causal_reasoning_draw_what_it_will_look_like.png" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
    <td><div style="max-height: 200px; overflow-y: auto;">
    1. <b>Source Image Analysis:</b> The source image shows a fluffy lime-green beanbag chair that looks evenly plump and undisturbed on a white background. 2. <b>Instruction Understanding:</b> The edit instruction asks for its appearance immediately after someone stood up from sitting there for a long time. 3. <b>Reasoning Process:</b> Prolonged weight compresses the fabric and internal fill, leaving a depressed seat area, wrinkles radiating outward, and a slowly recovering shape. 4. <b>Expected Visual Changes:</b> The visible result should be a noticeable dip and creasing where a person was seated. 5. <b>Elements to Preserve:</b> The background, beanbag color, lighting, and camera angle should remain unchanged while only the beanbag's shape shows the compression. 6. <b>Explicit Edit Prompt:</b> Edit the beanbag chair to show a clear seated depression in the center with surrounding wrinkles and slightly compressed fabric, while keeping the white background, lighting, and camera angle unchanged.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/editing/036_causal_reasoning_draw_what_it_will_look_like_result.jpeg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">Change the water to high-concentration saltwater</td>
    <td style="vertical-align: top;"><img src="./examples/editing/data/images_reasoning/042_physics_change_the_water_to_high-con.jpg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
    <td><div style="max-height: 200px; overflow-y: auto;">
    1. <b>Source Image Analysis:</b> The source image shows an egg resting at the bottom of a glass of water. 2. <b>Instruction Understanding:</b> The instruction asks to change the medium to high-concentration saltwater. 3. <b>Reasoning Process:</b> Saltwater is denser than fresh water, which increases the buoyant force on the egg. 4. <b>Expected Visual Changes:</b> As density increases, the egg will overcome gravity and float higher or suspend in the middle of the liquid. 5. <b>Elements to Preserve:</b> The glass and the egg's appearance should remain consistent, focusing on the shift in the egg's vertical position. 6. <b>Explicit Edit Prompt:</b> Edit the position of the egg so it is floating in the middle of the liquid instead of resting on the bottom, while keeping the glass and the egg's appearance unchanged.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/editing/042_physics_change_the_water_to_high-con_result.jpeg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
  <tr>
    <td style="vertical-align: top;">What the fruit looks like when ripe in the picture</td>
    <td style="vertical-align: top;"><img src="./examples/editing/data/images_reasoning/044_biology_what_the_fruit_looks_like_wh.jpg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
    <td><div style="max-height: 200px; overflow-y: auto;">
    1. <b>Source Image Analysis:</b> The source image shows green, unripe bananas. 2. <b>Instruction Understanding:</b> The instruction asks for the appearance of the fruit when ripe. 3. <b>Reasoning Process:</b> Ripening involves a breakdown of chlorophyll and the production of sugars, which turns the skin from green to yellow and often causes small brown sugar spots to appear. 4. <b>Expected Visual Changes:</b> The color and texture of the peel should transition to a ripe state. 5. <b>Elements to Preserve:</b> The shape of the bananas and the white background should remain constant. 6. <b>Explicit Edit Prompt:</b> Edit the green bananas to be bright yellow with small brown spots, while keeping the original shape and white background unchanged.</div></td>
    <td style="vertical-align: top;"><img src="./docs/assets/showcases/editing/044_biology_what_the_fruit_looks_like_wh_result.jpeg" style="max-width: 100%; max-height: 100%; object-fit: contain;"></td>
  </tr>
</table>

</details>

> 📸 **更多编辑样例：** 参见 [图像编辑样例集](./docs/showcases_CN.md#图像编辑)。

<details>
<summary>♻️ 图文交错生成（通用）</summary>

| |
| :---: |
| [<img alt="interleave case 05" src="./docs/assets/showcases/interleave/case_0005_matchgirl_warm_au.webp">](./docs/assets/showcases/interleave/case_0005_matchgirl_warm_au.webp) |
| [<img alt="interleave case 06" src="./docs/assets/showcases/interleave/case_0006_orange_cat_travel.webp">](./docs/assets/showcases/interleave/case_0006_orange_cat_travel.webp) |

</details>


<details>
<summary>♻️ 图文交错生成（推理）</summary>

| |
| :---: |
| [<img alt="interleave reasoning case" src="./docs/assets/showcases/interleave/reasoning.png">](./docs/assets/showcases/interleave/reasoning.png) |

</details>

> 📸 **更多图文交错样例：** 参见 [图文交错生成样例集](./docs/showcases_CN.md#图文交错生成)。

<details>
<summary>📝 视觉理解（通用）</summary>

| |
| :---: |
| [<img alt="vqa general cases" src="./docs/assets/showcases/vqa/general_case.webp">](./docs/assets/showcases/vqa/general_case.webp) |

</details>

<details>
<summary>📝 视觉理解（智能体）</summary>

| |
| :---: |
| [<img alt="vqa agentic case" src="./docs/assets/showcases/vqa/agentic_case.webp">](./docs/assets/showcases/vqa/agentic_case.webp) |


</details>

> 📸 **更多视觉理解样例：** 参见 [视觉理解样例集](./docs/showcases_CN.md#视觉理解)。


<details>
<summary>🦾 视觉语言动作</summary>

[![YouTube](./docs/assets/showcases/vla/1.png)](https://www.youtube.com/watch?v=3mvBPPgv8vo)
[![YouTube](./docs/assets/showcases/vla/2.png)](https://www.youtube.com/watch?v=2QZY8gf0Vsk)
[![YouTube](./docs/assets/showcases/vla/3.png)](https://www.youtube.com/watch?v=tznVbuYf0yw)

</details>


## 📊 核心评测

<details>
<summary>📝 视觉理解</summary>

<p align="center">
  <img src="docs/assets/benchmarks/understanding.webp" alt="Understanding Benchmarks">
</p>

</details>

<details>
<summary>🖼️ 视觉生成</summary>

<p align="center">
  <img src="docs/assets/benchmarks/generation.webp" alt="Generation Benchmarks">
</p>

</details>

<details>
<summary>♻️ 视觉推理</summary>

<p align="center">
  <img src="docs/assets/benchmarks/interleaved.webp" alt="Interleaved Benchmarks">
</p>

</details>

> 评测脚本与榜单复现指南已提供在 [`evaluation`](./evaluation/README_CN.md)。


## ⚠️ 进行中的改进

尽管在各项任务上表现优异，当前版本仍有若干已知局限有待改进：

* **视觉理解**：
  当前模型支持的上下文长度最长为 **32K** tokens，在需要更长或更复杂视觉上下文的场景下可能受到限制。

* **人体生成**：
  对人体细粒度细节的处理仍有挑战，尤其是当人物在画面中占比较小，或与周围物体存在复杂交互时。

* **文字生成**：
  文字渲染有时会出现拼写错误、字符变形或格式不一致的问题，且对 prompt 的措辞较为敏感，在文字密集场景下尤为明显。(最佳实践请参见 [`提示词增强`](./docs/prompt_enhancement.md))

* **图文交错生成**：

  * 作为实验性功能，图文交错生成仍在持续演进中，性能可能尚未达到专用文生图（T2I）流程的水平。

  * **Beta 状态：** 强化学习尚未针对图像编辑、推理及图文交错任务进行专项优化，当前性能与 SFT 模型相当。

我们将上述方向列为持续迭代的重点，期待在后续版本中不断改进。


## 🛠️ 快速开始


### 🌐 使用 SenseNova-Studio

体验 SenseNova-U1 最便捷的方式是通过 **[SenseNova-Studio](https://unify.light-ai.top/)** —— 一个 🆓 免费的在线体验平台，无需安装、无需 GPU，直接在浏览器中即可试用。

> **注：** 为服务更多用户，U1-Fast 经过步数蒸馏和 CFG 蒸馏，专供信息图生成使用。


### 🦞 使用 SenseNova-Skills（OpenClaw）

将 SenseNova-U1 集成进自己的智能体或应用，最简单的方式是使用配套仓库 **[SenseNova-Skills (OpenClaw) 🦞](https://github.com/OpenSenseNova/SenseNova-Skills)**——它将 SenseNova-U1 封装为开箱即用的技能，并提供统一的工具调用接口。

> 安装与使用详情请参考 [SenseNova-Skills README](https://github.com/OpenSenseNova/SenseNova-Skills)。

<details>
<summary>✨ 通过我们 Skills 和 Studio 制作的有趣案例</summary>
<p align="center">
  <img src="docs/assets/showcases/t2i_infographic/u1-case2.webp" alt="Skill Cases">
</p>

</details>

### 🤗 使用 transformers 运行

> **环境准备：** 按照[安装指南](./docs/installation_CN.md)克隆仓库并用 [uv](https://github.com/astral-sh/uv) 安装依赖。

<details open>
<summary>📝 视觉理解</summary>

```bash
python examples/vqa/inference.py --model_path SenseNova/SenseNova-U1-8B-MoT --image examples/vqa/data/images/menu.jpg --question "My friend and I are dining together tonight. Looking at this menu, can you recommend a good combination of dishes for 2 people? We want a balanced meal — a mix of mains and maybe a starter or dessert. Budget-conscious but want to try the highlights." --output outputs/answer.txt --max_new_tokens 8192 --do_sample --temperature 0.6 --top_p 0.95 --top_k 20 --repetition_penalty 1.05 --profile
```

</details>

> 批量推理、生成参数和 JSONL 格式请参见 [`examples/README_CN.md`](./examples/README_CN.md#视觉理解vqa)。

<details open>
<summary>🖼️ 文生图</summary>

```bash
python examples/t2i/inference.py --model_path SenseNova/SenseNova-U1-8B-MoT --prompt "这张信息图的标题是“SenseNova-U1”，采用现代极简科技矩阵风格。整体布局为水平三列网格结构，背景是带有极浅银灰色细密点阵的哑光纯白高级纸张纹理，画面长宽比为16:9。\n\n排版采用严谨的视觉层级：主标题使用粗体无衬线黑体字，正文使用清晰的现代等宽字体。配色方案极其克制，以纯白色为底，深炭黑为主视觉文字和边框，浅石板灰用于背景色块和次要信息区分，图标采用精致的银灰色线框绘制。\n\n在画面正上方居中位置，使用醒目的深炭黑粗体字排布着大标题“SenseNova-U1”。标题正下方是浅石板灰色的等宽字体副标题“新一代端到端统一多模态大模型家族”。\n\n画面主体分为左、中、右三个相等的垂直信息区块，区块之间通过充足的负空间进行物理隔离。\n\n左侧区块的主题是概述。顶部有一个银灰色线框绘制的、由放大镜和齿轮交织的图标，旁边是粗体小标题“Overview”。该区块内从上到下垂直排列着三个要点：第一个要点旁边是一个代表文档与照片重叠的极简图标，紧跟着文字“多模态模型家族，统一文本/图像理解和生成”。向下是由两个相连的同心圆组成的架构图标，配有文字“基于NEO-Unify架构（端到端统一理解和生成）”。最下方是一个带有斜线划掉的眼睛和漏斗形状的图标，明确指示文本“无需视觉编码器(VE)和变分自编码器(VAE)”。\n\n中间区块展示模型矩阵。顶部是一个包含两个分支节点的树状网络图标，旁边是粗体小标题“两个模型规格”。区块内分为上下两个包裹在浅石板灰色极细边框内的卡片。上方的卡片内画着一个代表高密度的实心几何立方体图标，大字标注“SenseNova-U1-8B-MoT”，下方是等宽字体说明“8B MoT 密集主干模型”。下方的卡片内画着一个带有闪电符号的网状发光大脑图标，大字标注“SenseNova-U1-A3B-MoT”，下方是等宽字体说明“A3B MoT 混合专家（MoE）主干模型”。在这两个独立卡片的正下方，左侧放置一个笑脸轮廓图标搭配文字“将在HF等平台公开”，右侧放置一个带有折角的书面报告图标搭配文字“将发布技术报告”。\n\n右侧区块呈现核心优势。顶部是一个代表巅峰的上升阶梯折线图图标，旁边是粗体小标题“Highlights”。该区块内部垂直分布着四个带有浅石板灰底色的长方形色块，每个色块内部左侧对应一个具体的图标，右侧为文字。第一个色块内是一个无缝相连的莫比乌斯环图标，配文“原生统一架构，无VE和VAE”。第二个色块内是一个顶端带有星星的奖杯图标，配文“单一统一模型在理解和生成任务上均达到SOTA性能”。第三个色块内是代表文本行与拍立得照片交替穿插的图标，配文“强大的原生交错推理能力（模型原生生成图像进行推理）”。最后一个色块内是一个被切分出一小块的硬币与详细饼状图结合的图标，配文“能生成复杂信息图表，性价比出色”。" --width 2720 --height 1536 --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 --output output.png --profile
```

</details>

> 默认分辨率为 2048×2048（1:1）。其它长宽比请参见[支持的分辨率档位](./examples/README_CN.md#推荐分辨率档位)。

> 当进行信息图生成时，建议先使用[提示词增强](./docs/prompt_enhancement_CN.md)以获得最佳效果。


<details open>
<summary>✏️ 图像编辑</summary>

```bash
python examples/editing/inference.py --model_path SenseNova/SenseNova-U1-8B-MoT --prompt "Change the animal's fur color to a darker shade." --image examples/editing/data/images/1.jpg --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 --output output_edited.png --profile --compare
```

</details>

> 💡 为获得最佳效果，建议在推理前将输入按原长宽比预缩放至约 2048×2048 分辨率（参见 [`examples/editing/resize_inputs.py`](./examples/editing/resize_inputs.py)）。


<details open>
<summary>♻️ 图文交错生成</summary>

```bash
python examples/interleave/inference.py --model_path SenseNova/SenseNova-U1-8B-MoT --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." --resolution "16:9" --output_dir outputs/interleave/ --stem demo --profile
```
</details>

> 批量推理、JSONL 格式、prompt 增强、分辨率档位及完整参数说明请参见 [`examples/README_CN.md`](./examples/README_CN.md)。


### ⚡ 使用 LightLLM + LightX2V 运行

面向生产环境的部署，我们在 **[LightLLM](https://github.com/ModelTC/lightllm)**（理解）和 **[LightX2V](https://github.com/ModelTC/lightx2v)**（生成）之上协同设计了一套专用推理栈。两个引擎以解耦方式运行，可以各自使用独立的并行策略与资源配额，中间通过低开销传输通道连接。

在单节点 `TP2 + CFG2` 配置下，该推理栈在 H100 / H200 上为 **2048×2048** 图像提供约 **~0.15 s/step**、**~9 s 端到端**的表现；相较 Triton 基线，我们基于 FA3 的混合掩码注意力带来 ~**2.4–3.2×** 的 prefill 加速。完整的单卡性能数据见 [`docs/inference_infra_CN.md`](./docs/inference_infra_CN.md)。

我们提供了官方 Docker 镜像，一行命令即可完成部署：

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
```

> ⚙️ **部署指南（Docker、启动参数、模式、量化、API 测试）：** 参见 [`docs/deployment_CN.md`](./docs/deployment_CN.md)。
>
> 📖 **完整架构设计与性能剖析：** 参见 [`docs/inference_infra_CN.md`](./docs/inference_infra_CN.md)。

<!-- ## 🖊️ Citation

```bibtex

``` -->

## 🌐 加入社区！

加入我们的社区，分享反馈、获取支持，并第一时间了解 SenseNova-U1 的最新进展 — 期待与你交流！

<div align="center">
<table>
  <tr>
    <td align="center"><b><a href="https://discord.gg/cxkwXWjp">Discord</a></b></td>
    <td align="center"><b>微信交流群</b></td>
  </tr>
  <tr>
    <td align="center"><a href="https://discord.gg/cxkwXWjp"><img src="docs/assets/discord_qr.webp" width="160"/></a></td>
    <td align="center"><img src="docs/assets/wechat_qr.webp" width="160"/></td>
  </tr>
</table>
</div>

## ⚖️ 许可证

本项目基于 [Apache 2.0 License](./LICENSE) 开源发布。
