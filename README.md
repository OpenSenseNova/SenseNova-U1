# SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-Unify Architecture

<p align="center">
  <strong>English</strong> | <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Coming-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/sensenova/sensenova-u1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://unify.light-ai.top/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20SenseNova_U1-Demo-Green" alt="SenseNova-U1 Demo"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <img src="docs/assets/teaser.png" alt="SenseNova-U1" width="900">
</p>

## 🌟 Overview

🚀 **SenseNova-U1**, a native unified paradigm (based on **[NEO-Unify](https://huggingface.co/blog/sensenova/neo-unify)**) where models no longer translate between modalities, but think and act across them natively. 
Multimodal AI is shifting from integration to unification—where capabilities emerge from within.
Our work sits within the pre-train–driven Chat era, yet gestures toward the next step: post-train–driven Agent era.


#### 🏗️ *Key Pillars:*      

- 🖼️ **Near-Lossless Visual Interface**: Preserving semantic richness + pixel fidelity (no VAEs or Vision Encoders) !  

- 🧠 **Native Mixture-of-Transformers**: Modality-agnostic reasoning with high efficiency and minimal conflict !   

- 🔗 **Unified End-to-End Learning**: Modeling directly on pixels + text from the first principles !   


#### ✨ *What This Unlocks:*

- 🏆 **Open-source SOTA efficiency**: U1 sets a new open-source SOTA for unified understanding and generation, and even at a small scale, delivers performance comparable to commercial models with outstanding cost efficiency.

- 📖 **Native interleaved image-text generation**: U1 can generate coherent interleaved text and images in a single flow, unlocking use case such as practical guides and travel diaries that combine clear communication with vivid storytelling, and turn complex information into easy-to-grasp visuals.

- 📰 **High-density information rendering**: U1 demonstrates strong capabilities in dense visual communication, generating richly structured layouts for knowledge illustration, posters, PPTs, comics, resumes, and other information-rich formats.

#### 🌍 *Beyond Multimodality:* 

- 🤖 Vision–Language–Action (VLA)
- 🌐 World Modeling (WM)


## 📣 Updated News

- `[2026.04.23]` Initial release of the weights for [SenseNova-U1-Mini-SFT](https://huggingface.co/sensenova/SenseNova-U1-Mini-Beta) and [SenseNova-U1-Mini-Beta](https://huggingface.co/sensenova/SenseNova-U1-Mini-Beta).  

- `[2026.04.23]` Initial release of the [inference code](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/README.md) for SenseNova-U1.   

## 📋 ToDo List

- [ ] Training code of SenseNova-U1 

- [ ] Final weights and technical report of SenseNova-U1


## 🦁 Model Zoo

| Model | Params | HF Weights |
| :---- | :------- | :--------- |
| SenseNova-U1-Mini-SFT | 8B MoT | [🤗 link](https://huggingface.co/sensenova/SenseNova-U1-Mini-SFT) |
| SenseNova-U1-Mini-Beta | 8B MoT | [🤗 link](https://huggingface.co/sensenova/SenseNova-U1-Mini-Beta) |
| SenseNova-U1-Flash-SFT | A3B MoT | 🤗 link |
| SenseNova-U1-Flash-Beta | A3B MoT | 🤗 link |

Note that the **SFT models** are trained in four stages: (1) *Understanding Warmup*, (2) *Generation Pre-training*, (3) *Unified Mid-training*, and (4) *Unified Supervised Fine-tuning*. The **Beta models** are obtained from the base model following an initial round of T2I reinforcement learning (RL) training.

## 🎨 Showcases

<details>
<summary>🖼️ Text-to-Image (General)</summary>

| | | |
| :---: | :---: | :---: |
| [<img width="300" alt="t2i general dense face hd 07" src="./docs/assets/showcases/t2i_general/16_9_dense_face_hd_07.webp">](./docs/assets/showcases/t2i_general/16_9_dense_face_hd_07.webp) | [<img width="300" alt="t2i general dense text rendering 18" src="./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_18.webp">](./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_18.webp) | [<img width="300" alt="t2i general dense text rendering 12" src="./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_12.webp">](./docs/assets/showcases/t2i_general/16_9_dense_text_rendering_12.webp) |
| [<img width="260" alt="t2i general face hd 13" src="./docs/assets/showcases/t2i_general/1_1_face_hd_13.webp">](./docs/assets/showcases/t2i_general/1_1_face_hd_13.webp) | [<img width="260" alt="t2i general face hd 17" src="./docs/assets/showcases/t2i_general/1_1_face_hd_17.webp">](./docs/assets/showcases/t2i_general/1_1_face_hd_17.webp) | [<img width="260" alt="t2i general face hd 07" src="./docs/assets/showcases/t2i_general/1_1_dense_artistic_10.webp">](./docs/assets/showcases/t2i_general/1_1_dense_artistic_10.webp) |
| [<img width="260" alt="t2i general landscape 06" src="./docs/assets/showcases/t2i_general/1_1_landscape_06.webp">](./docs/assets/showcases/t2i_general/1_1_landscape_06.webp) | [<img width="260" alt="t2i general dense landscape 12" src="./docs/assets/showcases/t2i_general/1_1_dense_landscape_12.webp">](./docs/assets/showcases/t2i_general/1_1_dense_landscape_12.webp) | [<img width="260" alt="t2i general landscape 07" src="./docs/assets/showcases/t2i_general/1_1_landscape_07.webp">](./docs/assets/showcases/t2i_general/1_1_landscape_07.webp) |
| [<img width="200" alt="t2i general portrait artistic 02 a" src="./docs/assets/showcases/t2i_general/9_16_dense_face_hd_10.webp">](./docs/assets/showcases/t2i_general/9_16_dense_face_hd_10.webp) | [<img width="200" alt="t2i general portrait artistic 02 b" src="./docs/assets/showcases/t2i_general/9_16_human_pose_11.webp">](./docs/assets/showcases/t2i_general/9_16_human_pose_11.webp) | [<img width="200" alt="t2i general portrait artistic 07" src="./docs/assets/showcases/t2i_general/9_16_artistic_07.webp">](./docs/assets/showcases/t2i_general/9_16_artistic_07.webp) |
| [<img width="200" alt="t2i general portrait text rendering 02" src="./docs/assets/showcases/t2i_general/9_16_sensenova_u1_31.webp">](./docs/assets/showcases/t2i_general/9_16_sensenova_u1_31.webp) | [<img width="200" alt="t2i general portrait dense landscape 05" src="./docs/assets/showcases/t2i_general/9_16_dense_landscape_05.webp">](./docs/assets/showcases/t2i_general/9_16_dense_landscape_05.webp) | [<img width="200" alt="t2i general portrait dense artistic 11" src="./docs/assets/showcases/t2i_general/9_16_dense_artistic_11.webp">](./docs/assets/showcases/t2i_general/9_16_dense_artistic_11.webp) |

</details>

<details>
<summary>🖼️ Text-to-Image (Reasoning)</summary>

<table>
  <tr>
    <th style="width: 20%">Original Text</th>
    <th style="width: 50%">Reasoning Process</th>
    <th style="width: 30%">Resulting Image</th>
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
</table>

</details>   

<details>
<summary>🖼️ Text-to-Image (Infographics)</summary>

| | | |
| :---: | :---: | :---: |
| [<img width="300" alt="t2i landscape 0001" src="./docs/assets/showcases/t2i_infographic/0001_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0001_2720x1536.webp) | [<img width="300" alt="t2i landscape 0002" src="./docs/assets/showcases/t2i_infographic/0002_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0002_2720x1536.webp) | [<img width="300" alt="t2i landscape 0003" src="./docs/assets/showcases/t2i_infographic/0003_2720x1536.webp">](./docs/assets/showcases/t2i_infographic/0003_2720x1536.webp) |
| [<img width="300" alt="t2i square 0004" src="./docs/assets/showcases/t2i_infographic/0004_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0004_2048x2048.webp) | [<img width="300" alt="t2i square 0005" src="./docs/assets/showcases/t2i_infographic/0005_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0005_2048x2048.webp) | [<img width="300" alt="t2i square 0006" src="./docs/assets/showcases/t2i_infographic/0006_2048x2048.webp">](./docs/assets/showcases/t2i_infographic/0006_2048x2048.webp) |
| [<img width="200" alt="t2i portrait 0007" src="./docs/assets/showcases/t2i_infographic/0007_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0007_1536x2720.webp) | [<img width="200" alt="t2i portrait 0008" src="./docs/assets/showcases/t2i_infographic/0008_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0008_1536x2720.webp) | [<img width="200" alt="t2i portrait 0009" src="./docs/assets/showcases/t2i_infographic/0009_1536x2720.webp">](./docs/assets/showcases/t2i_infographic/0009_1536x2720.webp) |

</details>  

> 📸 **More generation samples:** see [Image Generation Gallery](./docs/showcases.md#text-to-image). 


<details>
<summary>✏️ Image Editing (General)</summary>

| | |
| :---: | :---: |
| <div align="center"><a href="./examples/editing/data/images/1.webp"><img width="150" alt="editing input 1" src="./examples/editing/data/images/1.webp"></a> <a href="./docs/assets/showcases/editing/1_out.webp"><img width="150" alt="editing output 1" src="./docs/assets/showcases/editing/1_out.webp"></a><br><sub>Change the jacket of the person on the left to bright yellow.</sub></div> | <div align="center"><a href="./examples/editing/data/images/3.webp"><img width="150" alt="editing input 3" src="./examples/editing/data/images/3.webp"></a> <a href="./docs/assets/showcases/editing/3_out.webp"><img width="150" alt="editing output 3" src="./docs/assets/showcases/editing/3_out.webp"></a><br><sub>在小狗头上放一个花环，并且把图片变为吉卜力风格。</sub></div> |
| <div align="center"><a href="./examples/editing/data/images/2.webp"><img width="150" alt="editing input 2" src="./examples/editing/data/images/2.webp"></a> <a href="./docs/assets/showcases/editing/2_out.webp"><img width="150" alt="editing output 2" src="./docs/assets/showcases/editing/2_out.webp"></a><br><sub>Make the person in the image smile.</sub></div> | <div align="center"><a href="./examples/editing/data/images/4.webp"><img width="150" alt="editing input 4" src="./examples/editing/data/images/4.webp"></a> <a href="./docs/assets/showcases/editing/4_out.webp"><img width="150" alt="editing output 4" src="./docs/assets/showcases/editing/4_out.webp"></a><br><sub>Add a bouquet of flowers.</sub></div> |
| <div align="center"><a href="./examples/editing/data/images/8.webp"><img width="150" alt="editing input 8" src="./examples/editing/data/images/8.webp"></a> <a href="./docs/assets/showcases/editing/8_out.webp"><img width="150" alt="editing output 8" src="./docs/assets/showcases/editing/8_out.webp"></a><br><sub>Replace the man with a woman.</sub></div> | <div align="center"><a href="./examples/editing/data/images/6.webp"><img width="150" alt="editing input 6" src="./examples/editing/data/images/6.webp"></a> <a href="./docs/assets/showcases/editing/6_out.webp"><img width="150" alt="editing output 6" src="./docs/assets/showcases/editing/6_out.webp"></a><br><sub>Replace the text "WARFIGHTER" to "BATTLEFIELD" in the bold orange-red font.</sub></div> | 

</details>


<details>
<summary>✏️ Image Editing (Reasoning)</summary>

<table>
  <tr>
    <th style="width: 20%">Original Text</th>
    <th style="width: 30%">Original Image</th>
    <th style="width: 20%">Reasoning Process</th>
    <th style="width: 30%">Resulting Image</th>
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
    1. <b>Source Image Analysis:</b> The source image shows a fluffy lime-green beanbag chair that looks evenly plump and undisturbed on a white background. 2. <b>Instruction Understanding:</b> The edit instruction asks for its appearance immediately after someone stood up from sitting there for a long time. 3. <b>Reasoning Process:</b> Prolonged weight compresses the fabric and internal fill, leaving a depressed seat area, wrinkles radiating outward, and a slowly recovering shape. 4. <b>Expected Visual Changes:</b> The visible result should be a noticeable dip and creasing where a person was seated. 5. <b>Elements to Preserve:</b> The background, beanbag color, lighting, and camera angle should remain unchanged while only the beanbag’s shape shows the compression. 6. <b>Explicit Edit Prompt:</b> Edit the beanbag chair to show a clear seated depression in the center with surrounding wrinkles and slightly compressed fabric, while keeping the white background, lighting, and camera angle unchanged.</div></td>
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

> 📸 **More editing samples:** see [Image Editing Gallery](./docs/showcases.md#image-editing). 

<details>
<summary>♻️ Interleaved Generation</summary>

| |
| :---: |
| [<img alt="interleave case 03" src="./docs/assets/showcases/interleave/case_03.webp">](./docs/assets/showcases/interleave/case_03.webp) |
| [<img alt="interleave case 04" src="./docs/assets/showcases/interleave/case_04.webp">](./docs/assets/showcases/interleave/case_04.webp) |

</details>

> 📸 **More interleaved samples:** see [Interleaved Generation Gallery](./docs/showcases.md#interleaved-generation).

<details>
<summary>📝 Visual Understanding</summary>

| |
| :---: |
| [<img alt="vqa agentic case" src="./docs/assets/showcases/vqa/agentic_case.webp">](./docs/assets/showcases/vqa/agentic_case.webp) |
| [<img alt="vqa general cases" src="./docs/assets/showcases/vqa/general_case.webp">](./docs/assets/showcases/vqa/general_case.webp) |

</details>

> 📸 **More understanding samples:** see [Visual Understanding Gallery](./docs/showcases.md#visual-understanding). 


## 📊 Key Benchmarks

<details>
<summary>📝 Visual Understanding</summary>

<p align="center">
  <img src="docs/assets/benchmarks/understanding.webp" alt="Understanding Benchmarks">
</p>

</details>

<details>
<summary>🖼️ Visual Generation</summary>

<p align="center">
  <img src="docs/assets/benchmarks/generation.webp" alt="Generation Benchmarks">
</p>


</details>

<details>
<summary>♻️ Interleaved Generation</summary>

<p align="center">
  <img src="docs/assets/benchmarks/interleaved.webp" alt="Interleaved Benchmarks">
</p>

</details>

> Evaluation scripts and benchmark reproduction guides are added in [`evaluation`](./evaluation/README.md).


## 🛠️ Quick Start


### 🌐 Use with SenseNova-Studio

The fastest way to experience SenseNova-U1 is through **[SenseNova-Studio](https://unify.light-ai.top/)** — a 🆓 free online playground where you can try the model directly in your browser, no installation or GPU required.


### 🦞 Use with SenseNova-Skills (OpenClaw)

The easiest way to integrate SenseNova-U1 into your own agent or application is through our companion repository **[SenseNova-Skills (OpenClaw) 🦞](https://github.com/OpenSenseNova/SenseNova-Skills)**, which ships SenseNova-U1 as a ready-to-use skill with a unified tool-calling interface.

> Refer to the [SenseNova-Skills README](https://github.com/OpenSenseNova/SenseNova-Skills) for installation and usage details.

<details>
<summary>✨ Some interesting cases produced through our Skills and Studio</summary>
<p align="center">
  <img src="docs/assets/showcases/t2i_infographic/u1-case.webp" alt="Interleaved Benchmarks">
</p>

</details>

### 🤗 Run with transformers

> **Setup:** Follow the [Installation Guide](./docs/installation.md) to clone the repo and install dependencies with uv.

<details>
<summary>📝 Visual Understanding</summary>

```bash
python examples/vqa/inference.py --model_path SenseNova/SenseNova-U1-Mini-Beta --image examples/vqa/data/images/menu.jpg --question "My friend and I are dining together tonight. Looking at this menu, can you recommend a good combination of dishes for 2 people? We want a balanced meal — a mix of mains and maybe a starter or dessert. Budget-conscious but want to try the highlights." --output outputs/answer.txt --max_new_tokens 8192 --do_sample --temperature 0.6 --top_p 0.95 --top_k 20 --repetition_penalty 1.05 --profile
```

</details>

> See [`examples/README.md`](./examples/README.md#visual-understanding-vqa) for batched inference, generation parameters, and JSONL format.

<details>
<summary>🖼️ Text-to-Image</summary>

```bash
python examples/t2i/inference.py --model_path SenseNova/SenseNova-U1-Mini-Beta --prompt "这张信息图的标题是“SenseNova-U1”，采用现代极简科技矩阵风格。整体布局为水平三列网格结构，背景是带有极浅银灰色细密点阵的哑光纯白高级纸张纹理，画面长宽比为16:9。\n\n排版采用严谨的视觉层级：主标题使用粗体无衬线黑体字，正文使用清晰的现代等宽字体。配色方案极其克制，以纯白色为底，深炭黑为主视觉文字和边框，浅石板灰用于背景色块和次要信息区分，图标采用精致的银灰色线框绘制。\n\n在画面正上方居中位置，使用醒目的深炭黑粗体字排布着大标题“SenseNova-U1”。标题正下方是浅石板灰色的等宽字体副标题“新一代端到端统一多模态大模型家族”。\n\n画面主体分为左、中、右三个相等的垂直信息区块，区块之间通过充足的负空间进行物理隔离。\n\n左侧区块的主题是概述。顶部有一个银灰色线框绘制的、由放大镜和齿轮交织的图标，旁边是粗体小标题“Overview”。该区块内从上到下垂直排列着三个要点：第一个要点旁边是一个代表文档与照片重叠的极简图标，紧跟着文字“多模态模型家族，统一文本/图像理解和生成”。向下是由两个相连的同心圆组成的架构图标，配有文字“基于NEO-Unify架构（端到端统一理解和生成）”。最下方是一个带有斜线划掉的眼睛和漏斗形状的图标，明确指示文本“无需视觉编码器(VE)和变分自编码器(VAE)”。\n\n中间区块展示模型矩阵。顶部是一个包含两个分支节点的树状网络图标，旁边是粗体小标题“两个模型版本”。区块内分为上下两个包裹在浅石板灰色极细边框内的卡片。上方的卡片内画着一个代表高密度的实心几何立方体图标，大字标注“SenseNova-U1-Mini”，下方是等宽字体说明“18B参数密集模型”。下方的卡片内画着一个带有闪电符号的网状发光大脑图标，大字标注“SenseNova-U1-Flash”，下方是等宽字体说明“38B参数，3B激活的混合专家(MoE)模型”。在这两个独立卡片的正下方，左侧放置一个笑脸轮廓图标搭配文字“将在HF等平台公开”，右侧放置一个带有折角的书面报告图标搭配文字“将发布技术报告”。\n\n右侧区块呈现核心优势。顶部是一个代表巅峰的上升阶梯折线图图标，旁边是粗体小标题“Highlights”。该区块内部垂直分布着四个带有浅石板灰底色的长方形色块，每个色块内部左侧对应一个具体的图标，右侧为文字。第一个色块内是一个无缝相连的莫比乌斯环图标，配文“原生统一架构，无VE和VAE”。第二个色块内是一个顶端带有星星的奖杯图标，配文“单一统一模型在理解和生成任务上均达到SOTA性能”。第三个色块内是代表文本行与拍立得照片交替穿插的图标，配文“强大的原生交错推理能力（模型原生生成图像进行推理）”。最后一个色块内是一个被切分出一小块的硬币与详细饼状图结合的图标，配文“能生成复杂信息图表，成本仅为商业模型的1/10”。" --width 2048 --height 2048 --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 --output output.png --profile
```

</details>

> Default resolution is 2048×2048 (1:1). See [supported resolution buckets](./examples/README.md#supported-resolution-buckets) for other aspect ratios.


<details>
<summary>✏️ Image Editing</summary>

```bash
python examples/editing/inference.py --model_path SenseNova/SenseNova-U1-Mini-Beta --prompt "Change the animal's fur color to a darker shade." --image examples/editing/data/images/1.jpg --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 --output output_edited.png --profile --compare
```

</details>

> 💡 Pre-resize inputs to ~2048×2048 resolution with orginal aspect ratio before inference for best quality (see [`examples/editing/resize_inputs.py`](./examples/editing/resize_inputs.py)).


<details>

<summary>♻️ Interleaved Generation</summary>

```bash
python examples/interleave/inference.py --model_path SenseNova/SenseNova-U1-Mini-Beta --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." --resolution "16:9" --output_dir outputs/interleave/ --stem demo --profile
```
</details>

> See [`examples/README.md`](./examples/README.md) for batched inference, JSONL format, prompt enhancement, resolution buckets, and full flag reference.


### ⚡ Run with LightLLM + LightX2V

For production serving, we co-design a dedicated inference stack on top of **[LightLLM](https://github.com/ModelTC/lightllm)** (understanding) and **[LightX2V](https://github.com/ModelTC/lightx2v)** (generation). The two engines are disaggregated so that each path can use its own parallelism and resource budget, with a low-overhead transfer channel in between.

On a single node with `TP2 + CFG2`, this stack delivers roughly **~0.15 s/step** and **~9 s end-to-end** for a **2048×2048** image on H100 / H200, with a ~**2.4–3.2×** prefill speedup from our FA3-based hybrid-mask attention over the Triton baseline. Full per-GPU performance are reported in [`docs/inference_infra.md`](./docs/inference_infra.md).

An official docker image is provided for one-command deployment:

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
```

> ⚙️ **Deployment guide (Docker, launch flags, modes, quantization, API test):** see [`docs/deployment.md`](./docs/deployment.md).
>
> 📖 **Full design and performance profiling:** see [`docs/inference_infra.md`](./docs/inference_infra.md).

<!-- ## 🖊️ Citation

```bibtex

``` -->

## ⚖️ License

This project is released under the [Apache 2.0 License](./LICENSE).
