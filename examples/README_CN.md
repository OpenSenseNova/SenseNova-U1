# 示例

本目录提供 SenseNova-U1 的参考推理脚本。所有脚本均刻意保持自包含——除 `sensenova_u1` 包本身外，仅依赖 `torch`、`transformers`、`pillow`、`numpy`（以及可选的 `tqdm` / `flash-attn`）。

每个任务位于独立的子目录下，并配有对应的 `data/` 示例输入目录：

```
examples/
├── README.md
├── t2i/                       # 文生图
│   ├── inference.py
│   └── data/
│       ├── samples.jsonl
│       ├── samples_reasoning.jsonl
│       └── samples_infographic.jsonl
├── editing/                   # 图像编辑（it2i）
│   ├── inference.py
│   ├── resize_inputs.py       # 离线预缩放工具（推荐）
│   └── data/
│       ├── samples.jsonl
│       ├── samples_reasoning.jsonl
│       ├── images/
│       └── images_reasoning/
├── interleave/                # 图文交错生成（可直接运行）
│   ├── inference.py
│   ├── run.sh
│   └── data/
│       ├── samples.jsonl
│       ├── samples_reasoning.jsonl
│       ├── images/
│       └── images_reasoning/
└── vqa/                       # 视觉理解 / VQA
    ├── inference.py
    └── data/
        ├── samples.jsonl
        └── images/
```

## 文生图（Text-to-Image）

单条 prompt 推理：

```bash
python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "这张信息图的标题是“SenseNova-U1”，采用现代极简科技矩阵风格。整体布局为水平三列网格结构，背景是带有极浅银灰色细密点阵的哑光纯白高级纸张纹理，画面长宽比为16:9。\n\n排版采用严谨的视觉层级：主标题使用粗体无衬线黑体字，正文使用清晰的现代等宽字体。配色方案极其克制，以纯白色为底，深炭黑为主视觉文字和边框，浅石板灰用于背景色块和次要信息区分，图标采用精致的银灰色线框绘制。\n\n在画面正上方居中位置，使用醒目的深炭黑粗体字排布着大标题“SenseNova-U1”。标题正下方是浅石板灰色的等宽字体副标题“新一代端到端统一多模态大模型家族”。\n\n画面主体分为左、中、右三个相等的垂直信息区块，区块之间通过充足的负空间进行物理隔离。\n\n左侧区块的主题是概述。顶部有一个银灰色线框绘制的、由放大镜和齿轮交织的图标，旁边是粗体小标题“Overview”。该区块内从上到下垂直排列着三个要点：第一个要点旁边是一个代表文档与照片重叠的极简图标，紧跟着文字“多模态模型家族，统一文本/图像理解和生成”。向下是由两个相连的同心圆组成的架构图标，配有文字“基于NEO-Unify架构（端到端统一理解和生成）”。最下方是一个带有斜线划掉的眼睛和漏斗形状的图标，明确指示文本“无需视觉编码器(VE)和变分自编码器(VAE)”。\n\n中间区块展示模型矩阵。顶部是一个包含两个分支节点的树状网络图标，旁边是粗体小标题“两个模型规格”。区块内分为上下两个包裹在浅石板灰色极细边框内的卡片。上方的卡片内画着一个代表高密度的实心几何立方体图标，大字标注“SenseNova-U1-8B-MoT”，下方是等宽字体说明“8B MoT 密集主干模型”。下方的卡片内画着一个带有闪电符号的网状发光大脑图标，大字标注“SenseNova-U1-A3B-MoT”，下方是等宽字体说明“A3B MoT 混合专家（MoE）主干模型”。在这两个独立卡片的正下方，左侧放置一个笑脸轮廓图标搭配文字“将在HF等平台公开”，右侧放置一个带有折角的书面报告图标搭配文字“将发布技术报告”。\n\n右侧区块呈现核心优势。顶部是一个代表巅峰的上升阶梯折线图图标，旁边是粗体小标题“Highlights”。该区块内部垂直分布着四个带有浅石板灰底色的长方形色块，每个色块内部左侧对应一个具体的图标，右侧为文字。第一个色块内是一个无缝相连的莫比乌斯环图标，配文“原生统一架构，无VE和VAE”。第二个色块内是一个顶端带有星星的奖杯图标，配文“单一统一模型在理解和生成任务上均达到SOTA性能”。第三个色块内是代表文本行与拍立得照片交替穿插的图标，配文“强大的原生交错推理能力（模型原生生成图像进行推理）”。最后一个色块内是一个被切分出一小块的硬币与详细饼状图结合的图标，配文“能生成复杂信息图表，性价比出色”。" \
  --width 2720 --height 1536 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output output.png \
  --profile
```

通过 JSONL 文件批量推理（每行必须包含 `prompt` 字段；`width` / `height` / `seed` 为可选字段）：

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

可参考 [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) 中的精简起步样例。完整参数列表请运行 `python examples/t2i/inference.py --help` 查看。

面向信息图（infographic）的批量生成示例：

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples_infographic.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

可参考 [`t2i/data/samples_infographic.jsonl`](./t2i/data/samples_infographic.jsonl) 复现信息图展示样例。

### T2I 推理模式（think mode）

模型支持在扩散去噪前先进行一段**推理**：会先自回归生成 `<think>...</think>`，随后再生成图像。

单条 prompt（输出图像 + 推理文本）：

```bash
python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "A male peacock trying to attract a female" \
  --width 2048 --height 2048 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --seed 42 \
  --think \
  --print_think \
  --output outputs/peacock.png
```

该命令会写出 `outputs/peacock.think.txt`，也支持用 `--think_output /path/to/reasoning.txt` 指定保存路径，或用 `--print_think` 直接打印到标准输出。

```bash
python examples/t2i/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples_reasoning.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --seed 42 \
    --think \
    --print_think \
    --profile
```

JSONL 模式下，可在单条样本里设置 `"think": true`（即使全局没传 `--think` 也会对该条启用）；也可以直接传全局 `--think` 对所有样本启用。

### 推荐分辨率档位

SenseNova-U1 在约 2K 像素的分辨率档位上训练。尽管支持任意的 `--width` / `--height`，但对未训练过的尺寸组合，生成质量可能有所下降。

| 宽高比         | 宽 × 高                  |
| :----------- | :------------------------ |
| 1:1          | 2048 × 2048               |
| 16:9 / 9:16  | 2720 × 1536 / 1536 × 2720 |
| 3:2 / 2:3    | 2496 × 1664 / 1664 × 2496 |
| 4:3 / 3:4    | 2368 × 1760 / 1760 × 2368 |
| 2:1 / 1:2    | 2880 × 1440 / 1440 × 2880 |
| 3:1 / 1:3    | 3456 × 1152 / 1152 × 3456 |

### 信息图场景的 Prompt 增强

对于较短的 prompt——特别是**信息图**生成——可以在推理前先用一个能力更强的 LLM 对 prompt 进行改写增强，从而显著提升画面信息密度、排版还原度以及布局的遵循程度。加上 `--enhance` 参数即可开启：

```bash
# export U1_ENHANCE_API_KEY=sk-...                # required
# defaults target Gemini 3.1 Pro via its OpenAI-compatible endpoint;
# override any of these to point at SenseNova / Claude / Kimi 2.5 etc.:
# export U1_ENHANCE_BACKEND=chat_completions   # or 'anthropic'
# export U1_ENHANCE_ENDPOINT=https://...chat/completions
# export U1_ENHANCE_MODEL=gemini-3.1-pro

python examples/t2i/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "如何制作咖啡的教程" \
  --enhance --print_enhance \
  --output output.png
```

详细说明请参见 [`docs/prompt_enhancement_CN.md`](../docs/prompt_enhancement_CN.md)。

## 图像编辑（it2i）

单次编辑：

```bash
python examples/editing/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "Change the jacket of the person on the left to bright yellow." \
  --image examples/editing/data/images/1.jpg \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited.png \
  --profile --compare
```

通过 JSONL 文件批量编辑（每行必须包含 `prompt` 以及 `image` 路径；`seed` / `type` 为可选字段；`image` 也可以是路径列表以传入多张参考图；若某行同时提供 `width` 和 `height`，则会覆盖该样本上命令行的默认值）：

```bash
python examples/editing/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 \
    --profile --compare
```

输出分辨率共支持两种模式：

- **自动模式（默认）**：不传 `--width / --height` 时，输出分辨率跟随第一张输入图，通过 `smart_resize` 保持宽高比，总像素数归一化到 `--target_pixels`（默认 `2048 * 2048`），且 H / W 均对齐到 32 的倍数。
- **显式指定**：传入 `--width W --height H`（二者均须为 32 的倍数）。2048 × 2048 是一个通用场景下的稳妥选择。

为了获得更好的编辑质量，建议尽量提供高分辨率输入/参考图。
显存有限时可使用 `--input_max_pixels auto`：1-2 张图保持单图最高2048 × 2048，超过 2 张时把两张 2048 × 2048 的总预算均分给所有输入图。
这个策略是基于“两张 2048 × 2048 参考图”的显存预算给出的保守默认值；
可根据自己的显卡显存和编辑效果调节，也可传入整数，例如 `1048576` 表示单张输入图最高约 1024 × 1024。
外层 resize-to-budget 默认开启；可传 `--no-do-resize` 跳过这一步，但模型原生图像预处理仍会应用自己的输入尺寸限制。
脚本会在生成前打印每张输入图的上限。

CFG 默认值：`--cfg_scale 4.0`（文本引导强度），`--img_cfg_scale 1.0`（默认关闭图像 CFG）。完整参数列表请运行 `python examples/editing/inference.py --help` 查看。

可参考 [`editing/data/samples.jsonl`](./editing/data/samples.jsonl) 中的精简起步样例。


## 图文交错生成（Interleave）

`examples/interleave/inference.py` 调用的是 `model.interleave_gen`，可在**单次响应中交错生成文本与图像**。模型会先输出一段 `<think>...</think>` 的推理块，在其中生成中间图像，最后再给出简洁的答复。涵盖以下三种用法的启动脚本见 [`interleave/run.sh`](./interleave/run.sh)。

**输出文件：** 每个样本会写出 `<stem>.txt`（生成的文本）以及对应每张图像的 `<stem>_image_<i>.png`；在 `--jsonl` 模式下还会额外生成一份 `results.jsonl` 清单。

**分辨率：** 若通过 `--image` 或 JSONL 中的 `image` 字段提供了输入图像，输出分辨率会跟随第一张输入图（经 `smart_resize` 对齐到 32 的倍数），并覆盖 `--resolution` / `--width` / `--height` 的设置。

### 1) 单样本，仅文本 prompt
```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

### 2) 单样本，文本 prompt + 输入图像

```bash
python examples/interleave/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --prompt "<image>\n图文交错生成小猫游览故宫的场景" \
  --image examples/interleave/data/images/image0.jpg \
  --output_dir outputs/interleave/text_image \
  --stem demo_text_image
```

### 3) 通过 JSONL 批量推理

每行代表一个样本：

```json
{"prompt": "..."}
{"prompt": "...", "image": ["a.jpg"]}
```

```bash
python examples/interleave/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/interleave/data/samples.jsonl \
    --resolution "16:9" \
    --output_dir outputs/interleave/jsonl
```

[`interleave/data/samples.jsonl`](./interleave/data/samples.jsonl) 提供了一份包含两条样本（一条纯文本、一条图像条件）的起步文件。

## 视觉理解（VQA）

单图问答，启用采样：

```bash
python examples/vqa/inference.py \
  --model_path SenseNova/SenseNova-U1-8B-MoT \
  --image examples/vqa/data/images/menu.jpg \
  --question "My friend and I are dining together tonight. Looking at this menu, can you recommend a good combination of dishes for 2 people? We want a balanced meal — a mix of mains and maybe a starter or dessert. Budget-conscious but want to try the highlights." \
  --output outputs/menu_answer.txt \
  --max_new_tokens 8192 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.05 \
  --profile
```

如果省略 `--do_sample`（连同相关的采样参数），则切换为确定性的贪婪解码。

通过 JSONL 文件批量问答（每行必须包含 `image` 与 `question` 字段；`id` 为可选）：

```bash
python examples/vqa/inference.py \
    --model_path SenseNova/SenseNova-U1-8B-MoT \
    --jsonl examples/vqa/data/samples.jsonl \
    --output_dir outputs/vqa/ \
    --max_new_tokens 8192 \
    --do_sample \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --repetition_penalty 1.05 \
    --profile
```

结果会写入 `outputs/vqa/answers.jsonl`，每行一个 JSON 对象，包含 `id`、`image`、`question` 和 `answer` 字段。

起步样例可参考 [`vqa/data/samples.jsonl`](./vqa/data/samples.jsonl)。

### 生成参数

| Flag | Default | 说明 |
| :--- | :------ | :---------- |
| `--max_new_tokens` | 1024 | 生成响应的最大长度 |
| `--do_sample` | off (greedy) | 启用采样 |
| `--temperature` | 0.7 | 采样温度（需配合 `--do_sample` 使用） |
| `--top_p` | 0.9 | 核采样（Nucleus）阈值（需配合 `--do_sample` 使用） |
| `--top_k` | None | Top-k 采样（需配合 `--do_sample` 使用） |
| `--repetition_penalty` | None | 重复惩罚系数 |

完整参数列表请运行 `python examples/vqa/inference.py --help` 查看。
