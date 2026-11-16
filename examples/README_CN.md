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

## 单卡显存模式

U1.5 单卡推理先用 `--vram_mode balanced`；省略这个参数时，CLI 仍使用 `full`。四个参考脚本（t2i / interleave / editing / vqa）都支持下表模式。`balanced` / `low` 会占用较多系统内存：U1.5 的 RTX 5090 测试中，进程 CPU RSS 峰值约 52.2 GiB。可先按 64 GB 系统内存规划，具体测试范围和显存数据见[U1.5 显存与耗时实测](../docs/u1.5_inference_CN.md)。

| `--vram_mode` | 行为 |
|---------------|------|
| `full`        | 整模型常驻 GPU，不做 offload。（默认）|
| `fast`        | 异步预取，并在 GPU 显存预算内常驻 generation 层。|
| `low`         | 每层同步 CPU<->GPU 交换，降低权重显存占用。|
| `balanced`    | 异步预取，尽量让数据传输与计算重叠。U1.5 的推荐起始模式。|

四个脚本共用以下 `fast` 常驻参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fast_vram_fraction` | `0.90` | 物理显存比例形式的自动预算。|
| `--fast_vram_headroom_gib` | `2` | 为后续分配保留的可复用显存余量。|
| `--fast_activation_reserve_gib` | `4` | decoder 后续激活预留。|
| `--fast_vram_budget_gib` | 未设置 | 覆盖 fraction 的绝对预算。|

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A cinematic mountain village at sunrise" \
  --vram_mode balanced \
  --output output.png
```

## 文生图（Text-to-Image）

单条 prompt 推理：

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "一张四步手冲咖啡信息图，浅色背景，标题为「手冲咖啡」" \
  --width 2720 --height 1536 \
  --vram_mode balanced \
  --batch_size 1 \
  --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
  --output output.png \
  --profile
```

旧版信息图的[原始提示词](t2i/data/u1_infographic_prompt.txt)保留作案例输入，其中的发布文案不代表当前版本。

通过 JSONL 文件批量推理（每行必须包含 `prompt` 字段；`width` / `height` / `seed` 为可选字段）：

```bash
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

可参考 [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) 中的精简起步样例。完整参数列表请运行 `python examples/t2i/inference.py --help` 查看。

面向信息图（infographic）的批量生成示例：

```bash
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
    --jsonl examples/t2i/data/samples_infographic_V2.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile
```

可参考 [`t2i/data/samples_infographic_V2.jsonl`](./t2i/data/samples_infographic_V2.jsonl) 运行信息图展示 prompt。模型细节可见 [Infographic Model](../docs/u1_infographic_model_CN.md)。

信息图 8-step LoRA 批量生成（V1.0 LoRA）

```bash
hf download sensenova/SenseNova-U1-8B-MoT-LoRAs \
  SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors \
  --local-dir ./models/SenseNova-U1-8B-MoT-LoRAs

python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT-Infographic \
    --lora_path ./models/SenseNova-U1-8B-MoT-LoRAs/SenseNova-U1-8B-MoT-Infographic-LoRA-8step-V1.0.safetensors \
    --jsonl examples/t2i/data/samples_infographic.jsonl \
    --output_dir outputs/ \
    --cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 8 \
    --profile
```

上述 8-step 信息图 LoRA 面向 `SenseNova-U1-8B-MoT-Infographic` 发布，也可搭配 `SenseNova-U1-8B-MoT-Infographic-V2` 使用以获得更快生成速度，但可能带来美观度下降、文字模糊或重复等问题。追求最佳质量时，建议使用完整 V2 checkpoint。

### T2I 推理模式（think mode）

模型支持在扩散去噪前先进行一段**推理**：会先自回归生成 `<think>...</think>`，随后再生成图像。

单条 prompt（输出图像 + 推理文本）：

```bash
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
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
    --model_path sensenova/SenseNova-U1-8B-MoT \
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

`--enhance` 会先调用外部模型扩写提示词，再将结果交给 U1 生成图片。此步骤需要 API 密钥，也会增加一次请求。后端配置见[提示词增强说明](../docs/prompt_enhancement_CN.md)。

```bash
export U1_ENHANCE_API_KEY="your-api-key"

python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "如何制作咖啡的教程" \
  --enhance --print_enhance \
  --output output.png
```

## 图像编辑（it2i）

单次编辑：

```bash
python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "Change the jacket of the person on the left to bright yellow." \
  --image examples/editing/data/images/1.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited.png \
  --profile --compare
```

通过 JSONL 文件批量编辑（每行必须包含 `prompt` 以及 `image` 路径；`seed` / `type` 为可选字段；`image` 也可以是路径列表以传入多张参考图；若某行同时提供 `width` 和 `height`，则会覆盖该样本上命令行的默认值）：

```bash
python examples/editing/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
    --timestep_shift 3.0 --num_steps 50 \
    --profile --compare
```

### 信息图编辑

使用 `SenseNova-U1-8B-MoT-Infographic-V3` 进行信息图编辑：

```bash
python examples/editing/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT-Infographic-V3 \
  --prompt "将信息图的主标题修改为‘SenseNova-U1 V3’，保持其他内容和布局不变。" \
  --image docs/assets/showcases/t2i_infographic/0004.webp \
  --cfg_scale 4.0 --img_cfg_scale 1.0 --cfg_norm none \
  --timestep_shift 3.0 --num_steps 50 \
  --output edited_infographic.png \
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

用 `--max_images 1` 可将单次生成限制为最多一张图（默认上限为 10 张），便于比较单图推理耗时。提示词仍有可能只返回文本而不出图。

**分辨率：** 若通过 `--image` 或 JSONL 中的 `image` 字段提供了输入图像，输出分辨率会跟随第一张输入图（经 `smart_resize` 对齐到 32 的倍数），并覆盖 `--resolution` / `--width` / `--height` 的设置。

### 1) 单样本，仅文本 prompt
```bash
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/text \
  --stem demo_text
```

### 2) 单样本，文本 prompt + 输入图像

```bash
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
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
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/interleave/data/samples.jsonl \
    --resolution "16:9" \
    --output_dir outputs/interleave/jsonl
```

[`interleave/data/samples.jsonl`](./interleave/data/samples.jsonl) 提供了一份包含两条样本（一条纯文本、一条图像条件）的起步文件。

## 视觉理解（VQA）

单图问答，启用采样：

```bash
python examples/vqa/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
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
    --model_path sensenova/SenseNova-U1-8B-MoT \
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
