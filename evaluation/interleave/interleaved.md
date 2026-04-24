# interleave_inference

用于运行多模态 Benchmark 的推理脚本。

本文档用于说明各基准测试脚本的目录结构、依赖、默认参数及推荐运行方式。

## 目录结构

当前目录是：

```text
evaluation/interleave
```

实际脚本位于以下子目录中：

- `BabyVision/infer_babyvision.py`
- `OpenING/infer_opening.py`
- `Unimmmu/inference_unimmmu.py`
- `Realunify/inference_realunify.py`
- `Realunify/inference_realunify_ueg.py`

除非特别说明，下面所有命令都默认从仓库根目录进入 `evaluation/interleave` 后执行：

```bash
cd evaluation/interleave
```

如果从其他目录执行，请将文中的仓库相对路径替换为本地环境中的可访问路径。

## 使用前提

运行前请确认以下事项：

1. 你所在目录是 `evaluation/interleave`，否则相对路径可能解析错。
2. API Benchmark 需要可访问的服务地址；本地模型 Benchmark 需要可访问的 `--model_path` 和数据文件路径。
3. 部分代码默认值依赖特定环境中的内网地址或占位路径，不建议作为通用默认配置直接使用。推荐通过命令行参数或环境变量显式指定关键路径和服务地址。

## 依赖

BabyVision / OpenING（API 后端模式）：

```bash
python3 -m pip install requests regex tqdm
```

如果本地启用了图片重采样，BabyVision 额外安装：

```bash
python3 -m pip install pillow
```

Unimmmu / RealUnify（本地模型加载）：

```bash
python3 -m pip install torch torchvision transformers pillow numpy tqdm
```

## 环境检查

如需确认脚本能够正常启动并完成参数解析，可先执行：

```bash
python3 BabyVision/infer_babyvision.py --help
python3 OpenING/infer_opening.py --help
python3 Unimmmu/inference_unimmmu.py --help
python3 Realunify/inference_realunify.py --help
python3 Realunify/inference_realunify_ueg.py --help
```

如果执行过程中出现 `ModuleNotFoundError`，请先完成依赖安装，再执行后续推理任务。

## 默认值说明

文中区分两种“默认值”：

- 代码默认值：脚本里 `argparse` / 环境变量真正写死的默认值
- 推荐运行值：为减少环境差异带来的歧义，建议显式传入的参数

建议优先参考“推荐运行命令”，避免过度依赖代码默认值。

## BabyVision

### 脚本位置

```text
BabyVision/infer_babyvision.py
```

### 代码默认值

`infer_babyvision.py` 的代码默认值如下：

```json
{
  "data_path": "./babyvision_data/meta_data.jsonl",
  "image_root": "./babyvision_data",
  "output_dir": "./babyvision_results",
  "generate_urls": "http://127.0.0.1:8000/generate",
  "model_name": "local-model",
  "workers": 32,
  "max_retries": 3,
  "backend_max_retries": 20,
  "request_timeout": 600,
  "max_new_tokens": 32768,
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "min_pixels": 2097152,
  "max_pixels": 16777216
}
```

上述相对路径均相对于当前工作目录，建议从 `evaluation/interleave` 目录执行。

### 推荐运行命令

如果数据已放置在默认位置，且本地已启动 `/generate` 服务：

```bash
python3 BabyVision/infer_babyvision.py
```

推荐显式传入关键路径和服务地址：

```bash
python3 BabyVision/infer_babyvision.py \
  --data-path /path/to/meta_data.jsonl \
  --image-root /path/to/babyvision_images \
  --output-dir ./babyvision_results \
  --generate-urls http://127.0.0.1:8000/generate
```

### 常用环境变量

```bash
export BABYVISION_MODEL_NAME=local-model
export BABYVISION_DATA_PATH=./babyvision_data/meta_data.jsonl
export BABYVISION_IMAGE_ROOT=./babyvision_data
export BABYVISION_OUTPUT_DIR=./babyvision_results
export BABYVISION_GENERATE_URLS=http://127.0.0.1:8000/generate
export BABYVISION_WORKERS=32
export BABYVISION_MAX_RETRIES=3
export BABYVISION_BACKEND_MAX_RETRIES=20
export BABYVISION_REQUEST_TIMEOUT=600
```

### 输出格式

输出为 JSONL，每行一条样本结果，文件名格式：

```text
babyvision_<model_name>.jsonl
```

每条结果主要字段包括：

- `taskId`
- `type`
- `subtype`
- `ansType`
- `question`
- `answer`
- `model`
- `model_response`
- `extracted_answer`

### 运行行为

- 选择题会自动整理为 `(A) (B) (C)` 形式。
- 最终答案从模型输出中的 `<answer>...</answer>` 提取。
- 支持断点续跑；如果结果文件已存在，会按 `taskId` 跳过已完成样本。
- 若输入图片缺失、请求重试耗尽、已有输出 JSONL 损坏，脚本会失败并返回非零退出码。
- 执行结束后会打印失败汇总，便于批处理排查。

## OpenING

### 脚本位置

```text
OpenING/infer_opening.py
```

### 注意事项

OpenING 脚本包含一组代码默认值，但其中部分默认值依赖特定环境中的内网地址和绝对路径，不适合作为通用默认配置直接使用。

建议如下：

- 在具备相同环境配置时，可以复用这些默认值
- 在其他环境中，建议显式传入 `--mode`、`--api_backend`、`--meta_path`、`--data_file_name`、`--save_dir`，并通过环境变量或命令行传入服务地址

### 代码默认值

`infer_opening.py` 的代码默认值如下：

```json
{
  "mode": "stream_interleave",
  "api_backend": "generate",
  "model": "neo_chat",
  "temperature": 0.8,
  "top_p": 0.95,
  "max_tokens": 4096,
  "request_timeout": 600,
  "generate_max_retries": 20,
  "generate_placeholder_images": true,
  "seed": 200,
  "stream_seed": 500,
  "image_aspect_ratio": "16:9",
  "image_size": "1K",
  "image_type": "jpeg",
  "image_width": 1920,
  "image_height": 1088,
  "enable_thinking": true,
  "num_shards": 1,
  "shard_index": 0,
  "parallel_requests": 4,
  "retry_short_outputs": 2,
  "opening_step_prompt_style": "none"
}
```

另外，脚本中的这些环境默认值通常也会影响运行：

- `LIGHTLLM_BASE_URL`
- `OPENAI_API_KEY`
- `LIGHTLLM_MODEL`
- `LIGHTLLM_GENERATE_URLS`
- `LIGHTLLM_IMAGE_OUT`
- `OPENING_META_PATH`
- `OPENING_DATA_FILE_NAME`
- `OPENING_SAVE_DIR`

其中，仓库当前代码中的 `LIGHTLLM_BASE_URL`、`LIGHTLLM_GENERATE_URLS`、`OPENING_META_PATH`、`OPENING_SAVE_DIR` 为环境相关默认值，建议在实际运行时显式覆盖。

### 推荐运行命令

运行 OpenING Benchmark 时，需要显式指定：

```text
--mode opening
```

使用 OpenAI 兼容接口：

```bash
export LIGHTLLM_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
export LIGHTLLM_MODEL=local-model

python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend openai \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results
```

使用 `/generate` 后端：

```bash
export LIGHTLLM_GENERATE_URLS=http://127.0.0.1:8000/generate
export LIGHTLLM_MODEL=local-model

python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend generate \
  --parallel_requests 4 \
  --enable_thinking \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_tokens 4096 \
  --request_timeout 600 \
  --seed 200 \
  --stream_seed 500 \
  --image_aspect_ratio 16:9 \
  --image_size 1K \
  --image_type jpeg \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results
```

只跑前 20 条：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend generate \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results \
  --limit 20
```

分片运行：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend generate \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results \
  --num_shards 8 \
  --shard_index 0
```

覆盖已有结果：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend generate \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results \
  --overwrite
```

### 占位图逻辑

`generate` 后端默认开启占位图逻辑。

这意味着：

- 如果后端只返回图片标记但没有真实图片数据，脚本会按需补 1x1 PNG 占位图
- OpenING 模式下若文本或图像步数仍然明显不足，任务依然会失败

如需禁用占位图逻辑：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --disable-generate-placeholder-images
```

### 常用环境变量

下面这组更适合作为通用模板：

```bash
export LIGHTLLM_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
export LIGHTLLM_MODEL=local-model
export LIGHTLLM_GENERATE_URLS=http://127.0.0.1:8000/generate
export LIGHTLLM_IMAGE_OUT=./generated_images
export OPENING_META_PATH=/path/to/OpenING-benchmark
export OPENING_DATA_FILE_NAME=test_data.jsonl
export OPENING_SAVE_DIR=./opening_results
export OPENING_MAX_TOKENS=4096
export OPENING_IMAGE_WIDTH=1920
export OPENING_IMAGE_HEIGHT=1088
export OPENING_ENABLE_THINKING=true
export OPENING_GENERATE_PLACEHOLDER_IMAGES=true
```

### 输入格式

OpenING 读取 JSONL。单条样本示例：

```json
{
  "total_uid": "sample_001",
  "conversations": [
    {
      "input": [
        {"text": "step 1 input", "image": "./images/a.jpg"}
      ]
    },
    {
      "output": [
        {"text": "step 1 output", "image": "./images/b.jpg"}
      ]
    }
  ]
}
```

脚本会使用：

- `conversations[0].input` 作为模型输入
- `conversations[1].output` 的步数和图片需求作为完整性检查依据

### 输出格式

每条样本输出一个 JSON：

```text
<save_dir>/<total_uid>.json
```

如果模型生成了图片，还会保存图片文件，例如：

```text
<save_dir>/<total_uid>-o_0.jpg
<save_dir>/<total_uid>-o_1.png
```

### 运行行为

- `--api_backend generate` 目前只支持 `--mode stream_interleave` 和 `--mode opening`。
- OpenING 会根据 GT 的输出 step 数和图片需求检查结果是否完整，不够时会重试。
- 输入图像缺失、参数非法、数据文件不存在、样本推理失败时，脚本会返回非零退出码。
- 执行结束后会打印成功/跳过/失败统计和失败汇总。

## Unimmmu

### 脚本位置

```text
Unimmmu/inference_unimmmu.py
```

### 推理模式

- `i2t`：多图理解（`model.chat`），只产出文字
- `interleave`：多模态推理（`model.interleave_gen`），同时产出文字和图片

### 参数说明

`--model_path` 是必填参数。

`--data_path` 虽然在代码里有默认值，但默认值是占位路径：

```text
<DATA_ROOT>/unimmmu/vqa/unimmmu_direct.jsonl
```

因此，实际运行时应显式传入 `--data_path`。

### 单 GPU

```bash
python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_i2t \
  --inference_mode i2t \
  --limit 5
```

```bash
python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50 \
  --limit 5
```

### 多 GPU（torchrun）

```bash
torchrun --nproc_per_node=2 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50
```

### 大模型 device_map 模式

当模型放不进单卡时，使用 `--device_map auto` 让 HuggingFace 自动分配多卡：

```bash
python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --device_map auto \
  --max_memory_per_gpu_gb 60 \
  --cfg_scale 4.0 \
  --num_steps 50
```

如需启用多进程分片，可配合 `--num_shards` / `--shard_rank` 使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --device_map auto \
  --max_memory_per_gpu_gb 60 \
  --num_shards 4 \
  --shard_rank 0
```

```bash
CUDA_VISIBLE_DEVICES=2,3 python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --device_map auto \
  --max_memory_per_gpu_gb 60 \
  --num_shards 4 \
  --shard_rank 1
```

分片任务全部完成后，可执行以下命令合并结果：

```bash
python3 Unimmmu/merge_shards.py \
  --data_path /path/to/unimmmu_direct.jsonl \
  --shard_dir ./output/unimmmu_interleave/shards \
  --output_file ./output/unimmmu_interleave/unimmmu_results.jsonl
```

### 断点续跑

```bash
python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --resume
```

### 默认推理参数

```json
{
  "min_pixels": 1048576,
  "max_pixels": 4194304,
  "cfg_scale": 4.0,
  "img_cfg_scale": 1.0,
  "cfg_interval": [0.0, 1.0],
  "cfg_norm": "none",
  "num_steps": 50,
  "timestep_shift": 3.0,
  "seed": 42
}
```

### 输出格式

输出为 JSONL（`unimmmu_results.jsonl`），每行一条样本。主要字段：

- `hash_uid`
- `task`
- `model_response`
- `inference_mode`
- `generated_images`

### 评分

推理产出的 JSONL 请使用 Unimmmu 官方 benchmark 评测工具进行打分。

## RealUnify (GEU)

### 脚本位置

```text
Realunify/inference_realunify.py
```

### 推理模式

- `step`：两步流水线，先编辑图片（it2i），再回答问题（i2t）
- `interleave`：直接使用多模态推理能力，模型内部完成编辑和回答

### 参数说明

`--model_path` 是必填参数。

`--data_path` 虽然在代码里有默认值，但默认值是占位路径：

```text
<DATA_ROOT>/RealUnify/GEU_step_processed.jsonl
```

因此，实际运行时应显式传入 `--data_path`。

### 单 GPU

```bash
python3 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_step \
  --inference_mode step \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50 \
  --limit 5
```

```bash
python3 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
  --num_steps 50 \
  --limit 5
```

### 多 GPU（torchrun）

```bash
torchrun --nproc_per_node=2 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
  --num_steps 50
```

### 大模型 device_map 模式

```bash
python3 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --device_map auto \
  --max_memory_per_gpu_gb 60 \
  --cfg_scale 4.0 \
  --num_steps 50
```

如需启用多进程分片，可配合 `--num_shards` / `--shard_rank` 使用；全部完成后使用 `Realunify/merge_shards.py` 合并结果。

### 固定输出图片尺寸

默认使用 smart_resize 自适应。如需强制指定正方形输出：

```bash
python3 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --target_image_size 1024 \
  --device_map auto
```

### 默认推理参数

```json
{
  "min_pixels": 1048576,
  "max_pixels": 4194304,
  "cfg_scale": 4.0,
  "img_cfg_scale": 1.0,
  "cfg_interval": [0.0, 1.0],
  "cfg_norm": "none",
  "num_steps": 50,
  "timestep_shift": 3.0,
  "seed": 42
}
```

### 输出格式

输出为 JSONL（`realunify_results.jsonl`）。主要字段：

- `hash_uid`
- `task_type`
- `model_response`
- `answer`
- `generated_image`
- `generated_images`

### 评分

推理产出的 JSONL 请使用 RealUnify 官方 benchmark 评测工具进行打分。

## RealUnify (UEG)

### 脚本位置

```text
Realunify/inference_realunify_ueg.py
```

### 推理模式

- `understand_t2i`：先用文字理解精炼 prompt，再 T2I 生成图片
- `interleave`：直接 interleave 生成
- `t2i`：直接用预处理后的 prompt 做 T2I 生成

### 参数说明

`--model_path`、`--output_dir`、`--inference_mode` 都是必填参数。

`--data_path` 虽然在代码里有默认值，但默认值是占位路径：

```text
<DATA_ROOT>/RealUnify/UEG_step.json
```

因此，实际运行时应显式传入 `--data_path`。

### 快速开始

```bash
python3 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_understand_t2i \
  --inference_mode understand_t2i \
  --cfg_scale 4.0 \
  --num_steps 50
```

```bash
python3 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --timestep_shift 3.0 \
  --num_steps 50
```

```bash
python3 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_t2i \
  --inference_mode t2i \
  --cfg_scale 4.0 \
  --num_steps 50
```

### 多 GPU（torchrun）

```bash
torchrun --nproc_per_node=2 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --num_steps 50
```

### 输出格式

同时输出 JSONL（`ueg_results.jsonl`）和 JSON（`ueg_results.json`）。主要字段：

- `index`
- `task_type`
- `generated_image`
- `question_list`

### 评分

推理产出的 JSON / JSONL 请使用 RealUnify 官方 benchmark 评测工具进行打分。

## 评测流程概览

典型的完整评测流程如下：

```bash
MODEL_PATH=/path/to/hf_model

torchrun --nproc_per_node=2 Realunify/inference_realunify.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50

python3 Realunify/inference_realunify_ueg.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_understand_t2i \
  --inference_mode understand_t2i \
  --cfg_scale 4.0 \
  --num_steps 50

torchrun --nproc_per_node=2 Unimmmu/inference_unimmmu.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50
```

以上三个 benchmark 之间无依赖，可并行执行。

## 退出码

- `0`：全部成功，或仅有显式跳过
- `1`：启动校验失败，或运行过程中存在样本失败

## 常见问题

- 缺少 `requests`、`regex`、`tqdm`：先安装依赖。
- BabyVision 报数据文件不存在：检查 `--data-path` 或 `BABYVISION_DATA_PATH`。
- OpenING 报数据文件不存在：检查 `--meta_path`、`--data_file_name` 或对应环境变量。
- 本地模型脚本直接读取到 `<DATA_ROOT>/...`：说明你没有显式传 `--data_path`，而是命中了占位默认值。
- 输入图片缺失：检查 JSON / JSONL 里的相对路径，以及 `meta_path`、`image_root`、数据根目录是否正确。
- 结果被跳过：通常是输出已存在。OpenING 可加 `--overwrite`，BabyVision / Unimmmu / RealUnify 可结合已有输出和 `--resume` 处理。
