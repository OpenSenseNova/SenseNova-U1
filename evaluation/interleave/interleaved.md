# interleave_inference

用于运行多模态 Benchmark 推理脚本。

## 目录与执行方式

当前目录：

```text
evaluation/interleave
```

脚本列表：

- `BabyVision/infer_babyvision.py`
- `OpenING/infer_opening.py`
- `Unimmmu/inference_unimmmu.py`
- `Realunify/inference_realunify.py`
- `Realunify/inference_realunify_ueg.py`

除非特别说明，以下命令均默认从仓库根目录进入本目录后执行：

```bash
cd evaluation/interleave
```

如果从其他目录执行，请将文中的相对路径替换为本地环境中的可访问路径。

## 使用前提

运行前请确认：

1. 当前工作目录为 `evaluation/interleave`。
2. API 类 Benchmark 需要可访问的服务地址。
3. 本地模型类 Benchmark 需要可访问的 `--model_path` 与数据文件路径。
4. 建议显式指定关键路径和服务地址，不依赖环境相关默认值。

## 依赖

API 后端模式（BabyVision / OpenING）：

```bash
python3 -m pip install requests regex tqdm
```

如需本地图片重采样，额外安装：

```bash
python3 -m pip install pillow
```

本地模型加载模式（Unimmmu / RealUnify）：

```bash
python3 -m pip install torch torchvision transformers pillow numpy tqdm
```

## 环境检查

可先执行以下命令确认脚本能够启动并完成参数解析：

```bash
python3 BabyVision/infer_babyvision.py --help
python3 OpenING/infer_opening.py --help
python3 Unimmmu/inference_unimmmu.py --help
python3 Realunify/inference_realunify.py --help
python3 Realunify/inference_realunify_ueg.py --help
```

如出现 `ModuleNotFoundError`，请先安装依赖。

## 默认值说明

文中涉及两类默认值：

- 代码默认值：脚本中 `argparse` 或环境变量定义的默认值
- 推荐运行值：为减少环境差异，建议显式传入的参数

对于本地模型类脚本，`--data_path` 的代码默认值通常为占位路径，实际运行时应显式传入。

## BabyVision

### 代码默认值

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

### 推荐运行命令

默认位置运行：

```bash
python3 BabyVision/infer_babyvision.py
```

显式指定关键参数：

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

### 输出与行为

- 输出文件：`babyvision_<model_name>.jsonl`
- 主要字段：`taskId`、`type`、`subtype`、`ansType`、`question`、`answer`、`model`、`model_response`、`extracted_answer`
- 支持断点续跑；已有结果会按 `taskId` 跳过
- 输入图片缺失、请求失败或输出损坏时，脚本返回非零退出码

## OpenING

### 注意事项

- `infer_opening.py` 的部分代码默认值依赖特定环境，不建议直接依赖
- 运行 OpenING Benchmark 时，应显式指定 `--mode opening`
- 建议显式传入 `--api_backend`、`--meta_path`、`--data_file_name`、`--save_dir`

### 代码默认值

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
  "opening_step_prompt_style": "none",
  "meta_path": "./OpenING-benchmark",
  "data_file_name": "test_data.jsonl",
  "save_dir": "./opening_results",
  "output_dir": "./neo_chat_images"
}
```

### 推荐运行命令

OpenAI 兼容接口：

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

`/generate` 后端：

```bash
export LIGHTLLM_GENERATE_URLS=http://127.0.0.1:8000/generate
export LIGHTLLM_MODEL=local-model

python3 OpenING/infer_opening.py \
  --mode opening \
  --api_backend generate \
  --meta_path /path/to/OpenING-benchmark \
  --data_file_name test_data.jsonl \
  --save_dir ./opening_results
```

附加参数示例：

- 仅运行前 20 条：`--limit 20`
- 分片运行：`--num_shards 8 --shard_index 0`
- 覆盖已有结果：`--overwrite`
- 关闭占位图逻辑：`--disable-generate-placeholder-images`

### 常用环境变量

```bash
export LIGHTLLM_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
export LIGHTLLM_MODEL=local-model
export LIGHTLLM_GENERATE_URLS=http://127.0.0.1:8000/generate
export LIGHTLLM_IMAGE_OUT=./generated_images
export OPENING_META_PATH=./OpenING-benchmark
export OPENING_DATA_FILE_NAME=test_data.jsonl
export OPENING_SAVE_DIR=./opening_results
export OPENING_MAX_TOKENS=4096
export OPENING_IMAGE_WIDTH=1920
export OPENING_IMAGE_HEIGHT=1088
export OPENING_ENABLE_THINKING=true
export OPENING_GENERATE_PLACEHOLDER_IMAGES=true
```

### 输入、输出与行为

输入示例：

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

- 输入使用 `conversations[0].input`
- 完整性检查基于 `conversations[1].output`
- 每条样本输出为 `<save_dir>/<total_uid>.json`
- 如生成图片，文件名示例为 `<save_dir>/<total_uid>-o_0.jpg`
- `--api_backend generate` 仅支持 `--mode stream_interleave` 和 `--mode opening`

## Unimmmu

### 推理模式

- `i2t`：多图理解，仅输出文本
- `interleave`：多模态推理，同时输出文本和图片

### 参数与默认值

- `--model_path` 为必填参数
- `--data_path` 的代码默认值为占位路径，实际运行时应显式传入

默认推理参数：

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

### 推荐运行命令

单 GPU：

```bash
python3 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --num_steps 50
```

多 GPU：

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

`device_map` 模式：

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

分片合并：

```bash
python3 Unimmmu/merge_shards.py \
  --data_path /path/to/unimmmu_direct.jsonl \
  --shard_dir ./output/unimmmu_interleave/shards \
  --output_file ./output/unimmmu_interleave/unimmmu_results.jsonl
```

### 输出

- 结果文件：`unimmmu_results.jsonl`
- 主要字段：`hash_uid`、`task`、`model_response`、`inference_mode`、`generated_images`

## RealUnify (GEU)

### 推理模式

- `step`：先编辑图片，再回答问题
- `interleave`：直接进行多模态推理

### 参数与默认值

- `--model_path` 为必填参数
- `--data_path` 的代码默认值为占位路径，实际运行时应显式传入

默认推理参数：

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

### 推荐运行命令

单 GPU：

```bash
python3 Realunify/inference_realunify.py \
  --model_path /path/to/model \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
  --num_steps 50
```

多 GPU：

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

`device_map` 模式：

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

如需固定输出图片尺寸，可增加：

```text
--target_image_size 1024
```

### 输出

- 结果文件：`realunify_results.jsonl`
- 主要字段：`hash_uid`、`task_type`、`model_response`、`answer`、`generated_image`、`generated_images`

## RealUnify (UEG)

### 推理模式

- `understand_t2i`：先理解并精炼 prompt，再生成图片
- `interleave`：直接进行 interleave 生成
- `t2i`：直接使用预处理后的 prompt 生成图片

### 参数说明

- `--model_path`、`--output_dir`、`--inference_mode` 为必填参数
- `--data_path` 的代码默认值为占位路径，实际运行时应显式传入

### 推荐运行命令

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

### 输出

- 输出文件：`ueg_results.jsonl`、`ueg_results.json`
- 主要字段：`index`、`task_type`、`generated_image`、`question_list`

## 评测流程概览

典型流程如下：

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

以上三个 Benchmark 之间无依赖，可并行执行。

## 退出码

- `0`：全部成功，或仅有显式跳过
- `1`：启动校验失败，或运行过程中存在样本失败

## 常见问题

- 缺少 `requests`、`regex`、`tqdm`：请先安装依赖
- 数据文件不存在：检查 `--data_path`、`--meta_path`、`--data_file_name` 或对应环境变量
- 本地模型脚本读取到 `<DATA_ROOT>/...`：说明命中了占位默认值，请显式传入 `--data_path`
- 输入图片缺失：检查数据中的相对路径和数据根目录设置
- 结果被跳过：通常是输出已存在；可结合 `--overwrite` 或 `--resume` 处理
