# interleave_inference

用于运行多模态 Benchmark 的推理与评测脚本。

## 目录与执行方式

当前目录：

```text
evaluation/interleave
```

脚本列表：

- `BabyVision/infer_babyvision.py`
- `BabyVision/eval_babyvision.py`
- `BabyVision/compute_score.py`
- `OpenING/infer_opening.py`
- `OpenING/eval_opening.py`
- `OpenING/summarize_GPT_scores.py`
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

API 后端模式（BabyVision）：

```bash
python3 -m pip install requests regex tqdm
```

如需本地图片重采样，额外安装：

```bash
python3 -m pip install pillow
```

如需运行 BabyVision / OpenING 的评测脚本，额外安装：

```bash
python3 -m pip install openai pandas matplotlib pillow
```

本地模型加载模式（OpenING / Unimmmu / RealUnify）：

```bash
python3 -m pip install torch torchvision transformers pillow numpy tqdm
```

## 环境检查

可先执行以下命令确认脚本能够启动并完成参数解析：

```bash
python3 BabyVision/infer_babyvision.py --help
python3 BabyVision/eval_babyvision.py --help
python3 BabyVision/compute_score.py --help
python3 OpenING/infer_opening.py --help
python3 OpenING/eval_opening.py --help
python3 OpenING/summarize_GPT_scores.py --help
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

### 评测

BabyVision 的答案抽取与判分脚本为：

```text
BabyVision/eval_babyvision.py
```

示例命令：

```bash
python3 BabyVision/eval_babyvision.py \
  --input /path/to/babyvision_local-model.jsonl \
  --output /path/to/babyvision_local-model_eval.jsonl \
  --endpoint https://your-judge-endpoint \
  --api-key your_api_key \
  --model gpt-4.1 \
  --workers 16 \
  --retries 3
```

参数说明：

- `--input`：待评测的 BabyVision 推理结果文件
- `--output`：评测结果输出文件；不传时默认生成 `<input>_eval.jsonl` 或 `<input>_eval.json`
- `--endpoint`：judge 服务地址；也可通过 `BABYVISION_JUDGE_ENDPOINT` 或 `AZURE_OPENAI_ENDPOINT` 提供
- `--api-key`：judge API key；也可通过 `BABYVISION_JUDGE_API_KEY` 或 `AZURE_OPENAI_API_KEY` 提供
- `--api-version`：judge API version；默认 `2025-01-01-preview`
- `--model`：judge 模型名，默认 `gpt-4.1`
- `--workers`：并发请求数
- `--retries`：单条样本失败后的重试次数
- `--extractor`：答案抽取策略，可选 `llm`、`rule_then_llm`、`rule_only`
- `--force`：即使已有 `extracted_answer` 或 `LLMJudgeResult` 也强制重算
- `--judge-only`：仅对已有 `extracted_answer` 的样本做判分，不再重新抽取答案

### 算分

BabyVision 的准确率汇总脚本为：

```text
BabyVision/compute_score.py
```

示例命令：

```bash
python3 BabyVision/compute_score.py \
  /path/to/babyvision_local-model_eval.jsonl
```

输出说明：

- 输出整体准确率
- 输出 Type 维度平均准确率
- 输出 Subtype 维度平均准确率
- 支持同时传入多个结果文件，并汇总均值与标准差

## OpenING

### 使用说明

- 当前版本采用 `transformers` 本地模型推理，不再依赖 LightLLM 或 OpenAI 兼容接口
- 运行 OpenING Benchmark 时，应显式指定 `--mode opening`
- `--model_path` 为必填参数
- 建议显式传入 `--model_path`、`--meta-path`、`--data-file-name`、`--save_dir`
- 如需读取 `s3://...` 路径图像，可额外安装 `aoss_client` 并设置 `AOSS_CONF_PATH`

### 关键参数

- 推荐配置：`cfg_scale=4.0`、`img_cfg_scale=1.0`、`timestep_shift=3.0`、`cfg_interval=0 1.0`、`num_steps=50`
- 推荐生成设置：`max_new_tokens=4096`、`max_generation_pixels=4194304`、`oom_retry_max_pixels=1048576`
- 推荐输出尺寸：`image_width=1920`、`image_height=1088`
- 推荐提示策略：`opening_step_prompt_style=can_be`
- 推荐随机种子：`seed=42`

### 推荐命令

单卡示例：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --model_path /path/to/model \
  --save_dir /path/to/OpenING/gen_outputs/opening_output \
  --meta-path /path/to/OpenING-benchmark \
  --data-file-name test_data.jsonl \
  --think_mode think \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 3.0 \
  --cfg_interval 0 1.0 \
  --num_steps 50 \
  --max_new_tokens 4096 \
  --max_generation_pixels 4194304 \
  --oom_retry_max_pixels 1048576 \
  --image_width 1920 \
  --image_height 1088 \
  --opening_step_prompt_style can_be \
  --retry_short_outputs 0 \
  --seed 42
```

推荐的单机 8 卡分片运行方式：

```bash
mkdir -p logs
for LOCAL_RANK in 0 1 2 3 4 5 6 7; do
  echo "Starting shard ${LOCAL_RANK} on GPU ${LOCAL_RANK}"
  CUDA_VISIBLE_DEVICES=${LOCAL_RANK} python3 OpenING/infer_opening.py \
    --mode opening \
    --model_path /path/to/model \
    --save_dir /path/to/OpenING/gen_outputs/opening_output \
    --meta-path /path/to/OpenING-benchmark \
    --data-file-name test_data.jsonl \
    --think_mode think \
    --num_shards 8 \
    --shard_index ${LOCAL_RANK} \
    --cfg_scale 4.0 \
    --img_cfg_scale 1.0 \
    --timestep_shift 3.0 \
    --cfg_interval 0 1.0 \
    --num_steps 50 \
    --max_new_tokens 4096 \
    --max_generation_pixels 4194304 \
    --oom_retry_max_pixels 1048576 \
    --image_width 1920 \
    --image_height 1088 \
    --opening_step_prompt_style can_be \
    --retry_short_outputs 0 \
    --seed 42 \
    > logs/opening_shard${LOCAL_RANK}.log 2>&1 &
done
wait
```

说明：

- `--think_mode think` 表示仅运行 think 模式；如需同时产出 think 和 no_think 结果，可使用 `--think_mode think no_think`
- `--num_shards` 与 `--shard_index` 用于样本分片，适合单机多卡或多进程并行
- `--image_width` 与 `--image_height` 同时控制保存尺寸，并会向下对齐到模型要求的倍数
- `--max_generation_pixels` 为初始生成像素上限，`--oom_retry_max_pixels` 用于 CUDA OOM 后降分辨率重试
- `--opening_step_prompt_style can_be` 为当前默认推荐值

附加参数示例：

- 仅运行前 20 条：`--limit 20`
- 只处理指定 UID：`--uid_file /path/to/uid_list.txt`
- 覆盖已有结果：`--overwrite`
- 指定系统提示词文件：`--system_prompt_path /path/to/system_prompt.txt`
- 切换到 annotation config 模式：`--mode annotation_config --input_json_path /path/to/config.json`

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
- 如生成图片，文件名示例为 `<save_dir>/<total_uid>-o-0.jpg`
- 当 `--think_mode` 传入多个值时，脚本会分别写入对应目录；若 `save_dir` 以 `_output` 结尾，目录名会自动扩展为 `*_think_output`、`*_no_think_output`

### 评测

OpenING 支持基于 GPT judge 的主观评测。仓库内评测脚本为：

```text
OpenING/eval_opening.py
```

推荐从 OpenING 仓库根目录执行，并显式指定输出目录、结果文件、并发数和保存间隔：

```bash
cd /path/to/OpenING

export OPENING_JUDGE_BASE_URL=http://127.0.0.1:8000
export OPENING_JUDGE_API_KEY=your_api_key

python3 /path/to/SenseNova-U1/evaluation/interleave/OpenING/eval_opening.py \
  --mode output_dir \
  --opening_root /path/to/OpenING \
  --output_dir /path/to/OpenING/gen_outputs/opening_output \
  --output_file /path/to/OpenING/Interleaved_Arena/gpt-score_results_opening_output.json \
  --workers 4 \
  --save_every 10
```

如推理阶段使用 `--think_mode think no_think`，则可将 `--output_dir` 指向自动生成的
`*_think_output` 或 `*_no_think_output` 目录分别评测。

参数说明：

- `--mode output_dir`：按模型输出目录进行评测
- `--opening_root`：OpenING 工作目录根路径
- `--benchmark_dir`：可选，显式指定 `OpenING-benchmark` 目录
- `--output_dir`：待评测的 OpenING 输出目录，支持单个 `*_output` 目录或包含多个模型输出目录的父目录
- `--output_file`：GPT judge 结果保存路径
- `--api_base_url`：可选，覆盖 judge API base URL；也可通过 `OPENING_JUDGE_BASE_URL` 提供
- `--api_key`：可选，覆盖 judge API key；也可通过 `OPENING_JUDGE_API_KEY` 或 `OPENAI_API_KEY` 提供
- `--judge_model`：可选，judge 模型名，默认 `gpt-4o`
- `--workers`：并发请求数
- `--save_every`：每处理多少条样本落盘一次
- `--no_resume`：不复用已有评测结果，强制重新评测
- `--retry_invalid_scores`：仅重试已有结果中分数结构无效的样本
- `--limit`：仅评测前 N 条待处理任务

### 算分

OpenING 的 GPT judge 结果可使用以下脚本汇总：

```text
OpenING/summarize_GPT_scores.py
```

示例命令：

```bash
python3 OpenING/summarize_GPT_scores.py \
  --input_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_opening_output.json \
  --output_csv /path/to/OpenING/Interleaved_Arena/model_score_summaries.csv \
  --filtered_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_filtered.json
```

输出说明：

- `--input_json`：GPT judge 原始结果 JSON
- `--output_csv`：各模型维度均分与总分汇总表
- `--filtered_json`：过滤掉无效分数后的结果 JSON
- `--plot`：可选，展示排行榜表格

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
- 数据文件不存在：检查 `--data_path`，或 OpenING 的 `--meta-path`、`--data-file-name`
- 本地模型脚本读取到 `<DATA_ROOT>/...`：说明命中了占位默认值，请显式传入 `--data_path`
- 输入图片缺失：检查数据中的相对路径和数据根目录设置
- 结果被跳过：通常是输出已存在；可结合 `--overwrite` 或 `--resume` 处理
