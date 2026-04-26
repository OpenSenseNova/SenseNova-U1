# interleave_inference

用于运行多模态 Benchmark 的推理与评测脚本。

## 目录与执行方式

### 目录

```text
evaluation/interleave
```

### 脚本列表

- `BabyVision/infer_babyvision.py`
- `BabyVision/eval_babyvision.py`
- `BabyVision/compute_score.py`
- `OpenING/infer_opening.py`
- `OpenING/eval_opening.py`
- `OpenING/summarize_GPT_scores.py`
- `Unimmmu/inference_unimmmu.py`
- `Unimmmu/calculate_score.py`
- `Unimmmu/merge_shards.py`
- `Realunify/inference_realunify.py`
- `Realunify/inference_realunify_ueg.py`
- `Realunify/calculate_score.py`
- `Realunify/calculate_score_ueg.py`
- `Realunify/merge_shards.py`

### 执行方式

除非特别说明，以下命令默认从仓库根目录进入本目录后执行：

```bash
cd evaluation/interleave
```

若从其他目录执行，请将相对路径替换为本地可访问路径。

## 使用前提

### 说明

- 当前工作目录为 `evaluation/interleave`
- API 类 Benchmark 需要可访问的服务地址
- 本地模型类 Benchmark 需要可访问的 `--model_path` 与数据文件路径
- 建议显式传入关键路径和服务地址

## 依赖

### API 后端模式

适用于 BabyVision：

```bash
python3 -m pip install requests regex tqdm
```

如需本地图片重采样：

```bash
python3 -m pip install pillow
```

### 评测脚本依赖

如需运行 BabyVision / OpenING 的评测脚本：

```bash
python3 -m pip install openai pandas matplotlib pillow
```

### 本地模型加载模式

适用于 OpenING 推理、Unimmmu、RealUnify：

```bash
python3 -m pip install torch torchvision transformers pillow numpy tqdm
```

## 环境检查

### 示例命令

可先用以下命令检查脚本能否启动并完成参数解析：

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

### 说明

如出现 `ModuleNotFoundError`，先安装依赖。

## 默认值说明

### 说明

文中涉及两类默认值：

- 代码默认值：脚本中 `argparse` 或环境变量定义的默认值
- 推荐运行值：为减少环境差异，建议显式传入的参数

本地模型类脚本里的 `--data_path` 默认值通常是占位路径，实际运行请显式传入。

## BabyVision

### 推理

#### 示例命令

```bash
python3 BabyVision/infer_babyvision.py \
  --model-name local-model \
  --data-path /path/to/meta_data.jsonl \
  --image-root /path/to/babyvision_images \
  --output-dir ./output/babyvision_understand \
  --generate-urls http://127.0.0.1:8000/generate \
  --workers 32 \
  --max-retries 3 \
  --backend-max-retries 20 \
  --request-timeout 600 \
  --max-new-tokens 32768 \
  --no-do-sample \
  --temperature 0 \
  --top-p 0.95 \
  --repetition-penalty 1.05 \
  --min-pixels 262144 \
  --max-pixels 4194304
```

#### 参数说明

- `--model-name`：结果中的模型名，也用于输出文件名
- `--data-path`：BabyVision 的 `meta_data.jsonl` 路径
- `--image-root`：图片根目录；样本相对路径会拼到该目录下
- `--output-dir`：推理结果输出目录
- `--generate-urls`：一个或多个 `/generate` 地址，多个地址用英文逗号分隔
- `--workers`：并发线程数
- `--max-retries`：单条样本最大重试次数
- `--backend-max-retries`：单次后端请求最大重试次数
- `--request-timeout`：单次 HTTP 请求超时，单位秒
- `--max-new-tokens`：生成的最大 token 数
- `--do-sample`：启用采样生成
- `--temperature`：采样温度
- `--top-p`：top-p 采样阈值
- `--repetition-penalty`：重复惩罚系数
- `--min-pixels`：预处理后的最小图片像素
- `--max-pixels`：预处理后的最大图片像素

#### 输出与行为说明

- 输出文件：`babyvision_<model_name>.jsonl`
- 字段：`taskId`、`type`、`subtype`、`ansType`、`question`、`answer`、`model`、`model_response`、`extracted_answer`
- 支持断点续跑；已完成的 `taskId` 会跳过
- 输入图片缺失、请求失败或输出损坏时，脚本返回非零退出码

### 评测

#### 示例命令

BabyVision 的答案抽取与判分脚本为：

```text
BabyVision/eval_babyvision.py
```

```bash
python3 BabyVision/eval_babyvision.py \
  --input ./output/babyvision_understand/babyvision_local-model.jsonl \
  --output  ./output/babyvision_understand/babyvision_local-model_eval.jsonl \
  --endpoint https://your-judge-endpoint \
  --api-key your_api_key \
  --model gpt-4.1 \
  --force \
  --workers 16 \
  --retries 3 
```

#### 参数说明

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

#### 输出与行为说明

- 输出文件默认为 `<input>_eval.jsonl` 或 `<input>_eval.json`
- 结果中会补充答案抽取与 judge 判分字段
- 支持并发调用 judge，并可通过 `--force` 或 `--judge-only` 控制重跑

### 算分

#### 示例命令

BabyVision 的准确率汇总脚本为：

```text
BabyVision/compute_score.py
```

```bash
python3 BabyVision/compute_score.py \
  ./output/babyvision_understand/babyvision_local-model_eval.jsonl
```

#### 参数说明

- 位置参数：一个或多个已完成评测的 BabyVision 结果文件

#### 输出与行为说明

- 输出整体准确率
- 输出 Type 平均准确率
- 输出 Subtype 平均准确率
- 支持同时传入多个结果文件，并汇总均值与标准差

## OpenING

### 推理

#### 示例命令

单卡示例：

```bash
python3 OpenING/infer_opening.py \
  --mode opening \
  --model_path /path/to/model \
  --save_dir ./output/opening_interleave/opening_output \
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

单机 8 卡分片运行示例：

```bash
mkdir -p logs
for LOCAL_RANK in 0 1 2 3 4 5 6 7; do
  echo "Starting shard ${LOCAL_RANK} on GPU ${LOCAL_RANK}"
  CUDA_VISIBLE_DEVICES=${LOCAL_RANK} python3 OpenING/infer_opening.py \
    --mode opening \
    --model_path /path/to/model \
    --save_dir ./output/opening_interleave/opening_output \
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

#### 参数说明

- 采用 `transformers` 本地推理
- `--mode opening`：运行 OpenING Benchmark；切换到 `annotation_config` 模式可用 `--mode annotation_config --input_json_path /path/to/config.json`
- `--model_path`：模型路径，必填
- `--meta-path` 与 `--data-file-name`：分别指定 OpenING Benchmark 根目录与 JSONL 文件名
- `--save_dir`：结果输出目录
- `--think_mode`：可传 `think`、`no_think`，也可同时传两个值分别产出两套结果
- `--cfg_scale`、`--img_cfg_scale`、`--timestep_shift`、`--cfg_interval`、`--num_steps`：示例命令中分别使用 `4.0`、`1.0`、`3.0`、`0 1.0`、`50`
- `--max_new_tokens`、`--max_generation_pixels`、`--oom_retry_max_pixels`：示例命令中分别使用 `4096`、`4194304`、`1048576`
- `--image_width` 与 `--image_height`：示例命令中使用 `1920x1088`；实际会向下对齐到模型要求的倍数
- `--opening_step_prompt_style can_be`：推荐步数提示策略
- `--num_shards` 与 `--shard_index`：样本分片参数
- `--limit`：仅运行前 N 条样本，便于 smoke test
- `--uid_file`：仅处理文件中列出的 `total_uid`
- `--overwrite`：覆盖已有结果
- `--system_prompt_path`：读取外部系统提示词文件
- 如需读取 `s3://...` 路径下的图像，可额外安装 `aoss_client` 并设置 `AOSS_CONF_PATH`

#### 输出与行为说明

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

#### 示例命令

OpenING 支持基于 GPT judge 的主观评测。仓库内评测脚本为：

```text
OpenING/eval_opening.py
```

可在任意目录执行。为与全文保持一致，以下示例仍假设当前工作目录为 `evaluation/interleave`，并显式传 `--opening_root`：

```bash
export OPENING_JUDGE_BASE_URL=http://127.0.0.1:8000
export OPENING_JUDGE_API_KEY=your_api_key

python3 OpenING/eval_opening.py \
  --mode output_dir \
  --opening_root /path/to/OpenING \
  --output_dir ./output/opening_interleave/opening_output \
  --output_file /path/to/OpenING/gpt-score_results_opening_output.json \
  --workers 4 \
  --save_every 10
```

如推理阶段使用 `--think_mode think no_think`，可分别将 `--output_dir` 指向自动生成的 `*_think_output` 和 `*_no_think_output` 目录进行评测。

#### 参数说明

- `--mode output_dir`：按模型输出目录进行评测
- `--opening_root`：OpenING 工作目录根路径；默认取当前工作目录，若已在 OpenING 根目录执行可省略
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

#### 输出与行为说明

- 输出为 GPT judge 结果 JSON，由 `--output_file` 指定
- 可按单个 `*_output` 目录或模型输出父目录批量评测
- 支持断点续评和仅重试无效分数样本

### 算分

#### 示例命令

OpenING 的 GPT judge 结果可使用以下脚本汇总：

```text
OpenING/summarize_GPT_scores.py
```

```bash
python3 OpenING/summarize_GPT_scores.py \
  --input_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_opening_output.json \
  --output_csv /path/to/OpenING/Interleaved_Arena/model_score_summaries.csv \
  --filtered_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_filtered.json
```

#### 参数说明

- `--input_json`：GPT judge 原始结果 JSON
- `--output_csv`：各模型维度均分与总分汇总表
- `--filtered_json`：过滤掉无效分数后的结果 JSON
- `--plot`：可选，展示排行榜表格

#### 输出与行为说明

- 输出汇总 CSV，便于比较不同模型得分
- 可输出过滤后的 JSON，便于后续复核
- 开启 `--plot` 时可直接渲染排行榜表格

## Unimmmu

### 推理

#### 示例命令

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
torchrun --nproc_per_node=2 --master_port=29503 Unimmmu/inference_unimmmu.py \
  --model_path /path/to/model \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
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

#### 参数说明

- `--inference_mode`：可选 `i2t` 或 `interleave`
- `--model_path`：模型路径，必填
- `--data_path`：数据文件路径；默认值通常是占位路径，实际运行请显式传入
- `--output_dir`：结果输出目录
- `--cfg_scale`、`--img_cfg_scale`、`--cfg_interval`、`--cfg_norm`、`--num_steps`、`--timestep_shift`：控制 interleave 生成行为
- `--min_pixels` 与 `--max_pixels`：控制输入图片缩放范围
- `--target_image_size`：如需固定正方形生成尺寸，可显式指定
- `--resume`：按 `hash_uid` 跳过已完成样本
- `--num_shards` 与 `--shard_rank`：手动分片参数
- `--device_map auto`：让 HuggingFace 自动分配多卡；使用该模式时请单进程运行
- `--max_memory_per_gpu_gb`：配合 `--device_map` 使用，限制单卡显存上限
- `--limit`：仅运行前 N 条样本

#### 输出与行为说明

- 结果文件：`unimmmu_results.jsonl`
- 字段：`hash_uid`、`task`、`model_response`、`inference_mode`、`generated_images`
- `interleave` 模式生成的图片会保存到 `<output_dir>/images/<task>/`
- 使用 `--num_shards` 与 `--shard_rank` 手动分片时，中间结果写入 `<output_dir>/shards/`
- 使用 `--resume` 时，脚本会按 `hash_uid` 自动跳过已完成样本
- 当前实现中，`--resume` 发生在手动分片之前；若要稳定补跑单个 shard，建议删除该 shard 输出后不带 `--resume` 重跑

### 评测

#### 示例命令

- 无，直接进入算分

#### 参数说明

- 无

#### 输出与行为说明

- 无独立输出

### 算分

#### 示例命令

```bash
python3 Unimmmu/calculate_score.py \
  --input_file ./output/unimmmu_interleave/unimmmu_results.jsonl \
  --output_dir ./output/unimmmu_interleave/scores \
  --benchmark_path /path/to/image_text_agent
```

#### 参数说明

- `--input_file`：推理结果 JSONL
- `--output_dir`：得分输出目录；默认使用输入文件所在目录
- `--benchmark_path`：必填，需要指向包含 `evaluation/` 目录的 benchmark 仓库
- `--use_tools`：可选，可用于 geometry 的 tool_call 模式评分

#### 输出与行为说明

- 脚本会加载推理结果并打印任务分布
- 实际 scorer 来自外部 benchmark 仓库，而不是当前目录
- 评分结果会输出到 `--output_dir`

## RealUnify (GEU)

### 推理

#### 示例命令

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
torchrun --nproc_per_node=2 --master_port=29501 Realunify/inference_realunify.py \
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

分片合并：

```bash
python3 Realunify/merge_shards.py \
  --data_path /path/to/GEU_step_processed.jsonl \
  --shard_dir ./output/realunify_interleave/shards \
  --output_file ./output/realunify_interleave/realunify_results.jsonl
```

如需固定输出图片尺寸，可额外传入：

```text
--target_image_size 1024
```

#### 参数说明

- `--inference_mode`：可选 `step` 或 `interleave`
- `--model_path`：模型路径，必填
- `--data_path`：数据文件路径；默认值通常是占位路径，实际运行请显式传入
- `--output_dir`：结果输出目录
- `--cfg_scale`、`--img_cfg_scale`、`--cfg_interval`、`--cfg_norm`、`--num_steps`、`--timestep_shift`：控制编辑与生成行为
- `--min_pixels` 与 `--max_pixels`：控制输入图片缩放范围
- `--target_image_size`：如需固定正方形输出尺寸，可显式指定
- `--resume`：按 `hash_uid` 跳过已完成样本
- `--num_shards` 与 `--shard_rank`：手动分片参数
- `--device_map auto`：让 HuggingFace 自动分配多卡；使用该模式时请单进程运行
- `--max_memory_per_gpu_gb`：配合 `--device_map` 使用，限制单卡显存上限
- `--limit`：仅运行前 N 条样本

#### 输出与行为说明

- 结果文件：`realunify_results.jsonl`
- 字段：`hash_uid`、`task_type`、`model_response`、`answer`、`generated_image`、`generated_images`
- `step` 模式会写出 `generated_image`，其值为 `[input_image, edited_image]`
- `interleave` 模式会写出生成图片列表 `generated_images`
- 使用 `--num_shards` 与 `--shard_rank` 手动分片时，中间结果写入 `<output_dir>/shards/`
- 使用 `--resume` 时，脚本会按 `hash_uid` 自动跳过已完成样本
- 当前实现中，`--resume` 发生在手动分片之前；若要稳定补跑单个 shard，建议删除该 shard 输出后不带 `--resume` 重跑

### 评测

#### 示例命令

- 无，直接进入算分

#### 参数说明

- 无

#### 输出与行为说明

- 无独立输出

### 算分

#### 示例命令

```bash
python3 Realunify/calculate_score.py \
  --input_file ./output/realunify_interleave/realunify_results.jsonl \
  --output_file ./output/realunify_interleave/realunify_scores.json
```

#### 参数说明

- `--input_file`：推理结果 JSONL
- `--output_file`：可选，保存汇总结果 JSON 的路径

#### 输出与行为说明

- 脚本会优先从 `<answer>...</answer>` 中提取答案
- 如果没有 `<answer>` 标签，会回退到 `model_response` 中首个 `A/B/C/D` 字母
- 输出按 `task_type` 与 overall 两个层级汇总准确率

## RealUnify (UEG)

### 推理

#### 示例命令

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

#### 参数说明

- `--inference_mode`：可选 `understand_t2i`、`interleave`、`t2i`
- `--model_path`、`--output_dir`、`--inference_mode`：必填参数
- `--data_path`：数据文件路径；默认值通常是占位路径，实际运行请显式传入
- `--cfg_scale`、`--img_cfg_scale`、`--cfg_interval`、`--cfg_norm`、`--num_steps`、`--timestep_shift`：控制生成行为
- `--target_image_size`：如需固定正方形输出尺寸，可显式指定
- `--resume`：跳过已完成样本
- `--limit`：仅运行前 N 条样本
- `--device_map auto`：让 HuggingFace 自动分配多卡；使用该模式时请单进程运行
- `--max_memory_per_gpu_gb`：配合 `--device_map` 使用，限制单卡显存上限
- 当前脚本没有 `--num_shards` / `--shard_rank` 手动分片参数；多进程切分依赖 `torchrun` 提供的 distributed rank

#### 输出与行为说明

- 输出文件：`ueg_results.jsonl`、`ueg_results.json`
- 字段：`index`、`task_type`、`generated_image`、`question_list`
- 结果会保留生成图片路径与后续问答列表，供 judge 评分
- 当前实现使用 `index` 生成 `hash_uid` 供 `--resume` 使用；若原始数据存在重复 `index`，续跑与去重行为会按该键生效

### 评测

#### 示例命令

- 无，直接进入算分

#### 参数说明

- 无

#### 输出与行为说明

- 无独立输出

### 算分

#### 示例命令

```bash
python3 Realunify/calculate_score_ueg.py \
  --input_file ./output/ueg_interleave/ueg_results.json
```

#### 参数说明

- `--input_file`：`ueg_results.json` 或 `.jsonl`
- `--num_workers`：评分并发数

#### 输出与行为说明

- 当前脚本是评分骨架，需自行提供 `GeminiAPI` judge 封装
- 如果未补充 judge wrapper，脚本会在运行时抛出 `NotImplementedError`
- 正常完成后会输出带 judge 结果的 `_scored.json`

## 评测流程概览

### 示例命令

示例流程：

```bash
MODEL_PATH=/path/to/hf_model

torchrun --nproc_per_node=2 --master_port=29501 Realunify/inference_realunify.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/GEU_step_processed.jsonl \
  --output_dir ./output/realunify_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
  --num_steps 50

python3 Realunify/inference_realunify_ueg.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_understand_t2i \
  --inference_mode understand_t2i \
  --cfg_scale 4.0 \
  --num_steps 50

torchrun --nproc_per_node=2 --master_port=29503 Unimmmu/inference_unimmmu.py \
  --model_path ${MODEL_PATH} \
  --data_path /path/to/unimmmu_direct.jsonl \
  --output_dir ./output/unimmmu_interleave \
  --inference_mode interleave \
  --cfg_scale 4.0 \
  --img_cfg_scale 1.0 \
  --timestep_shift 1.0 \
  --num_steps 50
```

### 说明

- 以上示例主要覆盖本地模型类 Benchmark 的常见执行路径
- `RealUnify (GEU)`、`RealUnify (UEG)`、`Unimmmu` 三者之间无依赖，可并行执行
- `BabyVision` 与 `OpenING` 的完整流程请分别参考各自章节中的 `推理 / 评测 / 算分`

## 退出码

### 说明

- `0`：全部成功，或仅有显式跳过
- `1`：启动校验失败，或运行过程中存在样本失败

## 常见问题

### 说明

- 缺少 `requests`、`regex`、`tqdm`：请先安装依赖
- 数据文件不存在：检查 `--data_path`，或 OpenING 的 `--meta-path`、`--data-file-name`
- 本地模型脚本读取到 `<DATA_ROOT>/...`：说明命中了占位默认值，请显式传入 `--data_path`
- 输入图片缺失：检查数据中的相对路径和数据根目录设置
- 结果被跳过：通常是输出已存在；可结合 `--overwrite` 或 `--resume` 处理
