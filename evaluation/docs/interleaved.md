# Interleaved Generation Evaluation

Reproduction guide for SenseNova-U1 on interleaved generation benchmarks.

The benchmark scripts live under `evaluation/interleave/`:

- `BabyVision/` — API-based multimodal understanding with answer extraction and judge scoring
- `OpenING/` — local-model interleaved generation with GPT-based judging
- `Unimmmu/` — local-model interleaved generation with external score computation
- `Realunify/` — local-model interleaved generation for GEU and UEG

## 1. Overview

```
┌──────────────────────┐
│ evaluation/interleave│
└──────────┬───────────┘
           │
           ├── BabyVision   ── API inference ── extract / judge ── aggregate score
           ├── OpenING      ── local inference ─ GPT judge       ── summarize
           ├── Unimmmu      ── local inference ─ external scorer
           └── RealUnify    ── local inference ─ rule / judge scoring
```

1. `BabyVision` sends requests to one or more `/generate` endpoints and writes JSONL predictions.
2. `OpenING`, `Unimmmu`, and `RealUnify` load the model locally through `transformers`.
3. Some benchmarks have a separate judge or score-aggregation step after inference.
4. Most scripts support resume-friendly reruns through existing outputs, explicit `--resume`, or shard merging.

All commands below assume:

```bash
cd evaluation/interleave
```

## 2. Benchmark Matrix

| Benchmark | Inference backend | Evaluation backend | Primary outputs |
| --- | --- | --- | --- |
| `BabyVision` | HTTP `/generate` API | extraction + LLM judge | `babyvision_<model>.jsonl`, `*_eval.jsonl` |
| `OpenING` | local model | GPT judge + CSV summary | per-sample JSON, generated images, judge JSON |
| `Unimmmu` | local model | external scorer | `unimmmu_results.jsonl`, generated images |
| `RealUnify (GEU)` | local model | rule-based scorer | `realunify_results.jsonl`, score JSON |
| `RealUnify (UEG)` | local model | user-provided judge wrapper | `ueg_results.json[l]` |

For local-model benchmarks, pass the real dataset path explicitly instead of relying on placeholder defaults such as `<DATA_ROOT>/...`.

## 3. BabyVision

`BabyVision` is the API-backed benchmark in this suite. The typical flow is inference first, then extraction and judge scoring, then score aggregation.

Reference inference command:

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

| Argument | Meaning |
| --- | --- |
| `--data-path` | Path to `meta_data.jsonl`. |
| `--image-root` | Root directory used to resolve sample image paths. |
| `--generate-urls` | One or more `/generate` endpoints, comma-separated. |
| `--workers` | Concurrent request workers. |
| `--max-retries`, `--backend-max-retries` | Retry budget on the sample side and backend-request side. |
| `--request-timeout` | Per-request timeout in seconds. |
| `--min-pixels`, `--max-pixels` | Image preprocessing bounds. |

Inference writes `babyvision_<model_name>.jsonl`. Completed `taskId`s are skipped automatically on rerun.

Reference evaluation command:

```bash
python3 BabyVision/eval_babyvision.py \
  --input ./output/babyvision_understand/babyvision_local-model.jsonl \
  --output ./output/babyvision_understand/babyvision_local-model_eval.jsonl \
  --endpoint https://your-judge-endpoint \
  --api-key your_api_key \
  --model gpt-4.1 \
  --force \
  --workers 16 \
  --retries 3
```

`eval_babyvision.py` performs answer extraction and judge scoring. `--endpoint` and `--api-key` can also come from environment variables. Use `--force` to recompute existing records, or `--judge-only` to score records that already have `extracted_answer`.

Reference score command:

```bash
python3 BabyVision/compute_score.py \
  ./output/babyvision_understand/babyvision_local-model_eval.jsonl
```

The score script reports overall accuracy plus per-`type` and per-`subtype` results.

## 4. OpenING

`OpenING` runs local-model interleaved generation and then scores the outputs with a GPT judge.

Reference single-GPU inference command:

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

Reference 8-shard single-node command:

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

| Argument | Meaning |
| --- | --- |
| `--model_path` | Local model path. |
| `--meta-path` | Dataset root. |
| `--data-file-name` | Dataset JSONL file under the benchmark root. |
| `--save_dir` | Output directory for per-sample JSON and images. |
| `--think_mode` | `think`, `no_think`, or both. |
| `--cfg_interval`, `--max_generation_pixels`, `--oom_retry_max_pixels` | Main generation and OOM-retry controls. |
| `--num_shards`, `--shard_index` | Manual sharding for multi-process runs. |

Each sample is saved as `<save_dir>/<total_uid>.json`, and generated images use names such as `<save_dir>/<total_uid>-o-0.jpg`.

Reference judge command:

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

`eval_opening.py` supports both a single model-output directory and a parent directory containing multiple outputs. Existing judge results are reused by default; `--retry_invalid_scores` retries only malformed score records.

Reference summary command:

```bash
python3 OpenING/summarize_GPT_scores.py \
  --input_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_opening_output.json \
  --output_csv /path/to/OpenING/Interleaved_Arena/model_score_summaries.csv \
  --filtered_json /path/to/OpenING/Interleaved_Arena/gpt-score_results_filtered.json
```

This step converts raw judge results into a comparison-friendly CSV and can optionally emit a filtered JSON with invalid scores removed.

## 5. Unimmmu

`Unimmmu` supports both understanding-only and interleaved generation paths, but the interleaved mode is the one covered here.

Reference inference command:

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

Reference multi-GPU command:

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

Reference `device_map` command:

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

Reference shard merge command:

```bash
python3 Unimmmu/merge_shards.py \
  --data_path /path/to/unimmmu_direct.jsonl \
  --shard_dir ./output/unimmmu_interleave/shards \
  --output_file ./output/unimmmu_interleave/unimmmu_results.jsonl
```

| Argument | Meaning |
| --- | --- |
| `--inference_mode` | Use `interleave` for the benchmark covered here. |
| `--data_path` | Benchmark JSONL path. |
| `--output_dir` | Root directory for JSONL results and generated images. |
| `--resume` | Skip completed `hash_uid`s. |
| `--num_shards`, `--shard_rank` | Manual data sharding. |
| `--device_map auto` | Single-process multi-GPU loading via Hugging Face. |

The main output file is `unimmmu_results.jsonl`. Interleaved images are written under `<output_dir>/images/<task>/`. In the current implementation, resume is applied before shard selection, so rerunning one shard is most reliable after deleting that shard's outputs and rerunning without `--resume`.

Reference score command:

```bash
python3 Unimmmu/calculate_score.py \
  --input_file ./output/unimmmu_interleave/unimmmu_results.jsonl \
  --output_dir ./output/unimmmu_interleave/scores \
  --benchmark_path /path/to/image_text_agent
```

`calculate_score.py` delegates the actual scoring logic to the external benchmark repository pointed to by `--benchmark_path`.

## 6. RealUnify (GEU)

The GEU script supports both `step` and `interleave` modes. The interleaved path is the main one for SenseNova-U1 benchmarking.

Reference inference command:

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

Reference multi-GPU command:

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

Reference `device_map` command:

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

Reference shard merge command:

```bash
python3 Realunify/merge_shards.py \
  --data_path /path/to/GEU_step_processed.jsonl \
  --shard_dir ./output/realunify_interleave/shards \
  --output_file ./output/realunify_interleave/realunify_results.jsonl
```

The main result file is `realunify_results.jsonl`. In `step` mode, `generated_image` stores `[input_image, edited_image]`; in `interleave` mode, the generated sequence is stored under `generated_images`. As with `Unimmmu`, resume currently happens before manual shard selection.

If you want a fixed output image size, pass `--target_image_size 1024`.

Reference score command:

```bash
python3 Realunify/calculate_score.py \
  --input_file ./output/realunify_interleave/realunify_results.jsonl \
  --output_file ./output/realunify_interleave/realunify_scores.json
```

The scorer first tries to extract the answer from `<answer>...</answer>` and otherwise falls back to the first `A/B/C/D` letter found in `model_response`.

## 7. RealUnify (UEG)

The UEG script exposes `understand_t2i`, `interleave`, and `t2i` inference modes.

Reference `understand_t2i` command:

```bash
python3 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_understand_t2i \
  --inference_mode understand_t2i \
  --cfg_scale 4.0 \
  --num_steps 50
```

Reference `interleave` command:

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

Reference `t2i` command:

```bash
python3 Realunify/inference_realunify_ueg.py \
  --model_path /path/to/model \
  --data_path /path/to/UEG_step.json \
  --output_dir ./output/ueg_t2i \
  --inference_mode t2i \
  --cfg_scale 4.0 \
  --num_steps 50
```

Unlike the GEU script, this one does not expose manual `--num_shards` or `--shard_rank` flags. Multi-process splitting relies on the distributed rank provided by `torchrun`. The output preserves generated image paths together with the follow-up `question_list`.

Reference score command:

```bash
python3 Realunify/calculate_score_ueg.py \
  --input_file ./output/ueg_interleave/ueg_results.json
```

`calculate_score_ueg.py` is only a scaffold in the current repository. It expects a user-provided `GeminiAPI` judge wrapper and otherwise raises `NotImplementedError`.

## 8. Running the Evaluation

A typical local-model evaluation flow looks like this:

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

`RealUnify (GEU)`, `RealUnify (UEG)`, and `Unimmmu` are independent and can run in parallel. `BabyVision` and `OpenING` each have their own inference-plus-evaluation pipeline as described above.

## 9. Troubleshooting

- Dataset file not found: check `--data_path`, or for `OpenING`, verify `--meta-path` and `--data-file-name`.
- A path still contains `<DATA_ROOT>`: the script is using a placeholder default; pass the real path explicitly.
- Samples are unexpectedly skipped: outputs already exist; review `--resume`, `--overwrite`, or shard outputs.
- Rerunning a single shard gives odd behavior: for `Unimmmu` and `RealUnify (GEU)`, delete that shard's outputs first and rerun without `--resume`.
- UEG scoring fails immediately: `calculate_score_ueg.py` needs a user-supplied `GeminiAPI` wrapper.
