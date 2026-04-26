#!/usr/bin/env bash
set -euo pipefail

# Large-model launcher for RealUnify.
# Default: 2 GPUs per worker, multiple workers on one node, manual shard split.

STEP="${STEP:-3000_ema}"
MODEL_NAME="${MODEL_NAME:-<YOUR_MODEL_NAME>}"
MODEL_PATH="${MODEL_PATH:-<MODEL_ROOT>/${MODEL_NAME}/hf_step${STEP}}"

# Conda environment (modify to match your setup)
# source <CONDA_ROOT>/etc/profile.d/conda.sh
# conda activate <YOUR_ENV>
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

echo "launcher which python: $(which python)"
echo "launcher PYTHON_BIN: $PYTHON_BIN"

img_cfg_scale="${IMG_CFG_SCALE:-1.0}"
cfg_scale="${CFG_SCALE:-4.0}"
cfg_interval_start="${CFG_INTERVAL_START:-0.0}"
cfg_interval_end="${CFG_INTERVAL_END:-1.0}"
timestep_shift="${TIMESTEP_SHIFT:-3.0}"
cfg_norm="${CFG_NORM:-none}"
num_steps="${NUM_STEPS:-50}"
min_pixels="${MIN_PIXELS:-1048576}"
max_pixels="${MAX_PIXELS:-4194304}"
max_memory_per_gpu_gb="${MAX_MEMORY_PER_GPU_GB:-60}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
INFERENCE_MODE="${INFERENCE_MODE:-interleave}"
RESUME="${RESUME:-0}"
LIMIT="${LIMIT:-}"
TARGET_IMAGE_SIZE="${TARGET_IMAGE_SIZE:-}"

CFG_INTERVAL_TAG=$(printf "%s_%s" "$cfg_interval_start" "$cfg_interval_end" | sed "s/-/neg/g; s/\.//g")
TIMESTEP_TAG=$(printf "%s" "$timestep_shift" | sed "s/-/neg/g; s/\.//g")
TARGET_IMAGE_TAG=""
if [[ -n "$TARGET_IMAGE_SIZE" ]]; then
  TARGET_IMAGE_TAG="_imgsz_${TARGET_IMAGE_SIZE//[^0-9xX]/_}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-<OUTPUT_ROOT>/realunify/${MODEL_NAME}_${STEP}_cfg_${img_cfg_scale//./}_${cfg_scale//./}_interval_${CFG_INTERVAL_TAG}_ts_${TIMESTEP_TAG}_norm_${cfg_norm}${TARGET_IMAGE_TAG}/${INFERENCE_MODE}}"
DATA_PATH="${DATA_PATH:-<DATA_ROOT>/RealUnify/GEU_step_processed.jsonl}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/worker_logs}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${OUTPUT_DIR}/realunify_results.jsonl}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

build_device_groups() {
  local devices_csv="$1"
  local per_worker="$2"
  local joined_group

  if (( per_worker <= 0 )); then
    echo "GPUS_PER_WORKER must be positive, got $per_worker"
    exit 1
  fi

  IFS=',' read -r -a ALL_VISIBLE_DEVICES <<< "$devices_csv"
  if (( ${#ALL_VISIBLE_DEVICES[@]} == 0 )); then
    echo "No visible devices were provided."
    exit 1
  fi
  if (( ${#ALL_VISIBLE_DEVICES[@]} % per_worker != 0 )); then
    echo "Visible GPU count ${#ALL_VISIBLE_DEVICES[@]} is not divisible by GPUS_PER_WORKER=$per_worker"
    exit 1
  fi

  DEVICE_GROUPS=()
  for ((i=0; i<${#ALL_VISIBLE_DEVICES[@]}; i+=per_worker)); do
    joined_group=$(IFS=,; echo "${ALL_VISIBLE_DEVICES[*]:i:per_worker}")
    DEVICE_GROUPS+=("$joined_group")
  done
}

build_device_groups "$CUDA_VISIBLE_DEVICES" "$GPUS_PER_WORKER"
TOTAL_NUM_SHARDS="${#DEVICE_GROUPS[@]}"

echo "MODEL_PATH=$MODEL_PATH"
echo "INFERENCE_MODE=$INFERENCE_MODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPUS_PER_WORKER=$GPUS_PER_WORKER"
echo "TOTAL_NUM_SHARDS=$TOTAL_NUM_SHARDS"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TARGET_IMAGE_SIZE=${TARGET_IMAGE_SIZE:-<smart_resize>}"

pids=()
for shard_rank in "${!DEVICE_GROUPS[@]}"; do
  worker_devices="${DEVICE_GROUPS[$shard_rank]}"
  worker_log="$LOG_DIR/realunify_shard_${shard_rank}.log"
  echo "Launching shard ${shard_rank}/${TOTAL_NUM_SHARDS} on GPUs ${worker_devices}. Log: ${worker_log}"
  (
    export CUDA_VISIBLE_DEVICES="$worker_devices"
    cmd=("$PYTHON_BIN" Realunify/inference_realunify.py
      --model_path "$MODEL_PATH"
      --data_path "$DATA_PATH"
      --output_dir "$OUTPUT_DIR"
      --min_pixels "$min_pixels"
      --max_pixels "$max_pixels"
      --cfg_scale "$cfg_scale"
      --img_cfg_scale "$img_cfg_scale"
      --cfg_interval "$cfg_interval_start" "$cfg_interval_end"
      --cfg_norm "$cfg_norm"
      --timestep_shift "$timestep_shift"
      --num_steps "$num_steps"
      --inference_mode "$INFERENCE_MODE"
      --device_map auto
      --max_memory_per_gpu_gb "$max_memory_per_gpu_gb"
      --num_shards "$TOTAL_NUM_SHARDS"
      --shard_rank "$shard_rank")

    if [[ "$RESUME" == "1" ]]; then
      cmd+=(--resume)
    fi
    if [[ -n "$LIMIT" ]]; then
      cmd+=(--limit "$LIMIT")
    fi
    if [[ -n "$TARGET_IMAGE_SIZE" ]]; then
      cmd+=(--target_image_size "$TARGET_IMAGE_SIZE")
    fi

    "${cmd[@]}"
  ) >"$worker_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "At least one RealUnify worker failed. Check logs under $LOG_DIR."
  exit 1
fi

python Realunify/merge_shards.py \
  --data_path "$DATA_PATH" \
  --shard_dir "$OUTPUT_DIR/shards" \
  --output_file "$MERGED_OUTPUT"

python Realunify/calculate_score.py \
  --input_file "$MERGED_OUTPUT"

echo "RealUnify large-model run finished."
echo "Merged results: $MERGED_OUTPUT"
