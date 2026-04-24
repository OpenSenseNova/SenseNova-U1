#!/usr/bin/env bash
# Conda activate hooks may read unset variables, so avoid nounset here.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "$SCRIPT_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Generation settings
MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1-8B-MoT-SFT}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?BENCHMARK_ROOT must point at the CVTG-2K dataset root}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sensenova/cvtg}"

# TextCrafter evaluation settings (required only when RUN_EVAL=1)
TEXTCRAFTER_ROOT="${TEXTCRAFTER_ROOT:-}"
PADDLEOCR_SOURCE_DIR="${PADDLEOCR_SOURCE_DIR:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
RESULT_FILE="${RESULT_FILE:-${OUTPUT_DIR}/CVTG_results.json}"

LAUNCH_MODE="${LAUNCH_MODE:-device_map_multi}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
if [[ -z "${RUN_EVAL+x}" ]]; then
  if (( NUM_NODES > 1 )); then
    RUN_EVAL=0
  else
    RUN_EVAL=1
  fi
fi

GPUS="${GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-2}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
MAX_MEMORY_PER_GPU_GB="${MAX_MEMORY_PER_GPU_GB:-70}"

IMAGE_SIZE="${IMAGE_SIZE:-2048}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-1.0}"
LOG_DIR="${LOG_DIR:-cvtg_worker_logs/node_${NODE_RANK}}"
CVTG_SUBSETS="${CVTG_SUBSETS:-CVTG,CVTG-Style}"
CVTG_AREAS="${CVTG_AREAS:-2,3,4,5}"
TARGET_KEYS="${TARGET_KEYS:-}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --benchmark_root "$BENCHMARK_ROOT"
  --output_dir "$OUTPUT_DIR"
  --image_size "$IMAGE_SIZE"
  --num_steps "$NUM_STEPS"
  --cfg_scale "$CFG_SCALE"
  --timestep_shift "$TIMESTEP_SHIFT"
  --subsets "$CVTG_SUBSETS"
  --areas "$CVTG_AREAS"
)

EXTRA_ARGS=()
if [[ -n "$TARGET_KEYS" ]]; then
  EXTRA_ARGS+=(--target_keys "$TARGET_KEYS")
fi

count_visible_gpus() {
  local devices_csv="$1"
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "$devices_csv"
  echo "${#_VISIBLE_DEVICES[@]}"
}

ensure_libgl1() {
  if ldconfig -p 2>/dev/null | grep -q 'libGL\.so\.1'; then
    return 0
  fi

  echo "libGL.so.1 not found. Install it before running CVTG evaluation (e.g. 'apt-get install libgl1')."
  return 1
}

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

if (( NUM_NODES <= 0 )); then
  echo "NUM_NODES must be positive, got $NUM_NODES"
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if (( RUN_GENERATION != 0 )); then
  if [[ "$LAUNCH_MODE" == "device_map" ]]; then
    export CUDA_VISIBLE_DEVICES
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
    TRANSFORMERS_VERBOSITY=error python eval_cvtg.py \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      --device_map "$DEVICE_MAP" \
      --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
      --num_shards "$NUM_NODES" \
      --shard_rank "$NODE_RANK"
  elif [[ "$LAUNCH_MODE" == "device_map_multi" ]]; then
    mkdir -p "$LOG_DIR"
    build_device_groups "$CUDA_VISIBLE_DEVICES" "$GPUS_PER_WORKER"
    LOCAL_NUM_SHARDS="${#DEVICE_GROUPS[@]}"
    TOTAL_NUM_SHARDS=$((NUM_NODES * LOCAL_NUM_SHARDS))
    NODE_SHARD_OFFSET=$((NODE_RANK * LOCAL_NUM_SHARDS))
    pids=()

    for shard_rank in "${!DEVICE_GROUPS[@]}"; do
      worker_devices="${DEVICE_GROUPS[$shard_rank]}"
      global_shard_rank=$((NODE_SHARD_OFFSET + shard_rank))
      worker_log="$LOG_DIR/cvtg_shard_${global_shard_rank}.log"
      echo "Launching global shard ${global_shard_rank}/${TOTAL_NUM_SHARDS} on node ${NODE_RANK}/${NUM_NODES} using GPUs ${worker_devices}. Log: ${worker_log}"
      (
        export CUDA_VISIBLE_DEVICES="$worker_devices"
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
        TRANSFORMERS_VERBOSITY=error python eval_cvtg.py \
          "${COMMON_ARGS[@]}" \
          "${EXTRA_ARGS[@]}" \
          --device_map "$DEVICE_MAP" \
          --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
          --num_shards "$TOTAL_NUM_SHARDS" \
          --shard_rank "$global_shard_rank"
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
      echo "At least one CVTG worker failed. Check logs under $LOG_DIR."
      exit 1
    fi
  elif [[ "$LAUNCH_MODE" == "ddp" ]]; then
    TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$GPUS" eval_cvtg.py \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    echo "Unsupported LAUNCH_MODE=$LAUNCH_MODE. Use device_map, device_map_multi, or ddp."
    exit 1
  fi
fi

if (( RUN_EVAL != 0 )); then
  if [[ -z "$TEXTCRAFTER_ROOT" ]]; then
    echo "TEXTCRAFTER_ROOT must be set to run the TextCrafter evaluation stage."
    exit 1
  fi
  if [[ ! -d "$TEXTCRAFTER_ROOT" ]]; then
    echo "TEXTCRAFTER_ROOT does not exist: $TEXTCRAFTER_ROOT"
    exit 1
  fi
  if [[ ! -d "$BENCHMARK_ROOT" ]]; then
    echo "BENCHMARK_ROOT does not exist: $BENCHMARK_ROOT"
    exit 1
  fi
  ensure_libgl1 || exit 1
  if [[ ! -d "$HOME/.paddleocr" ]]; then
    if [[ -z "$PADDLEOCR_SOURCE_DIR" || ! -d "$PADDLEOCR_SOURCE_DIR" ]]; then
      echo "PaddleOCR cache at \$HOME/.paddleocr is missing and PADDLEOCR_SOURCE_DIR is not a valid directory."
      exit 1
    fi
    cp -r "$PADDLEOCR_SOURCE_DIR" "$HOME/.paddleocr"
  fi

  EVAL_GPUS="${EVAL_GPUS:-$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")}"
  cd "$TEXTCRAFTER_ROOT"
  TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$EVAL_GPUS" unified_metrics_eval.py \
    --benchmark_dir "$BENCHMARK_ROOT" \
    --result_dir "$OUTPUT_DIR" \
    --output_file "$RESULT_FILE" \
    --cache_dir "$HF_CACHE_DIR"
fi
