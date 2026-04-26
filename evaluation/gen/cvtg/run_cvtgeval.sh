#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "$SCRIPT_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Generation settings
MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1-8B-MoT}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sensenova/cvtg}"

# TextCrafter evaluation settings (used only when RUN_EVAL=1)
PADDLEOCR_SOURCE_DIR="${PADDLEOCR_SOURCE_DIR:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
RESULT_FILE="${RESULT_FILE:-${OUTPUT_DIR}/CVTG_results.json}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY_PER_GPU_GB="${MAX_MEMORY_PER_GPU_GB:-70}"

IMAGE_SIZE="${IMAGE_SIZE:-2048}"
SAVE_SIZE="${SAVE_SIZE:-}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-7.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-1.0}"
CVTG_SUBSETS="${CVTG_SUBSETS:-CVTG,CVTG-Style}"
CVTG_AREAS="${CVTG_AREAS:-2,3,4,5}"
TARGET_KEYS="${TARGET_KEYS:-}"

GEN_ARGS=(
  --model_path "$MODEL_PATH"
  --benchmark_root "$BENCHMARK_ROOT"
  --output_dir "$OUTPUT_DIR"
  --image_size "$IMAGE_SIZE"
  --num_steps "$NUM_STEPS"
  --cfg_scale "$CFG_SCALE"
  --timestep_shift "$TIMESTEP_SHIFT"
  --subsets "$CVTG_SUBSETS"
  --areas "$CVTG_AREAS"
  --device_map "$DEVICE_MAP"
  --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB"
)
if [[ -n "$TARGET_KEYS" ]]; then
  GEN_ARGS+=(--target_keys "$TARGET_KEYS")
fi
if [[ -n "$SAVE_SIZE" ]]; then
  GEN_ARGS+=(--save_size "$SAVE_SIZE")
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

mkdir -p "$OUTPUT_DIR"

if (( RUN_GENERATION != 0 )); then
  export CUDA_VISIBLE_DEVICES
  TRANSFORMERS_VERBOSITY=error python eval_cvtg.py "${GEN_ARGS[@]}"
fi

if (( RUN_EVAL != 0 )); then
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
  TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$EVAL_GPUS" \
    "${SCRIPT_DIR}/unified_metrics_eval.py" \
    --benchmark_dir "$BENCHMARK_ROOT" \
    --result_dir "$OUTPUT_DIR" \
    --output_file "$RESULT_FILE" \
    --cache_dir "$HF_CACHE_DIR"
fi
