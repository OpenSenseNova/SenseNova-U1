#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "$SCRIPT_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRANSFORMERS_VERBOSITY=error

# Generation settings
MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1-8B-MoT}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sensenova/tiif}"

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
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
SAVE_SIZE="${SAVE_SIZE:-}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
CFG_NORM="${CFG_NORM:-global}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRID_ROWS="${GRID_ROWS:-1}"
GRID_COLS="${GRID_COLS:-1}"

TIIFBENCH_SPLIT="${TIIFBENCH_SPLIT:-testmini}"
TIIFBENCH_PROMPT_DIR="${TIIFBENCH_PROMPT_DIR:-${SCRIPT_DIR}/data/${TIIFBENCH_SPLIT}_prompts}"
TIIFBENCH_EVAL_PROMPT_DIR="${TIIFBENCH_EVAL_PROMPT_DIR:-${SCRIPT_DIR}/data/${TIIFBENCH_SPLIT}_eval_prompts}"
TIIFBENCH_EVAL_MODEL="${TIIFBENCH_EVAL_MODEL:-gpt-4o}"
TIIFBENCH_AZURE_ENDPOINT="${TIIFBENCH_AZURE_ENDPOINT:-}"
TIIFBENCH_API_VERSION="${TIIFBENCH_API_VERSION:-2025-01-01-preview}"
TIIFBENCH_SPECIFIC_FILE="${TIIFBENCH_SPECIFIC_FILE:-}"
TIIFBENCH_EVAL_MODEL_TAG="${TIIFBENCH_EVAL_MODEL_TAG:-sensenova-u1}"

if (( NUM_NODES <= 0 )); then
  echo "NUM_NODES must be positive, got $NUM_NODES"
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"
  exit 1
fi
if [[ ! -d "$TIIFBENCH_PROMPT_DIR" ]]; then
  echo "Prompt dir not found: $TIIFBENCH_PROMPT_DIR"
  exit 1
fi
if [[ ! -d "$TIIFBENCH_EVAL_PROMPT_DIR" ]]; then
  echo "Eval prompt dir not found: $TIIFBENCH_EVAL_PROMPT_DIR"
  exit 1
fi

BENCH_NAME="TIIFBench-${TIIFBENCH_SPLIT}"
IMAGE_DIR="${OUTPUT_DIR}/${BENCH_NAME}"
RESULTS_DIR="${OUTPUT_DIR}/tiifbench-${TIIFBENCH_SPLIT}_results"

mkdir -p "$OUTPUT_DIR"

if (( RUN_GENERATION != 0 )); then
  GEN_ARGS=(
    --model_path "$MODEL_PATH"
    --output_dir "$IMAGE_DIR"
    --resolution "$IMAGE_SIZE"
    --cfg_scale "$CFG_SCALE"
    --cfg_norm "$CFG_NORM"
    --timestep_shift "$TIMESTEP_SHIFT"
    --num_steps "$NUM_STEPS"
    --batch_size "$BATCH_SIZE"
    --rows "$GRID_ROWS"
    --cols "$GRID_COLS"
    --input_folder "$TIIFBENCH_PROMPT_DIR"
    --num_shards "$NUM_NODES"
    --shard_rank "$NODE_RANK"
  )
  if [[ -n "$TIIFBENCH_SPECIFIC_FILE" ]]; then
    GEN_ARGS+=(--specific_file "$TIIFBENCH_SPECIFIC_FILE")
  fi
  if [[ -n "$SAVE_SIZE" ]]; then
    GEN_ARGS+=(--save_size "$SAVE_SIZE")
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$GPUS" eval_tiif.py "${GEN_ARGS[@]}"
fi

if (( RUN_EVAL == 0 )); then
  exit 0
fi
if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
  echo "Skipping TIIFBench eval on node ${NODE_RANK}; eval runs only on node 0."
  exit 0
fi
if [[ -z "${API_KEY:-}" ]]; then
  echo "API_KEY is required for TIIFBench evaluation."
  exit 1
fi

mkdir -p "$RESULTS_DIR"
EVAL_JSON_DIR="${RESULTS_DIR}/eval_json"
mkdir -p "$EVAL_JSON_DIR"

eval_cmd=(
  python "${SCRIPT_DIR}/eval/eval_with_vlm_mp.py"
  --jsonl_dir "$TIIFBENCH_EVAL_PROMPT_DIR"
  --image_dir "$IMAGE_DIR"
  --eval_model "$TIIFBENCH_EVAL_MODEL_TAG"
  --output_dir "$EVAL_JSON_DIR"
  --model "$TIIFBENCH_EVAL_MODEL"
  --api_key "$API_KEY"
  --api_version "$TIIFBENCH_API_VERSION"
)
if [[ -n "$TIIFBENCH_AZURE_ENDPOINT" ]]; then
  eval_cmd+=(--azure_endpoint "$TIIFBENCH_AZURE_ENDPOINT")
fi

"${eval_cmd[@]}"

python "${SCRIPT_DIR}/eval/summary_results.py" --input_dir "$EVAL_JSON_DIR"
python "${SCRIPT_DIR}/eval/summary_dimension_results.py" \
  --input_excel "${EVAL_JSON_DIR}/result_summary.xlsx" \
  --output_txt "${RESULTS_DIR}/result_summary_dimension.txt"
