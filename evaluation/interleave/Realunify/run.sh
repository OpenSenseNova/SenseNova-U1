#!/bin/bash
# NEO Model Inference for RealUnify Benchmark
# Usage examples for different scenarios

# ============================================================================
# Configuration
# ============================================================================

# Model path (override via env: MODEL_NAME=xxx STEP=yyy bash run.sh ...)
STEP="${STEP:-<STEP_TAG>}"
MODEL_NAME="${MODEL_NAME:-<YOUR_MODEL_NAME>}"
MODEL_PATH="${MODEL_PATH:-<MODEL_ROOT>/${MODEL_NAME}/hf_step${STEP}}"

# Conda environment (modify to match your setup)
# export PATH=<CONDA_ROOT>/bin:$PATH
# source <CONDA_ROOT>/etc/profile.d/conda.sh
# conda activate <YOUR_ENV>
which python

# Output directory

img_cfg_scale=1.0
cfg_scale=4.0
cfg_interval_start=0.0
cfg_interval_end=1.0           # 1.5/4.0
timestep_shift=3.0
cfg_norm=none
CFG_INTERVAL_TAG=$(printf "%s_%s" "$cfg_interval_start" "$cfg_interval_end" | sed "s/-/neg/g; s/\.//g")
TIMESTEP_TAG=$(printf "%s" "$timestep_shift" | sed "s/-/neg/g; s/\.//g")
OUTPUT_DIR=<OUTPUT_ROOT>/realunify/${MODEL_NAME}_${STEP}_cfg_${img_cfg_scale//./}_${cfg_scale//./}_interval_${CFG_INTERVAL_TAG}_ts_${TIMESTEP_TAG}_norm_${cfg_norm}

# Data path (default)
DATA_PATH="${DATA_PATH:-<DATA_ROOT>/RealUnify/GEU_step_processed.jsonl}"
UEG_DATA_PATH="${UEG_DATA_PATH:-<DATA_ROOT>/RealUnify/UEG_step.json}"

num_steps=50
seed=42

# ============================================================================
# Example 1: Single GPU - Test run with limited samples
# ============================================================================

run_test() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/test \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --limit 5 \
        --seed 42
}

# ============================================================================
# Example 2: Single GPU - Full run
# ============================================================================

run_single_gpu() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --seed 42
}

# ============================================================================
# Example 3: Multi-GPU with torchrun
# ============================================================================

run_multi_gpu() {
    NUM_GPUS=${1:-8}

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --seed 42
}

# ============================================================================
# Example 4: Resume from checkpoint
# ============================================================================

run_resume() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --resume \
        --seed 42
}

# ============================================================================
# Example 5: Custom parameters
# ============================================================================

run_custom() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/custom \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale 2.0 \
        --img_cfg_scale 1.5 \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 100 \
        --seed 42
}

# ============================================================================
# Example 6: Calculate score
# ============================================================================

run_score() {
    INPUT_FILE=${1:-"${OUTPUT_DIR}/realunify_results.jsonl"}
    OUTPUT_FILE=${2:-""}

    if [ -n "${OUTPUT_FILE}" ]; then
        python Realunify/calculate_score.py \
            --input_file ${INPUT_FILE} \
            --output_file ${OUTPUT_FILE}
    else
        python Realunify/calculate_score.py \
            --input_file ${INPUT_FILE}
    fi
}

# ============================================================================
# Example 7: Interleave mode - Test run
# ============================================================================

run_interleave_test() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/interleave_test \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --limit 5 \
        --seed 42
}

# ============================================================================
# Example 8: Interleave mode - Single GPU
# ============================================================================

run_interleave() {
    python Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/interleave \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --seed 42
}

# ============================================================================
# Example 9: Interleave mode - Multi-GPU
# ============================================================================

run_interleave_multi() {
    NUM_GPUS=${1:-8}

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29501 \
        Realunify/inference_realunify.py \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/interleave \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --seed 42
}

# ============================================================================
# UEG Benchmark
# ============================================================================

run_ueg() {
    local mode=${1:-understand_t2i}
    python Realunify/inference_realunify_ueg.py \
        --model_path ${MODEL_PATH} \
        --data_path ${UEG_DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/ueg_${mode} \
        --inference_mode ${mode} \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps ${num_steps} \
        --seed ${seed}
}

run_ueg_test() {
    local mode=${1:-understand_t2i}
    python Realunify/inference_realunify_ueg.py \
        --model_path ${MODEL_PATH} \
        --data_path ${UEG_DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/ueg_${mode}_test \
        --inference_mode ${mode} \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps ${num_steps} \
        --seed ${seed} \
        --limit 5
}

run_ueg_multi() {
    local mode=${1:-understand_t2i}
    local num_gpus=${2:-8}
    torchrun \
        --nproc_per_node=${num_gpus} \
        --master_port=29502 \
        Realunify/inference_realunify_ueg.py \
        --model_path ${MODEL_PATH} \
        --data_path ${UEG_DATA_PATH} \
        --output_dir ${OUTPUT_DIR}/ueg_${mode} \
        --inference_mode ${mode} \
        --cfg_scale ${cfg_scale} \
        --img_cfg_scale ${img_cfg_scale} \
        --cfg_interval ${cfg_interval_start} ${cfg_interval_end} \
        --cfg_norm ${cfg_norm} \
        --timestep_shift ${timestep_shift} \
        --num_steps ${num_steps} \
        --seed ${seed}
}

run_ueg_score() {
    local input_file=${1}
    local num_workers=${2:-16}
    python Realunify/calculate_score_ueg.py \
        --input_file ${input_file} \
        --num_workers ${num_workers}
}

# ============================================================================
# Main
# ============================================================================

# Parse command line arguments
case "$1" in
    "test")
        run_test
        run_score "${OUTPUT_DIR}/test/realunify_results.jsonl"
        ;;
    "single")
        run_interleave_multi 8
        run_score "${OUTPUT_DIR}/interleave/realunify_results.jsonl"
        ;;
    "multi")
        run_multi_gpu ${2}
        run_score "${OUTPUT_DIR}/realunify_results.jsonl"
        ;;
    "resume")
        run_resume
        run_score "${OUTPUT_DIR}/realunify_results.jsonl"
        ;;
    "custom")
        run_custom
        run_score "${OUTPUT_DIR}/custom/realunify_results.jsonl"
        ;;
    "score")
        run_score ${2} ${3}
        ;;
    "interleave_test")
        run_interleave_test
        run_score "${OUTPUT_DIR}/interleave_test/realunify_results.jsonl"
        ;;
    "interleave")
        run_interleave
        run_score "${OUTPUT_DIR}/interleave/realunify_results.jsonl"
        ;;
    "interleave_multi")
        run_interleave_multi ${2}
        run_score "${OUTPUT_DIR}/interleave/realunify_results.jsonl"
        ;;
    "device_map_multi")
        GPUS_PER_WORKER="${2:-2}" INFERENCE_MODE="${INFERENCE_MODE:-interleave}" \
        bash Realunify/launch_device_map_multi.sh
        ;;
    "ueg")
        run_ueg "${2:-understand_t2i}"
        ;;
    "ueg_test")
        run_ueg_test "${2:-understand_t2i}"
        ;;
    "ueg_multi")
        run_ueg_multi "${2:-understand_t2i}" "${3:-8}"
        ;;
    "ueg_score")
        run_ueg_score "${2}" "${3:-16}"
        ;;
    *)
        echo "NEO Model Inference for RealUnify Benchmark"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  test             - Run test with 5 samples (two-step mode)"
        echo "  single           - Default run on 8 GPUs in interleave mode"
        echo "  multi            - Run on multiple GPUs (default: 8, two-step mode)"
        echo "                     Usage: $0 multi [num_gpus]"
        echo "  resume           - Resume from previous run"
        echo "  custom           - Run with custom parameters"
        echo "  score            - Calculate score from results"
        echo "                     Usage: $0 score [input_file] [output_file]"
        echo "  interleave_test  - Run test with 5 samples (interleave mode)"
        echo "  interleave       - Run on single GPU (interleave mode)"
        echo "  interleave_multi - Run on multiple GPUs (interleave mode)"
        echo "                     Usage: $0 interleave_multi [num_gpus]"
        echo "  device_map_multi - Large-model run, default 2 GPUs per worker with shard merge"
        echo "                     Usage: $0 device_map_multi [gpus_per_worker]"
        echo ""
        echo "  UEG Commands:"
        echo "  ueg              - UEG benchmark inference"
        echo "                     Usage: $0 ueg [understand_t2i|interleave|t2i]"
        echo "  ueg_test         - UEG test with 5 samples"
        echo "                     Usage: $0 ueg_test [understand_t2i|interleave|t2i]"
        echo "  ueg_multi        - UEG multi-GPU inference"
        echo "                     Usage: $0 ueg_multi [mode] [num_gpus]"
        echo "  ueg_score        - Score UEG results (Gemini judge)"
        echo "                     Usage: $0 ueg_score <results.json> [num_workers]"
        echo ""
        echo "Modes:"
        echo "  two-step:   Edit image (it2i) -> Answer question (i2t)"
        echo "  interleave: Direct multimodal reasoning with thinking"
        echo ""
        echo "Note: Output image size automatically matches input image size."
        echo "      Use --min_pixels/--max_pixels to control resize constraints."
        echo ""
        echo "Before running, modify MODEL_PATH in this script."
        echo "Default timestep_shift: ${timestep_shift}"
        ;;
esac
