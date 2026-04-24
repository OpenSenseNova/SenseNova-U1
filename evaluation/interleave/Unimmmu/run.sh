#!/bin/bash
# Inference helper for the Unimmmu benchmark
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
OUTPUT_DIR="${OUTPUT_DIR:-<OUTPUT_ROOT>/unimmmu/${MODEL_NAME}_${STEP}_cfg_${img_cfg_scale//./}_${cfg_scale//./}_interval_${CFG_INTERVAL_TAG}_ts_${TIMESTEP_TAG}_norm_${cfg_norm}}"

# Data path
DATA_PATH="${DATA_PATH:-<DATA_ROOT>/unimmmu/vqa/unimmmu_direct.jsonl}"
BENCHMARK_PATH="${BENCHMARK_PATH:-}"

# ============================================================================
# i2t mode - Test run with limited samples
# ============================================================================

run_i2t_test() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/i2t_test" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --inference_mode i2t \
        --limit 5 \
        --seed 42
}

# ============================================================================
# i2t mode - Single GPU full run
# ============================================================================

run_i2t() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/i2t" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --inference_mode i2t \
        --seed 42
}

# ============================================================================
# i2t mode - Multi-GPU
# ============================================================================

run_i2t_multi() {
    NUM_GPUS=${1:-8}

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29502 \
        Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/i2t" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --inference_mode i2t \
        --seed 42
}

# ============================================================================
# i2t mode - Resume
# ============================================================================

run_i2t_resume() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/i2t" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --inference_mode i2t \
        --resume \
        --seed 42
}

# ============================================================================
# Interleave mode - Test run
# ============================================================================

run_interleave_test() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/interleave_test" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale "${cfg_scale}" \
        --img_cfg_scale "${img_cfg_scale}" \
        --cfg_interval "${cfg_interval_start}" "${cfg_interval_end}" \
        --cfg_norm "${cfg_norm}" \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --limit 5 \
        --seed 42
}

# ============================================================================
# Interleave mode - Single GPU
# ============================================================================

run_interleave() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/interleave" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale "${cfg_scale}" \
        --img_cfg_scale "${img_cfg_scale}" \
        --cfg_interval "${cfg_interval_start}" "${cfg_interval_end}" \
        --cfg_norm "${cfg_norm}" \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --seed 42
}

# ============================================================================
# Interleave mode - Multi-GPU
# ============================================================================

run_interleave_multi() {
    NUM_GPUS=${1:-8}

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29503 \
        Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/interleave" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale "${cfg_scale}" \
        --img_cfg_scale "${img_cfg_scale}" \
        --cfg_interval "${cfg_interval_start}" "${cfg_interval_end}" \
        --cfg_norm "${cfg_norm}" \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --seed 42
}

# ============================================================================
# Interleave mode - Resume
# ============================================================================

run_interleave_resume() {
    python Unimmmu/inference_unimmmu.py \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/interleave" \
        --min_pixels 1048576 \
        --max_pixels 4194304 \
        --cfg_scale "${cfg_scale}" \
        --img_cfg_scale "${img_cfg_scale}" \
        --cfg_interval "${cfg_interval_start}" "${cfg_interval_end}" \
        --cfg_norm "${cfg_norm}" \
        --num_steps 50 \
        --inference_mode interleave \
        --timestep_shift 1.0 \
        --resume \
        --seed 42
}

# ============================================================================
# Calculate score
# ============================================================================

run_score() {
    INPUT_FILE=${1:-"${OUTPUT_DIR}/i2t/unimmmu_results.jsonl"}
    OUTPUT_SCORE_DIR=${2:-""}

    if [ -z "${BENCHMARK_PATH}" ]; then
        echo "Skipping Unimmmu scoring: set BENCHMARK_PATH to the benchmark repository path."
        return 0
    fi

    if [ -n "${OUTPUT_SCORE_DIR}" ]; then
        python Unimmmu/calculate_score.py \
            --input_file "${INPUT_FILE}" \
            --output_dir "${OUTPUT_SCORE_DIR}" \
            --benchmark_path "${BENCHMARK_PATH}"
    else
        python Unimmmu/calculate_score.py \
            --input_file "${INPUT_FILE}" \
            --benchmark_path "${BENCHMARK_PATH}"
    fi
}

# ============================================================================
# Main
# ============================================================================

# Parse command line arguments
case "$1" in
    "i2t_test")
        run_i2t_test
        run_score "${OUTPUT_DIR}/i2t_test/unimmmu_results.jsonl"
        ;;
    "i2t")
        run_i2t
        run_score "${OUTPUT_DIR}/i2t/unimmmu_results.jsonl"
        ;;
    "i2t_multi")
        run_i2t_multi "${2}"
        run_score "${OUTPUT_DIR}/i2t/unimmmu_results.jsonl"
        ;;
    "i2t_resume")
        run_i2t_resume
        run_score "${OUTPUT_DIR}/i2t/unimmmu_results.jsonl"
        ;;
    "interleave_test")
        run_interleave_test
        run_score "${OUTPUT_DIR}/interleave_test/unimmmu_results.jsonl"
        ;;
    "interleave")
        run_interleave
        run_score "${OUTPUT_DIR}/interleave/unimmmu_results.jsonl"
        ;;
    "interleave_multi")
        run_interleave_multi "${2}"
        run_score "${OUTPUT_DIR}/interleave/unimmmu_results.jsonl"
        ;;
    "interleave_resume")
        run_interleave_resume
        run_score "${OUTPUT_DIR}/interleave/unimmmu_results.jsonl"
        ;;
    "device_map_multi")
        GPUS_PER_WORKER="${2:-2}" INFERENCE_MODE="${INFERENCE_MODE:-i2t}" \
        bash Unimmmu/launch_device_map_multi.sh
        ;;
    "score")
        run_score "${2:-}" "${3:-}"
        ;;
    "" )
        run_interleave_multi 8
        run_score "${OUTPUT_DIR}/interleave/unimmmu_results.jsonl"
        ;;
    *)
        echo "Inference helper for the Unimmmu benchmark"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  i2t_test          - Test with 5 samples (i2t mode)"
        echo "  i2t               - Single GPU full run (i2t mode)"
        echo "  i2t_multi         - Multi-GPU run (i2t mode)"
        echo "                      Usage: $0 i2t_multi [num_gpus]"
        echo "  i2t_resume        - Resume previous i2t run"
        echo "  interleave_test   - Test with 5 samples (interleave mode)"
        echo "  interleave        - Single GPU full run (interleave mode)"
        echo "  interleave_multi  - Multi-GPU run (interleave mode)"
        echo "                      Usage: $0 interleave_multi [num_gpus]"
        echo "  interleave_resume - Resume previous interleave run"
        echo "  device_map_multi  - Large-model run, default 2 GPUs per worker with shard merge"
        echo "                      Usage: $0 device_map_multi [gpus_per_worker]"
        echo "  score             - Calculate score from results"
        echo "                      Usage: $0 score [input_file] [output_dir]"
        echo ""
        echo "Modes:"
        echo "  i2t:        Multi-image understanding only (model.chat)"
        echo "  interleave: Multimodal reasoning with thinking (model.interleave_gen)"
        echo ""
        echo "Task Types (524 total):"
        echo "  geometry: 140 (calculation/proving, 1 image)"
        echo "  jigsaw:   150 (binary choice, 3 images)"
        echo "  maze:     150 (path finding, 1 image)"
        echo "  sliding:   84 (puzzle solving, 2 images)"
        echo ""
        echo "Set BENCHMARK_PATH before enabling Unimmmu scoring."
        echo ""
        echo "Before running, modify MODEL_PATH in this script."
        echo "Default timestep_shift: ${timestep_shift}"
        ;;
esac
