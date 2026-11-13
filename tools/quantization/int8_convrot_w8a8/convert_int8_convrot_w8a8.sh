#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set manually:
U1_SOURCE_DIR="${U1_SOURCE_DIR:-/path/to/hf_checkpoint}"
U1_OUTPUT_DIR="${U1_OUTPUT_DIR:-./outputs/quantization}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "$SCRIPT_DIR/convert_w8a8.py" \
  --model-path "$U1_SOURCE_DIR" \
  --output "${U1_OUTPUT_DIR}/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/int8-convrot/SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors"
