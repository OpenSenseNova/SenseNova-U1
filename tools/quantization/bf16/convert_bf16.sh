#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set manually:
U1_SOURCE_DIR="${U1_SOURCE_DIR:-/path/to/hf_checkpoint}"
U1_OUTPUT_DIR="${U1_OUTPUT_DIR:-./outputs/quantization}"

python "$SCRIPT_DIR/convert_bf16.py" \
  --model-path "$U1_SOURCE_DIR" \
  --output "${U1_OUTPUT_DIR}/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/bf16/SenseNova-U1.6-BF16.safetensors"
