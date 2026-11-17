#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

U1_SOURCE_DIR="${U1_SOURCE_DIR:-/path/to/hf_checkpoint}"
U1_OUTPUT_DIR="${U1_OUTPUT_DIR:-./outputs/quantization}"
COMFYUI_BF16_OUTPUT="${COMFYUI_BF16_OUTPUT:-$U1_OUTPUT_DIR/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/checkpoints/SenseNova-U1.6-BF16.safetensors}"
COMFYUI_BF16_REPORT="${COMFYUI_BF16_REPORT:-${COMFYUI_BF16_OUTPUT%.safetensors}.conversion.json}"

python "$SCRIPT_DIR/convert_bf16.py" \
  "$U1_SOURCE_DIR" \
  "$COMFYUI_BF16_OUTPUT" \
  --report "$COMFYUI_BF16_REPORT"
