#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

U1_OUTPUT_DIR="${U1_OUTPUT_DIR:-./outputs/quantization}"
COMFYUI_BF16_OUTPUT="${COMFYUI_BF16_OUTPUT:-$U1_OUTPUT_DIR/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/checkpoints/SenseNova-U1.6-BF16.safetensors}"
COMFYUI_INT8_OUTPUT="${COMFYUI_INT8_OUTPUT:-$U1_OUTPUT_DIR/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/checkpoints/SenseNova-U1.6-INT8-ConvRot.safetensors}"
COMFYUI_INT8_REPORT="${COMFYUI_INT8_REPORT:-${COMFYUI_INT8_OUTPUT%.safetensors}.tsv}"

embedding_args=()
if [[ "${COMFYUI_QUANTIZE_EMBEDDINGS:-1}" == "0" || "${COMFYUI_QUANTIZE_EMBEDDINGS:-1}" == "false" ]]; then
  embedding_args+=(--no-quant-embeddings)
else
  embedding_args+=(--quant-embeddings)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "$SCRIPT_DIR/convert_int8_convrot.py" \
  "${COMFYUI_BF16_OUTPUT}" \
  "${COMFYUI_INT8_OUTPUT}" \
  "${embedding_args[@]}" \
  --verify-report "$COMFYUI_INT8_REPORT"
