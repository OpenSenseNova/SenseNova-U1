#!/usr/bin/env bash
set -euo pipefail

# Generate images using SenseNova-U1-8B-MoT.
MODEL_PATH="sensenova/SenseNova-U1-8B-MoT"
IMAGE_OUTPUT_DIR="outputs/sensenova/bizgeneval"
EVAL_OUTPUT_DIR="outputs/sensenova/bizgeneval_eval"

# Evaluation settings
API_BASE="http://your-api-base/v1"
API_KEY="your-api-key"
JUDGE_MODEL="gemini-3-pro-preview"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

python evaluation/gen/bizgeneval/gen_images_bizgeneval.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${IMAGE_OUTPUT_DIR}"

python evaluation/gen/bizgeneval/eval_images_bizgeneval.py \
  --image-dir "${IMAGE_OUTPUT_DIR}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --api-base "${API_BASE}" \
  --api-key "${API_KEY}" \
  --judge-model "${JUDGE_MODEL}"
