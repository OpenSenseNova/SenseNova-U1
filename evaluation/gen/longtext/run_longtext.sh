#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="sensenova/SenseNova-U1-8B-MoT"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

python evaluation/gen/longtext/gen_images_longtext.py \
  --model-path "${MODEL_PATH}" \
  --output-dir outputs/longtext/en \
  --lang en

python evaluation/gen/longtext/eval_images_longtext.py \
  --image-dir outputs/longtext/en \
  --output-dir outputs/longtext/en_eval \
  --mode en

python evaluation/gen/longtext/gen_images_longtext.py \
  --model-path "${MODEL_PATH}" \
  --output-dir outputs/longtext/zh \
  --lang zh

python evaluation/gen/longtext/eval_images_longtext.py \
  --image-dir outputs/longtext/zh \
  --output-dir outputs/longtext/zh_eval \
  --mode zh
