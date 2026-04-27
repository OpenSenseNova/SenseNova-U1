#!/usr/bin/env bash
# Download SenseNova-U1 weights from HuggingFace into <repo_root>/models/.
#
# Usage:
#   bash evaluation/easi/scripts/download_weights.sh 8b-mot   # sensenova/SenseNova-U1-8B-MoT (reasoning)
#
# Requires: .venv-lightllm activated (has huggingface_hub installed).
# First-time use: `uv pip install huggingface_hub` if `hf` command not found.
# Optional: export HF_TOKEN=... if the repo gates downloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-lightllm"
MODELS_DIR="${REPO_ROOT}/models"
mkdir -p "${MODELS_DIR}"

# Auto-activate .venv-lightllm if it exists and we're not already in it.
# Avoids picking up `hf` from an arbitrary venv that may lack hf_transfer etc.
if [ -d "${VENV_DIR}" ] && { [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV}" != "${VENV_DIR}" ]; }; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# If HF_HUB_ENABLE_HF_TRANSFER=1 is set but hf_transfer isn't importable, fall
# back to plain HTTP downloads rather than crashing mid-file.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" = "1" ]; then
  if ! python -c "import hf_transfer" >/dev/null 2>&1; then
    echo "[warn] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer not installed — disabling for this run" >&2
    unset HF_HUB_ENABLE_HF_TRANSFER
  fi
fi

# Prefer the new `hf` CLI (huggingface_hub >= 0.34). Fall back to huggingface-cli.
if command -v hf >/dev/null 2>&1; then
  DL="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL="huggingface-cli download"
else
  echo "[error] neither 'hf' nor 'huggingface-cli' found. Activate .venv-lightllm and run: uv pip install huggingface_hub" >&2
  exit 1
fi

download() {
  local repo_id="$1"
  local local_dir="${MODELS_DIR}/$(basename "${repo_id}")"
  echo "[download] ${repo_id} -> ${local_dir}"
  ${DL} "${repo_id}" --local-dir "${local_dir}"
}

target="${1:-8b-mot}"
case "${target}" in
  8b-mot) download "sensenova/SenseNova-U1-8B-MoT" ;;
  *)
    echo "[error] unknown target: ${target}. Use: 8b-mot" >&2
    exit 1
    ;;
esac

echo "[done] weights at ${MODELS_DIR}"
