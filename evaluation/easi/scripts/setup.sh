#!/usr/bin/env bash
# Set up everything needed to run SenseNova-U1 visual understanding benchmarks:
# LightLLM serving stack + EASI benchmark client + VLMEvalKit endpoint
# registration.
#
# Idempotent: safe to re-run. Each step checks whether it already ran.
#
# What it does (full run):
#   1. Sanity-checks host prereqs (uv, libnuma, nvidia driver).             [lightllm]
#   2. Initializes submodules (LightLLM, EASI, EASI/VLMEvalKit, EASI/lmms-eval).
#   3. Creates .venv-lightllm/ Python 3.10 venv at the repo root.           [lightllm]
#   4. Installs pinned LightLLM deps (strips unbuildable nixl + cchardet).  [lightllm]
#   5. Installs vllm 0.16.0 + LightLLM editable + pandas.                   [lightllm]
#   6. Applies local patches from evaluation/easi/lightllm-stack/patches/.  [lightllm]
#   7. Verifies LightLLM imports + api_server CLI.                          [lightllm]
#   8. Installs EASI benchmark client venv (delegates to EASI's own setup.sh,
#      which creates evaluation/easi/EASI/.venv with Python 3.11 and installs
#      both VLMEvalKit and lmms-eval backends + flash-attn).
#   9. Registers SenseNova-U1 endpoints in VLMEvalKit:
#      - copies evaluation/easi/config/sensenova_models.py into
#        evaluation/easi/EASI/VLMEvalKit/vlmeval/sensenova_models.py
#      - applies evaluation/easi/patches/easi_sensenova_config.patch (7-line hook).
#      Edit config/sensenova_models.py then re-run this script to propagate.
#      Point `api_base` at your own endpoint (localhost, docker host, remote infra
#      team endpoint, etc.) — nothing in EASI itself assumes the server is local.
#
# Flags:
#   --skip-lightllm  skip steps 1, 3-7 — DON'T install the LightLLM serving stack
#                    (use when you already have a SenseNova-U1 OpenAI-compatible
#                    endpoint elsewhere: docker, remote infra team, etc.)
#   --skip-easi      skip step 8 (EASI client venv install — slow, builds flash-attn)
#   --skip-register  skip step 9 (VLMEvalKit endpoint registration)
#
# Usage:
#   bash evaluation/easi/scripts/setup.sh                         # full install
#   bash evaluation/easi/scripts/setup.sh --skip-lightllm         # bring your own endpoint
#   bash evaluation/easi/scripts/setup.sh --skip-easi             # lightllm only
#   bash evaluation/easi/scripts/setup.sh --skip-register         # no config.py patch
set -euo pipefail

SKIP_LIGHTLLM=0
SKIP_EASI=0
SKIP_REGISTER=0
for arg in "$@"; do
  case "${arg}" in
    --skip-lightllm) SKIP_LIGHTLLM=1 ;;
    --skip-easi)     SKIP_EASI=1 ;;
    --skip-register) SKIP_REGISTER=1 ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[error] unknown flag: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [ "${SKIP_LIGHTLLM}" = "1" ] && [ "${SKIP_EASI}" = "1" ] && [ "${SKIP_REGISTER}" = "1" ]; then
  echo "[error] --skip-lightllm + --skip-easi + --skip-register leaves nothing to do" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EASI_ROOT}/../.." && pwd)"

STACK_DIR="${EASI_ROOT}/lightllm-stack"
LIGHTLLM_DIR="${STACK_DIR}/LightLLM"
VENV_DIR="${REPO_ROOT}/.venv-lightllm"
PATCHES_DIR="${STACK_DIR}/patches"
REQ_OUT="/tmp/lightllm-req.txt"

EASI_DIR="${EASI_ROOT}/EASI"
EASI_VENV="${EASI_DIR}/.venv"
EASI_VLMEVAL_DIR="${EASI_DIR}/VLMEvalKit"
EASI_PATCHES_DIR="${EASI_ROOT}/patches"
EASI_CONFIG_DIR="${EASI_ROOT}/config"

log() { echo "[setup] $*"; }
err() { echo "[error] $*" >&2; }

# -------------------------------------------------------------------------
# 1) Host prereqs (LightLLM-side only)
# -------------------------------------------------------------------------
if [ "${SKIP_LIGHTLLM}" = "1" ]; then
  log "skipping LightLLM setup (--skip-lightllm)"
  log "  (assumes you already have a SenseNova-U1 OpenAI-compatible endpoint —"
  log "   point config/sensenova_models.py at it before running step 10)"

  # Still need uv for the EASI venv delegation below, and git to init EASI submodule.
  if [ "${SKIP_EASI}" = "0" ] && ! command -v uv >/dev/null 2>&1; then
    err "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
else
  log "checking host prereqs..."

  if ! command -v uv >/dev/null 2>&1; then
    err "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  if ! ldconfig -p 2>/dev/null | grep -q libnuma.so.1; then
    err "libnuma.so.1 not found. Install system package:"
    err "  sudo apt-get install -y libnuma1 libnuma-dev"
    exit 1
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found. NVIDIA driver required."
    exit 1
  fi

  driver_ok="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F. '{print ($1 >= 550)}')"
  if [ "${driver_ok}" != "1" ]; then
    err "NVIDIA driver < 550.x detected. torch 2.9.1+cu128 requires >= 550."
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    exit 1
  fi
fi

# -------------------------------------------------------------------------
# 2) Initialize submodules
# -------------------------------------------------------------------------
# With --skip-lightllm we only need EASI (+ its nested VLMEvalKit/lmms-eval).
# Otherwise init everything recursively.
if [ "${SKIP_LIGHTLLM}" = "1" ]; then
  log "initializing EASI submodule (skipping LightLLM)..."
  (cd "${REPO_ROOT}" && git submodule update --init --recursive evaluation/easi/EASI)
else
  log "initializing submodules (recursive)..."
  (cd "${REPO_ROOT}" && git submodule update --init --recursive)

  if [ ! -f "${LIGHTLLM_DIR}/setup.py" ]; then
    err "LightLLM submodule still empty after init: ${LIGHTLLM_DIR}"
    err "  run: cd ${REPO_ROOT} && git submodule status"
    exit 1
  fi
fi

# -------------------------------------------------------------------------
# 3) Create LightLLM venv (.venv-lightllm)
# -------------------------------------------------------------------------
if [ "${SKIP_LIGHTLLM}" = "0" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    log "creating Python 3.10 venv at ${VENV_DIR}..."
    uv venv -p 3.10 "${VENV_DIR}"
  else
    log "venv already exists at ${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  uv pip install --quiet --upgrade pip
fi

# -------------------------------------------------------------------------
# 4-6) LightLLM deps, patches, verify
# -------------------------------------------------------------------------
if [ "${SKIP_LIGHTLLM}" = "0" ]; then
  log "filtering upstream requirements (dropping nixl, cchardet)..."
  grep -v "^nixl\|^cchardet" "${LIGHTLLM_DIR}/requirements.txt" > "${REQ_OUT}"

  # Skip reinstall if lightllm is already importable and key pins match.
  # Also verify the editable install points at the current LIGHTLLM_DIR (not a
  # stale pre-move path).
  lightllm_path="$(python -c "import lightllm, os; print(os.path.dirname(os.path.dirname(lightllm.__file__)))" 2>/dev/null || echo "")"
  if python -c "import lightllm, vllm, torch, flashinfer, sgl_kernel, xformers, pandas" >/dev/null 2>&1 \
     && [ "${lightllm_path}" = "${LIGHTLLM_DIR}" ]; then
    log "all required packages already installed — skipping pip phase"
  else
    if [ -n "${lightllm_path}" ] && [ "${lightllm_path}" != "${LIGHTLLM_DIR}" ]; then
      log "existing lightllm install points at stale path ${lightllm_path} — reinstalling"
    fi
    log "installing LightLLM requirements (torch 2.9.1+cu128, flashinfer, sgl-kernel, xformers, ...)"
    uv pip install -r "${REQ_OUT}"

    log "installing vllm 0.16.0..."
    uv pip install --no-cache-dir vllm==0.16.0

    log "installing LightLLM (editable)..."
    uv pip install --no-cache-dir -e "${LIGHTLLM_DIR}"

    log "installing missing transitive deps (pandas)..."
    uv pip install --quiet pandas
  fi

  if [ -d "${PATCHES_DIR}" ] && ls "${PATCHES_DIR}"/*.patch >/dev/null 2>&1; then
    log "applying patches in ${PATCHES_DIR}..."
    for p in "${PATCHES_DIR}"/*.patch; do
      name="$(basename "${p}")"
      if (cd "${LIGHTLLM_DIR}" && git apply --reverse --check "${p}" >/dev/null 2>&1); then
        log "  ${name}: already applied"
        continue
      fi
      if (cd "${LIGHTLLM_DIR}" && git apply --check "${p}" >/dev/null 2>&1); then
        (cd "${LIGHTLLM_DIR}" && git apply "${p}")
        log "  ${name}: applied"
      else
        err "  ${name}: patch does not apply cleanly — file may have drifted upstream"
        err "         inspect manually: cd ${LIGHTLLM_DIR} && git apply --check ${p}"
        exit 1
      fi
    done
  else
    log "no patches to apply"
  fi

  log "verifying imports..."
  python - <<'PY'
import torch, flashinfer, sgl_kernel, xformers, vllm, lightllm
print(f"  torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  vllm       {vllm.__version__}")
print(f"  flashinfer ok")
print(f"  sgl_kernel ok")
print(f"  xformers   ok")
print(f"  lightllm   ok")
PY

  log "verifying LightLLM CLI..."
  python -m lightllm.server.api_server --help | head -3
fi

# -------------------------------------------------------------------------
# 7) Install EASI benchmark client venv (separate from .venv-lightllm)
# -------------------------------------------------------------------------
# Delegates to EASI's own setup.sh, which creates EASI/.venv with Python 3.11
# and installs -e VLMEvalKit -e lmms-eval + pinned deps + flash-attn.
if [ "${SKIP_EASI}" = "1" ]; then
  log "skipping EASI client install (--skip-easi)"
elif [ -d "${EASI_VENV}" ] && "${EASI_VENV}/bin/python" -c "import vlmeval" >/dev/null 2>&1; then
  log "EASI client venv already has vlmeval — skipping install"
else
  log "installing EASI client deps (creates ${EASI_VENV}, may take several minutes)..."
  log "  (delegating to ${EASI_DIR}/scripts/setup.sh)"
  # Deactivate LightLLM venv if active so EASI's setup.sh creates its own cleanly.
  deactivate 2>/dev/null || true
  (cd "${EASI_DIR}" && bash scripts/setup.sh)
  # Re-activate LightLLM venv only if we set it up earlier.
  if [ "${SKIP_LIGHTLLM}" = "0" ] && [ -d "${VENV_DIR}" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
fi

# -------------------------------------------------------------------------
# 8) Register local LightLLM endpoints with EASI's VLMEvalKit
# -------------------------------------------------------------------------
# Two-step wire-up, both idempotent:
#   (a) copy evaluation/easi/config/sensenova_models.py into VLMEvalKit/vlmeval/
#   (b) apply evaluation/easi/patches/easi_sensenova_config.patch to
#       VLMEvalKit/vlmeval/config.py so it imports those entries and updates
#       supported_VLM at module load.
# Edit the endpoint/port/max_tokens values in config/sensenova_models.py
# then re-run this script to propagate.
if [ "${SKIP_REGISTER}" = "1" ]; then
  log "skipping endpoint registration (--skip-register)"
elif [ ! -d "${EASI_VLMEVAL_DIR}" ]; then
  log "skipping endpoint registration (VLMEvalKit submodule not initialized)"
else
  log "registering SenseNova-U1 endpoints in VLMEvalKit..."

  # (a) copy the editable config module
  src="${EASI_CONFIG_DIR}/sensenova_models.py"
  dst="${EASI_VLMEVAL_DIR}/vlmeval/sensenova_models.py"
  if [ ! -f "${src}" ]; then
    err "  missing ${src} — can't register endpoints"
    exit 1
  fi
  cp -f "${src}" "${dst}"
  log "  copied sensenova_models.py -> ${dst}"

  # (b) apply the config.py hook patch (idempotent)
  patch_file="${EASI_PATCHES_DIR}/easi_sensenova_config.patch"
  if [ ! -f "${patch_file}" ]; then
    err "  missing ${patch_file}"
    exit 1
  fi
  if (cd "${EASI_VLMEVAL_DIR}" && git apply --reverse --check "${patch_file}" >/dev/null 2>&1); then
    log "  easi_sensenova_config.patch: already applied"
  elif (cd "${EASI_VLMEVAL_DIR}" && git apply --check "${patch_file}" >/dev/null 2>&1); then
    (cd "${EASI_VLMEVAL_DIR}" && git apply "${patch_file}")
    log "  easi_sensenova_config.patch: applied"
  else
    err "  easi_sensenova_config.patch does not apply cleanly — inspect manually:"
    err "    cd ${EASI_VLMEVAL_DIR} && git apply --check ${patch_file}"
    exit 1
  fi
fi

log "done."
log ""
log "next steps:"
if [ "${SKIP_LIGHTLLM}" = "1" ]; then
  log "  - point config/sensenova_models.py at YOUR endpoint, then propagate:"
  log "      \$EDITOR evaluation/easi/config/sensenova_models.py"
  log "      bash evaluation/easi/scripts/setup.sh --skip-lightllm --skip-easi"
  log ""
  log "  - run a benchmark:"
  log "      source evaluation/easi/EASI/.venv/bin/activate"
  log "      cd evaluation/easi/EASI"
  log "      python scripts/submissions/run_easi_eval.py --model SenseNova-U1-8B-MoT-Local --benchmarks blink"
else
  log "  - launch server (weights auto-downloaded on first call):"
  log "      bash evaluation/easi/scripts/serve.sh                 # 8b-mot → port 8000"
  log "      # or multi-replica DP (same script, DP env flips mode):"
  log "      DP=4 TP=2 bash evaluation/easi/scripts/serve.sh"
  log ""
  log "  - run a benchmark (from a second shell, after server is up):"
  log "      source evaluation/easi/EASI/.venv/bin/activate"
  log "      cd evaluation/easi/EASI"
  log "      python scripts/submissions/run_easi_eval.py --model SenseNova-U1-8B-MoT-Local --benchmarks blink"
fi
