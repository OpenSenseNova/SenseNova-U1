#!/usr/bin/env bash
# Launch LightLLM OpenAI-compatible server(s) for SenseNova-U1.
#
# Two modes, picked automatically by DP:
#
#   DP=1 (default)  — single LightLLM instance bound directly to the canonical
#                     per-model port (8b-mot → 8000).
#                     No load balancer, no extra process.
#
#   DP>1            — DP tp-sharded replicas on backend ports (8100+, step 10)
#                     + a Python load balancer on the canonical port.
#                     Clients always hit the same port regardless of DP, so
#                     VLMEvalKit config.py never needs to change.
#
# Total GPUs used = DP × TP, assigned contiguously from GPU 0 unless GPUS set.
#
# Usage:
#   bash evaluation/easi/scripts/serve.sh                               # DP=1, 8b-mot, GPUs 0-1, port 8000
#   TP=8 GPUS=0,1,2,3,4,5,6,7 bash evaluation/easi/scripts/serve.sh     # single big instance
#   DP=4 TP=2 bash evaluation/easi/scripts/serve.sh                     # 4 replicas × tp=2, LB on 8000
#
# Env vars:
#   MODEL               8b-mot                             (default: 8b-mot)
#   MODEL_DIR           explicit path, overrides MODEL     (default: ./models/SenseNova-U1-8B-MoT)
#   DP                  # of replicas                      (default: 1)
#   TP                  tensor parallel degree / replica   (default: 2)
#   GPUS                CSV of CUDA_VISIBLE_DEVICES ids    (default: 0,1,...,DP*TP-1)
#   HOST                bind address                       (default: 0.0.0.0)
#   LB_PORT             canonical client-facing port       (default: 8000)
#                       (alias: PORT — honored for backcompat when DP=1)
#   BACKEND_BASE_PORT   first backend port when DP>1       (default: 8100, step 10)
#   MAX_LEN             --max_req_total_len                (default: 32768)
#   MEM_FRAC            --mem_fraction                     (default: 0.85)
#   MODEL_NAME          advertised model name              (default: per-model)
#   REASONING           --reasoning_parser                 (default: qwen3)
#                         qwen3: strips <think>...</think> into reasoning_content
#                         qwen3-thinking: force reasoning even w/o <think> tag
#                         "" (empty): disable parser (raw content)
#   NO_AUTO_DL          1 = skip weight auto-download      (default: unset)
#   DETAIL_LOG          1 = --detail_log (per-request DEBUG: timing, prompt, tokens, ViT)
#   LIGHTLLM_LOG_LEVEL  debug|info|warning|error           (default: info)
#   LOG_DIR             dir for replica log files (DP>1)   (default: evaluation/easi/logs/)
#
# Guardrails:
#   - DP*TP > #visible GPUs → fail fast (or set GPUS=... explicitly)
#   - GPUS count must equal DP*TP
#   - Any port (LB + backends) already in use → fail fast with owner PID when discoverable
#   - DP=1 vs DP>1 are mutually exclusive on the canonical port — don't run both
#     concurrently for the same MODEL
#
# Notes:
#   - Ctrl-C / SIGTERM cascade-kills all replicas + LB when DP>1.
#   - Per-replica logs at $LOG_DIR/lightllm-<MODEL>-<port>.log (DP>1 only).
#   - LB passes through health probes + streaming; least in-flight balancing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EASI_ROOT}/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv-lightllm"

# ---------------------------------------------------------------------------
# 1) Resolve model defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-8b-mot}"
case "${MODEL}" in
  8b-mot)
    DEFAULT_DIR="${REPO_ROOT}/models/SenseNova-U1-8B-MoT"
    DEFAULT_LB_PORT=8000
    DEFAULT_BACKEND_BASE=8100
    DEFAULT_MODEL_NAME="sensenova-u1-8b-mot"
    DEFAULT_REASONING="qwen3"
    ;;
  *)
    echo "[error] MODEL must be '8b-mot' (got: ${MODEL})" >&2
    exit 1
    ;;
esac

MODEL_DIR="${MODEL_DIR:-${DEFAULT_DIR}}"
DP="${DP:-1}"
TP="${TP:-2}"
HOST="${HOST:-0.0.0.0}"
# PORT is kept as an alias for LB_PORT for backcompat. LB_PORT wins if both set.
LB_PORT="${LB_PORT:-${PORT:-${DEFAULT_LB_PORT}}}"
BACKEND_BASE_PORT="${BACKEND_BASE_PORT:-${DEFAULT_BACKEND_BASE}}"
MAX_LEN="${MAX_LEN:-32768}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
REASONING="${REASONING-${DEFAULT_REASONING}}"    # unset → default; "" → user disable
DETAIL_LOG="${DETAIL_LOG:-0}"
LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-info}"
LOG_DIR="${LOG_DIR:-${EASI_ROOT}/logs}"

# ---------------------------------------------------------------------------
# 2) Integer validation
# ---------------------------------------------------------------------------
case "${DP}${TP}${LB_PORT}${BACKEND_BASE_PORT}" in *[!0-9]*)
  echo "[error] DP, TP, LB_PORT, BACKEND_BASE_PORT must all be integers" >&2
  exit 1
  ;;
esac
if [ "${DP}" -lt 1 ] || [ "${TP}" -lt 1 ]; then
  echo "[error] DP and TP must be >= 1 (got DP=${DP} TP=${TP})" >&2
  exit 1
fi

# When DP>1, LB_PORT must not overlap the backend range.
if [ "${DP}" -gt 1 ]; then
  backend_end=$((BACKEND_BASE_PORT + 10 * (DP - 1)))
  if [ "${LB_PORT}" -ge "${BACKEND_BASE_PORT}" ] && [ "${LB_PORT}" -le "${backend_end}" ] \
     && [ $(( (LB_PORT - BACKEND_BASE_PORT) % 10 )) -eq 0 ]; then
    echo "[error] LB_PORT=${LB_PORT} collides with backend port range ${BACKEND_BASE_PORT}..${backend_end} (step 10)" >&2
    echo "        move LB_PORT or BACKEND_BASE_PORT so they don't overlap" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 3) GPU sanity
# ---------------------------------------------------------------------------
NEED=$(( DP * TP ))

if [ -n "${GPUS:-}" ]; then
  gpu_count="$(echo "${GPUS}" | awk -F, '{print NF}')"
  if [ "${gpu_count}" != "${NEED}" ]; then
    echo "[error] GPUS has ${gpu_count} entries but DP*TP=${NEED}" >&2
    echo "        provide exactly ${NEED} GPU IDs, or unset GPUS to auto-assign" >&2
    exit 1
  fi
  IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi not found — can't auto-detect GPUs. Set GPUS=..." >&2
    exit 1
  fi
  avail="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [ "${avail}" -lt "${NEED}" ]; then
    echo "[error] DP*TP = ${NEED} GPUs required but only ${avail} visible to nvidia-smi" >&2
    echo "        reduce DP or TP, or set GPUS=... to a subset of available GPUs" >&2
    exit 1
  fi
  GPU_ARR=()
  for i in $(seq 0 $((NEED - 1))); do GPU_ARR+=("${i}"); done
fi

# ---------------------------------------------------------------------------
# 4) Venv activation
# ---------------------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "[error] venv not found at ${VENV_DIR}" >&2
  echo "        run: bash evaluation/easi/scripts/setup.sh" >&2
  exit 1
fi
if [ -z "${VIRTUAL_ENV:-}" ] || [ "${VIRTUAL_ENV}" != "${VENV_DIR}" ]; then
  echo "[serve] activating ${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# ---------------------------------------------------------------------------
# 5) Weight check (once, before any replica forks)
# ---------------------------------------------------------------------------
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  if [ "${NO_AUTO_DL:-0}" = "1" ]; then
    echo "[error] ${MODEL_DIR}/config.json missing" >&2
    echo "        set NO_AUTO_DL=0 or run: bash evaluation/easi/scripts/download_weights.sh ${MODEL}" >&2
    exit 1
  fi
  echo "[serve] config.json missing at ${MODEL_DIR} — downloading ${MODEL}..."
  bash "${SCRIPT_DIR}/download_weights.sh" "${MODEL}"
  if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "[error] download appears to have failed — still no ${MODEL_DIR}/config.json" >&2
    echo "        check HF_TOKEN, network, and org membership for SenseNova" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 6) Pre-flight port probe
# ---------------------------------------------------------------------------
# Catches stale servers on any port we plan to bind. Avoids the
# "replica crash-binds, but zombie on port fakes LB health" failure mode.
port_in_use() {
  local port="$1"
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    exec 3<&-
    return 0
  fi
  return 1
}

ports_to_check=()
if [ "${DP}" -eq 1 ]; then
  # Single-instance mode: bind the canonical port directly.
  ports_to_check+=("${LB_PORT}")
else
  # Multi-replica mode: LB on canonical + each backend.
  ports_to_check+=("${LB_PORT}")
  for i in $(seq 0 $((DP - 1))); do
    ports_to_check+=($((BACKEND_BASE_PORT + i * 10)))
  done
fi

busy=()
for p in "${ports_to_check[@]}"; do
  port_in_use "${p}" && busy+=("${p}")
done

if [ "${#busy[@]}" -gt 0 ]; then
  echo "[error] port(s) already in use: ${busy[*]}" >&2
  for p in "${busy[@]}"; do
    echo "  port ${p}:" >&2
    ss -lntp 2>/dev/null | awk -v pat=":${p} " '$4 ~ pat {print "    " $0}' >&2 \
      || lsof -i":${p}" 2>/dev/null | sed 's/^/    /' >&2 \
      || echo "    (couldn't identify owner; try: sudo lsof -i:${p} or sudo ss -lntp | grep :${p})" >&2
  done
  echo "[error] stop the conflicting process(es) first, or override LB_PORT / BACKEND_BASE_PORT" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 7) Helper: launch one LightLLM instance (blocking unless backgrounded by caller)
# ---------------------------------------------------------------------------
launch_lightllm() {
  local gpus="$1" port="$2"
  local extra=()
  [ -n "${REASONING}" ] && extra+=(--reasoning_parser "${REASONING}")
  [ "${DETAIL_LOG}" = "1" ] && extra+=(--detail_log)
  env CUDA_VISIBLE_DEVICES="${gpus}" \
      LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL}" \
      python -m lightllm.server.api_server \
      --model_dir "${MODEL_DIR}" \
      --model_name "${MODEL_NAME}" \
      --model_owner "sensenova" \
      --host "${HOST}" \
      --port "${port}" \
      --tp "${TP}" \
      --max_req_total_len "${MAX_LEN}" \
      --mem_fraction "${MEM_FRAC}" \
      --trust_remote_code \
      "${extra[@]}"
}

# ---------------------------------------------------------------------------
# 8) Dispatch: single-instance or multi-replica + LB
# ---------------------------------------------------------------------------
if [ "${DP}" -eq 1 ]; then
  # --- single instance, direct on LB_PORT ---
  mkdir -p "${LOG_DIR}"
  PIDFILE="${LOG_DIR}/serve.${MODEL}.pids"
  set -m

  # Self-heal stale PIDs from prior run (same as multi-replica path).
  if [ -f "${PIDFILE}" ]; then
    stale="$(cat "${PIDFILE}" 2>/dev/null | tr '\n' ' ')"
    for pg in ${stale}; do
      [ -z "${pg}" ] && continue
      kill -TERM -"${pg}" 2>/dev/null || true
    done
    sleep 1
    for pg in ${stale}; do
      [ -z "${pg}" ] && continue
      kill -KILL -"${pg}" 2>/dev/null || true
    done
    pkill -KILL -f "lightllm\.server\.api_server.*${MODEL_DIR}" 2>/dev/null || true
    rm -f "${PIDFILE}"
  fi

  echo "[serve] model=${MODEL_NAME} dir=${MODEL_DIR}"
  echo "[serve] GPUS=${GPUS:-$(IFS=,; echo "${GPU_ARR[*]}")} TP=${TP} port=${LB_PORT} reasoning=${REASONING:-off} detail_log=${DETAIL_LOG} log_level=${LIGHTLLM_LOG_LEVEL}"

  _gpus_csv="$(IFS=,; echo "${GPU_ARR[*]}")"
  ( launch_lightllm "${_gpus_csv}" "${LB_PORT}" ) &
  LIGHTLLM_PID=$!
  echo "${LIGHTLLM_PID}" > "${PIDFILE}"

  _cleaning_up=0
  cleanup() {
    [ "${_cleaning_up}" = "1" ] && return
    _cleaning_up=1
    echo ""
    echo "[serve] shutting down..."
    kill -TERM -"${LIGHTLLM_PID}" 2>/dev/null || kill -TERM "${LIGHTLLM_PID}" 2>/dev/null || true
    local waited=0
    while [ "${waited}" -lt 10 ]; do
      kill -0 "${LIGHTLLM_PID}" 2>/dev/null || break
      sleep 1; waited=$((waited + 1))
    done
    kill -KILL -"${LIGHTLLM_PID}" 2>/dev/null || true
    kill -KILL "${LIGHTLLM_PID}" 2>/dev/null || true
    pkill -KILL -P $$ 2>/dev/null || true
    pkill -KILL -f "lightllm\.server\.api_server.*${MODEL_DIR}" 2>/dev/null || true
    rm -f "${PIDFILE}"
    echo "[serve] cleanup done"
  }
  trap cleanup EXIT INT TERM HUP

  wait "${LIGHTLLM_PID}"
  exit $?
fi

# --- multi-replica + LB ---
mkdir -p "${LOG_DIR}"
PIDFILE="${LOG_DIR}/serve.${MODEL}.pids"

# Enable job control so each backgrounded subshell gets its own process group
# (pgid == subshell pid). We kill the whole group on cleanup — catches every
# LightLLM worker / router / zmq / visual server / detokenizer child.
set -m

# ---------------------------------------------------------------------------
# Self-heal: if a previous run left a PID file, try to clean its processes
# ---------------------------------------------------------------------------
if [ -f "${PIDFILE}" ]; then
  echo "[serve] stale PID file found: ${PIDFILE}"
  stale_pgs="$(cat "${PIDFILE}" 2>/dev/null | tr '\n' ' ')"
  for pg in ${stale_pgs}; do
    [ -z "${pg}" ] && continue
    # Negative PID = signal entire process group
    kill -TERM -"${pg}" 2>/dev/null || true
  done
  sleep 1
  for pg in ${stale_pgs}; do
    [ -z "${pg}" ] && continue
    kill -KILL -"${pg}" 2>/dev/null || true
  done
  # Belt + suspenders: kill any lightllm server still tied to this MODEL_DIR
  pkill -KILL -f "lightllm\.server\.api_server.*${MODEL_DIR}" 2>/dev/null || true
  rm -f "${PIDFILE}"
  echo "[serve] stale processes cleaned; continuing..."
fi

echo "[serve] multi-replica mode: DP=${DP} TP=${TP} total_gpus=${NEED}"
echo "[serve] GPU assignment: ${GPU_ARR[*]}"
echo "[serve] backend ports (step 10): $(for i in $(seq 0 $((DP-1))); do echo -n "$((BACKEND_BASE_PORT + i*10)) "; done)"
echo "[serve] LB listens on :${LB_PORT}  (canonical port for ${MODEL})"
echo "[serve] replica logs: ${LOG_DIR}/"
echo "[serve] reasoning=${REASONING:-off} detail_log=${DETAIL_LOG} log_level=${LIGHTLLM_LOG_LEVEL}"

REPLICA_PIDS=()    # pid of each backgrounded subshell (== pgid with job control)
BACKENDS=()
for i in $(seq 0 $((DP - 1))); do
  port=$((BACKEND_BASE_PORT + i * 10))
  start=$((i * TP))
  end=$((start + TP - 1))
  gpus=""
  for j in $(seq "${start}" "${end}"); do
    gpus="${gpus}${gpus:+,}${GPU_ARR[$j]}"
  done
  log="${LOG_DIR}/lightllm-${MODEL}-${port}.log"
  echo "[serve] launching replica $((i+1))/${DP}: GPUS=${gpus} PORT=${port} → ${log}"
  # With `set -m` above, each backgrounded subshell gets its own pgid (== $!).
  # LightLLM's fork-spawned children inherit that pgid, so `kill -- -$pid` on
  # cleanup hits the whole tree (router, tp workers, visual server, zmq, ...).
  ( launch_lightllm "${gpus}" "${port}" ) >"${log}" 2>&1 &
  REPLICA_PIDS+=("$!")
  BACKENDS+=("http://localhost:${port}")
done

BACKENDS_CSV="$(IFS=,; echo "${BACKENDS[*]}")"

# Persist PIDs/PGIDs for self-heal on next run.
{ for pid in "${REPLICA_PIDS[@]}"; do echo "${pid}"; done; } > "${PIDFILE}"

# ---------------------------------------------------------------------------
# Hardened cleanup: SIGTERM process groups → wait → SIGKILL → pattern sweep
# ---------------------------------------------------------------------------
_cleaning_up=0
cleanup() {
  [ "${_cleaning_up}" = "1" ] && return
  _cleaning_up=1
  echo ""
  echo "[serve] shutting down..."

  local all_pids=("${REPLICA_PIDS[@]}" "${LB_PID:-}")

  # 1) SIGTERM the entire process group of each replica + LB.
  for pid in "${all_pids[@]}"; do
    [ -z "${pid}" ] && continue
    kill -TERM -"${pid}" 2>/dev/null \
      || kill -TERM "${pid}" 2>/dev/null || true
  done

  # 2) Poll up to 10s for graceful exit.
  local waited=0
  while [ "${waited}" -lt 10 ]; do
    local alive=0
    for pid in "${all_pids[@]}"; do
      [ -z "${pid}" ] && continue
      kill -0 "${pid}" 2>/dev/null && { alive=1; break; }
    done
    [ "${alive}" = "0" ] && break
    sleep 1
    waited=$((waited + 1))
  done

  # 3) SIGKILL any survivors (groups, then individual pids).
  for pid in "${all_pids[@]}"; do
    [ -z "${pid}" ] && continue
    kill -KILL -"${pid}" 2>/dev/null || true
    kill -KILL "${pid}" 2>/dev/null || true
  done

  # 4) Belt + suspenders: sweep any remaining children of this shell and any
  #    lightllm procs tied to our MODEL_DIR (catches orphans that escaped pg).
  pkill -KILL -P $$ 2>/dev/null || true
  pkill -KILL -f "lightllm\.server\.api_server.*${MODEL_DIR}" 2>/dev/null || true
  pkill -KILL -f "${SCRIPT_DIR}/lb.py" 2>/dev/null || true

  rm -f "${PIDFILE}"
  echo "[serve] cleanup done"
}
# EXIT covers normal exit + unexpected errors (set -e); INT/TERM/HUP covers signals.
trap cleanup EXIT INT TERM HUP

echo "[serve] starting LB (backends: ${BACKENDS_CSV})"
( env BACKENDS="${BACKENDS_CSV}" LB_PORT="${LB_PORT}" python "${SCRIPT_DIR}/lb.py" ) &
LB_PID=$!
echo "${LB_PID}" >> "${PIDFILE}"

wait "${LB_PID}"
# EXIT trap fires cleanup
