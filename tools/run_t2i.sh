#!/usr/bin/env bash
# Layout: <workspace>/tools/run_t2i.sh and <workspace>/env.sh.
# Runs are created before loading env.sh, beneath <workspace>/runs/.
set -euo pipefail
exec 3>&2  # Preserve terminal diagnostics even when EXIT occurs inside a redirected block.
RUN_DIR=''
STAGE=bootstrap
STAGE_LOG='(terminal only)'
finish() {
  local rc=$? record_rc=0 result=failed
  trap - EXIT
  set +e
  if (( rc == 0 )); then result=automatic_checks_passed; fi
  if [[ -n "$RUN_DIR" ]]; then
    printf '%s\n' "$STAGE" > "$RUN_DIR/stage.txt" || record_rc=1
    printf '%s\n' "$rc" > "$RUN_DIR/script_exit_code.txt" || record_rc=1
    printf '%s\n' "$result" > "$RUN_DIR/status.txt" || record_rc=1
    if (( record_rc != 0 )); then
      printf 'WARNING: could not persist all final status files.\n' >&3
      if (( rc == 0 )); then rc=74; fi
      # Best effort: never knowingly leave writable status files reporting success.
      printf '%s\n' "$rc" > "$RUN_DIR/script_exit_code.txt" || :
      printf 'failed\n' > "$RUN_DIR/status.txt" || :
    fi
  fi
  printf 'Script exit code: %s; stage: %s\n' "$rc" "$STAGE" >&3
  printf 'Run directory: %s\nStage log: %s\n' "${RUN_DIR:-NOT CREATED (see terminal)}" "$STAGE_LOG" >&3
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stage() {
  STAGE=$1
  STAGE_LOG="$RUN_DIR/$2"
  printf '%s\n' "$STAGE" > "$RUN_DIR/stage.txt"
  printf '%s\n' "$STAGE" >> "$RUN_DIR/stage_history.txt"
  printf 'Stage: %s\n' "$STAGE"
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
mkdir -p -- "$WORKSPACE_DIR/runs"
RUN_DIR="$(mktemp -d "$WORKSPACE_DIR/runs/t2i-XXXXXXXX")"
export RUN_DIR
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'running\n' > "$RUN_DIR/status.txt"
printf 'no\n' > "$RUN_DIR/inference_started.txt"
# No latest_run.txt is written or read. Use this invocation's printed directory.

stage load_environment environment_load.log
source "$WORKSPACE_DIR/env.sh" > "$STAGE_LOG" 2>&1
stage preflight preflight.log
{
  : "${U15_ROOT:?missing U15_ROOT}"
  configured_root="$(cd -- "$U15_ROOT" && pwd)"
  if [[ "$configured_root" != "$WORKSPACE_DIR" ]]; then
    printf 'U15_ROOT must match the script parent directory: %s\n' "$WORKSPACE_DIR" >&2
    exit 1
  fi
  : "${REPO_DIR:?missing REPO_DIR}" "${CODE_COMMIT:?missing CODE_COMMIT}"
  : "${MODEL_DIR:?missing MODEL_DIR}" "${MODEL_ID:?missing MODEL_ID}"
  : "${MODEL_REV:?missing MODEL_REV}" "${UV:?missing UV}" "${PY:?missing PY}"
  : "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
  cd -- "$REPO_DIR"
  actual_commit="$(git rev-parse HEAD)"
  if [[ "$actual_commit" != "$CODE_COMMIT" ]]; then
    printf 'Code commit mismatch: expected %s, got %s\n' "$CODE_COMMIT" "$actual_commit" >&2
    exit 1
  fi
  tracked_changes="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$tracked_changes" ]]; then
    printf 'Tracked working tree is dirty:\n%s\n' "$tracked_changes" >&2
    exit 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    printf 'Missing model directory: %s\n' "$MODEL_DIR" >&2
    exit 1
  fi
} > "$STAGE_LOG" 2>&1

stage collect_environment environment.txt
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  git rev-parse HEAD
  printf 'model=%s\nrevision=%s\n' "$MODEL_ID" "$MODEL_REV"
  printf 'model_dir=%s\n' "$MODEL_DIR"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  "$UV" --version
  "$PY" --version
  nvidia-smi
} > "$STAGE_LOG" 2>&1
stage freeze freeze.log
"$UV" pip freeze --python "$PY" > "$RUN_DIR/freeze.txt" 2> "$STAGE_LOG"
stage snapshot snapshot.log
{
  cp -- "$WORKSPACE_DIR/env.sh" "$RUN_DIR/env_used.sh"
  cp -- "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" "$RUN_DIR/run_used.sh"
} > "$STAGE_LOG" 2>&1

stage gpu_check gpu_check.log
"$PY" "$SCRIPT_DIR/verify_env.py" --gpus 1 > "$STAGE_LOG" 2>&1
stage resource_check resource_check.json
MODE="${U15_VRAM_MODE:-full}"
case "$MODE" in full|low|balanced|fast) ;; *) echo "Invalid U15_VRAM_MODE: $MODE" >&3; exit 2 ;; esac
"$PY" "$SCRIPT_DIR/memory_preflight.py" --model-dir "$MODEL_DIR" \
  --vram-mode "$MODE" --device cuda:0 > "$STAGE_LOG" 2>&1
stage prepare_command command.sh
PROMPT="${U15_PROMPT:-A red ceramic mug on a wooden table, soft daylight, simple background.}"
STEPS=50
CFG=4.0
if [[ -n "${U15_LORA_PATH:-}" ]]; then
  test -f "$U15_LORA_PATH"
  expected_lora=f4190cabd01fa261a45be853275c483d3a8069295c52cf01f7b2b31f4356f201
  actual_lora="$(sha256sum -- "$U15_LORA_PATH")"
  [[ "${actual_lora%% *}" == "$expected_lora" ]] || { echo 'V2 LoRA hash mismatch' >&3; exit 1; }
  STEPS=8
  CFG=1.0
fi
printf '%s\n' "$PROMPT" > "$RUN_DIR/prompt.txt"
cmd=(
  "$PY" -u examples/t2i/inference.py
  --model_path "$MODEL_DIR"
  --prompt "$PROMPT"
  --output "$RUN_DIR/first.png"
  --width 2048 --height 2048
  --num_steps "$STEPS" --batch_size 1 --seed 42
  --cfg_scale "$CFG" --cfg_norm none
  --timestep_shift 3.0 --cfg_interval 0.0 1.0
  --device cuda:0 --dtype bfloat16
  --attn_backend sdpa --vram_mode "$MODE"
  --profile
)
if [[ -n "${U15_LORA_PATH:-}" ]]; then cmd+=(--lora_path "$U15_LORA_PATH"); fi
if [[ "$MODE" == fast ]]; then
  cmd+=(--fast_vram_budget_gib "${U15_FAST_BUDGET_GIB:-18}"
        --fast_vram_headroom_gib "${U15_FAST_HEADROOM_GIB:-2}"
        --fast_activation_reserve_gib "${U15_FAST_ACTIVATION_GIB:-4}")
fi
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
stage inference inference.log
printf 'yes\n' > "$RUN_DIR/inference_started.txt"
start=$SECONDS
set +e
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/inference.log"
statuses=("${PIPESTATUS[@]}")
set -e
rc="${statuses[0]}"
log_rc="${statuses[1]}"
printf '%s\n' "$rc" > "$RUN_DIR/exit_code.txt"
printf '%s\n' "$log_rc" > "$RUN_DIR/log_exit_code.txt"
printf '%s\n' "$((SECONDS - start))" > "$RUN_DIR/wall_seconds.txt"
if (( rc != 0 || log_rc != 0 )); then
  printf 'Inference process exit code: %s; inference tee exit code: %s\n' "$rc" "$log_rc" >&2
  exit 1
fi
stage output_check output_check.txt
set +e
"$PY" - <<'CHECK' 2>&1 | tee "$RUN_DIR/output_check.txt"
import hashlib
import os
from pathlib import Path
from PIL import Image
if not __debug__:
    raise SystemExit("Run without -O or PYTHONOPTIMIZE")
p = Path(os.environ["RUN_DIR"]) / "first.png"
assert p.is_file() and p.stat().st_size > 0, str(p)
with Image.open(p) as im:
    assert im.format == "PNG", im.format
    im.verify()
with Image.open(p) as im:
    im.load()
    assert im.size == (2048, 2048), im.size
    print("PNG_CHECK_OK", im.size, im.mode)
print("SHA256", hashlib.sha256(p.read_bytes()).hexdigest())
print("OUTPUT", p)
CHECK
check_statuses=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${check_statuses[0]}" > "$RUN_DIR/output_check_exit_code.txt"
printf '%s\n' "${check_statuses[1]}" > "$RUN_DIR/output_check_log_exit_code.txt"
if (( check_statuses[0] != 0 || check_statuses[1] != 0 )); then
  printf 'Output check exit code: %s; output-check tee exit code: %s\n' "${check_statuses[0]}" "${check_statuses[1]}" >&2
  exit 1
fi
stage complete output_check.txt
printf 'Manual visual review still required: %s/first.png\n' "$RUN_DIR"
