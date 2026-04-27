# evaluation/easi/ — SenseNova-U1 visual understanding benchmarking

Self-contained subpackage for running benchmark harnesses (VLMEvalKit, EASI, lmms-eval) against a locally-served SenseNova-U1 model.

## Layout

| Path | What |
| :--- | :--- |
| `EASI/` | git submodule → `EvolvingLMMs-Lab/EASI@main`. Contains `VLMEvalKit` + `lmms-eval` as its own nested submodules |
| `lightllm-stack/LightLLM/` | git submodule → `ModelTC/LightLLM@neo_plus_clean` |
| `lightllm-stack/patches/` | local fixes applied on top of LightLLM |
| `config/sensenova_models.py` | **editable** VLMEvalKit model entries — edit to tweak endpoint URLs, ports, `max_tokens`, etc. |
| `patches/easi_sensenova_config.patch` | 7-line hook added to `VLMEvalKit/vlmeval/config.py` that imports the above |
| `scripts/setup.sh` | one-shot install: submodules, deps, venv, patches, VLMEvalKit wire-up, verify |
| `scripts/serve.sh` | launch LightLLM server. `DP=1` (default) → single instance. `DP>1` → N replicas + LB on the canonical port |
| `scripts/lb.py` | least-in-flight HTTP LB used by `serve.sh` when `DP>1` |
| `scripts/serve_lb.sh` | deprecated shim that forwards to `serve.sh` |
| `scripts/download_weights.sh` | standalone HF weight fetcher |

Weights land in `<repo_root>/models/` (gitignored). Serving venv `<repo_root>/.venv-lightllm/` (gitignored, separate from the main repo `.venv`).

## Quickstart

```bash
# one-time host lib (needs sudo)
sudo apt-get install -y libnuma1 libnuma-dev

# install EVERYTHING — LightLLM stack + EASI client + endpoint registration.
# Idempotent. First run takes several minutes (builds flash-attn for the EASI venv).
bash evaluation/easi/scripts/setup.sh

# launch server — auto-downloads weights on first run
MODEL=8b-mot GPUS=0,1 TP=2 bash evaluation/easi/scripts/serve.sh   # → localhost:8000

# run a benchmark from a second shell
source evaluation/easi/EASI/.venv/bin/activate
cd evaluation/easi/EASI
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --benchmarks blink \
  --api-nproc 16
```

Setup modes:

```bash
bash evaluation/easi/scripts/setup.sh                   # full install
bash evaluation/easi/scripts/setup.sh --skip-lightllm   # bring your own endpoint
bash evaluation/easi/scripts/setup.sh --skip-easi       # LightLLM side only
bash evaluation/easi/scripts/setup.sh --skip-register   # no auto VLMEvalKit wiring
```

### Bring-your-own endpoint

If you already have a SenseNova-U1 OpenAI-compatible endpoint (docker container on another host, infra team API, production deployment, etc.), skip the LightLLM install entirely:

#### 1. Install only EASI + the VLMEvalKit wiring

```bash
bash evaluation/easi/scripts/setup.sh --skip-lightllm
```

Skips: host prereq checks, `.venv-lightllm` creation, LightLLM dep install, LightLLM patches, api_server CLI verification. LightLLM submodule is NOT initialized. Only the EASI submodule (+ its nested VLMEvalKit / lmms-eval) gets pulled.

#### 2. Point `config/sensenova_models.py` at your endpoint

Edit the `entries` dict — change `api_base` to wherever your endpoint actually lives. Rename the dict key if you want a clearer label in `run_easi_eval.py --model …`. Full schema + `GPT4V` kwargs documented in [`GUIDE.md`](GUIDE.md) — see the "Configuring a custom OpenAI-compatible endpoint" section.

Quick example — single remote endpoint:

```python
# evaluation/easi/config/sensenova_models.py
from functools import partial
from vlmeval.api.gpt import GPT4V  # type: ignore[import-not-found]

entries = {
    "SenseNova-U1-8B-MoT-Prod": partial(
        GPT4V,
        model="sensenova-u1-8b-mot",
        api_base="https://your.host.example.com/v1/chat/completions",
        key="sk-your-real-token-or-dummy",
        temperature=0,
        max_tokens=32768,
        retry=10,
        verbose=False,
    ),
}
```

#### 3. Propagate the edit into VLMEvalKit

```bash
bash evaluation/easi/scripts/setup.sh --skip-lightllm --skip-easi
```

This re-copies `sensenova_models.py` into `EASI/VLMEvalKit/vlmeval/`. Fast — no reinstalls.

#### 4. Run benchmarks

Same as the local-server path — just use whatever key you set in step 2:

```bash
source evaluation/easi/EASI/.venv/bin/activate
cd evaluation/easi/EASI

# single benchmark
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Prod \
  --output-dir eval_results_prod_viewspatial \
  --api-nproc 16 \
  --benchmarks viewspatial

# full EASI-8 suite
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Prod \
  --output-dir eval_results_prod \
  --api-nproc 16

# multiple specific benchmarks
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Prod \
  --benchmarks viewspatial,blink,3dsrbench \
  --api-nproc 16 \
  --output-dir eval_results_prod
```

Tune `--api-nproc` based on your endpoint's capacity. Remote endpoints with rate limits: start low (4-8) and ramp up. Strong production backends behind an LB: 32-64.

#### Notes

- `scripts/serve.sh` won't work under `--skip-lightllm` — that's the point. Your endpoint lives elsewhere.
- If the endpoint exposes multiple SenseNova-U1 variants (or you have several endpoints to benchmark side-by-side), add multiple entries to `sensenova_models.py` — each gets its own model key.
- The `GPT4V` wrapper auto-handles HTTP retries (on 5xx / timeout) and chunks images as base64 data URIs; no client-side prep needed.

## Model → port map

| `MODEL` arg | HF repo | Server port | Reasoning parser |
| :--- | :--- | :---: | :--- |
| `8b-mot` (default) | `sensenova/SenseNova-U1-8B-MoT` | 8000 | `qwen3` (strips `<think>`) |

Override the port with `PORT=<n>`.

## Multi-replica (DP) serving behind a load balancer

`serve.sh` auto-switches to multi-replica mode when `DP > 1`: launches N tp-sharded LightLLM replicas on backend ports + a Python load balancer on the canonical port. **Same port as `DP=1`** — VLMEvalKit config never needs to change when scaling up/down.

```bash
# 4 replicas × tp=2 on 8 GPUs — higher throughput for many short requests
DP=4 TP=2 bash evaluation/easi/scripts/serve.sh

# 2 replicas × tp=4
DP=2 TP=4 bash evaluation/easi/scripts/serve.sh
```

Only one serve.sh process per model at a time — both `DP=1` and `DP>1` bind the same canonical port.

Port layout:

| MODEL | LB (client-facing) | Backends |
| :--- | :---: | :--- |
| `8b-mot` | 8000 | 8100, 8110, 8120, 8130 (step 10) |

Override with `LB_PORT=...` or `BACKEND_BASE_PORT=...`.

Direct hits to a backend port (e.g. `http://localhost:18000/v1/models`) still work — useful for debugging one specific replica.

Sanity-check guardrails (fail fast, no partial launches):
- `DP * TP <= # visible GPUs` (from `nvidia-smi`) unless `GPUS=...` overrides
- If `GPUS=...` provided, must contain exactly `DP * TP` entries
- `LB_PORT` must not collide with any backend port in `[BACKEND_BASE_PORT, BACKEND_BASE_PORT + 10*(DP-1)]`
- `MODEL` must be `8b-mot`
- `DP`, `TP`, `LB_PORT`, `BACKEND_BASE_PORT` must be integers ≥ 1
- **Pre-flight port probe**: every port (LB + all backends) must be free. Stale processes from a previous run are detected before any replica is launched, with `ss -lntp` / `lsof` output naming the owner when possible

Balancing: least in-flight. Streaming passthrough. Per-request timeout 30 min (override `LB_REQUEST_TIMEOUT`).

Health: each replica probed every 10s via `GET /v1/models`. Unhealthy backends skipped but not evicted; rejoin when probes pass. Monitor at `GET http://localhost:<LB_PORT>/_lb/status`.

Per-replica logs land at `evaluation/easi/logs/lightllm-<MODEL>-<port>.log` (gitignored). Override with `LOG_DIR=...`. Ctrl-C / SIGTERM on the `serve.sh` shell cascade-kills all replicas + LB.

### Process hygiene

serve.sh won't leak zombie GPU procs:

- Each replica + LB is launched via `setsid` into its own process group. Cleanup hits `kill -TERM -$pgid` → 10 s grace → `kill -KILL -$pgid` → the whole tree (router, tp workers, visual server, zmq, detokenizer) goes down.
- Trap covers `EXIT INT TERM HUP`, not just signals — catches unexpected errors too.
- Belt + suspenders: `pkill -P $$` (our direct children) + `pkill -f "lightllm.server.api_server.*$MODEL_DIR"` (escaped orphans tied to this model) run after the grace period.
- PID file at `$LOG_DIR/serve.<MODEL>.pids` records pgids. On next `serve.sh` launch, any stale entries get TERM+KILL automatically before the new run starts.

If you still end up with zombies (container crashed / SIGKILL'd), run:

```bash
pkill -KILL -f "lightllm.server.api_server.*SenseNova-U1-Mini"
```

and for GPU mem held by processes in another PID namespace (container got torn down and recreated), only a host-side `kill` or pod restart can recover.

### Debugging / verbose logging

`serve.sh` accepts:

| Env | Effect |
| :--- | :--- |
| `DETAIL_LOG=1` | Adds `--detail_log` to LightLLM — logs per-request timing, prompt text, token IDs, and (for multimodal) per-image ViT inference timings |
| `LIGHTLLM_LOG_LEVEL=debug` | Drops LightLLM's root logger to DEBUG. Everything that's wrapped in `logger.debug(...)` now prints (lots of internal signal: KV cache state, router scheduling, detokenization). Default is `info` |

```bash
# verbose single instance
DETAIL_LOG=1 LIGHTLLM_LOG_LEVEL=debug \
  MODEL=8b-mot GPUS=0,1 TP=2 bash evaluation/easi/scripts/serve.sh

# verbose multi-replica (env flows to every replica)
DETAIL_LOG=1 LIGHTLLM_LOG_LEVEL=debug \
  DP=4 TP=2 MODEL=8b-mot bash evaluation/easi/scripts/serve.sh

# tail one replica's log
tail -f evaluation/easi/logs/lightllm-8b-mot-18000.log
```

Use when a benchmark reports 100% API failures or suspicious per-sample outputs — full HTTP/tokenization trace surfaces on the server side.

## Running benchmarks

Activate the EASI client venv and work from `evaluation/easi/EASI`:

```bash
source evaluation/easi/EASI/.venv/bin/activate
cd evaluation/easi/EASI
```

### Single benchmark
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --output-dir eval_results_sensenova-u1-8b-mot_viewspatial \
  --api-nproc 16 \
  --benchmarks viewspatial
```

### Full EASI-8 suite (omit `--benchmarks`)
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --output-dir eval_results_sensenova-u1-8b-mot \
  --api-nproc 16
```

### Multiple benchmarks at once
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --benchmarks viewspatial,blink,3dsrbench \
  --api-nproc 16 \
  --output-dir eval_results_sensenova-u1-8b-mot
```

### Benchmark keys (EASI-8 core)

`vsi_bench`, `mmsi_bench`, `mindcube_tiny`, `viewspatial`, `site_image`, `site_video`, `blink`, `3dsrbench`, `embspatial`.

Aliases: `site` = `site_image + site_video`. Group name: `sitebench`. Extra (opt-in via `--include-extra`): `mmsi_video_bench`, `omnispatial_(manual_cot)`, `spar_bench`, `vsi_debiased`.

### Useful flags

| Flag | Purpose |
| :--- | :--- |
| `--api-nproc N` | Concurrent HTTP requests to the LightLLM server. 16-32 on 8× H100 with `tp=8`. Lower if timeouts/500s |
| `--no-judge` | Skip LLM-judge re-eval, trust `exact_matching` scores. Faster; less accurate for free-form answers |
| `--rerun` | Force re-evaluation, bypass resume |
| `--verbose` | Print per-sample model responses |
| `--include-extra` | Also run the extra (non-EASI-8) benchmarks |
| `--submit` | Push results to EASI leaderboard (requires `HF_TOKEN`) |
| `--nproc N` | torchrun DP (for local-model backends only, ignored for API endpoints) |

## Tweaking VLMEvalKit endpoint config

Endpoints live in `config/sensenova_models.py` — a committed, in-repo Python module with the `partial(GPT4V, ...)` entries. Edit it, re-run `setup.sh` (idempotent, a few seconds if `--skip-easi`), and `supported_VLM` picks up the change on next interpreter start.

```bash
$EDITOR evaluation/easi/config/sensenova_models.py     # change max_tokens / temperature / URLs
bash evaluation/easi/scripts/setup.sh --skip-easi       # propagate
```

Full schema of the `entries` dict and `GPT4V` kwarg reference (including `img_detail`, `timeout`, `system_prompt`, thinking-mode subclass pattern, remote-endpoint examples): [`GUIDE.md` §4](GUIDE.md#configuring-a-custom-openai-compatible-endpoint).

The patch at `patches/easi_sensenova_config.patch` only adds a 7-line `from .sensenova_models import entries; supported_VLM.update(entries)` hook to `VLMEvalKit/vlmeval/config.py` — you shouldn't need to touch it.

## Full guide

See [`GUIDE.md`](GUIDE.md) — covers host prereqs, dependency filtering, patch workflow, VLMEvalKit wiring, thinking-mode handling, troubleshooting.

## Image generation benchmarks

Not covered here. Requires the full LightLLM + LightX2V stack (via the upstream Docker image `lightx2v/lightllm_lightx2v:20260407` — see [`../../docs/deployment_CN.md`](../../docs/deployment_CN.md)). LightX2V pins `torch<=2.8.0` which conflicts with LightLLM's `torch==2.9.1`, so image-gen serving lives in a separate environment.
