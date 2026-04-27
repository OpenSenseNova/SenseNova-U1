# EASI Benchmarking — Full Guide

End-to-end setup to run visual-understanding benchmarks (VLMEvalKit, EASI, lmms-eval) against SenseNova-U1 by exposing the model as an **OpenAI-compatible HTTP endpoint** via the LightLLM inference server, then pointing the benchmark toolkit at `/v1/chat/completions`.

This is the comprehensive reference. For the quickstart + file layout, see [`README.md`](README.md).

This guide covers a **native (no Docker) install** on a Linux host with NVIDIA GPUs. The upstream-supported path is the Docker image `lightx2v/lightllm_lightx2v:20260407` documented in [`docs/deployment_CN.md`](../../docs/deployment_CN.md) — use that when Docker/nvidia-container-toolkit is available. This native recipe is the fallback for sandboxed environments (containers, chroot, clusters without privileged pod access).

## Why LightLLM (not `transformers`)

`examples/*/inference.py` scripts use the `transformers` backend, which is fine for one-off inference but not servable:

- `NEOChatModel` is registered only with `AutoModel` / `AutoConfig`, not with `AutoModelForImageTextToText` or an `AutoProcessor`. `transformers serve` (4.57+) and `text-generation-inference` both dispatch on those mappings — they will not discover SenseNova-U1.
- Inference uses a custom `model.chat(tokenizer, pixel_values, question, gen_cfg, grid_hw=...)` signature (`src/sensenova_u1/models/neo_unify/modeling_neo_chat.py:1732`), not the standard `processor(images, text) → model.generate()` flow that serving stacks expect.
- vLLM / SGLang have no built-in NEO-Unify model class; porting is a weeks-long task.

LightLLM has native `neo_chat` + `neo_chat_moe` model implementations (`lightllm/models/neo_chat/`, `lightllm/models/neo_chat_moe/`) and exposes an OpenAI-compatible `/v1/chat/completions`, which is exactly what benchmark toolkits consume.

**Skipped in this guide: LightX2V.** LightX2V is only imported when `--enable_multimodal_x2i` is passed (see `lightllm/server/x2i_server/manager.py`). Visual understanding benchmarks do not generate images, so we omit that flag and avoid LightX2V's `torch<=2.8.0` pin (which conflicts with LightLLM's `torch==2.9.1` requirement). Image-generation benchmarks — when those scripts land — will need the full LightLLM + LightX2V stack via the Docker image.

---

## 1) Host prerequisites

| Item | Required |
| :--- | :--- |
| OS | Linux (x86_64) |
| NVIDIA driver | ≥ 550.x (recommended 550.90.07+) |
| GPU | Hopper / Ampere class with compute capability ≥ 80. Verified on H100 80GB |
| Python | 3.10 (matches LightLLM's Dockerfile) |
| `uv` | `uv >= 0.9`. Install: <https://docs.astral.sh/uv/getting-started/installation/> |
| System lib | `libnuma1`, `libnuma-dev` (required by `sgl-kernel`) |

Install the system lib once (needs sudo):

```bash
sudo apt-get install -y libnuma1 libnuma-dev
```

The CUDA runtime is shipped inside the `torch==2.9.1+cu128` wheel — you do NOT need a matching CUDA toolkit on the host as long as the host driver supports CUDA 12.8+ (forward-compatible from 550.x).

---

## 2) One-shot install

`evaluation/easi/scripts/setup.sh` installs the full pipeline in one idempotent run:

| Phase | What |
| :--- | :--- |
| 1 | Host prereq checks (`uv`, `libnuma`, driver) |
| 2 | Recursive submodule init (LightLLM, EASI, EASI/VLMEvalKit, EASI/lmms-eval) |
| 3 | `.venv-lightllm/` Python 3.10 venv at repo root |
| 4-6 | Pinned LightLLM deps + vllm + editable LightLLM + transitive fixes (pandas) |
| 7 | Apply local patches from `evaluation/easi/lightllm-stack/patches/` |
| 8 | Verify LightLLM imports + api_server CLI |
| 9 | **EASI client venv** — delegates to `EASI/scripts/setup.sh` (Py 3.11, VLMEvalKit + lmms-eval + flash-attn) |
| 10 | **Endpoint registration** — sitecustomize.py injector into the EASI venv |

```bash
bash evaluation/easi/scripts/setup.sh
```

Flags:
- `--skip-lightllm` — skip phases 1, 3-7 (DON'T install LightLLM). Use when you have a SenseNova-U1 OpenAI-compatible endpoint already (docker elsewhere, remote infra, etc.). Edit `config/sensenova_models.py` to point `api_base` at your endpoint
- `--skip-easi` — skip phase 9 (flash-attn build is slow; useful for fast reruns when only touching the LightLLM side)
- `--skip-register` — skip phase 10 (no auto endpoint wiring)

Re-running is safe — each step checks whether it already ran. If you just want to understand what happens under the hood, sections 2a–2f below spell out phases 2-8.

### 2a. LightLLM submodule

LightLLM is pinned as a git submodule at `evaluation/easi/lightllm-stack/LightLLM`, tracking branch `neo_plus_clean` (the branch that contains NEO-Unify model support). `evaluation/easi/lightllm-stack/patches/` holds any local fixes applied on top.

```bash
git submodule update --init evaluation/easi/lightllm-stack/LightLLM
```

To bump to a newer LightLLM commit later:

```bash
cd evaluation/easi/lightllm-stack/LightLLM
git fetch origin neo_plus_clean && git checkout origin/neo_plus_clean
cd -
git add evaluation/easi/lightllm-stack/LightLLM   # records the new submodule SHA
```

> `LightX2V` can be cloned alongside for image-generation workloads later. Unused for VQA-only serving; not included as a submodule.

### 2b. Create the serving venv

Keep this venv separate from the main `.venv` used by `examples/*/inference.py` — LightLLM pins `torch==2.9.1` while the main env uses torch 2.8.

```bash
cd /path/to/SenseNova-U1
uv venv -p 3.10 .venv-lightllm
source .venv-lightllm/bin/activate
uv pip install --upgrade pip
```

`.venv-lightllm/` is already in `.gitignore`.

### 2c. Strip non-installable pins from `requirements.txt`

The upstream `LightLLM/requirements.txt` includes two packages that fail in a clean environment:

- **`nixl==0.8.0`** — not published to PyPI; built from source inside the Dockerfile against a custom UCX build. Only used by `--run_mode nixl_prefill/nixl_decode` (PD-disaggregation over RDMA) which we are not running.
- **`cchardet==2.1.7`** — archived upstream, build fails on modern `setuptools`. Optional character-detection helper; `chardet` is pulled in transitively as a fallback.

```bash
grep -v "^nixl\|^cchardet" evaluation/easi/lightllm-stack/LightLLM/requirements.txt > /tmp/lightllm-req.txt
```

### 2d. Install

```bash
# pinned deps (torch 2.9.1+cu128, flashinfer, sgl-kernel, xformers, triton, ...)
uv pip install -r /tmp/lightllm-req.txt

# vllm is a hard runtime dep (used for shared utilities)
uv pip install --no-cache-dir vllm==0.16.0

# LightLLM itself, editable
uv pip install --no-cache-dir -e evaluation/easi/lightllm-stack/LightLLM

# transitive dep missing from upstream requirements
uv pip install pandas
```

### 2e. Apply local patches

See [§8 Known patches](#8-known-patches). `setup.sh` applies these automatically; if you installed by hand:

```bash
cd evaluation/easi/lightllm-stack/LightLLM
for p in ../patches/*.patch; do git apply "$p"; done
```

### 2f. Verify

```bash
python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import flashinfer, sgl_kernel, xformers, vllm, lightllm
print('flashinfer', 'sgl-kernel', 'xformers ok', 'vllm', vllm.__version__)
print('lightllm ok')
"

python -m lightllm.server.api_server --help | head -5
```

Expected:

```
torch 2.9.1+cu128 cuda True
flashinfer sgl-kernel xformers ok vllm 0.16.0
lightllm ok
usage: api_server.py [-h] ...
```

### Features we skip (and when you'd re-enable them)

| Feature | Re-enable when | How |
| :--- | :--- | :--- |
| **FlashMLA** | Running DeepSeek MLA-style attention (SenseNova-U1 uses Qwen3 GQA, not MLA) | Follow `evaluation/easi/lightllm-stack/LightLLM/docker/Dockerfile:49-53` |
| **DeepEP + NVSHMEM** | Multi-node MoE with InfiniBand GPUs | `Dockerfile:78-100`; needs root for gdrcopy |
| **NIXL + custom UCX** | PD-disaggregated serving (`--run_mode nixl_*`) | `Dockerfile:102-138`; needs root, RDMA stack |
| **LightMem** | Disk KV-cache offload | `Dockerfile:60-64` |
| **LightX2V (image gen)** | Running image-generation benchmarks | Install `evaluation/easi/lightllm-stack/LightX2V` in a separate venv with `torch<=2.8.0`, or use the upstream Docker image |

---

## 3) Launch the server

Helper script: `evaluation/easi/scripts/serve.sh`. Auto-activates `.venv-lightllm`, auto-downloads model weights on first run if missing, and picks a per-model default port so both models can run concurrently without clashing.

### Model → port mapping

| `MODEL` value | HF repo | Port | `--reasoning_parser` default | Advertised `model` name |
| :--- | :--- | :---: | :--- | :--- |
| `8b-mot` *(default)* | `sensenova/SenseNova-U1-8B-MoT` | 8000 | `qwen3` (strips `<think>…</think>`) | `sensenova-u1-8b-mot` |

### Defaults
```bash
# 8b-mot, GPUs 0-1, tp=2, port 8000
bash evaluation/easi/scripts/serve.sh
```

### Max throughput on 8× H100 (single model)
```bash
MODEL=8b-mot GPUS=0,1,2,3,4,5,6,7 TP=8 bash evaluation/easi/scripts/serve.sh
```

### Env vars (full list)

| Var | Default | Notes |
| :--- | :--- | :--- |
| `MODEL` | `8b-mot` | `8b-mot`. Ignored if `MODEL_DIR` is set |
| `MODEL_DIR` | `./models/SenseNova-U1-Mini-<Beta\|SFT>` | Absolute path overrides |
| `GPUS` | `0,1` | Comma-separated `CUDA_VISIBLE_DEVICES` |
| `TP` | `2` | Tensor-parallel degree; must equal `GPUS` count |
| `HOST` | `0.0.0.0` | |
| `PORT` | per-model (8000 / 8001) | Overrides the default port from the table above |
| `MAX_LEN` | `32768` | `--max_req_total_len` |
| `MEM_FRAC` | `0.85` | `--mem_fraction` — fraction of GPU mem for KV cache |
| `MODEL_NAME` | per-model | Advertised via `/v1/models`; benchmark client `model` field must match |
| `REASONING` | per-model | `--reasoning_parser`. `qwen3` for beta, disabled for sft. Set to empty string to disable on beta |
| `NO_AUTO_DL` | unset | Set to `1` to skip auto-download when model dir is missing (error out instead) |

### First-launch warmup

Triton / CUDA kernels compile on first request; the first `/v1/chat/completions` call can take **several minutes** to return. Subsequent calls are cached and fast. Health-check with:

```bash
# after startup log shows "Uvicorn running on http://0.0.0.0:8000"
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sensenova-u1-8b-mot","messages":[{"role":"user","content":"hi"}]}' | head -c 500
```

For a proper multimodal smoke test (image + text), use [`examples/serving/client.py`](../../examples/serving/client.py):

```bash
source .venv                           # the main sensenova_u1 venv has the `requests` client deps
python examples/serving/client.py \
  --mode vqa \
  --prompt "Describe this image." \
  --image_path examples/vqa/data/images/menu.jpg \
  --url http://localhost:8000/v1
```

---

### Standalone weight download (if you want to pre-fetch)

`evaluation/easi/scripts/serve.sh` calls this for you automatically. To run it independently:

```bash
source .venv-lightllm/bin/activate     # hf CLI lives in this venv
bash evaluation/easi/scripts/download_weights.sh 8b-mot
```

Weights land at `./models/SenseNova-U1-<Mini|Flash>-Beta/`. `./models/` is gitignored. Set `export HF_TOKEN=hf_...` if the HF repo is gated.

---

## 4) Point VLMEvalKit / EASI at the endpoint

The server speaks OpenAI `/v1/chat/completions`. Any OpenAI-compatible client works unchanged. For VLMEvalKit the `GPT4V` wrapper (`vlmeval/api/gpt.py`) is the right client class.

EASI is checked out as a submodule at `evaluation/easi/EASI` (tracking `EvolvingLMMs-Lab/EASI@main`). Its VLMEvalKit fork is at `evaluation/easi/EASI/VLMEvalKit` — nested submodule.

VLMEvalKit's `run.py --config <json>` flag is mutually exclusive with `--data/--model`, which is what EASI's `run_easi_eval.py` uses. So the model entry must exist in `vlmeval.config.supported_VLM` at import time.

### Tracked-file wire-up (what `setup.sh` does)

Two files you care about:

| Path | Role |
| :--- | :--- |
| `evaluation/easi/config/sensenova_models.py` | **Editable source of truth** for endpoint URLs, ports, `max_tokens`, temperature, retry, etc. Tweak here, commit, done |
| `evaluation/easi/patches/easi_sensenova_config.patch` | 7-line patch that appends an import hook to `VLMEvalKit/vlmeval/config.py` |

`setup.sh` Phase 10:
1. Copies `config/sensenova_models.py` → `EASI/VLMEvalKit/vlmeval/sensenova_models.py`
2. Applies `patches/easi_sensenova_config.patch` to `EASI/VLMEvalKit/vlmeval/config.py` — adds:
   ```python
   try:
       from .sensenova_models import entries as _sensenova_u1_entries
       supported_VLM.update(_sensenova_u1_entries)
   except Exception as _e:
       import sys; print(f"[sensenova-u1-config] failed: {_e}", file=sys.stderr)
   ```

Both steps idempotent — patch apply is reverse-checked before applying, copy is always overwrite (cheap). Tweak the editable module, re-run setup.sh, done.

### Tweak workflow

```bash
# edit endpoint/port/max_tokens/temperature:
$EDITOR evaluation/easi/config/sensenova_models.py

# propagate to VLMEvalKit:
bash evaluation/easi/scripts/setup.sh --skip-easi   # (--skip-easi skips the slow EASI venv check)
```

Verify:

```bash
source evaluation/easi/EASI/.venv/bin/activate
python -c 'from vlmeval.config import supported_VLM; print([k for k in supported_VLM if "SenseNova-U1-" in k])'
# ['SenseNova-U1-8B-MoT-Local']
```

### Why not edit `config.py` directly?

The VLMEvalKit submodule is pinned to a specific upstream SHA. Any direct edit becomes dirty submodule state that doesn't survive `git submodule update`. The tracked-file + patch pattern is git-friendly: edits live in the parent SenseNova-U1 repo and re-apply cleanly after submodule bumps.

### Configuring a custom OpenAI-compatible endpoint

`config/sensenova_models.py` is a plain Python module containing a `entries: dict[str, partial[GPT4V]]` top-level variable. Each entry maps a **model key** (what you pass to `run_easi_eval.py --model …`) to a partially-applied `GPT4V` client bound to an endpoint.

Template:

```python
from functools import partial
from vlmeval.api.gpt import GPT4V  # type: ignore[import-not-found]

entries = {
    "<YourModelName>": partial(
        GPT4V,
        model="<advertised-model-name>",                    # must match the server's `model` field
        api_base="http://<host>:<port>/v1/chat/completions",
        key="<api-key-or-dummy>",
        temperature=0,                                      # 0 = greedy (deterministic)
        max_tokens=8192,                                    # higher for thinking models
        retry=10,                                           # per-request retries on 5xx / timeout
        verbose=False,
    ),
}
```

### `GPT4V` kwargs reference

| Kwarg | Type | Notes |
| :--- | :--- | :--- |
| `model` | str | Value echoed to the server in the request `model` field. Must match `--model_name` on the server side (LightLLM: `MODEL_NAME` env var; vLLM: `--served-model-name`, etc.) |
| `api_base` | str | Full path to the chat-completions endpoint, including `/v1/chat/completions`. Works for any OpenAI-compatible server (LightLLM, vLLM, SGLang, TGI, OpenRouter, Anthropic-via-openai-shim, etc.) |
| `key` | str | Bearer token. Use `"dummy"` for auth-less local servers |
| `temperature` | float | `0` for deterministic benchmarking; set > 0 if a benchmark needs sampling |
| `max_tokens` | int | Generation cap. Thinking models (e.g. SenseNova-U1-8B-MoT) need ≥ 8192 so they don't truncate mid-`<think>` |
| `top_p` | float | Nucleus sampling cutoff; default 1.0 (no trim) |
| `retry` | int | HTTP-level retries on 5xx / timeout. 10 is generous |
| `wait` | float | Seconds between retries; defaults to exponential backoff |
| `verbose` | bool | Per-request logging; leave False for benchmark runs |
| `img_detail` | `"low"` / `"high"` / `"auto"` | Passed through to `image_url.detail` — only matters for servers that honor it (GPT-4o). LightLLM ignores |
| `timeout` | int | Per-request timeout in seconds. Defaults to 60 — bump to 180+ for thinking models on slow hardware |
| `system_prompt` | str | Prepended as the system message. Leave unset unless a benchmark demands a specific persona |

Full kwarg list: `evaluation/easi/EASI/VLMEvalKit/vlmeval/api/gpt.py`.

### Examples

**Local LightLLM (default)** — what `setup.sh` ships out of the box:
```python
"SenseNova-U1-8B-MoT-Local": partial(
    GPT4V,
    model="sensenova-u1-8b-mot",
    api_base="http://localhost:8000/v1/chat/completions",
    key="dummy", temperature=0, max_tokens=32768, retry=10, verbose=False,
),
```

**Remote endpoint (infra team or production)**:
```python
"SenseNova-U1-8B-MoT-Prod": partial(
    GPT4V,
    model="sensenova-u1-8b-mot",
    api_base="https://sensenova-u1.internal.example.com/v1/chat/completions",
    key="sk-your-real-token",
    temperature=0, max_tokens=32768, retry=5, verbose=False,
),
```

**Endpoint that needs `enable_thinking` toggled off** (subclass pattern):
```python
from vlmeval.api.gpt import GPT4V

class _SenseNovaNoThinking(GPT4V):
    def generate_inner(self, inputs, **kwargs):
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        return super().generate_inner(inputs, **kwargs)

entries = {
    "SenseNova-U1-8B-MoT-Local-NoThink": partial(
        _SenseNovaNoThinking,
        model="sensenova-u1-8b-mot",
        api_base="http://localhost:8000/v1/chat/completions",
        key="dummy", temperature=0, max_tokens=2048, retry=10, verbose=False,
    ),
}
```

After any edit: `bash evaluation/easi/scripts/setup.sh --skip-lightllm --skip-easi` to propagate into `VLMEvalKit/vlmeval/sensenova_models.py`, then your new model key is available via `run_easi_eval.py --model <YourModelName>`.

### Running benchmarks

```bash
source evaluation/easi/EASI/.venv/bin/activate
cd evaluation/easi/EASI
```

**Single benchmark**:
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --output-dir eval_results_sensenova-u1-8b-mot_viewspatial \
  --api-nproc 16 \
  --benchmarks viewspatial
```

**Full EASI-8 suite** (omit `--benchmarks`):
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --output-dir eval_results_sensenova-u1-8b-mot \
  --api-nproc 16
```

**Multiple benchmarks**:
```bash
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --benchmarks viewspatial,blink,3dsrbench \
  --api-nproc 16 \
  --output-dir eval_results_sensenova-u1-8b-mot
```

Benchmark keys (EASI-8): `vsi_bench, mmsi_bench, mindcube_tiny, viewspatial, site_image, site_video, blink, 3dsrbench, embspatial`. Alias `site` expands to `site_image + site_video`.

Useful flags: `--no-judge` (trust exact_matching), `--rerun` (skip resume), `--verbose` (per-sample output), `--include-extra` (add non-EASI-8 benches), `--submit` (push to leaderboard — needs `HF_TOKEN`).

`--api-nproc` = concurrent HTTP requests to LightLLM. 16-32 on 8× H100 with `tp=8`. Lower if you see timeouts / 500s.

### Plain VLMEvalKit

Same `GPT4V` wrapper pattern — just add the entry to whichever config file the standalone VLMEvalKit uses, then run `python run.py --model <key> --data <bench>`.

### Thinking-mode toggle

If a benchmark requires disabling (or forcing) the `<think>…</think>` reasoning block, subclass `GPT4V` and inject `chat_template_kwargs` into the payload. Pattern already used for Qwen3.5-VL at `evaluation/easi/EASI/VLMEvalKit/vlmeval/config.py:127-137`:

```python
class _SenseNovaNoThinking(GPT4V):
    def generate_inner(self, inputs, **kwargs):
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        return super().generate_inner(inputs, **kwargs)
```

### `max_tokens` and thinking mode

When thinking is on (default), the `<think>...</think>` block alone can run thousands of tokens. If generation hits `max_tokens` before `</think>`, the reasoning parser returns an **empty `content`** (all the text is trapped in `reasoning_content`, the final answer never gets emitted). For VLMEvalKit, this shows up as blank model outputs / parser failures.

- For thinking-mode benchmarks: set `max_tokens >= 8192` in the `GPT4V` partial, or higher for multi-hop reasoning benches. The EASI config example above uses `max_tokens=8192`.
- For benchmarks that don't need reasoning: disable thinking via the `_SenseNovaNoThinking` subclass pattern and drop `max_tokens` back to `2048` to save latency.

---

## 5) Troubleshooting

### `libnuma.so.1: cannot open shared object file`
`sgl_kernel` dynamic dep. Install system lib:
```bash
sudo apt-get install -y libnuma1 libnuma-dev
```

### `nixl` build fails / `cchardet` build fails
Strip them from the requirements file as shown in §2c. Neither is needed for single-node serving. `setup.sh` already does this.

### `ModuleNotFoundError: No module named 'pandas'`
Transitive dep of `lightllm/models/neo_chat_moe/vision_process.py` not declared in upstream `requirements.txt`. Install manually: `uv pip install pandas`. `setup.sh` handles this automatically.

### `jinja2.exceptions.UndefinedError: 'list object' has no attribute 'startswith'`
HF chat template called on OpenAI-style multimodal content list. Fixed by the `build_prompt_flatten_content.patch` in §6. If you freshly cloned LightLLM and skipped `setup.sh`, apply manually:
```bash
cd evaluation/easi/lightllm-stack/LightLLM
git apply ../patches/build_prompt_flatten_content.patch
```

### `iptables: Permission denied` (during `docker run`)
You are inside an unprivileged container (Kubernetes pod, LXC, chroot). Docker-in-Docker is blocked. Use this native recipe instead — no Docker needed.

### Server launches but first request hangs several minutes
Expected. Triton/CUDA kernel compilation on first invocation. Subsequent calls are cached. Confirm by tailing the server log for `compiling` / `compiled` messages.

### `CUDA out of memory`
Options, in order of impact:
- Reduce `MEM_FRAC` (e.g. `0.7` leaves more headroom).
- Lower `MAX_LEN` (`--max_req_total_len`) — kv-cache scales with it.
- Increase `TP` and give more GPUs.
- On VLMEvalKit side, lower `--api-nproc` to cap concurrent active requests.

### Model loads but `/v1/chat/completions` 404s
Check the model served: `curl http://localhost:8000/v1/models`. The `model` field in the request body must match the value there (by default `sensenova-u1`, set via `MODEL_NAME` env var).

### torch version conflicts when activating both venvs
They're deliberately separate. `.venv` uses torch 2.8 (for SenseNova-U1 transformers inference); `.venv-lightllm` uses torch 2.9.1 (LightLLM's pin). Activate only one at a time — never source both into the same shell.

---

## 6) Known patches

Local fixes for LightLLM bugs/gaps we've hit. Tracked under `evaluation/easi/lightllm-stack/patches/`. `setup.sh` applies these automatically and skips if already applied.

| Patch | Fixes | Target file |
| :--- | :--- | :--- |
| `build_prompt_flatten_content.patch` | OpenAI multimodal `content` lists crash the HF chat template (`'list object' has no attribute 'startswith'`). Adds `_flatten_multimodal_content` step that rewrites list-form content into a string with `<image>`/`<audio>` placeholders before `apply_chat_template`; `NeoChatTokenizer.encode` later expands them to `<img>...</img>` with injected image-token IDs. | `lightllm/server/build_prompt.py` |

### Recovering after `git submodule update` in `evaluation/easi/lightllm-stack/LightLLM/`

Git submodule updates reset the working tree to the pinned SHA, discarding any applied patches. Re-apply:

```bash
bash evaluation/easi/scripts/setup.sh            # idempotent; re-applies any drifted patches
# or, manually:
cd evaluation/easi/lightllm-stack/LightLLM
for p in ../patches/*.patch; do
  git apply --reverse --check "$p" 2>/dev/null || git apply "$p"
done
```

If upstream file moves cause `git apply --check` to fail, the patch needs regenerating against the new file — inspect the conflict, reapply the logic manually, and regenerate with `git diff <file> > ../patches/<name>.patch`.

### Contributing fixes upstream

These patches are candidates for upstream PRs to <https://github.com/ModelTC/LightLLM>. The multimodal flatten fix is model-agnostic and should benefit every `/v1/chat/completions` multimodal client.

---

## 7) Recap — shortest path

```bash
# one-time host prereq (needs sudo)
sudo apt-get install -y libnuma1 libnuma-dev

# full install: LightLLM stack + EASI client venv + endpoint registration
bash evaluation/easi/scripts/setup.sh

# launch (auto-downloads weights on first run)
bash evaluation/easi/scripts/serve.sh            # 8b-mot → port 8000

# benchmark (second shell, after server up)
source evaluation/easi/EASI/.venv/bin/activate
cd evaluation/easi/EASI
python scripts/submissions/run_easi_eval.py \
  --model SenseNova-U1-8B-MoT-Local \
  --benchmarks blink --api-nproc 16
```
