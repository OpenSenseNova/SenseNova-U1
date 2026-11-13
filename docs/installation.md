# Installation

This guide covers the Python environment for running SenseNova-U1 and U1.5 locally with the `transformers` backend.

> The repository root is the **inference project**. Training has an independent
> project, lockfile, and virtual environment under [`training/`](../training/README.md);
> do not install both dependency sets into the same environment.

> **Software versions:** Python 3.11, torch 2.8, CUDA 12.8 (cu128). Update `pyproject.toml` index URLs if your driver requires a different CUDA version.

Use [**uv**](https://docs.astral.sh/uv/) with the committed `uv.lock` to install the reference environment.

> uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

## 1. Clone the repository

```bash
git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

## 2. Install dependencies with uv

```bash
uv --project . sync --locked
source .venv/bin/activate
```

The `sensenova_u1` package is installed in editable mode, so the canonical [NEO-Unify model](../src/sensenova_u1/models/neo_unify/) is automatically registered with `transformers.Auto*` at import time.

> **Older NVIDIA drivers:** the default index is CUDA 12.8. If your driver
> does not support cu128, change `[tool.uv.sources]` / `[[tool.uv.index]]`
> in `pyproject.toml` to e.g. `https://download.pytorch.org/whl/cu126` (and
> adjust the pinned torch / torchvision versions accordingly) before
> running `uv lock` and then `uv --project . sync --locked`.

## Optional: flash-attn

`flash-attn` is declared as an optional extra;
without it the model transparently falls back to torch SDPA;
once flash-attn is importable the runtime picks it automatically (`--attn_backend auto`).

```bash
# (a) Build from source via PyPI
uv --project . sync --locked --extra flash

# (b) Install a prebuilt CUDA wheel matching your torch + Python
uv pip install /path/to/flash_attn-2.8.3+cu12torch28cxx11abitrue-cp311-cp311-*.whl
```

With the environment active, choose a model guide: [U1.5](u1.5_model.md), [U1.5 Preview](u1.5_preview.md#run-u15-preview), [U1 infographic](u1_infographic_model.md#run-infographic-v3), or [original U1](u1_model.md). Each guide links its own run commands and examples.
