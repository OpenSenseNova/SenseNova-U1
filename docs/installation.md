# Installation (Transformers Inference)

This guide covers setting up the Python environment for running SenseNova-U1 locally with the `transformers` backend.

> **Software versions:** Python 3.11, torch 2.8, CUDA 12.8 (cu128). Update `pyproject.toml` index URLs if your driver requires a different CUDA version.

We recommend [**uv**](https://docs.astral.sh/uv/) to manage the Python environment.

> uv installation guide: <https://docs.astral.sh/uv/getting-started/installation/>

## 1. Clone the repository

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

## 2. Install dependencies with uv

```bash
uv sync
source .venv/bin/activate
```

The `sensenova_u1` package is installed in editable mode, so the canonical [NEO-Unify model](../src/sensenova_u1/models/neo_unify/) is automatically registered with `transformers.Auto*` at import time.

> **Older NVIDIA drivers:** the default index is CUDA 12.8. If your driver
> does not support cu128, change `[tool.uv.sources]` / `[[tool.uv.index]]`
> in `pyproject.toml` to e.g. `https://download.pytorch.org/whl/cu126` (and
> adjust the pinned torch / torchvision versions accordingly) before
> running `uv sync`.

## Optional: flash-attn

`flash-attn` is declared as an optional extra;
without it the model transparently falls back to torch SDPA;
once flash-attn is importable the runtime picks it automatically (`--attn_backend auto`).

```bash
# (a) Build from source via PyPI
uv sync --extra flash

# (b) Install a prebuilt CUDA wheel matching your torch + Python
uv pip install /path/to/flash_attn-2.8.3+cu12torch28cxx11abitrue-cp311-cp311-*.whl
```
