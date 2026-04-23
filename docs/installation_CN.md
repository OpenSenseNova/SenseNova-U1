# 安装指南（Transformers 推理）

本指南介绍如何搭建 Python 环境，以使用 `transformers` 后端在本地运行 SenseNova-U1。

> **软件版本：** Python 3.11、torch 2.8、CUDA 12.8（cu128）。如果本机驱动需要其他 CUDA 版本，请相应修改 `pyproject.toml` 中的 index URL。

我们推荐使用 [**uv**](https://docs.astral.sh/uv/) 管理 Python 环境。

> uv 安装指南：<https://docs.astral.sh/uv/getting-started/installation/>

## 1. 克隆仓库

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

## 2. 使用 uv 安装依赖

```bash
uv sync
source .venv/bin/activate
```

`sensenova_u1` 会以可编辑模式安装，因此在 import 时，标准的 [NEO-Unify 模型](../src/sensenova_u1/models/neo_unify/) 会自动注册到 `transformers.Auto*` 接口。

> **较旧的 NVIDIA 驱动：** 默认 index 对应 CUDA 12.8。若驱动不支持 cu128，请先将
> `pyproject.toml` 中的 `[tool.uv.sources]` / `[[tool.uv.index]]` 改为例如
> `https://download.pytorch.org/whl/cu126`（并同步调整 torch / torchvision 的固定版本），
> 再执行 `uv sync`。

## 可选：flash-attn

`flash-attn` 以可选依赖（extra）的形式提供：未安装时模型会自动回退到 torch SDPA；一旦可以 import flash-attn，运行时就会自动启用（`--attn_backend auto`）。

```bash
# (a) 通过 PyPI 从源码编译
uv sync --extra flash

# (b) 安装与当前 torch + Python 匹配的预编译 CUDA wheel
uv pip install /path/to/flash_attn-2.8.3+cu12torch28cxx11abitrue-cp311-cp311-*.whl
```
