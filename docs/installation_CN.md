# 安装指南

本指南介绍使用 `transformers` 后端在本地运行 SenseNova-U1 和 U1.5 所需的 Python 环境。

> 仓库根目录是**推理项目**。训练在 [`training/`](../training/README.md) 下维护独立的项目、
> 锁文件和虚拟环境；不要把两套依赖安装到同一个环境中。

> **软件版本：** Python 3.11、torch 2.8、CUDA 12.8（cu128）。如果本机驱动需要其他 CUDA 版本，请相应修改 `pyproject.toml` 中的 index URL。

使用 [**uv**](https://docs.astral.sh/uv/) 和仓库中的 `uv.lock` 安装参考环境。

> uv 安装指南：<https://docs.astral.sh/uv/getting-started/installation/>

## 1. 克隆仓库

```bash
git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-U1.git
cd SenseNova-U1
```

## 2. 使用 uv 安装依赖

```bash
uv --project . sync --locked
source .venv/bin/activate
```

`sensenova_u1` 会以可编辑模式安装，因此在 import 时，标准的 [NEO-Unify 模型](../src/sensenova_u1/models/neo_unify/) 会自动注册到 `transformers.Auto*` 接口。

> **较旧的 NVIDIA 驱动：** 默认 index 对应 CUDA 12.8。若驱动不支持 cu128，请先将
> `pyproject.toml` 中的 `[tool.uv.sources]` / `[[tool.uv.index]]` 改为例如
> `https://download.pytorch.org/whl/cu126`（并同步调整 torch / torchvision 的固定版本），
> 再执行 `uv lock` 和 `uv --project . sync --locked`。

## 可选：flash-attn

`flash-attn` 以可选依赖（extra）的形式提供：未安装时模型会自动回退到 torch SDPA；一旦可以 import flash-attn，运行时就会自动启用（`--attn_backend auto`）。

```bash
# (a) 通过 PyPI 从源码编译
uv --project . sync --locked --extra flash

# (b) 安装与当前 torch + Python 匹配的预编译 CUDA wheel
uv pip install /path/to/flash_attn-2.8.3+cu12torch28cxx11abitrue-cp311-cp311-*.whl
```

激活环境后，选择对应的模型指南：[U1.5](u1.5_model_CN.md)、[U1.5 Preview](u1.5_preview_CN.md#运行-u15-preview)、[U1 信息图](u1_infographic_model_CN.md#运行信息图-v3)或[原版 U1](u1_model_CN.md)。各页分别提供运行命令和案例入口。
