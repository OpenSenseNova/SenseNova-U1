# SenseNova U1.5 推理指南

面向正式完整模型 **SenseNova-U1.5-8B-MoT**，提供从环境准备、权重校验到首张图片生成的推理流程，以及低显存、多卡和服务部署参考。

[快速开始](#快速开始) · [推理示例](#推理示例) · [文档导航](#文档导航) · [官方项目](https://github.com/OpenSenseNova/SenseNova-U1) · [模型卡](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/blob/9feeeab8a2792514d109cd34589342a2cc1d4ab2/README.md)

## 项目介绍

SenseNova U1.5 是原生统一多模态模型，官方提供文生图、图像编辑、视觉问答和图文交错生成入口。模型能力与示例见[官方介绍](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/README.md)。

本页是 SenseNova-U1 仓库中的**中文 U1.5 推理指南**。为保留固定版本的运行流程，安装步骤会在独立工作目录中检出指定代码 commit、下载固定模型快照，并复制配套工具；不会切换当前文档分支。Python 推理主线以文生图为起点，便于先检查环境、资源和输出，再扩展到其他任务。

这套指南提供：

- **固定版本的准备流程**：使用上游 `uv.lock` 安装依赖，检查 GPU 环境，校验完整权重分片。
- **可重复运行的文生图入口**：每次生成独立输出目录，保留命令、环境、日志、退出码和图片检查结果。
- **不同资源条件下的推理路线**：单卡 full、CPU offload、双卡/四卡模块分配，以及 8step V2 LoRA。
- **分开的部署与集成说明**：API 服务、ComfyUI 和 GGUF 各有独立文档，不混用环境与运行命令。

## 环境与模型

主线面向 **Linux x86_64 + NVIDIA CUDA GPU**，GPU 需要原生 BF16 支持，驱动需兼容 PyTorch cu128。使用预编译 Torch 与 SDPA，不要求预先安装 CUDA Toolkit 或 FlashAttention。完整系统条件见[环境与模型准备](installation.md)。

| 项目 | 固定基线 |
| --- | --- |
| 正式模型 | `sensenova/SenseNova-U1.5-8B-MoT` |
| 上游代码 commit | `94bf26c94ab7c30211dcc29466152d0302c7a262` |
| 模型 revision | `9feeeab8a2792514d109cd34589342a2cc1d4ab2` |
| Python / PyTorch | Python 3.11 / PyTorch 2.8.0+cu128 |
| 默认推理 | BF16 + SDPA，2048×2048，batch 1，50 steps，seed 42 |

首次下载完整快照约需 **35.1 GB** 磁盘空间，环境、缓存和输出另算。完整 BF16 参数本身约 **32.66 GiB**，运行时还需要激活和缓存空间，不能按模型名中的“8B”直接估算显存。单卡 full、offload 与多卡的资源差异见[推理模式选择](inference.md)。

请使用上述正式完整模型。Preview、SFT、LoRA 和 GGUF 文件不能直接替代本页基线。Windows、macOS/Metal、CPU 和其他加速器不在本指南主线范围内。

## 快速开始

### 1. 首次准备环境和权重

**首次使用必须先完成[环境与模型准备](installation.md)第 1–4 节，再执行下面的出图命令。**从本仓库根目录开始，该目录应同时包含 `README.md`、`doc/`、`tools/` 和 `doc/SHA256SUMS`。

安装页会完成以下准备：

1. 校验指南文件，选择可写工作盘，将辅助工具复制到 `U15_ROOT/tools/`，生成 `U15_ROOT/env.sh`。
2. 将官方源码检出到固定 commit，并在 `U15_ROOT/SenseNova-U1/.venv/` 安装锁定依赖。
3. 下载完整模型快照，执行环境、权重、配置和 tokenizer 检查。

默认 `U15_ROOT` 为 `$HOME/sensenova-u15`；自定义工作盘时，后续命令中的 `env.sh` 路径也要相应修改。已有部署应恢复原配置，不覆盖原来的 `env.sh` 或源码目录。

继续前，安装输出应包含 `ENVIRONMENT_CHECK_OK`、8 行 `SHA256_OK`、`MODEL_FILES_CHECK_OK` 和 `CONFIG_TOKENIZER_OK`。任一步报错都先停止排查。

`env.sh` 由安装步骤生成，仓库中不预置该文件。**运行时请使用 `U15_ROOT/tools/` 下的脚本，不要直接在仓库根目录执行 `bash tools/run_t2i.sh`。**

### 2. 生成第一张图片

以下使用单卡 `full` 模式。先确认只有一张获准 GPU 可见，且显存能容纳完整权重及运行开销。显存不足时先看下方 [CPU offload 示例](#使用-cpu-offload)，不要直接启动 full 模式。

在专用 Bash 终端执行；所有后续示例也使用这份配置：

```bash
set -euo pipefail
U15_ENV="$HOME/sensenova-u15/env.sh"  # 自定义工作盘请改为实际路径。
if [[ ! -f "$U15_ENV" ]]; then
  printf '未找到 %s，请先完成 doc/installation.md 第 1–4 节。\n' "$U15_ENV" >&2
  exit 1
fi
source "$U15_ENV"
U15_PROMPT="A red ceramic mug on a wooden table, soft daylight, simple background." \
U15_VRAM_MODE=full U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

脚本会检查代码版本、GPU 和资源条件，再调用[官方 T2I 入口](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/examples/t2i/inference.py)。默认不启用 thinking、提示增强或 LoRA；本次生成目标是木桌上的红色陶瓷杯。

### 3. 检查本次输出

终端会打印本次 `Run directory`，路径形如 `$U15_ROOT/runs/t2i-XXXXXXXX/`。请检查这个目录，不要用旧图片判断本次运行是否成功。

- `first.png`：生成图片，应能完整打开，尺寸为 2048×2048。
- `inference.log`：模型推理日志；`command.sh` 和 `environment.txt` 记录本次命令与环境。
- `script_exit_code.txt`：完整脚本的退出码，应为 `0`；`output_check.txt` 应包含 `PNG_CHECK_OK`。

最后人工查看图片，确认主体符合提示词，不是全黑、噪声或占位图。自动文件检查不评价图像质量。完整验收条件与失败阶段定位见[推理指南](inference.md)和[常见问题与验收](troubleshooting.md)。

## 推理示例

以下示例均以安装完成为前提。新开 Bash 终端后，先恢复配置；自定义工作盘请修改路径：

```bash
source "$HOME/sensenova-u15/env.sh"
```

### 修改提示词

显存充足的单卡可继续使用 full 模式：

```bash
U15_PROMPT="A blue ceramic teapot on a white tabletop, soft studio lighting, product photograph." \
U15_VRAM_MODE=full U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

每次调用都会创建新的输出目录。其余采样参数保持基线设置，便于比较提示词变化；脚本不支持采样断点恢复。

### 使用 CPU offload

单卡显存不足、主存充足时，可先尝试 `balanced`：

```bash
U15_PROMPT="A red ceramic mug on a wooden table, soft daylight, simple background." \
U15_VRAM_MODE=balanced U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

`low`、`balanced`、`fast` 均使用完整 BF16 权重，只改变 CPU/GPU 间的权重调度，不是量化。固定锁页主存权重下界约 **32.51 GiB**，实际还需额外空间，不能靠普通 swap 补足。脚本会先做资源检查，通过也不保证出图成功。模式选择、预算与限制见[单卡低显存推理](inference.md#3-单卡低显存)。

### 更多任务与运行方式

| 目标 | 入口与说明 |
| --- | --- |
| 双卡或四卡文生图 | [多卡推理](inference.md#4-双卡或四卡推理)：使用 `run_t2i_multigpu.py` 分配完整模块，不是张量并行 |
| 8 步采样 | [8step V2 LoRA](inference.md#5-8step-v2-lora)：在正式基座上加载指定适配器，不减少基座体积 |
| 图像编辑、视觉问答、图文交错生成 | [官方任务示例](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/examples/README.md)：在 `$REPO_DIR` 内运行对应入口，逐项验证输入输出 |
| 提供理解与生成 API | [服务部署](serving.md)：独立的 LightLLM + LightX2V 环境与拓扑配置 |
| 节点工作流或量化运行时 | [ComfyUI](integrations/comfyui.md) / [GGUF](integrations/gguf.md)：分别确认模型版本、文件格式与运行时兼容性 |

单卡卸载不能与 `device_map` 叠加。扩展到其他任务、更换硬件或调整部署拓扑后，请分别执行对应验收。

## 指南结构

```text
.
├── README.md / README_CN.md      # 项目介绍与 U1.5 指南入口
├── doc/
│   ├── README.md                # 中文 U1.5 文档索引
│   ├── quickstart.md            # 本页：运行条件与首次推理
│   ├── installation.md          # 环境、固定版本与模型准备
│   ├── inference.md             # Python 推理与资源模式
│   ├── serving.md               # 独立 API 服务部署
│   ├── troubleshooting.md       # 常见问题与验收
│   ├── SHA256SUMS               # 本指南和工具的文件校验清单
│   └── integrations/
│       ├── comfyui.md           # ComfyUI 入口与兼容性边界
│       └── gguf.md              # GGUF 与替代运行时
└── tools/
    ├── verify_env.py            # 依赖、GPU、BF16 与 SDPA 检查
    ├── verify_model.py          # 正式模型文件与分片校验
    ├── memory_preflight.py      # 权重和资源前检
    ├── run_t2i.sh               # 单卡文生图、日志与输出检查
    └── run_t2i_multigpu.py       # 双卡/四卡模块分配推理
```

上面仅列出本指南涉及的文件。仓库原有的 `src/`、`examples/`、`docs/`、`apps/` 和 `training/` 等目录保持不变。为复用固定基线，安装流程会在 `$REPO_DIR` 创建单独的代码检出；模型位于 `$MODEL_DIR`，运行配置、缓存和输出位于 `$U15_ROOT`。不要在当前文档分支直接切换到基线 commit，否则会丢失新指南入口。

## 文档导航

| 我想做什么 | 阅读文档 |
| --- | --- |
| 从零安装，确认硬件和完整模型是否就绪 | [环境与模型准备](installation.md) |
| 调整显存模式、使用多卡或 8step V2 LoRA | [项目推理](inference.md) |
| 启动可接受请求的理解与生成服务 | [服务部署](serving.md) |
| 处理安装失败、OOM、无图或验收问题 | [常见问题与验收](troubleshooting.md) |
| 接入 ComfyUI 工作流 | [ComfyUI](integrations/comfyui.md) |
| 评估量化权重与其他运行时 | [GGUF 与替代运行时](integrations/gguf.md) |

模型能力和任务参数以固定版本的官方实现与模型卡为依据。升级版本后，请重新执行对应检查。代码与权重的使用条件见[上游许可证](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/LICENSE)与[模型卡](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/blob/9feeeab8a2792514d109cd34589342a2cc1d4ab2/README.md)；第三方运行时请分别查看其许可证。
