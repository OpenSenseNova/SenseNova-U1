# 环境与模型准备

完成本页后，继续[项目推理](inference.md)。

## 1. 运行条件

- Linux x86_64、glibc ≥ 2.28，已安装 Git、Bash、Python 3、venv/pip。
- NVIDIA 驱动可运行 Torch 2.8.0 cu128，目标 GPU 支持原生 BF16；容器或调度器已授予 GPU 访问权。
- 工作盘可写且持久化。正式快照约 **35.1 GB**，环境、缓存、临时文件和输出另算；完整实验工作盘可按 **100 GB 以上**规划，这不是运行门槛。
- 可访问 GitHub、PyPI、PyTorch cu128 源和 Hugging Face。不要关闭 TLS 校验或绕过组织网络策略。

主线使用预编译 Torch 和 SDPA，无需预先安装 CUDA Toolkit 或 FlashAttention。Windows、macOS/Metal、CPU、AMD、XPU、ARM 不在本指南覆盖范围内。

**容量建议不是测得的最低要求。实际显存、主存占用与性能取决于设备、任务和推理配置，请按目标环境确认。**

完整 BF16 参数体积为 **32.6575 GiB**，不能按模型名“8B”估成 16 GB。单张 32 GiB 卡、两张 16 GiB 卡均小于完整参数体积，不能容纳全部 BF16 权重。

`low/balanced/fast` 的固定锁页主存权重下界约 **32.5072 GiB**，还需加载、缓存和运行空间；普通 swap 不能补足。主存可从 64 GiB 级配置评估，大显存基准或复杂任务可评估 128 GiB，这些不是测得最低值。`fast` 保留部分 GPU 权重也不会释放这份 CPU 副本。资源依据见 [权重索引](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/blob/9feeeab8a2792514d109cd34589342a2cc1d4ab2/model.safetensors.index.json)和 [offload 实现](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/src/sensenova_u1/utils/layer_offload.py)。

## 2. 固定版本并准备目录

| 项目 | 本文基线 |
| --- | --- |
| 代码 commit | `94bf26c94ab7c30211dcc29466152d0302c7a262` |
| 模型 | `sensenova/SenseNova-U1.5-8B-MoT` |
| 模型 revision | `9feeeab8a2792514d109cd34589342a2cc1d4ab2` |
| Python / uv | 3.11 / 0.13.7 |
| Torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| Transformers / Accelerate | 5.15.0 / 1.14.0 |
| Hub / safetensors / Pillow | 1.27.0 / 0.8.0 / 12.0.0 |

依赖按固定仓库的 `uv.lock` 整体安装，不逐包拼装，也不混入 `training/` 环境。锁文件不锁 NVIDIA 驱动和 Python 补丁版本。不要直接切换到 main；升级时另建环境，重新检查依赖和推理结果。

在**专用 Bash 终端**中依次执行。报错就停止，修复后重试当前步骤。先进入本文件上一级、包含 `README.md`、`doc/` 和 `tools/` 的仓库根目录，将 `U15_ROOT` 改为可写模型盘路径：

```bash
set -euo pipefail
export GUIDE_DIR="$PWD"  # 当前目录必须含 README.md、doc/、tools/ 和 doc/SHA256SUMS。
(cd "$GUIDE_DIR" && sha256sum --check doc/SHA256SUMS)
export U15_ROOT="$HOME/sensenova-u15"
test ! -e "$U15_ROOT/env.sh"  # 已有部署请 source 原文件，不覆盖。
mkdir -p "$U15_ROOT"/{models,logs,runs,tools,cache}
cp "$GUIDE_DIR"/tools/{verify_env.py,verify_model.py,memory_preflight.py,run_t2i.sh,run_t2i_multigpu.py} "$U15_ROOT/tools/"
printf 'export U15_ROOT=%q\n' "$U15_ROOT" > "$U15_ROOT/env.sh"
cat >> "$U15_ROOT/env.sh" <<'ENV'
export CODE_COMMIT=94bf26c94ab7c30211dcc29466152d0302c7a262
export MODEL_ID=sensenova/SenseNova-U1.5-8B-MoT
export MODEL_REV=9feeeab8a2792514d109cd34589342a2cc1d4ab2
export REPO_DIR="$U15_ROOT/SenseNova-U1"
export MODEL_DIR="$U15_ROOT/models/$MODEL_REV"
export HF_HOME="$U15_ROOT/cache/huggingface"
export UV_CACHE_DIR="$U15_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$U15_ROOT/tooling/python"
export UV_PYTHON_BIN_DIR="$U15_ROOT/tooling/python-bin"
export UV="$U15_ROOT/tooling/bin/uv"
export PY="$REPO_DIR/.venv/bin/python"
export HF="$REPO_DIR/.venv/bin/hf"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
ENV
source "$U15_ROOT/env.sh"
bash -n "$U15_ROOT/tools/run_t2i.sh"
df -h "$U15_ROOT"
free -h
ldd --version
nvidia-smi
nvidia-smi -L
```

`env.sh` 只保存版本和路径，不放 token。HOME 只读时改用获准的可写工作盘，保留上面两个 `UV_PYTHON_*` 设置。

GPU 分配遵从调度器或容器的 `CUDA_VISIBLE_DEVICES`；未设置时模板默认卡 0。单卡路线只保留一张获准 GPU，多卡路线明确分配 2/4 张。进程内 `cuda:0` 是第一张可见卡，不一定是宿主机物理卡 0。

## 3. 安装和环境检查

```bash
test ! -e "$REPO_DIR"  # 已有目录先检查，或另建工作目录。
git clone https://github.com/OpenSenseNova/SenseNova-U1.git "$REPO_DIR"
git -C "$REPO_DIR" checkout --detach "$CODE_COMMIT"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
git -C "$REPO_DIR" status --short

python3 -m venv "$U15_ROOT/tooling"
"$U15_ROOT/tooling/bin/python" -m pip install --index-url https://pypi.org/simple uv==0.13.7
"$UV" --version | tee "$U15_ROOT/logs/uv-version.txt"
"$UV" python install 3.11
cd "$REPO_DIR"
"$UV" --project . sync --locked --python 3.11 2>&1 | tee "$U15_ROOT/logs/install.log"
"$UV" pip check --python "$PY" 2>&1 | tee "$U15_ROOT/logs/dependency-check.log"
"$UV" pip freeze --python "$PY" > "$U15_ROOT/logs/freeze.txt"
"$PY" --version
"$PY" "$U15_ROOT/tools/verify_env.py" 2>&1 | tee "$U15_ROOT/logs/environment-check.log"
"$PY" examples/t2i/inference.py --help > "$U15_ROOT/logs/t2i-help.txt"
```

`git status --short` 应为空；detached HEAD 正常。环境检查逐卡执行原生 BF16 和 SDPA 小算子，应输出 `ENVIRONMENT_CHECK_OK`。`nvidia-smi` 正常、cu128 安装成功或 `--help` 可用都不能替代此检查。无 GPU 机器只能执行下一节的模型文件检查。

## 4. 下载并校验完整模型

```bash
mkdir -p "$MODEL_DIR"
"$HF" download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" \
  2>&1 | tee "$U15_ROOT/logs/download.log"
"$PY" "$U15_ROOT/tools/verify_model.py" 2>&1 | tee "$U15_ROOT/logs/model-check.log"
cd "$REPO_DIR"
"$PY" - <<'PY'
import os
import sensenova_u1
from transformers import AutoConfig, AutoTokenizer
p = os.environ["MODEL_DIR"]
c = AutoConfig.from_pretrained(p, local_files_only=True)
t = AutoTokenizer.from_pretrained(p, local_files_only=True)
assert c.model_type == "neo_chat", c.model_type
print("CONFIG_TOKENIZER_OK", type(c).__name__, type(t).__name__)
PY
```

应有 8 行 `SHA256_OK`、`MODEL_FILES_CHECK_OK` 和 `CONFIG_TOKENIZER_OK`。检查器完整读取 8 个权重分片核对 SHA256，也检查小文件存在、大小及 JSON 语法；小文件未在该脚本中做远端 hash 比对。

下载中断后重跑相同 ID、revision 和目录的命令，可复用已完成文件。不要换成 Preview、SFT 或 LoRA；它们不等于本文正式完整基座。推理 CLI 没有 `--revision` 参数，所以先下载固定快照，再传本地路径。

环境检查与模型校验通过后，按[推理指南](inference.md)运行；故障排查见[常见问题](troubleshooting.md)。

## 来源

- [固定版本安装说明](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/docs/installation_CN.md)、[依赖锁文件](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/uv.lock)
- [正式模型快照](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/tree/9feeeab8a2792514d109cd34589342a2cc1d4ab2)
