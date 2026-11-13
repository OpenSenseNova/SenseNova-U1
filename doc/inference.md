# SenseNova U1.5 推理

本文只覆盖项目 Python 推理入口。先完成[环境与模型准备](installation.md)，再在专用 Bash 终端恢复配置：

```bash
source "$HOME/sensenova-u15/env.sh"  # 自定义工作盘请改为实际路径。
```

默认使用正式完整基座、BF16 + SDPA、2048×2048、batch 1。以下容量仅供选型，实际资源需求与运行结果以目标环境为准。

## 1. 选择模式

| 资源或目标 | 首测路线 | 关键条件 |
| --- | --- | --- |
| 48/80 GB 级单卡 | `full` | 完整权重之外还要留激活和缓存空间 |
| 24/32 GB 级单卡 | `balanced`，必要时 `low` | 主存充足，先做资源前检 |
| 12/16 GB 级单卡 | `low` 实验 | 2K 激活仍可能超限，不保证能出图 |
| 两张/四张 24 GB 级卡 | `full + device_map=balanced` | 分模块放置，检查实际 map 和每卡余量 |
| 已能出图，希望减少采样步数 | 正式基座 + 8step V2 LoRA | 不减少基座体积，另验质量和合并内存 |

## 2. 单卡推理

确认只有一张获准 GPU 可见，再运行：

```bash
U15_VRAM_MODE=full U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

[运行脚本](../tools/run_t2i.sh)调用官方 T2I 入口：**2048×2048、batch 1、50 steps、seed 42、CFG 4.0、cfg_norm none、CFG interval 0–1、shift 3.0、BF16、SDPA**。默认生成木桌上的红色陶瓷杯，不启用 thinking、提示增强或 LoRA。

每次生成新的 `runs/t2i-XXXXXXXX/`，终端打印本次 `Run directory`。验收时检查：

1. `script_exit_code.txt`、`exit_code.txt`、`log_exit_code.txt` 和两个 `output_check_*exit_code.txt` 均为 0。
2. `output_check.txt` 含 `PNG_CHECK_OK`，本次 `first.png` 可完整解码，尺寸 2048×2048。
3. **人工打开图片**，确认提示词内容合理，不是全黑、噪声或占位图。文件检查不评价图像质量。

新终端恢复原配置即可复跑；使用自定义工作盘时相应修改第一行：

```bash
source "$HOME/sensenova-u15/env.sh"
U15_PROMPT="A red cube on a white background, studio photograph." \
U15_VRAM_MODE=full U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

脚本每次重新加载模型，不支持采样断点恢复。不要重放旧 `command.sh`，其中输出路径仍指向旧目录。修改尺寸时须同时修改生成参数与输出检查，并重新验收。

## 3. 单卡低显存

本分支复用同一环境和完整权重。三种模式仍为 BF16，不是量化：`low` 同步逐层搬运；`balanced` 预取后续两层；`fast` 另按预算驻留部分生成权重。

先检查主存和单卡环境：

```bash
"$PY" "$U15_ROOT/tools/verify_env.py" --gpus 1
"$PY" "$U15_ROOT/tools/memory_preflight.py" \
  --model-dir "$MODEL_DIR" --vram-mode balanced --device cuda:0 \
  --output "$U15_ROOT/logs/preflight-balanced.json"
```

返回 2 就停止。返回 0 只表示未触发已知前置问题，不保证容量充足；还要核对嵌套 cgroup、调度限额和其他进程。主存足够时先测试 `balanced`；需要测试 `low` 时，再把下行模式改为 `low` 单独重跑：

```bash
U15_VRAM_MODE=balanced U15_LORA_PATH= bash "$U15_ROOT/tools/run_t2i.sh"
```

`balanced` 稳定后可试 `fast`。下例按 24 GiB 物理显存设置初始预算，运行时仍须检查峰值，不适用于更小的卡：

```bash
U15_VRAM_MODE=fast U15_LORA_PATH= \
U15_FAST_BUDGET_GIB=18 U15_FAST_HEADROOM_GIB=2 U15_FAST_ACTIVATION_GIB=4 \
  bash "$U15_ROOT/tools/run_t2i.sh"
```

预算不是进程显存硬上限。**不要将这些卸载模式与 `device_map` 叠加**；固定 loader 会清除非 full 模式下的 map。主存不足时从 balanced 改 low 也不能解决固定锁页权重下界。

## 4. 双卡或四卡推理

此路线使用 **full + balanced device map**，把完整模块放到不同卡，不是张量并行。不要用 `torchrun` 包住单卡 CLI，否则可能只是启动多份模型。

先按调度分配设置恰好 2/4 张可见 GPU，不覆盖已有分配。下例默认双卡；四卡仅替换前两行的值，再执行一次：

```bash
N=2
BUDGET='0=18GiB,1=18GiB,cpu=0GiB'
# 四卡改为：N=4
# BUDGET='0=12GiB,1=12GiB,2=12GiB,3=12GiB,cpu=0GiB'
(
  set -euo pipefail
  cd "$REPO_DIR"
  LAUNCH_LOG="$(mktemp "$U15_ROOT/runs/launch-${N}gpu-XXXXXXXX.log")"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -u "$U15_ROOT/tools/run_t2i_multigpu.py" \
    --repo-dir "$REPO_DIR" --model-dir "$MODEL_DIR" --run-root "$U15_ROOT/runs" \
    --gpus "$N" --device-map balanced --max-memory "$BUDGET" --headroom-gib 4 \
    2>&1 | tee "$LAUNCH_LOG"
)
```

两组预算均按 24 GB 级卡设置。`max_memory` 主要约束权重分配，4 GiB 预留不保证足以容纳激活和缓存，需根据实际峰值调整。

[多卡运行脚本](../tools/run_t2i_multigpu.py)逐卡检查环境、先规划再加载，拒绝计划或实际 map 落到 CPU/disk、未使用全部目标 GPU，以及参数设备不一致。`cpu=0GiB` 本身不能保证不退到 disk；`max_memory` 也不是峰值上限。

保留本次 `planned_device_map.json`、`hf_device_map.json`、`result.json`、日志和图片。自动生成检查通过后仍须人工看图；`complete_acceptance_passed` 默认保持 false。P2P 结果只作记录，真实互联性能另测。脚本使用固定 Transformers 内部 API，不能跨版本直接套用。

## 5. 8step V2 LoRA

先完成基础模型单卡推理，再加载 LoRA。LoRA 只减少采样步数，不压缩基座，合并还需额外主存；适配器兼容性和输出质量需单独检查。

```bash
(
  set -euo pipefail
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  LORA_DIR="$U15_ROOT/models/u15-lora"
  LORA_FILE=SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors
  "$HF" download sensenova/SenseNova-U1.5-8B-MoT-LoRAs "$LORA_FILE" \
    --revision f33b8fe0216e6f10e5aed58ebbfdd05f02826732 --local-dir "$LORA_DIR"
  printf '%s  %s\n' f4190cabd01fa261a45be853275c483d3a8069295c52cf01f7b2b31f4356f201 \
    "$LORA_DIR/$LORA_FILE" | sha256sum --check --strict
)
U15_VRAM_MODE=balanced \
U15_LORA_PATH="$U15_ROOT/models/u15-lora/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors" \
  bash "$U15_ROOT/tools/run_t2i.sh"
```

最后一步要求单卡。脚本校验指定 [V2 文件](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/blob/f33b8fe0216e6f10e5aed58ebbfdd05f02826732/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors)的 hash，切换为 **8 steps、CFG 1.0、cfg_norm none、shift 3.0**。只把基础模型步数改成 8 不是 V2 配方，也不能据此保证 50/8 倍端到端加速。

## 6. 其他任务与服务

编辑、VQA、交错和 thinking 使用[官方任务入口](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/examples/README.md)，先查看各入口的 `--help`，再独立验证输入输出。T2I 首图不代表其他任务通过。提示增强可能向外部 API 发送输入并产生费用，本指南默认不启用。

需要提供 API 时，转到独立的[服务部署](serving.md)；问题排查与完整验收见[常见问题与验收](troubleshooting.md)。

## 来源

- [T2I 入口](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/examples/t2i/inference.py)
- [权重加载与设备映射](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/src/sensenova_u1/utils/checkpoint_loading.py)
