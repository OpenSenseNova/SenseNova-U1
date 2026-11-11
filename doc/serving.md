# SenseNova U1.5 多卡服务

使用独立的 **LightLLM + LightX2V** 环境提供理解与生成服务。开始前，先按 [环境与模型准备](installation.md) 完成正式模型的下载与校验。

用于生产前，需在目标机器确认镜像依赖、FA3 内核、资源容量和真实请求结果。

## 1. 选择总卡数和拓扑

| 总卡数 | 模式 | 理解 | 生成 | 配置文件 |
| --- | --- | --- | --- | --- |
| 2 | separate | TP1，卡 0 | 单卡，卡 1 | `neopp_dense.json` |
| 2 | colocate | TP2，卡 0/1 | CFG2，卡 0/1 | `neopp_dense_parallel_cfg.json` |
| 4 | separate | TP2，卡 0/1 | CFG2，卡 2/3 | `neopp_dense_parallel_cfg.json` |
| 4 | colocate | TP2，卡 0/1 | CFG2×SEQ2，卡 0–3 | `neopp_dense_parallel_cfg_seq.json` |
| 4 | separate | TP2，卡 0/1 | 卡 2、3 各一个单卡副本 | `neopp_dense.json` |

生成 world = CFG 并行数 × SEQ 并行数；生成池有 4 卡、world 为 2 时，是两个 2 卡副本。separate 另占理解 TP×DP 张卡，所以四卡生成 CFG2×SEQ2 加理解 TP1 共至少 5 卡，加 TP2 共至少 6 卡。colocate 共享显存，理解 KV 与生成中间量可能争用，不能用权重体积除以卡数估算峰值。

三份配置均为非量化、50 步、CFG 4、shift 3、`flash_attn3`，不含 V2 LoRA。[FA3 官方说明](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#flashattention-3-beta-release)的 Hopper 实现要求 H100/H800、CUDA ≥ 12.3，建议 12.8。A100/4090 支持 FA2 不等于能运行此配置；其他卡型或替代后端需单独核对构建与内核。

## 2. 启动隔离容器

管理员须先准备 Docker、NVIDIA Container Toolkit 和 GPU 权限。恢复[环境与模型准备](installation.md)中的 `env.sh`，把 `SERVICE_GPU_IDS` 设为已获准使用的宿主机数字编号，不能直接填 UUID/MIG 字符串。

拉取官方镜像后记录并使用 digest，并确认它与下述固定源码的依赖兼容。以下沿用官方 host IPC/network，仅适合授权实验机，共享宿主机命名空间；服务绑定本机，不要直接暴露公网。

```bash
# 先 source 实际路径/env.sh，并 export SERVICE_GPU_IDS=获准的编号列表，例如 0,1。
(
  set -euo pipefail
  : "${MODEL_DIR:?先恢复主线 env.sh}"
  : "${SERVICE_GPU_IDS:?先设置获准的宿主机 GPU 数字编号}"
  export SERVICE_ROOT="$U15_ROOT/service"
  mkdir -p "$SERVICE_ROOT"
  TAG=lightx2v/lightllm_lightx2v:20260407
  docker pull "$TAG"
  IMAGE="$(docker image inspect --format '{{index .RepoDigests 0}}' "$TAG")"
  test -n "$IMAGE"
  printf '%s\n' "$IMAGE" | tee "$SERVICE_ROOT/image-digest.txt"
  docker run --rm -it --name u15-service-lab \
    --gpus "\"device=$SERVICE_GPU_IDS\"" --ipc=host --network host \
    --mount "type=bind,src=$MODEL_DIR,dst=/models/u15,readonly" \
    --mount "type=bind,src=$SERVICE_ROOT,dst=/u15-service" \
    "$IMAGE" /bin/bash
)
```

若容器名已占用，用 `docker ps -a --filter name=u15-service-lab` 检查，不自动删除。另选容器名时，下文 `docker exec` 也要同步修改。

## 3. 固定源码并检查依赖

在**容器内**执行。三个目标目录须为新目录；已有内容先检查，不覆盖。

```bash
set -euo pipefail
export MODEL_DIR=/models/u15
export LL=/u15-service/LightLLM
export LX=/u15-service/LightX2V
export U1=/u15-service/SenseNova-U1
for p in "$LL" "$LX" "$U1"; do test ! -e "$p"; done
git clone https://github.com/ModelTC/LightLLM.git "$LL"
git -C "$LL" checkout --detach dba4de76c3627a5a544334fdbff0e8717ddfe585
git clone https://github.com/ModelTC/LightX2V.git "$LX"
git -C "$LX" checkout --detach 9fd3d98c178d7a9430334683637cc8c31f3920ee
git clone https://github.com/OpenSenseNova/SenseNova-U1.git "$U1"
git -C "$U1" checkout --detach 94bf26c94ab7c30211dcc29466152d0302c7a262
export PYTHONPATH="$LL:$LX${PYTHONPATH:+:$PYTHONPATH}"
python --version
python -m pip freeze > /u15-service/freeze.txt
nvidia-smi
python -m lightllm.server.api_server --help > /u15-service/server-help.txt
python "$U1/examples/serving/client.py" --help > /u15-service/client-help.txt
python - <<'PY'
import json, os
from pathlib import Path
for name in ('neopp_dense.json', 'neopp_dense_parallel_cfg.json', 'neopp_dense_parallel_cfg_seq.json'):
    cfg = json.loads((Path(os.environ['LX']) / 'configs/neopp' / name).read_text())
    print(name, json.dumps(cfg, ensure_ascii=False))
    assert cfg['attn_type'] == 'flash_attn3'
    assert cfg['infer_steps'] == 50 and cfg['cfg_scale'] == 4.0
PY
```

帮助或 import 失败就停止，核对镜像与源码依赖，不盲目升级 Transformers，也不把主线 `.venv` 清单装进容器。官方未给出本三仓组合的完整可复用锁文件；修复后记录最终 freeze、内核与编译条件。配置检查没有加载 FA3，必须后续真实执行。

## 4. 启动一套服务

Docker 可能重新编号 GPU，下列参数使用**容器内编号**。先定义函数，再只选一条调用前台运行：

```bash
serve_u15() {
  local visible="$1" mode="$2" tp="$3" gen="$4" conf="$5"
  CUDA_VISIBLE_DEVICES="$visible" \
  PYTHONPATH="$LL:$LX${PYTHONPATH:+:$PYTHONPATH}" \
  python -m lightllm.server.api_server \
    --model_dir "$MODEL_DIR" --enable_multimodal_x2i \
    --x2i_server_deploy_mode "$mode" --x2i_server_used_gpus "$gen" \
    --x2v_gen_model_config "$LX/configs/neopp/$conf" \
    --host 127.0.0.1 --port 8000 --max_req_total_len 65536 \
    --mem_fraction 0.75 --tp "$tp" --dp 1
}

# 默认：总共 2 卡，理解 1 卡 + 生成 1 卡。
serve_u15 0,1 separate 1 1 neopp_dense.json 2>&1 | tee /u15-service/server.log

# 其他拓扑只替换上一行的 serve_u15 参数，保留日志管线：
# 2 卡共置：0,1 colocate 2 2 neopp_dense_parallel_cfg.json
# 4 卡分离：0,1,2,3 separate 2 2 neopp_dense_parallel_cfg.json
# 4 卡共置：0,1,2,3 colocate 2 4 neopp_dense_parallel_cfg_seq.json
# 4 卡分离双生成副本：0,1,2,3 separate 2 2 neopp_dense.json
```

65536 和 0.75 来自官方示例，是待测上限与内存比例，不是容量保证，尤其 colocate 需要留出生成空间。记录每个 worker 的可见卡、rank/world、端口和显存。该路线依赖本机共享资源，不代表自动支持跨机器部署。

## 5. 验证真实请求

服务保持运行，在宿主机第二个终端进入同一容器：

```bash
docker exec -it u15-service-lab /bin/bash
```

在第二个容器终端运行无输入图的 T2I 请求，并校验保存的 JSON 和图片：

```bash
(
  set -euo pipefail
  U1=/u15-service/SenseNova-U1
  OUT="$(mktemp -d /u15-service/request-XXXXXXXX)"
  python "$U1/examples/serving/client.py" \
    --mode t2i --url http://127.0.0.1:8000/v1 \
    --prompt 'A red ceramic mug on a wooden table, soft daylight, simple background.' \
    --no-enable-thinking --aspect-ratio 1:1 --image-size 2K \
    --height 2048 --width 2048 --seed 42 --out-dir "$OUT" \
    2>&1 | tee "$OUT/client.log"
  export OUT
  python - <<'CHECK'
import hashlib, json, os
from pathlib import Path
from PIL import Image
if not __debug__:
    raise SystemExit('Run without -O or PYTHONOPTIMIZE')
root = Path(os.environ['OUT'])
responses = list(root.glob('*_t2i_response.json'))
assert len(responses) == 1, f'Expected one response JSON, got {len(responses)}'
data = json.loads(responses[0].read_text())
assert not data.get('error'), data.get('error')
choices = data.get('choices') or []
assert len(choices) == 1, 'Expected one response choice'
message = choices[0].get('message') or {}
assert len(message.get('images') or []) == 1, 'Expected one image in response'
files = [p for p in root.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
assert len(files) == 1, f'Expected one saved image, got {len(files)}'
image = files[0]
with Image.open(image) as im:
    fmt, size = im.format, im.size
    im.verify()
with Image.open(image) as im:
    im.load()
    assert im.size == (2048, 2048), im.size
print('API_IMAGE_CHECK_OK', fmt, size, image)
print('SHA256', hashlib.sha256(image.read_bytes()).hexdigest())
print('Manual visual review still required')
CHECK
  printf '检查本次请求目录：%s\n' "$OUT"
)
```

客户端可能保存 JPEG，以实际解码结果为准。`API_IMAGE_CHECK_OK` 后还需人工看图，并核对 worker 日志。退出 0、HTTP 200 或 health ready 都不等于请求成功。该 T2I 客户端使用非流式响应；测试流式交错时另查完整 SSE 错误与正常结束。

## 6. 其他限制和上线前检查

- 固定客户端存在 `--image_path` 与 `args.image` 字段不一致，编辑和带图交错不能照搬为验收探针。VQA、编辑、交错和 thinking 均需单独测试。
- 理解 FP8 使用 `--quant_type fp8w8a8`，生成 FP8 另选 `neopp_dense_fp8.json`；只设置一侧不等于全服务量化。先运行非量化配置，再检查 FP8 权重、内核支持、资源占用和输出质量。
- 按[验收清单](troubleshooting.md)记录冷启动、重复生成、重启和资源峰值，再测试并发、取消、超时与恢复。以真实请求的结果为准。
- 对外服务还需鉴权、请求队列、并发与输入限额、日志隐私、进程隔离、监控和恢复策略。正式开放端口前单独审查容器与网络配置。

## 固定来源

- [官方部署说明](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/docs/deployment.md)、[官方客户端](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/examples/serving/client.py)
- [LightLLM manager](https://github.com/ModelTC/LightLLM/blob/dba4de76c3627a5a544334fdbff0e8717ddfe585/lightllm/server/x2i_server/manager.py)、[LightX2V adapter](https://github.com/ModelTC/LightLLM/blob/dba4de76c3627a5a544334fdbff0e8717ddfe585/lightllm/server/x2i_server/lightx2v/adapter.py)
- [LightX2V NeoPP](https://github.com/ModelTC/LightX2V/blob/9fd3d98c178d7a9430334683637cc8c31f3920ee/scripts/neopp/README.md)
