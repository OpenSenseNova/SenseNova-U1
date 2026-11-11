# 常见问题与验收

本文使用[环境与模型准备](installation.md)中的路径和变量。新终端先恢复实际工作盘中的 `env.sh`。

## 常见问题

**OOM 怎么处理？** 先分清加载阶段还是生成阶段。加载 OOM 时减步数不会缩小权重；优先选卸载或多卡。生成 OOM 再调整尺寸、batch 和模式，并重测峰值。不要把 35.1 GB 下载大小当显存要求。

**`nvidia-smi` 正常但 CUDA 检查失败？** 核对 `$PY` 是否为项目解释器、Torch 是否为 cu128、GPU mask 和容器权限。`nvidia-smi` 的 CUDA Version 是驱动支持信息，`torch.version.cuda` 是当前 Torch 构建信息。

**uv 不可用、HOME 只读或锁文件报错？** 检查 `env.sh` 中 `UV` 和两个 `UV_PYTHON_*` 路径；系统缺 venv/pip 时走正式包管理器，不用 `sudo pip`。锁文件冲突先核对 commit、源码改动和 uv 版本，不删除 `--locked`。

**找不到 `sensenova_u1` 或不识别 `neo_chat`？** 确认 `uv sync` 成功、使用 `$PY`，并先 `import sensenova_u1` 注册本地模型，不用 `trust_remote_code=True` 掩盖错误环境。

**下载报 401/403、SSL 或连接错误？** 核对 URL、revision、证书、机器时间、代理与授权。中断后复用同目录重试；不要关闭 TLS 校验，不在日志中写 token。

**SOCKS 代理报缺少 `socksio`？** 仅在已使用 SOCKS 且明确报此错时执行：

```bash
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple socksio==1.0.0
"$UV" pip check --python "$PY"
"$UV" pip freeze --python "$PY" > "$U15_ROOT/logs/freeze-with-proxy-support.txt"
```

这是官方锁外的可选补充。再次 `uv sync --locked` 会移除它，需要时重装；不修改官方锁文件，也不默认删除代理变量。

**没出图，从哪里排查？** 用终端打印的本次运行目录，先读 `stage.txt`、`script_exit_code.txt` 和该阶段日志。`inference_started.txt=no` 表示尚未进入推理；不能拿旧 PNG 或只有 `[saved]` 的日志当成功。多卡优先看 `result.json`、`traceback.txt` 和 actual map。

**为什么多卡仍只用一张，或出现跨设备错误？** 只设 `max_memory` 不会建立 map，`sequential` 也可能把模型放满第一张卡。使用多卡运行脚本检查全部参数的实际设备；不要用整模型 `.to(cuda:0)` 掩盖跨卡问题。

**能离线吗？** 先备齐依赖、配置、tokenizer 和权重。`HF_HUB_OFFLINE=1` 或本地加载只约束部分下载行为，不等于进程网络隔离；严格离线需在获准隔离环境重新测试。

**重启、固定 seed 或 HTTP 200 就算通过吗？** 均不能单独判断。重启要恢复持久化路径并重跑；跨后端固定 seed 不保证逐像素一致。服务还需检查响应错误、图片数量、解码、尺寸和人工内容，见[服务部署](serving.md)。

## 目标机验收

每种 GPU、模式、任务、精度和服务拓扑分别完成以下检查：

- [ ] 固定代码、权重和适配器身份；环境检查、文件校验通过。
- [ ] 记录 GPU/驱动、可见编号映射、互联、主存/cgroup 限额和完整依赖。
- [ ] 记录 prompt/seed、尺寸、steps/CFG、dtype/backend、卸载预算或 planned/actual map。
- [ ] 推理正常退出，产物可解码、尺寸和数量正确，人工看图通过。
- [ ] 分开测加载、冷启动、同进程重复生成和重启；建议同进程至少 3 次、至少 3 个固定 seed 对照质量。
- [ ] 分别记录每卡 PyTorch allocated/reserved 峰值、驱动侧显存采样、CPU RSS/cgroup peak 和耗时；未测留空。官方 profile 可能漏掉被重置前的加载峰值。
- [ ] 记录 OOM、跨设备异常、取消和恢复；要提供服务时另测并发和故障恢复。

反馈问题时附版本、机器/资源、完整命令、失败阶段、退出码、traceback、map 和输出检查结果；先去掉 token、代理凭据和私密输入。
