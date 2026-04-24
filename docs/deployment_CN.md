# LightLLM + LightX2V 部署

本文档介绍如何基于 Docker 镜像 `lightx2v/lightllm_lightx2v:20260407`，通过 LightLLM + LightX2V 部署 SenseNova-U1 推理服务。

## 1) 拉取并进入 Docker 镜像

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
docker run --gpus all --ipc=host --network host -it lightx2v/lightllm_lightx2v:20260407 /bin/bash
```

## 2) 在容器内克隆运行时依赖

镜像中自带的源码未必是最新版本，建议重新克隆这两个仓库，并将 LightLLM 切到已验证的分支：

```bash
git clone https://github.com/ModelTC/LightX2V.git
git clone https://github.com/ModelTC/LightLLM.git
cd LightLLM
git checkout neo_plus_clean
```

## 3) X2I 相关参数

在同一个 API 服务中开启图像生成时，用到以下参数：

- `--enable_multimodal_x2i`  
  开启图像生成能力。
- `--x2i_server_used_gpus`  
  分配给 X2I 生成服务的 GPU 数量。
- `--x2i_server_deploy_mode {colocate,separate}`  
  - `colocate`：理解与生成共用同一块可见 GPU 资源池。
  - `separate`：理解与生成拆分为独立服务，可分别占用不同的 GPU。
- `--x2i_use_naive_impl`  
  X2I 使用原生 PyTorch 实现，仅用于调试与测试，不建议在生产环境追求吞吐量时使用。

## 4) 部署模式

### 模式 A：`colocate`（单服务共用 GPU）

适合快速验证与简化运维。LLM 理解路径（`--tp`）与 X2I 生成路径（`--x2i_server_used_gpus`）从同一组可见 GPU 中分配资源。

示例（共 2 张 GPU）：
- 理解路径：`tp=2`
- 生成路径：`cfg=2`（在 `neopp_dense_parallel_cfg.json` 中配置）

```bash
PYTHONPATH=/workspace/LightX2V/ \
python -m lightllm.server.api_server \
  --model_dir $MODEL_DIR \
  --enable_multimodal_x2i \
  --x2i_server_deploy_mode colocate \
  --x2i_server_used_gpus 2 \
  --x2v_gen_model_config /workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg.json \
  --host 0.0.0.0 \
  --port 8000 \
  --max_req_total_len 65536 \
  --mem_fraction 0.75 \
  --tp 2
```

### 模式 B：`separate`（理解与生成分离部署）

`separate` 的思路与 LLM 服务中的 PD 分离类似：将不同阶段放到不同的 GPU 组上，避免长阶段拖慢短阶段。

在多模态场景下，图像生成通常是长阶段，而理解请求轻量且耗时较短。将两者分离后，即便生成 worker 被占满，理解请求依然能正常流转。

推荐的部署配置方案：

1. **默认方案（以稳定性为先）：理解 `tp=1` + 生成 1 GPU**
   - 理解：`--tp 1`
   - 生成：`--x2i_server_used_gpus 1`
   - 适合作为混合负载下的基线方案。pipeline 简单，又能避免理解与生成互相产生队头阻塞。

2. **理解加强方案：理解 `tp=2` + 生成 1 GPU**
   - 理解：`--tp 2`
   - 生成：`--x2i_server_used_gpus 1`
   - 适用于复杂 prompt 或高理解 QPS 成为瓶颈的场景。

3. **生成加强方案：理解 `tp=1/2` + 生成并行**
   - 理解：`--tp 1` 或 `--tp 2`
   - 生成方案 A（2 GPU）：`--x2i_server_used_gpus 2` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg.json`
   - 生成方案 B（4 GPU）：`--x2i_server_used_gpus 4` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg_seq.json`
   - 适用于生成延迟/吞吐量占主导的场景，也是最常见的扩容路径。

`separate` 模式启动 API 服务示例：

```bash
PYTHONPATH=/workspace/LightX2V/ \
python -m lightllm.server.api_server \
  --model_dir $MODEL_DIR \
  --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate \
  --x2i_server_used_gpus 1 \
  --x2v_gen_model_config /workspace/LightX2V/configs/neopp/neopp_dense.json \
  --host 0.0.0.0 \
  --port 8000 \
  --max_req_total_len 65536 \
  --mem_fraction 0.75 \
  --tp 2
```

## 5) 量化

`separate` 模式的另一个好处是，理解与生成可以各自采用独立的量化策略。

两条路径解耦后，可分别针对各自的质量与延迟目标进行调优：

1. **理解 FP16/BF16 + 生成 FP8**
   - 理解：不加量化参数，保持默认精度。
   - 生成：使用 FP8 生成配置，例如
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`
   - 推荐作为生产环境的默认量化方案。

2. **理解 FP8 + 生成 FP8**
   - 理解：添加 `--quant_type fp8w8a8`
   - 生成：使用 FP8 生成配置
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`
   - 适用于 GPU 显存或吞吐量吃紧的场景。

说明：
- `--quant_type fp8w8a8` 控制理解路径的量化精度。
- 生成侧的精度由 `--x2v_gen_model_config` 决定。

## 6) OpenAI 兼容 API

API 服务启动之后，可直接通过 LightLLM 暴露的 OpenAI 兼容端点发送请求。下面是一个最简的文生图示例：

```bash
python examples/serving/client.py \
  --mode t2i \
  --prompt "A cozy coffee shop storefront with infographic style."
```

更多模式（VQA、图像编辑、图文交错生成）及请求格式，详见 [`examples/serving/client.py`](../examples/serving/client.py)。
