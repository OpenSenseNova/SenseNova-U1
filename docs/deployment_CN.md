# U1 的 LightLLM + LightX2V 部署

本文档介绍如何基于 Docker 镜像 `lightx2v/lightllm_lightx2v:20260407`，通过 LightLLM + LightX2V 部署 SenseNova-U1 推理服务。

## 1) 拉取并进入 Docker 镜像

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
docker run --gpus all --ipc=host --network host -it lightx2v/lightllm_lightx2v:20260407 /bin/bash
```

## 2) 在容器内克隆运行时依赖

镜像中未必带有下方命令所用的源码。克隆两个仓库，并将 LightLLM 切到本示例使用的分支：

```bash
git clone https://github.com/ModelTC/LightX2V.git
git clone https://github.com/ModelTC/LightLLM.git
cd LightLLM
git checkout neo_plus_clean
```

上述命令使用的是上游可移动分支。部署时请记录两个仓库实际检出的 commit；只固定 Docker 镜像标签，无法固定容器内另行克隆的源码版本。

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

`separate` 将理解与生成进程放在不同的 GPU 组上。下面是三种资源配置；具体选择应按实际请求负载测量。

1. **理解 `tp=1`；生成 1 GPU**
   - 理解：`--tp 1`
   - 生成：`--x2i_server_used_gpus 1`

2. **理解 `tp=2`；生成 1 GPU**
   - 理解：`--tp 2`
   - 生成：`--x2i_server_used_gpus 1`

3. **理解 `tp=1/2`；生成 2 或 4 GPU**
   - 理解：`--tp 1` 或 `--tp 2`
   - 生成方案 A（2 GPU）：`--x2i_server_used_gpus 2` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg.json`
   - 生成方案 B（4 GPU）：`--x2i_server_used_gpus 4` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg_seq.json`

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

`separate` 模式下，理解侧和生成侧可以分别选择量化精度：

1. **理解 FP16/BF16 + 生成 FP8**
   - 理解：不加量化参数，保持默认精度。
   - 生成：使用 FP8 生成配置，例如
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`

2. **理解 FP8 + 生成 FP8**
   - 理解：添加 `--quant_type fp8w8a8`
   - 生成：使用 FP8 生成配置
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`

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
