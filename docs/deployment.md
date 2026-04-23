# LightLLM + LightX2V Deployment

This guide provides a practical deployment flow for serving SenseNova-U1 with
LightLLM + LightX2V using the Docker image
`lightx2v/lightllm_lightx2v:20260407`.

## 1) Pull and enter the Docker image

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
docker run --gpus all --ipc=host --network host -it lightx2v/lightllm_lightx2v:20260407 /bin/bash
```

## 2) Clone runtime dependencies inside the container

The image may not include the latest source trees. Clone both repositories and
pin LightLLM to the validated branch:

```bash
git clone https://github.com/ModelTC/LightX2V.git
git clone https://github.com/ModelTC/LightLLM.git
cd LightLLM
git checkout neo_plus_clean
```

## 3) X2I-related arguments

When enabling image generation in the same API server, use the following flags:

- `--enable_multimodal_x2i`  
  Enable image generation capability.
- `--x2i_server_used_gpus`  
  Number of GPUs reserved for the X2I generation server.
- `--x2i_server_deploy_mode {colocate,separate}`  
  - `colocate`: understanding and generation share the same visible GPU pool.
  - `separate`: understanding and generation are deployed as separate services, and
    can use different GPU sets.
- `--x2i_use_naive_impl`  
  Use the native/naive PyTorch backend for X2I (debugging/testing only, not for
  production throughput).

## 4) Deployment modes

### Mode A: `colocate` (single service, shared GPU pool)

Use this mode for quick validation and simpler operations. The LLM understanding
path (`--tp`) and X2I generation path (`--x2i_server_used_gpus`) consume resources
from the same visible GPUs.

Example (2 GPUs total):
- understanding path: `tp=2`
- generation path: `cfg=2` (configured in `neopp_dense_parallel_cfg.json`)

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

### Mode B: `separate` (understanding and generation decoupled)

`separate` is conceptually similar to PD-style decoupling in LLM serving: split
different stages onto different GPU groups so a long stage does not block the
short stage.

For multimodal serving, image generation is usually the long stage, while
understanding is short and lightweight. Separating them allows understanding
requests to keep flowing even when generation workers are busy.

Recommended deployment profiles:

1. **Default profile (continuity-first): Understanding `tp=1` + Generation 1 GPU**
   - Understanding: `--tp 1`
   - Generation: `--x2i_server_used_gpus 1`
   - Use as the baseline profile for mixed workloads. It keeps the pipeline simple
     while avoiding head-of-line blocking between understanding and generation.

2. **Understanding-expanded profile: Understanding `tp=2` + Generation 1 GPU**
   - Understanding: `--tp 2`
   - Generation: `--x2i_server_used_gpus 1`
   - Use when complex prompts or high understanding QPS become the bottleneck.

3. **Generation-expanded profile: Understanding `tp=1/2` + Generation parallel**
   - Understanding: `--tp 1` or `--tp 2`
   - Generation option A (2 GPUs): `--x2i_server_used_gpus 2` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg.json`
   - Generation option B (4 GPUs): `--x2i_server_used_gpus 4` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg_seq.json`
   - Use when generation latency/throughput dominates (most common scaling path).

Example launch (separate mode in API server):

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

## 5) Quantization

`separate` mode also enables independent quantization strategies for
understanding and generation.

Because understanding and generation are decoupled, you can tune quality/latency
for each path independently:

1. **Understanding FP16/BF16 + Generation FP8**
   - Understanding: no quantization flag (keep default precision)
   - Generation: use FP8 generation config, for example
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`
   - Recommended as the default quantized profile for production.

2. **Understanding FP8 + Generation FP8**
   - Understanding: add `--quant_type fp8w8a8`
   - Generation: use FP8 generation config
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`
   - Use when GPU memory/throughput is the primary constraint.

Notes:
- `--quant_type fp8w8a8` controls quantization on the understanding path.
- Generation-side precision is controlled by `--x2v_gen_model_config`.

## 6) OpenAI-compatible API

Once the API server is up, you can send requests through the OpenAI-compatible
endpoint exposed by LightLLM. A minimal text-to-image example:

```bash
python examples/serving/client.py \
  --mode t2i \
  --prompt "A cozy coffee shop storefront with infographic style."
```

See [`examples/serving/client.py`](../examples/serving/client.py) for more modes
(VQA, editing, interleaved) and request formats.