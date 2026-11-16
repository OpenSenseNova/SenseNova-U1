# U1 deployment with LightLLM + LightX2V

This guide provides a practical deployment flow for serving SenseNova-U1 with
LightLLM + LightX2V using the Docker image
`lightx2v/lightllm_lightx2v:20260407`.

## 1) Pull and enter the Docker image

```bash
docker pull lightx2v/lightllm_lightx2v:20260407
docker run --gpus all --ipc=host --network host -it lightx2v/lightllm_lightx2v:20260407 /bin/bash
```

## 2) Clone runtime dependencies inside the container

The image may not include the source trees used below. Clone both repositories
and select the LightLLM branch used by this example:

```bash
git clone https://github.com/ModelTC/LightX2V.git
git clone https://github.com/ModelTC/LightLLM.git
cd LightLLM
git checkout neo_plus_clean
```

These commands use moving upstream branches. Record the commit hashes of both
checkouts for your deployment; the Docker image tag alone does not fix the
versions of source code cloned inside it.

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

`separate` places understanding and generation processes on different GPU groups. The following resource configurations are examples; choose one based on measurements of your workload.

1. **Understanding `tp=1`; generation 1 GPU**
   - Understanding: `--tp 1`
   - Generation: `--x2i_server_used_gpus 1`

2. **Understanding `tp=2`; generation 1 GPU**
   - Understanding: `--tp 2`
   - Generation: `--x2i_server_used_gpus 1`

3. **Understanding `tp=1/2`; generation 2 or 4 GPUs**
   - Understanding: `--tp 1` or `--tp 2`
   - Generation option A (2 GPUs): `--x2i_server_used_gpus 2` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg.json`
   - Generation option B (4 GPUs): `--x2i_server_used_gpus 4` +
     `/workspace/LightX2V/configs/neopp/neopp_dense_parallel_cfg_seq.json`

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

In `separate` mode, you can set precision independently for understanding and generation:

1. **Understanding FP16/BF16 + Generation FP8**
   - Understanding: no quantization flag (keep default precision)
   - Generation: use FP8 generation config, for example
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`

2. **Understanding FP8 + Generation FP8**
   - Understanding: add `--quant_type fp8w8a8`
   - Generation: use FP8 generation config
     `/workspace/LightX2V/configs/neopp/neopp_dense_fp8.json`

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
