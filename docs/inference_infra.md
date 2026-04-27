# Inference Infrastructure

This document describes the inference infrastructure behind **SenseNova-U1**, built on top of **[LightLLM](https://github.com/ModelTC/lightllm)** and **[LightX2V](https://github.com/ModelTC/lightx2v)**.

## Overview

SenseNova-U1 is exposed as one unified multimodal model, but the understanding and generation paths exhibit different execution shapes in production. They tend to prefer different scheduling policies, parallelization strategies, and resource ratios, rather than a single shared serving configuration. When both are coupled inside one monolithic runtime, these choices become unnecessarily tied together, which can leave both paths operating away from their respective optimal points.

To avoid this coupling, SenseNova-U1 adopts a **disaggregated** architecture:

- **LightLLM** for understanding, text streaming, and control flow
- **LightX2V** for image generation

These two engines exchange generation state through pinned shared memory and high-performance transfer kernels. The handoff is lightweight, while each side can still run with its own optimal execution policy.

![LightLLM + LightX2V decoupled architecture](./assets/lightllm_x2v.png)

This design provides practical benefits in production:

- Independent parallelism (for example, understanding with `TP=2` (Tensor Parallel=2), generation
  with `CFG=2` (CFG Parallel=2) or `SP=2` (Sequence Parallel=2)).
- Independent resource allocation (different GPU counts and memory budgets).
- Independent scaling for text-heavy vs. image-heavy traffic.
- Better operational isolation and simpler performance tuning.

The same architecture can be deployed in two modes, depending on your hardware budget and traffic pattern:

- **Separate**: LightLLM and LightX2V run on different GPU groups.
- **Colocate**: LightLLM and LightX2V run as separate processes on the same GPU.

In most production setups, `Separate` is the default choice because it gives clearer bottleneck control and independent scaling. `Colocate` is useful for quick validation, generation-heave scenes, or smaller GPU setups.

### Attention for Multimodal Prefill of NEO-Unify

NEO-Unify's prefill attention is not standard causal attention. Text tokens remain causal, while image tokens attend to the full text prefix together with the entire image span. To support this hybrid masking pattern, we modified both attention implementations in our stack: the Triton kernel and the official FlashAttention3 (FA3) codebase. Our FA3 branch is available at [WANDY666/flash-attention](https://github.com/WANDY666/flash-attention).

Concretely, we introduced an optional image_token_tag argument that adjusts the mask row by row. Text rows keep the standard causal mask. Image rows, instead of using plain causal truncation, are allowed to attend to all preceding text tokens and all image tokens within the image span.

To preserve the causal-triangle speedup whenever possible, the kernel makes the decision per M-block. It OR-reduces the image_token_tag values inside the current block: if the block contains no image token, it keeps the standard causal K-range; if the block contains image tokens, it extends the K-range to cover the required image span. As a result, pure-text blocks still follow the normal causal path, while only the relevant blocks pay the extra work needed by the hybrid mask.

![NEO-Unify multimodal attention behavior](./assets/attn.png)

The overhead therefore does not depend on a fixed ratio, but on how image tokens are distributed across the sequence and across M-block boundaries. When image rows are concentrated in only part of the sequence, the extra work is correspondingly localized. For text-only requests, image_token_tag is empty, and the kernel falls back to vanilla FA3 with no additional overhead.
The benchmark below compares two implementations for Neo-style multimodal prefill:

- **Triton implementation**: easier to migrate into existing codebases, with lower
  integration cost and faster iteration.
- **FA3 implementation**: higher absolute performance on supported hardware.

<div align="center">

|  batch  | max_seq_len | image_token_num | triton (ms) | FA3 (ms) | speedup (×) |
|:-------:|:-----------:|:---------------:|:-----------:|:--------:|:-----------:|
|    8    |     4096    |       88        |    1.95     |   0.81   |  **2.41×**  |
|    8    |     8192    |       171       |    6.55     |   2.68   |  **2.45×**  |
|    8    |    65536    |       150       |   43.30     |  14.95   |  **2.90×**  |
|   16    |     4096    |       379       |    4.12     |   1.68   |  **2.46×**  |
|   16    |     8192    |       246       |   17.76     |   7.40   |  **2.40×**  |
|   16    |    65536    |       206       |  107.74     |  33.66   |  **3.20×**  |
|   32    |     4096    |       726       |    8.46     |   3.46   |  **2.44×**  |
|   32    |     8192    |       536       |   31.74     |  13.24   |  **2.40×**  |
|   32    |    65536    |       417       |  171.00     |  58.26   |  **2.94×**  |
|   64    |     4096    |       1170      |   16.08     |   6.88   |  **2.34×**  |
|   64    |     8192    |       1177      |   55.48     |  22.91   |  **2.42×**  |
|   64    |    65536    |       1291      |  348.89     | 124.82   |  **2.80×**  |
|  128    |     4096    |       2057      |   30.89     |  12.53   |  **2.47×**  |
|  128    |     8192    |       2196      |  104.73     |  43.22   |  **2.42×**  |
|  128    |    65536    |       2205      |  706.60     | 241.67   |  **2.92×**  |

</div>


### Deployment

For a concise deployment runbook (Docker image, startup command, and API tests),
see [`deployment.md`](./deployment.md).


### Generation Performance

The table below reports **2048x2048** image generation latency for
**SenseNova-U1-8B-MoT(NEO-Unify)**. Fill in measured numbers for each machine and deployment profile.
Note: TP2+CFG2 means Tensor Parallel=2 + CFG Parallel=2.

<div align="center">

| GPU  | Deployment Config | Per-step Latency (s/step) | End-to-end Latency (s) |
|:----:|:-----------------:|:-------------------------:|:----------------------:|
| H100 | TP2+CFG2 / colocate | 0.158 | 9.23 |
| H200 | TP2+CFG2 / colocate | 0.152 | 9.54 |
| 5090 | TP2+CFG2 / separate | 0.415 | 23.04 |
| L40S | TP2+CFG2 / separate | 0.443 | 25.62 |

</div>

In NEO-Unify, the KV cache for the generation stage is provided by the understanding module, so T2I (generation) and I2I (editing) have very similar runtime characteristics. For brevity, we report only T2I latency here.


### Cross-Model Speed Comparison

The table below compares the latency of a single diffusion step for
**2048x2048** image generation with **CFG enabled**. Unless otherwise noted,
all measurements are taken on **H100**; the `TP2+CFG2` result uses
`2x H100`.
Note: TP2+CFG2 means Tensor Parallel=2 + CFG Parallel=2.

<div align="center">

|       Model       | Understanding | Generation | Per-step latency (s/step) |
|:-----------------:|:-------------:|:----------:|:-------------------------:|
| Qwen-Image-2512   |      7B       |     20B    |           1.478           |
| Z-Image           |      4B       |     6B     |           1.110           |
| GLM-Image         |      9B       |     7B     |           1.394           |
| ERNIE-Image       |      8B       |     8B     |           1.565           |
| LongCat-Image     |      8B       |     6B     |           0.796           |
| SenseNova-U1-8B-MoT (Neo-Unify) | 8B | 8B | 0.312 |
| SenseNova-U1-8B-MoT (Neo-Unify, TP2+CFG2) | 8B | 8B | 0.158 |

</div>
