# GPU Memory Profiling

This document records VRAM usage and performance benchmarks for the SenseNova-U1-8B-MoT model across different inference tasks. All tests are run with the `--profile` flag on a single NVIDIA H100 80G GPU.

---

## Text-to-Image

Standard text-to-image inference without chain-of-thought.

```bash
python examples/t2i/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --output_dir outputs/ \
    --cfg_scale 4.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --profile
```

```
================================================================
Profile summary
================================================================
  config              : vram_mode=full, attn_backend=flash, dtype=bfloat16
  model load          :   88.857 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB, cpu RSS 5.59 GiB
  generations         : 1 call(s), 1 image(s) total, 22.108 s wall
  avg per image       :   22.108 s
  image tokens        : patch_size=32, avg 4096 tok/image (4096)
  throughput          :   185.27 tok/s
  generation peak mem : allocated 34.83 GiB, reserved 35.82 GiB, cpu RSS 5.59 GiB
================================================================
```

---

## Text-to-Image with Chain-of-Thought

Enables chain-of-thought reasoning (`--think`), where the model outputs its reasoning process before generating the image. Generation time and VRAM usage increase slightly.

```bash
python examples/t2i/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --output_dir outputs/ \
    --cfg_scale 4.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --profile \
    --think \
    --print_think
```

```
================================================================
Profile summary
================================================================
  config              : vram_mode=full, attn_backend=flash, dtype=bfloat16
  model load          :   82.060 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB, cpu RSS 5.58 GiB
  generations         : 1 call(s), 1 image(s) total, 38.342 s wall
  avg per image       :   38.342 s
  image tokens        : patch_size=32, avg 4096 tok/image (4096)
  throughput          :   106.83 tok/s
  generation peak mem : allocated 35.02 GiB, reserved 35.94 GiB, cpu RSS 5.58 GiB
================================================================
```

---

## Image Editing

Image editing requires both an input image and an editing instruction. Processing the additional input image results in higher peak VRAM usage compared to plain text-to-image.

```bash
python examples/editing/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --prompt "Change the man's coat to yellow." \
    --image examples/editing/data/images/1.webp \
    --cfg_scale 4.0 \
    --img_cfg_scale 1.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --output output_edited.png \
    --profile \
    --compare
```

```
================================================================
Profile summary
================================================================
  config              : vram_mode=full, attn_backend=flash, dtype=bfloat16
  model load          :   80.541 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB, cpu RSS 5.61 GiB
  generations         : 1 call(s), 1 image(s) total, 25.871 s wall
  avg per image       :   25.871 s
  image tokens        : patch_size=32, avg 4029 tok/image (4029)
  throughput          :   155.74 tok/s
  generation peak mem : allocated 39.50 GiB, reserved 41.32 GiB, cpu RSS 5.61 GiB
================================================================
```

---

## Interleaved Text-Image Generation

Interleaved generation produces multiple images and corresponding text in a single inference call. Per-image token count is lower, but overall VRAM usage and wall time are substantially higher.

```bash
python examples/interleave/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT/ \
    --prompt "I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial." \
    --resolution "16:9" \
    --output_dir outputs/interleave/ \
    --stem demo \
    --profile
```

```
================================================================
Profile summary
================================================================
  config              : vram_mode=full, attn_backend=flash, dtype=bfloat16
  model load          :   74.821 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB, cpu RSS 5.63 GiB
  generations         : 1 call(s), 6 image(s) total, 296.118 s wall
  avg per image       :   49.353 s
  image tokens        : patch_size=32, avg 2304 tok/image (2304)
  throughput          :    46.68 tok/s
  generation peak mem : allocated 49.22 GiB, reserved 69.18 GiB, cpu RSS 5.63 GiB
================================================================
```

---

## Task Comparison Summary

| Task        | Load Peak VRAM (GiB) | Gen Peak VRAM (GiB) | CPU RSS (GiB) | Avg Time (s) | Throughput (tok/s) |
|-------------|:--------------------:|:-------------------:|:-------------:|:------------:|:------------------:|
| t2i         | 32.77 / 33.10        | 34.83 / 35.82       | 5.59          | 22.108       | 185.27             |
| t2i-think   | 32.77 / 33.10        | 35.02 / 35.94       | 5.58          | 38.342       | 106.83             |
| editing     | 32.77 / 33.10        | 39.50 / 41.32       | 5.61          | 25.871       | 155.74             |
| interleave  | 32.77 / 33.10        | 49.22 / 69.18       | 5.63          | 49.353       |  46.68             |

> VRAM columns are formatted as `allocated / reserved`. CPU RSS is the peak RSS during the generation phase.

## Low-VRAM Inference (Text-to-Image as Example)

### VRAM Budget Cap (`--max_memory`)

The `--max_memory` parameter caps the GPU VRAM budget to simulate consumer-grade GPUs with varying VRAM capacities, covering 32 GB (e.g. RTX 5090), 24 GB (e.g. RTX 4090), 16 GB (e.g. RTX 4080), 12 GB (e.g. RTX 4070), and 8 GB (e.g. RTX 4060). Model layers exceeding the VRAM budget are automatically offloaded to CPU memory, so CPU RSS rises significantly as the GPU budget decreases.

> It is recommended to set `max_memory` slightly below the GPU's physical VRAM (e.g. use `26GiB`–`28GiB` for a 32 GB card) to leave enough headroom and avoid OOM errors during inference.

```bash
python examples/t2i/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --prompt "..." \
    --output_dir outputs/ \
    --cfg_scale 4.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --device_map auto \
    --max_memory "0=<N>GiB,cpu=80GiB" \
    --profile
```

| GPU Budget | Target GPU         | Load Peak VRAM (GiB) | Gen Peak VRAM (GiB) | Load CPU RSS (GiB) | Gen CPU RSS (GiB) | Avg Time (s) | Throughput (tok/s) |
|:----------:|:------------------:|:--------------------:|:-------------------:|:-----------------:|:-----------------:|:------------:|:------------------:|
| 27 GiB     | RTX 5090 (32 GB)   | 25.71 / 25.71        | 27.76 / 28.31       | 5.62              | 10.27             | 87.692       | 46.71              |
| 20 GiB     | RTX 4090 (24 GB)   | 18.52 / 18.52        | 20.58 / 21.12       | 5.59              | 19.50             | 174.961      | 23.41              |
| 13 GiB     | RTX 4080 (16 GB)   | 11.33 / 11.34        | 13.39 / 13.93       | 5.62              | 24.12             | 250.757      | 16.33              |
| 9 GiB      | RTX 4070 (12 GB)   | 7.74 / 7.74          | 9.79 / 10.33        | 5.55              | 28.76             | 290.039      | 14.12              |
| 7 GiB      | RTX 4060 (8 GB)    | 5.58 / 5.59          | 7.64 / 8.18         | 5.56              | 28.76             | 316.323      | 12.95              |

> VRAM columns are formatted as `allocated / reserved`. As the GPU budget decreases, model layers are progressively offloaded to CPU, causing CPU RSS to rise and inference throughput to drop.

### VRAM Optimization Mode (`--vram_mode`)

The `--vram_mode` parameter selects the VRAM optimization strategy, trading off inference speed against VRAM footprint.

```bash
python examples/t2i/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --prompt "..." \
    --output_dir outputs/ \
    --cfg_scale 4.0 \
    --cfg_norm none \
    --timestep_shift 3.0 \
    --num_steps 50 \
    --vram_mode <full|balanced|low> \
    --profile
```

| `--vram_mode` | Strategy                                                              | Load Peak VRAM (GiB) | Gen Peak VRAM (GiB) | Load CPU RSS (GiB) | Gen CPU RSS (GiB) | Avg Time (s) | Throughput (tok/s) |
|:-------------:|:----------------------------------------------------------------------|:--------------------:|:-------------------:|:-----------------:|:-----------------:|:------------:|:------------------:|
| `full`        | Entire model resident on GPU, no offload (default, fastest)           | 32.77 / 33.10        | 34.83 / 35.82       | 5.59              | 5.59              | 22.108       | 185.27             |
| `balanced`    | Async prefetch (H2D overlapped with compute), greatly reduced VRAM    | —                    | 6.78 / 12.50        | 0.95              | 47.18             | 112.394      | 36.44              |
| `low`         | Synchronous CPU↔GPU swap per layer, minimum GPU VRAM, slowest        | —                    | 5.34 / 5.85         | 0.98              | 47.22             | 130.191      | 31.46              |

> VRAM columns are formatted as `allocated / reserved`. `balanced` and `low` modes use lazy loading — no GPU VRAM is allocated during model load (shown as —); weights are swapped in on demand, causing CPU RSS to rise significantly during generation.
