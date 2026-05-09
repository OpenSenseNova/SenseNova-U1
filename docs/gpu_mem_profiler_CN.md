# 显存性能分析

本文档记录了 SenseNova-U1-8B-MoT 模型在不同推理任务下的显存占用与性能基准数据。所有测试均通过 `--profile` 参数启用，运行环境为单张 NVIDIA H100 80G GPU。

---

## 文生图

标准文生图推理，不启用思维链。

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

## 文生图（思维链）

启用思维链推理（`--think`），模型在生成图像前先输出推理过程，生成耗时和显存略有增加。

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

## 图像编辑

图像编辑任务需同时输入原图与编辑指令，因额外处理输入图像，生成峰值显存高于纯文生图。

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

## 图文交错生成

交错生成任务会在一次推理中产生多张图像与对应文字，单图 token 数较少但整体显存和耗时显著更高。

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

## 各任务显存对比

| 任务            | 加载峰值显存 (GiB) | 生成峰值显存 (GiB) | CPU 内存 (GiB) | 平均耗时 (s) | 吞吐量 (tok/s) |
|----------------|:-----------------:|:-----------------:|:-------------:|:-----------:|:-------------:|
| t2i            | 32.77 / 33.10     | 34.83 / 35.82     | 5.59          | 22.108      | 185.27        |
| t2i-think      | 32.77 / 33.10     | 35.02 / 35.94     | 5.58          | 38.342      | 106.83        |
| editing        | 32.77 / 33.10     | 39.50 / 41.32     | 5.61          | 25.871      | 155.74        |
| interleave     | 32.77 / 33.10     | 49.22 / 69.18     | 5.63          | 49.353      |  46.68        |

> 显存列格式为 `allocated / reserved`；CPU 内存为生成阶段 RSS 峰值。

## 低显存推理（以文生图为例）

### 显存上限约束（`--max_memory`）

通过 `--max_memory` 参数限制 GPU 可用显存上限，模拟不同显存规格的消费级 GPU，涵盖 32 GB（如 RTX 5090）、24 GB（如 RTX 4090）、16 GB（如 RTX 4080）、12 GB（如 RTX 4070）及 8 GB（如 RTX 4060）等典型配置。模型超出显存上限的部分将自动卸载至 CPU 内存，因此 CPU RSS 会随 GPU 预算降低而显著升高。

> 建议将 `max_memory` 设置为略低于 GPU 物理显存的值（例如 32 GB 显卡可设为 `26GiB`–`28GiB`），以预留足够的显存余量，避免推理过程中出现 OOM。

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

| GPU 预算   | 目标显卡           | 加载峰值显存 (GiB) | 生成峰值显存 (GiB) | 加载 CPU RSS (GiB) | 生成 CPU RSS (GiB) | 平均耗时 (s) | 吞吐量 (tok/s) |
|:----------:|:------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------:|:-------------:|
| 27 GiB     | RTX 5090 (32 GB)   | 25.71 / 25.71     | 27.76 / 28.31     | 5.62              | 10.27             | 87.692      | 46.71         |
| 20 GiB     | RTX 4090 (24 GB)   | 18.52 / 18.52     | 20.58 / 21.12     | 5.59              | 19.50             | 174.961     | 23.41         |
| 13 GiB     | RTX 4080 (16 GB)   | 11.33 / 11.34     | 13.39 / 13.93     | 5.62              | 24.12             | 250.757     | 16.33         |
| 9 GiB      | RTX 4070 (12 GB)   | 7.74 / 7.74       | 9.79 / 10.33      | 5.55              | 28.76             | 290.039     | 14.12         |
| 7 GiB      | RTX 4060 (8 GB)    | 5.58 / 5.59       | 7.64 / 8.18       | 5.56              | 28.76             | 316.323     | 12.95         |

> 显存列格式为 `allocated / reserved`；随 GPU 预算降低，模型层逐步卸载至 CPU，生成 CPU RSS 相应升高，推理吞吐量下降。

### 显存优化模式（`--vram_mode`）

通过 `--vram_mode` 参数切换显存优化策略，在推理速度与显存占用之间进行权衡。

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

| `--vram_mode` | 策略说明                                         | 加载峰值显存 (GiB) | 生成峰值显存 (GiB) | 加载 CPU RSS (GiB) | 生成 CPU RSS (GiB) | 平均耗时 (s) | 吞吐量 (tok/s) |
|:-------------:|:------------------------------------------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------:|:-------------:|
| `full`        | 整模型常驻 GPU，不做 offload（默认，速度最快）     | 32.77 / 33.10     | 34.83 / 35.82     | 5.59              | 5.59              | 22.108      | 185.27        |
| `balanced`    | 异步预取（H2D 与计算重叠），显存占用大幅降低       | —                 | 6.78 / 12.50      | 0.95              | 47.18             | 112.394     | 36.44         |
| `low`         | 每层同步 CPU↔GPU 交换，GPU 显存占用最小，速度最慢 | —                 | 5.34 / 5.85       | 0.98              | 47.22             | 130.191     | 31.46         |

> 显存列格式为 `allocated / reserved`；`balanced` 和 `low` 模式采用懒加载，模型权重加载阶段无 GPU 显存分配（以 — 表示），运行时按需换入，因此生成阶段 CPU RSS 显著升高。