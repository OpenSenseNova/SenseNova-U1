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
  model load          :  114.100 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB
  generations         : 1 call(s), 1 image(s) total, 22.981 s wall
  avg per image       :   22.981 s
  image tokens        : patch_size=32, avg 4096 tok/image (4096)
  throughput          :   178.23 tok/s
  generation peak mem : allocated 34.83 GiB, reserved 35.76 GiB
================================================================
```

---

## 文生图（思维链）

启用思维链推理（`--think`），模型在生成图像前先输出推理过程，生成耗时和显存略有增加。

```bash
python examples/t2i/inference.py \
    --model_path checkpoints/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
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
  model load          :  111.171 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB
  generations         : 1 call(s), 1 image(s) total, 39.608 s wall
  avg per image       :   39.608 s
  image tokens        : patch_size=32, avg 4080 tok/image (4080)
  throughput          :   103.01 tok/s
  generation peak mem : allocated 35.01 GiB, reserved 35.91 GiB
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
  model load          :  105.127 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB
  generations         : 1 call(s), 1 image(s) total, 27.192 s wall
  avg per image       :   27.192 s
  image tokens        : patch_size=32, avg 4029 tok/image (4029)
  throughput          :   148.17 tok/s
  generation peak mem : allocated 39.50 GiB, reserved 41.32 GiB
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
  model load          :  103.491 s
  load peak memory    : allocated 32.77 GiB, reserved 33.10 GiB
  generations         : 1 call(s), 1 image(s) total, 312.974 s wall
  avg per image       :  312.974 s
  image tokens        : patch_size=32, avg 2304 tok/image (2304)
  throughput          :     7.36 tok/s
  generation peak mem : allocated 49.35 GiB, reserved 68.63 GiB
================================================================
```

---

## 各任务显存对比

| 任务            | 加载峰值显存 (GiB) | 生成峰值显存 (GiB) | 平均耗时 (s) | 吞吐量 (tok/s) |
|----------------|:-----------------:|:-----------------:|:-----------:|:-------------:|
| t2i            | 32.77 / 33.10     | 34.83 / 35.76     | 22.981      | 178.23        |
| t2i-think      | 32.77 / 33.10     | 35.01 / 35.91     | 39.608      | 103.01        |
| editing        | 32.77 / 33.10     | 39.50 / 41.32     | 27.192      | 148.17        |
| interleave     | 32.77 / 33.10     | 49.35 / 68.63     | 312.974     |   7.36        |

> 显存列格式为 `allocated / reserved`。
