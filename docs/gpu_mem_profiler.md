# GPU Memory Profiler

This document records the GPU memory usage and performance benchmarks of the SenseNova-U1-8B-MoT model across different inference tasks. All tests are enabled via the `--profile` flag and were run on a single NVIDIA H100 80G GPU.

---

## Text-to-Image (t2i)

Standard text-to-image inference without chain-of-thought reasoning.

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

## Text-to-Image with Thinking (t2i-think)

Enables chain-of-thought reasoning via `--think`. The model outputs its reasoning process before generating the image, resulting in slightly higher latency and memory usage.

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

## Image Editing

Image editing requires both a reference image and an instruction as input. Processing the input image incurs additional memory overhead, resulting in a higher generation peak memory than pure text-to-image.

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

## Interleaved Text-Image Generation (interleave)

Interleaved generation produces multiple images interleaved with text in a single inference pass. Per-image token count is lower, but overall memory usage and latency are significantly higher.

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

## Memory Comparison

| Task        | Load Peak Memory (GiB) | Generation Peak Memory (GiB) | Avg Time (s) | Throughput (tok/s) |
|-------------|:----------------------:|:----------------------------:|:------------:|:------------------:|
| t2i         | 32.77 / 33.10          | 34.83 / 35.76                | 22.981       | 178.23             |
| t2i-think   | 32.77 / 33.10          | 35.01 / 35.91                | 39.608       | 103.01             |
| editing     | 32.77 / 33.10          | 39.50 / 41.32                | 27.192       | 148.17             |
| interleave  | 32.77 / 33.10          | 49.35 / 68.63                | 312.974      |   7.36             |

> Memory columns are formatted as `allocated / reserved`.
