# U1.6 权重转换与临时验证

[English](README.md) | 简体中文

六套脚本分别用于 BF16 合并、GGUF Q4/Q6/Q8、FP8、INT8 ConvRot。
使用步骤：放好文件夹 → 填写路径 → 运行脚本。

## 1. 文件夹怎么放

在完整 SenseNova-U1 项目中保留下面的结构：

```text
tools/quantization/
├── README.md
├── README_CN.md
├── bf16/
├── gguf_q4_0/
├── gguf_q6_k/
├── gguf_q8_0/
├── fp8_w8a8/
├── int8_convrot_w8a8/
├── inference.py
└── quant_runtime.py
```

每个转换子文件夹可以单独复制使用。里面的 `.sh`、`.py` 文件要一起保留，
不依赖其它转换子文件夹。推理入口需要完整项目。

转换依赖：

```bash
python -m pip install torch safetensors numpy gguf comfy-kitchen
```

## 2. 哪些参数需要改

在项目根目录设置下面两个路径，然后运行转换命令：

```bash
export U1_SOURCE_DIR="/data/models/hf_checkpoint"
export U1_OUTPUT_DIR="/data/SenseNova-U1/outputs/quantization"
```

- `U1_SOURCE_DIR`：原始 HF/EMA 文件夹，包含全部分片、`model.safetensors.index.json`、配置和 Tokenizer。
- `U1_OUTPUT_DIR`：结果总目录，脚本会在下面创建 `sensenova` 及各权重子文件夹。

也可以直接修改各 `.sh` 中 `# set manually:` 下的默认路径。
脚本使用当前环境的 `python`，不指定虚拟环境位置。

FP8/INT8 还需要选择 GPU：

```bash
export CUDA_VISIBLE_DEVICES="0"
```

## 3. 怎么运行

在项目根目录按需执行：

```bash
bash tools/quantization/bf16/convert_bf16.sh
bash tools/quantization/gguf_q4_0/convert_gguf_q4_0.sh
bash tools/quantization/gguf_q6_k/convert_gguf_q6_k.sh
bash tools/quantization/gguf_q8_0/convert_gguf_q8_0.sh
bash tools/quantization/fp8_w8a8/convert_fp8_w8a8.sh
bash tools/quantization/int8_convrot_w8a8/convert_int8_convrot_w8a8.sh
```

脚本不会覆盖已有权重。重复转换请设置新的 `U1_OUTPUT_DIR`。
中途失败留下的 `.partial` 文件也会阻止覆盖；确认不再需要后再手动清理。

## 4. 结果在哪里

```text
U1_OUTPUT_DIR/sensenova/
├── SenseNova-U1.6-8B-MoT-ComfyUI/
│   ├── bf16/SenseNova-U1.6-BF16.safetensors
│   ├── fp8/SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors
│   └── int8-convrot/SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors
└── SenseNova-U1.6-8B-MoT-GGUF/
    ├── Q4_0/SenseNova-U1.6-8B-MoT-Q4_0.gguf
    ├── Q6_K/SenseNova-U1.6-8B-MoT-Q6_K.gguf
    └── Q8_0/SenseNova-U1.6-8B-MoT-Q8_0.gguf
```

每个权重旁都会复制 `config.json`、`tokenizer_config.json`、`vocab.json`、
`merges.txt`、`special_tokens_map.json`、`added_tokens.json`；输入目录需要提供这些文件。
`conversion.json` 记录格式、张量数和文件 SHA256，FP8/INT8 另有 `quantization_config.json`。


## 5. 如何验证推理

推理依赖：基础包
```bash
python -m pip install torch torchvision triton transformers huggingface-hub accelerate safetensors tokenizers sentencepiece numpy pillow einops diffusers gguf comfy-kitchen packaging tqdm httpx psutil PyYAML
```
推理依赖：FlashAttention 在 PyTorch 安装完成后单独安装
```bash
python -m pip install flash-attn --no-build-isolation
```


苹果图示例（项目根目录执行，沿用上面设置的 `U1_OUTPUT_DIR`）：

```bash
python tools/quantization/inference.py \
  --model_path "$U1_OUTPUT_DIR/sensenova/SenseNova-U1.6-8B-MoT-ComfyUI/fp8/SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors" \
  --format fp8 \
  --prompt "A studio photograph of a red apple on a white table, soft daylight." \
  --width 1024 --height 1024 \
  --num_steps 50 --seed 42 \
  --cfg_scale 4 --timestep_shift 3 \
  --warmup 1 --repeats 3 \
  --output_dir "$U1_OUTPUT_DIR/validation/fp8"
```

验证 INT8 时换成对应权重文件，设置 `--format int8-convrot`，并换一个输出目录。BF16 使用 `--format bf16`，三个 GGUF 都使用 `--format gguf`；`--model_path` 均填写具体权重文件。

推理固定使用 FlashAttention、BF16 非量化层和全模型驻留 GPU。图片及耗时、显存记录保存在 `--output_dir`。
