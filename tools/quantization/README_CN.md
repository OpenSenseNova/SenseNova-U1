# U1.6 权重转换与临时验证

[English](README.md) | 简体中文

原生转换脚本用于 BF16 合并、GGUF Q4/Q6/Q8、FP8 和 INT8 ConvRot。
ComfyUI checkpoint 转换由官方 ComfyUI 集成维护，不属于本仓库的发布流程。
使用步骤：准备原始模型 → 填写路径 → 运行目标格式脚本。

推理只保留一条主路径：`examples/t2i/inference.py`、
`examples/editing/inference.py` 和 `examples/interleave/inference.py` 都使用
`src/sensenova_u1/utils/checkpoint_loading.py` 中的共享加载器；加载原生量化权重时
使用默认的 `--model_format auto` 或显式指定格式。`tools/quantization/inference.py`
只是复用同一个 T2I 引擎的量化冒烟验证封装，不再维护第二套模型运行时。

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
├── conversion_utils.py
└── inference.py
```

转换脚本依赖共享的 `conversion_utils.py`，请在完整项目中运行。
推理入口同样需要完整项目。

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

原生 INT8 转换器写入 `src/sensenova_u1/utils/quant_runtime.py` 使用的
`sensenova-w8a8-v2` 文件级 metadata。ComfyUI 转换有独立的发布流程和 metadata 格式。

脚本不会覆盖已有权重。重复转换请设置新的 `U1_OUTPUT_DIR`。
中途失败留下的 `.partial` 文件也会阻止覆盖；确认不再需要后再手动清理。

## 4. 结果在哪里

```text
U1_OUTPUT_DIR/sensenova/
├── SenseNova-U1.6-8B-MoT-Native/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   ├── SenseNova-U1.6-BF16.safetensors
│   ├── SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors
│   ├── SenseNova-U1.6-INT8-ConvRot-W8A8.safetensors
│   ├── SenseNova-U1.6-BF16.conversion.json
│   ├── SenseNova-U1.6-FP8-E4M3FN-W8A8.conversion.json
│   ├── SenseNova-U1.6-INT8-ConvRot-W8A8.conversion.json
│   ├── SenseNova-U1.6-FP8-E4M3FN-W8A8.quantization_config.json
│   ├── SenseNova-U1.6-INT8-ConvRot-W8A8.quantization_config.json
│   ├── manifest.json
│   └── ...
└── SenseNova-U1.6-8B-MoT-GGUF/
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── special_tokens_map.json
    ├── added_tokens.json
    ├── SenseNova-U1.6-8B-MoT-Q4_0.gguf
    ├── SenseNova-U1.6-8B-MoT-Q6_K.gguf
    ├── SenseNova-U1.6-8B-MoT-Q8_0.gguf
    ├── SenseNova-U1.6-8B-MoT-Q4_0.conversion.json
    ├── SenseNova-U1.6-8B-MoT-Q6_K.conversion.json
    ├── SenseNova-U1.6-8B-MoT-Q8_0.conversion.json
    ├── manifest.json
    └── ...
```

六个配置和 tokenizer 文件会复制到各格式目录中，用户可以直接下载完整的格式目录。
生成的 `manifest.json` 列出该目录中的所有产物。若输出路径符合上面的结构，
`U1_OUTPUT_DIR/sensenova/manifest.json` 还会列出 Native 和 GGUF 下的全部产物。

输入目录必须提供 `config.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`、
`special_tokens_map.json` 和 `added_tokens.json`，脚本会把它们复制到各格式目录。

每个产物都会生成独立的 `<artifact>.conversion.json`，记录格式、张量数量、源模型配置
哈希和文件 SHA256。FP8 和 INT8 还会生成独立的
`<artifact>.quantization_config.json`，不同模型的报告不会互相覆盖。


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
  --model_path "$U1_OUTPUT_DIR/sensenova/SenseNova-U1.6-8B-MoT-Native/SenseNova-U1.6-FP8-E4M3FN-W8A8.safetensors" \
  --format fp8 \
  --prompt "A studio photograph of a red apple on a white table, soft daylight." \
  --width 1024 --height 1024 \
  --num_steps 50 --seed 42 \
  --cfg_scale 4 --timestep_shift 3 \
  --warmup 1 --repeats 3 \
  --output_dir "$U1_OUTPUT_DIR/validation/fp8"
```

验证 INT8 时换成对应权重文件，设置 `--format int8-convrot`，并换一个输出目录。BF16 使用 `--format bf16`，三个 GGUF 都使用 `--format gguf`；`--model_path` 均填写具体权重文件，并保留同目录下的六个资源文件。

推理固定使用 FlashAttention、BF16 非量化层和全模型驻留 GPU。图片及耗时、显存记录保存在 `--output_dir`。

## 6. 发布已验证产物到 Hugging Face

发布和转换分开执行。发布脚本会先校验 `manifest.json`、产物级转换报告、量化配置以及六个配置和 tokenizer 文件，再上传到目标模型仓库。脚本从环境变量读取 token，不接受命令行中的明文 token。

在执行命令的 shell 中设置 token：

```bash
export HF_TOKEN="<huggingface-user-access-token>"
```

将完整的 `sensenova` 输出目录发布到模型仓库中的 `quantized` 目录：

```bash
python tools/quantization/publish_hf.py \
  --source-dir "$U1_OUTPUT_DIR/sensenova" \
  --repo-id sensenova/SenseNova-U1.6-8B-MoT \
  --path-in-repo quantized \
  --purpose "Publish U1.6 RC1 Native and GGUF artifacts" \
  --tag rc1 \
  --commit-message "release: publish U1.6 RC1 quantized artifacts"
```

使用 `--dry-run` 可以只校验目录并打印发布计划，不连接 Hugging Face。仅在明确要把已有 tag 移动到新 commit 时使用 `--replace-tag`。正式发布完成后，可以把同一个不可变 revision 标记为 `v1.0.0`。

## 7. 发布 LoRA 适配器到 Hugging Face

独立 LoRA 文件使用专用发布脚本。脚本会校验成对的
`.lora_down.weight` 和 `.lora_up.weight`、有限浮点值、rank 形状、兼容基座模型以及适配器 SHA256。未提供 README 时，脚本会生成 `lora_manifest.json` 和最小模型卡。

```bash
export HF_TOKEN="<huggingface-user-access-token>"

python tools/quantization/publish_lora_hf.py \
  --lora-path ./outputs/SenseNova-U1.6-example.safetensors \
  --repo-id sensenova/SenseNova-U1.6-example-LoRA \
  --base-model sensenova/SenseNova-U1.6-8B-MoT \
  --purpose "Publish U1.6 RC1 LoRA adapter" \
  --source-revision rc1 \
  --tag rc1 \
  --commit-message "release: publish U1.6 RC1 LoRA adapter"
```

适配器有经过审核的模型卡时使用 `--readme`。基座模型和对应 revision 会记录在 `lora_manifest.json` 中，作为适配器发布契约的一部分。
