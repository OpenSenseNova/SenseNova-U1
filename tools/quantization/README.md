# U1.6 权重转换与临时验证

六套转换工具直接读取原始 HF/EMA 分片，互不依赖。BF16 只合并已有的
BF16 分片，FP8、INT8 不需要先运行 BF16 转换。

当前版本用于 U1.6 8B MoT 的临时验证，按 1116 个原始张量检查覆盖范围。
FP8/INT8 量化 588 个 Attention/MLP 线性层，包含理解和图像生成分支；
GGUF 额外量化 `lm_head`，共 589 个权重，其余张量以 F32 保存。
其它模型结构尚未验证。

## 1. 文件夹怎么放

在完整 SenseNova-U1 项目中保留下面的结构：

```text
tools/quantization/
├── README.md
├── NOTICE.md
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

已验证环境：Linux、Python 3.11、PyTorch 2.8.0 + CUDA 12.8、NumPy 2.2.0、
Safetensors 0.8.0、GGUF 0.19.0、comfy-kitchen 0.2.41、H100。
BF16/GGUF 转换使用 CPU；FP8/INT8 使用 CUDA。读取分片后仍会在内存中保留
输出张量，完整模型转换需要足够的主机内存，建议至少 64 GB。

Q6_K 还需要编译 GGML。Linux 安装好 C/C++ 编译器、CMake、curl 后，在项目根目录运行一次：

```bash
bash tools/quantization/gguf_q6_k/build_ggml.sh
```

脚本下载并校验固定版本源码，在 `gguf_q6_k` 内生成 `libggml-base.so`。
源码、构建缓存和 `.so` 不提交到 Git。Q4/Q8 不需要这一步。

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

同时转换 FP8 和 INT8 时，在两个终端使用不同 GPU；两者都读取同一个原始分片目录。

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

FP8/INT8 使用本工具的 `sensenova-w8a8-v2` 格式，需要下面的配套加载器。
目录名沿用已有组织方式，不代表已验证官方 ComfyUI 兼容性。
GGUF 保留 SenseNova 张量名称，用本项目的 Python 加载器推理，
不代表 llama.cpp 已原生支持该模型。

## 5. 如何验证推理

先按项目的 [安装说明](../../docs/installation_CN.md) 安装基础推理环境。
量化推理还需要 `gguf`、`diffusers`、`comfy-kitchen`，本验证入口使用 FlashAttention：

```bash
python -m pip install gguf diffusers comfy-kitchen
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

INT8 换成对应权重文件、`--format int8-convrot` 和新的输出目录。
BF16 使用 `--format bf16`，Q4/Q6/Q8 都使用 `--format gguf`。
`--model_path` 均填写具体权重文件，旁边的配置和 Tokenizer 文件需要保留。

FP8 使用 `torch._scaled_mm`，INT8 使用 `torch._int_mm`，在矩阵乘法前动态量化激活。
GGUF 为 weight-only 路径，允许反量化。非量化层及模块间激活仍使用 BF16。
验证入口使用全模型驻留 GPU，固定 `cfg_norm=none`、`cfg_interval=(0, 1)`。
输出包含 PNG、生成耗时和显存峰值 JSON；耗时排除加载、预热和图片保存。
这些结果用于验证正确性，量化不保证加速，尤其 INT8 ConvRot 有旋转和激活量化开销。
