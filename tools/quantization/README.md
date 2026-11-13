# U1.6 Weight Conversion and Validation

English | [Chinese](README_CN.md)

The six scripts cover BF16 merging, GGUF Q4/Q6/Q8, FP8, and INT8 ConvRot.
To use them: place the files, set the paths, then run the scripts.

## 1. Directory layout

Keep this structure inside the complete SenseNova-U1 project:

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

Each conversion subdirectory can be copied and used independently. Keep its
`.sh` and `.py` files together; it does not depend on the other conversion
subdirectories. Inference requires the complete project.

Install the conversion dependencies:

```bash
python -m pip install torch safetensors numpy gguf comfy-kitchen
```

Q6_K also requires building GGML. On Linux, install a C/C++ compiler, CMake,
and curl, then run this once from the project root:

```bash
bash tools/quantization/gguf_q6_k/build_ggml.sh
```

The script downloads and verifies a pinned source version, then creates
`libggml-base.so` inside `gguf_q6_k`. Q4/Q8 do not need this step.

## 2. Set the paths

Set these two paths from the project root before running a conversion:

```bash
export U1_SOURCE_DIR="/data/models/hf_checkpoint"
export U1_OUTPUT_DIR="/data/SenseNova-U1/outputs/quantization"
```

- `U1_SOURCE_DIR`: the original HF/EMA directory, containing all weight shards,
  `model.safetensors.index.json`, configuration files, and tokenizer files.
- `U1_OUTPUT_DIR`: the output root. The scripts create `sensenova` and the
  corresponding weight subdirectories inside it.

Alternatively, edit the default paths below `# set manually:` in each `.sh`
file. The scripts use `python` from the current environment; they do not require
a specific virtual environment path.

Select a GPU for FP8/INT8:

```bash
export CUDA_VISIBLE_DEVICES="0"
```

## 3. Run a conversion

Run the scripts you need from the project root:

```bash
bash tools/quantization/bf16/convert_bf16.sh
bash tools/quantization/gguf_q4_0/convert_gguf_q4_0.sh
bash tools/quantization/gguf_q6_k/convert_gguf_q6_k.sh
bash tools/quantization/gguf_q8_0/convert_gguf_q8_0.sh
bash tools/quantization/fp8_w8a8/convert_fp8_w8a8.sh
bash tools/quantization/int8_convrot_w8a8/convert_int8_convrot_w8a8.sh
```

The scripts do not overwrite existing weights. Use a new `U1_OUTPUT_DIR` to
repeat a conversion. A `.partial` file left by an interrupted conversion also
blocks overwriting; remove it manually only when it is no longer needed.

## 4. Find the outputs

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

The scripts copy `config.json`, `tokenizer_config.json`, `vocab.json`,
`merges.txt`, `special_tokens_map.json`, and `added_tokens.json` next to each
checkpoint. These files must be present in the input directory.

`conversion.json` records the format, tensor counts, and file SHA256.
FP8/INT8 also produce `quantization_config.json`.

## 5. Validate inference

Install the base inference dependencies:

```bash
python -m pip install torch torchvision triton transformers huggingface-hub accelerate safetensors tokenizers sentencepiece numpy pillow einops diffusers gguf comfy-kitchen packaging tqdm httpx psutil PyYAML
```

Install FlashAttention separately after installing PyTorch:

```bash
python -m pip install flash-attn --no-build-isolation
```

FP8 apple example: run from the project root using the `U1_OUTPUT_DIR` set above.

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

For INT8, use its checkpoint file, set `--format int8-convrot`, and choose a new
output directory. Use `--format bf16` for BF16 and `--format gguf` for all three
GGUF formats. Always set `--model_path` to the checkpoint file itself.

Inference uses FlashAttention, BF16 for non-quantized layers, and keeps the full
model on the GPU. Images, timing results, and peak GPU memory usage are saved
under `--output_dir`.
