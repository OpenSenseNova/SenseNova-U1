# ComfyUI checkpoint conversion

These two exporters create the ComfyUI checkpoint formats for SenseNova-U1:

- `convert_bf16.py` streams the HF safetensors shards into one lossless BF16
  file. It preserves tensor names, shapes, dtypes, and payload bytes.
- `convert_int8_convrot.py` exports INT8 + ConvRot tensors with ComfyUI's
  per-module `.comfy_quant` metadata. It accepts the merged BF16 file, so run
  the BF16 wrapper first.

The wrappers read `config.env` when it exists. Start with
`config.env.example`, then run:

```bash
cp config.env.example config.env
# edit config.env
bash convert_bf16.sh
bash convert_int8_convrot.sh
```

The INT8 wrapper quantizes token embeddings by default, matching the generic
ComfyUI exporter. Set `COMFYUI_QUANTIZE_EMBEDDINGS=0` to keep `embed_tokens` in
BF16, which matches the layer selection of the native SenseNova converter.

The output is intended for ComfyUI's native mixed-precision loader. It is not
the file-level `sensenova-w8a8-v2` format consumed by
`src/sensenova_u1/utils/quant_runtime.py`.

Required packages are `torch`, `safetensors`, and `comfy-kitchen`; INT8
conversion requires a CUDA device.
