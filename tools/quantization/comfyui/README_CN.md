# ComfyUI 模型转换

这两个导出脚本生成 SenseNova-U1 的 ComfyUI checkpoint 格式：

- `convert_bf16.py` 流式合并 HF safetensors 分片，生成无损 BF16 单文件；
  tensor 名称、形状、dtype 和 payload 字节都保持不变。
- `convert_int8_convrot.py` 生成带 ComfyUI 每层 `.comfy_quant` 配置的
  INT8 + ConvRot 文件。它以合并后的 BF16 文件为输入，因此先运行 BF16
  wrapper。

wrapper 会在存在 `config.env` 时读取它。可以先复制配置模板：

```bash
cp config.env.example config.env
# 修改 config.env
bash convert_bf16.sh
bash convert_int8_convrot.sh
```

INT8 wrapper 默认会量化 token embedding，与通用 ComfyUI 导出器一致。
设置 `COMFYUI_QUANTIZE_EMBEDDINGS=0` 可以让 `embed_tokens` 保持 BF16，
这样量化层选择与原生 SenseNova 转换器一致。

输出文件用于 ComfyUI 原生 mixed-precision loader，不是
`src/sensenova_u1/utils/quant_runtime.py` 使用的 `sensenova-w8a8-v2` 文件级格式。

需要安装 `torch`、`safetensors` 和 `comfy-kitchen`；INT8 转换需要 CUDA
设备。
