# GGUF 与替代运行时

量化路线使用独立环境，不沿用[项目推理](../inference.md)的 BF16 运行配方。

GGUF 文件需要与运行时匹配，不能直接替换项目推理使用的 BF16 权重。使用前固定运行时版本和文件 revision/hash，并核对正式模型的 1,116 项 key/shape、实际量化类型、缺失项和非 Linear 层支持。

- 官方 Python GGUF、[ComfyUI 节点](comfyui.md)、llama.cpp 的文件不可视为通用。仅 LM GGUF 不含完整生成分支；给 `model_path` 一个完整目录也不会自动补齐缺失权重。
- GGUF 不与普通 V2 merge、`device_map/max_memory` 或项目推理的 layer offload 组合。
- 官方锁中 GGUF extra 为 `diffusers 0.37.1 + gguf 0.19.0`。单独环境检查导入、真实量化层和质量，不改动项目推理环境。
- [llama.cpp 转换路线](https://github.com/ggml-org/llama.cpp/blob/3ed7284f2d9e22b2d1a7d05c1416c00c5a9f0bc7/conversion/sensenova_u1.py)用于理解，需匹配 LM GGUF 与 mmproj；[stable-diffusion.cpp 指南](https://github.com/leejet/stable-diffusion.cpp/blob/f5df5f8b84f139f1e80496902da90798a0d92896/docs/sensenova_u1.md#L36-L45)的该路线仅覆盖非 thinking 文生图。使用这些运行时需另行安装和测试，不能复用项目推理的环境或运行命令。

## 来源

- [固定版本 GGUF loader](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/src/sensenova_u1/utils/gguf_loader.py)
- [固定版本依赖锁](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/uv.lock)
