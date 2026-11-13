# ComfyUI

ComfyUI 的节点、工作流和依赖在对应集成中维护，不使用[项目推理](../inference.md)的 Python 启动命令。

- 官方仓库提供 [ComfyUI 工作流示例](https://github.com/OpenSenseNova/SenseNova-U1/blob/94bf26c94ab7c30211dcc29466152d0302c7a262/apps/comfyui/example_workflows/infographic_series_t2i_edit.json)。该示例面向 Infographic 系列，不能直接当作正式 U1.5 的已验证配置。
- 使用社区节点时，先核对其支持的模型版本、权重格式和配套工作流，再按节点仓库的说明安装；不要混用 U1、U1.5 Preview 与正式 U1.5 权重。
- 节点使用的 GGUF 文件不一定兼容官方 Python 或其他运行时，具体限制见 [GGUF](gguf.md)。

正式 U1.5 的节点兼容性、资源需求和出图效果需在所选工作流中单独验证。
