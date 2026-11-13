# SenseNova U1.5 中文推理与部署指南

| 文档 | 内容 |
| --- | --- |
| [快速开始](quickstart.md) | 运行条件、首次推理、输出检查与文档导航 |
| [环境与模型准备](installation.md) | 运行条件、固定版本、依赖安装和权重校验 |
| [项目推理](inference.md) | 单卡、低显存、多卡、8step V2 LoRA 和任务入口 |
| [服务部署](serving.md) | LightLLM + LightX2V 的独立环境、拓扑与请求检查 |
| [常见问题与验收](troubleshooting.md) | 故障排查、目标机验收和反馈信息 |
| [ComfyUI](integrations/comfyui.md) | 节点与工作流的独立入口和兼容性边界 |
| [GGUF](integrations/gguf.md) | 量化文件、运行时适配和限制 |

本目录固定使用 U1.5 正式基座和代码 commit `94bf26c94ab7c30211dcc29466152d0302c7a262`。安装过程在独立工作目录检出该版本，不要求把当前文档分支切回旧 commit。仓库现有的 [安装文档](../docs/installation_CN.md)、[部署文档](../docs/deployment_CN.md)和其他版本资料仍保留。

代码和模型分别遵循对应上游项目的许可证；发布或提供服务前查看所用 revision 的许可证和模型卡。
