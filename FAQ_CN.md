# FAQ\-zh

本文档汇总 SenseNova U1 开源以来社区高频问题，按主题分类整理。如有未覆盖的问题，欢迎在 [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) 或 [Discord 社群](https://discord\.gg/Su5mnbYFWf)反馈。

> **相关链接**
> - [GitHub 仓库](https://github.com/OpenSenseNova/SenseNova-U1)
> - [技术报告](https://arxiv.org/abs/2605.12500)
> - [在线体验](https://unify.light-ai.top)
> - [Discord 社群](https://discord.gg/Su5mnbYFWf)

## 一、产品与体验

**Q：图文交错功能的邀请码在哪里申请？**

A：欢迎通过[在线体验平台](https://unify.light-ai.top)使用图文交错功能。

**Q：有可以直接在线体验的网站吗？**

A：欢迎访问[在线体验平台](https://unify.light-ai.top)。

**Q：U1 与国内外主流模型的对比数据在哪里查看？**

A：请参考[技术报告第五章（评测部分）](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1.pdf)。

**Q：能否支持九宫格/分镜图生成？例如输入参考图或参考风格，输出 9 宫格分镜图。**

A：从架构能力来说可以支持，但具体的功能将在后续的版本中体现，敬请期待。

**Q：官方 Studio 生成的信息图，与本地部署/API 的体验差别较大，是什么原因？**

A：官方 Studio 及 API 接入的模型更侧重于综合性价比，兼顾了高效率与优质产出，是大部分通用场景下的最优平衡解，且保持常态化更新。本地部署版本则提供更高的灵活性，便于用户根据个性化需求进行选择，以实现更具针对性的生成效果。

**Q：中文 Prompt 的效果优于英文 Prompt，会针对英文场景优化吗？**

A：近期会有针对性的优化工作，请关注后续版本发布。

**Q：信息图功能很实用，但其他图片类型（如写实人像）效果有限，后续会提升吗？**

A：信息图是目前模型的主要方向，同时也在持续迭代其他图片类型的生成质量，欢迎持续关注并给出建议。

## 二、API 与 Token Plan

**Q：API 大概什么时间放出？**

A：中国区已上线 [Token Plan](https://platform.sensenova.cn/token-plan)，支持每 5 小时 1500 次调用。其他国家/地区的 Token Plan 计划于 7 月上线，敬请期待。

**Q：如何获取 API Key 进行调用？**

A：建议通过 [Token Plan](https://platform.sensenova.cn/token-plan) 创建 API Key 调用。当前 SenseNova 6.7 Flash-Lite 免费开放，参考[文档](https://platform.sensenova.cn/docs)。

**Q：有没有更轻量、无框架绑定的方式，让 U1 直接兼容 OpenAI 接口规范？**

A：可以关注商汤的 Token Plan，一键调用 API，参考[文档](https://platform.sensenova.cn/docs)。

## 三、技术报告与论文

**Q：技术报告什么时候发布？**

A：欢迎阅读关于 SenseNova U1 的最新[技术报告](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1.pdf)。

**Q：arXiv 论文什么时候放出？**

A：欢迎阅读 [arXiv 上的论文](https://arxiv.org/abs/2605.12500)。

## 四、训练与微调

**Q：训练代码什么时候开源？**

A：SenseNova-U1 的全参数微调训练代码已于 5 月 21 日正式开源，详见[文档](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/training/README.md)。

**Q：训练时是否使用了视频数据（video data）？**

A：理解部分使用了 video data 进行预训练，但本次训练中视频数据配比相对较低，预期下一版本会提高比例。视频帧的处理方式大体为：在每一帧的 image token 前加一个时间戳，例如 `[mm:ss.ff]:<image>`。

**Q：LoRA 训练代码会开源吗？**

A：LoRA 训练代码的开源已在计划中，请关注我们后续的发布。

**Q：人像畸形问题用全参数微调，大概需要多少训练数据才能解决？**

A：这类问题更可能需要在训练的较早期让模型学会，单纯靠后期微调数据量收益有限。我们的下一版模型正在重点解决该问题。

**Q：视觉理解任务支持多图输入吗？HuggingFace 上只看到单图样例。**

A：支持多图。底层 `NEOChatModel.chat()` 已原生支持多图：在 `samples.jsonl` 里把 `image` 写成路径数组，并在 `question` 中手动放置对应数量的 `〈image〉` 占位符`（推荐 “Image-1: 〈image〉 换行 Image-2: 〈image〉 ...” 模板）`。示例：`{"id":"cmp","image":["a.jpg","b.jpg"],"question":"Image-1: 〈image〉` 换行 `Image-2: 〈image〉 换行 Compare them."}（实际使用时 〈image〉 即标准的 image 占位标签、“换行”即换行符 \n）`。若用 `examples/vqa/inference.py`，把 `answer()` 改成对每张图分别 `load_image_native` 后 `torch.cat` 出 `pixel_values` 与 `grid_hw` 并兼容 `image` 为 list 即可，其余无需改动。

**Q：RL（强化学习）部分的代码会开源吗？**

A：RL 代码的开源在计划中，欢迎持续关注。

## 五、部署与硬件

**Q：本地电脑如何调用？推荐什么部署框架？**

A：可本地部署 U1。框架建议：ollama（简单易用，但性能一般）；`llama.cpp`（配置稍复杂，但性能最佳）；vLLM（速度快，对 GPU 友好）。若不想自行部署，也可直接通过 [Token Plan](https://www.sensenova.cn/token-plan) 创建 API Key 调用。

**Q：生成 2K 图片需要多大显存？**

A：在 H100 上的测试结果为 35–37GB 显存。详见 [Issue #87](https://github.com/OpenSenseNova/SenseNova-U1/issues/87)。

**Q：模型效果最好的分辨率设置是多少？**

A：目前推荐设置为 2K，这也是内部研发时的主要 target 分辨率。

**Q：5090 32G 显卡可以部署 SenseNova-U1-8B-MoT 吗？**

A：可以部署。部分用户反馈调用 API 时速度偏慢、显存利用率偏低，该问题我们正在排查，欢迎在 [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) 反馈具体环境信息以便定位。

**Q：有没有本地部署的一键整合包？**

A：请参考[快速上手文档](https://github.com/OpenSenseNova/SenseNova-U1)进行部署。

**Q：是否支持 arm64/aarch64 架构？目前 uv 安装和 docker 镜像都是 amd64。**

A：官方默认依赖为 `torch 2.8 + cu128`，该组合暂无对应的 `aarch64 wheel`。可尝试改用 2.7.1 或 2.9 以上版本，PyTorch 官方提供了部分 [aarch64 wheel](https://download.pytorch.org/whl/cu128/torch/)。已有社区用户反馈使用 torch 2.9.0 在 aarch64 上安装成功（会有默认版本不一致的警告）。我们正在评估是否将默认版本切换为对 aarch64 支持更好的版本。

**Q：是否支持 vLLM 部署？**

A：目前已支持 vLLM 部署，请参考 [Issue #93](https://github.com/OpenSenseNova/SenseNova-U1/issues/93)。

**Q：是否支持 MLX？**

A：目前暂不支持 MLX。

**Q：是否支持 NVFP4 量化（如 DGX Spark 上 `LightLLM + LightX2V` 的文本 NVFP4 + 视觉全精度混合精度配置）？**

A：目前暂不支持 NVFP4。

**Q：是否支持昇腾卡/Atlas 加速卡/华为系国产卡部署？**

A：目前暂不支持，仅支持在 `Linux + CUDA` 平台上部署。Day-0 的国产芯片支持由对应芯片厂商主导，暂未获得统一的官方支持方案。

**Q：是否支持 arm64 版本的 `lightllm_lightx2v` 部署？**

A：目前 `lightllm_lightx2v` 暂不支持 arm64，仅支持 `Linux + CUDA` 平台。

**Q：内存/显存占用较大，配置不够的机器跑不起来，有更省资源的方案吗？**

A：该模型对内存/显存的需求确实较高，资源有限时建议使用 GGUF 的 `low/balanced` 档位。参考 [GitHub](https://github.com/OpenSenseNova/SenseNova-U1)。

**Q：GGUF 平衡模式 8G 能跑，但 full 模式（如 Q6）显存会爆，正常吗？**

A：是的，full 模式对显存要求较高，属于预期情况。资源有限时建议改用 GGUF 的 `low/balanced` 档，参考 [GitHub](https://github.com/OpenSenseNova/SenseNova-U1)。

**Q：有没有多卡（2 卡/4 卡）推理脚本？单卡爆显存。**

A：可以使用 `LightLLM + LightX2V` 进行多卡推理，已支持多卡多机。参考[部署文档](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md)。

**Q：对 Python 版本有要求吗？比如必须 3.11？**

A：模型本身并未限制 Python 版本。PyTorch 与 Python 版本之间存在绑定关系，若能自行搭建好环境，可以使用自己已有的 Python 版本。

**Q：ComfyUI 节点为什么没上传到官方 ComfyUI 节点 Registry？怎么安装？**

A：官方 ComfyUI 节点已发布并登记到 [ComfyUI 节点 Registry](https://registry.comfy.org/zh/nodes/ComfyUI-SenseNova-U1)。安装可参考[文档](https://github.com/OpenSenseNova/SenseNova-U1/tree/main/apps/comfyui)。

**Q：打开 `think_mode` 后会不会跑 1024 步、导致耗时过长？**

A：推理步数被限制为 50 步或 8 步，步数大小不会受 `think_mode` 影响。若实测耗时异常，欢迎在 [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) 反馈具体环境。

**Q：用 infographic 功能时，50 步反而比 8 步更快几分钟，正常吗？**

A：官方 Studio、API 与本地部署模型均已升级新版，建议在新版模型上用相同 Prompt 重新测试。若仍异常，欢迎在 [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) 反馈。

**Q：在 `XPU（Intel GPU）` 环境下速度很慢、`dtype/device` 设置后实际仍跑成 f32/cpu，怎么办？**

A：XPU 相关适配已更新，建议拉取最新代码重试。另外 infographic 模型用 f16 时可能出现 cast 报错，切换为 bf16 可改善。若仍有问题欢迎在 [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) 反馈。

**Q：模型运行需要什么硬件配置？**

A：bf16 模型打下 36G，所以至少需要机器显存超过这个大小，或者多卡推理。

**Q：双卡共 32G 显存，使用 `--x2i_server_deploy_mode colocat、tp=2` 时出现 OOM，如何设置参数以使用系统内存？**

A：bf16 模型大小36G，所以需要更大显存的机器。

**Q：如何成功部署 38B 模型？需要什么配置？**

A： 38B 的模型，76G 现存需求，建议双卡 80G 机器，或者 4 卡 32G 机器（如 5090）。

**Q：单卡运行需要修改哪些配置？tp 和 cfg 都改成 1 吗？**

A：都给 1 即可。

**Q：将脚本封装为 vLLM/SGLang 服务的方案？**

A：目前已支持 vLLM 部署，请参考 [Issue #93](https://github.com/OpenSenseNova/SenseNova-U1/issues/93)。

## 六、提示词

**Q：提示词应该怎么写？同样的提示词在不同平台生成的差别较大。**

A：可参考 [Prompt Gallery 的效果示例与提示词结构](https://github.com/OpenSenseNova/SenseNova-Skills/blob/main/docs/sn-infographic-examples_CN.md)。也建议开启思考模式，让模型先针对你的需求进行思考再生成，效果更稳定。

**Q：提示词增强能否使用本地模型？必须接入 API 吗？**

A：支持使用本地模型，参考[文档](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/prompt_enhancement_CN.md)。[提示词模板可参考](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/src/sensenova_u1/prompt_enhance/templates/infographic_system.md)。此外，[SenseNova 6.7 Flash-Lite](https://platform.sensenova.cn/token-plan) 当前免费，可用于提示词增强。

## 七、常见报错

**Q：用 lightx2v 镜像运行时，因模型目录缺少 `conversation.py`，报 `NeoChatTokenizer 加载失败 / self.conversation_module 为 None`，怎么解决？**

A：该问题已修复，请使用最新分支代码。详见 [Issue #170](https://github.com/OpenSenseNova/SenseNova-U1/issues/170)，参考[部署文档](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md)。

**Q：镜像容器内对话正常，但文生图报错：`NoneType object has no attribute get_conv_template`，怎么解决？**

A：该问题已修复，请使用最新分支代码，并参考[部署文档](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md)。

