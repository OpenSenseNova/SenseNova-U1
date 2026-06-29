# FAQ

This document collects the questions most frequently raised by the community since SenseNova U1 was open-sourced, organized by topic. If your question isn't covered here, feel free to reach out via [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues) or our [Discord community](https://discord.gg/Su5mnbYFWf).

> **Useful links**
> - [GitHub repository](https://github.com/OpenSenseNova/SenseNova-U1)
> - [Technical report](https://arxiv.org/abs/2605.12500)
> - [Try NOW](https://unify.light-ai.top/)
> - [Discord community](https://discord.gg/Su5mnbYFWf)
> 
> 💡 Note: The Free Token plan for global users will update in July. Stay tuned on Discord!

## 1. Product & Experience

**Q: Where can I request an invite code for the interleaved text-and-image feature?**

A: The interleaved text-and-image feature is now available. You can try it using the [SenseNova Interleaved demo platform](https://unify.light-ai.top/).

**Q: Is there a website where I can try it online directly?**

A: Yes, you're welcome to use our [online demo platform](https://unify.light-ai.top/).

**Q: Where can I find benchmark comparisons between U1 and other leading models?**

A: See Chapter 5 (Evaluation) of the [technical report](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1.pdf).

**Q: Can it generate a 3x3 grid or storyboard? For example, taking a reference image or style as input and producing a 9-panel storyboard.**

A: The architecture is capable of supporting this, but the feature itself will come in a future release. Stay tuned.

**Q: The infographics generated in the official Studio look quite different from what I get with local deployment or the API. Why is that?**

A: The models behind the official Studio and API are tuned for overall cost-effectiveness, balancing efficiency with output quality, which makes them the best all-around fit for most general use cases—and they're updated on a regular basis. The local deployment version offers more flexibility, letting you tailor the setup to your specific needs for more targeted results.

**Q: Chinese prompts work better than English ones. Will you optimize for English?**

A: Targeted improvements are coming soon—keep an eye out for upcoming releases.

**Q: The infographic feature is very useful, but other image types (such as photorealistic portraits) are limited. Will this improve?**

A: Infographics are the model's primary focus right now, but we're also continuously improving generation quality for other image types. We welcome your ongoing feedback and suggestions.

## 2. API & Token Plan

**Q: When will the API be released?**

A: The Token Plan is already live in mainland China, supporting 1,500 calls per 5-hour window. Token Plans for other countries and regions are scheduled to launch in July, stay tuned.

**Q: How do I get an API key to make calls?**

A: We recommend creating an API key through the Token Plan. SenseNova 6.7 Flash-Lite is currently free to use—you can sign up at [this website](https://platform.sensenova.cn/token-plan) (Chinese users only; global version coming in July). See [the documentation](https://platform.sensenova.cn/docs) for details.

**Q: Is there a lighter, framework-agnostic way to make U1 directly compatible with the OpenAI API spec?**

A: Take a look at SenseTime's Token Plan, which lets you call the API out of the box. [Documentation](https://platform.sensenova.cn/en/docs).

## 3. Technical Report & Paper

**Q: When will the technical report be published?**

A: It's already out—read it [here](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1.pdf).

**Q: When will the arXiv paper be released?**

A: It's already [online](https://arxiv.org/abs/2605.12500).

## 4. Training & Fine-tuning

**Q: When will the training code be open-sourced?**

A: [The full-parameter fine-tuning code for SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/training/README.md) was officially open-sourced on May 21.

**Q: Was video data used during training?**

A: Video data was used to pre-train the understanding component, though its share in this round of training was relatively small; we expect to increase the ratio in the next version. Video frames are handled roughly as follows: a timestamp is prepended to the image token of each frame—for example, `[mm:ss.ff]:<image>`. A more detailed description will be provided in a future technical report.

**Q: Will the LoRA training code be open-sourced?**

A: Open-sourcing the LoRA training code is on our roadmap. Watch for our upcoming releases.

**Q: For fixing distorted portraits with full-parameter fine-tuning, roughly how much training data is needed?**

A: Issues like this are better addressed by having the model learn them earlier in training; relying on later fine-tuning data alone yields limited returns. Our next model version is focused on solving this problem.

**Q: Do visual understanding tasks support multi-image input? I only see single-image examples on HuggingFace.**

A: Multi-image input is supported. The underlying `NEOChatModel.chat()` natively handles multiple images: in `samples.jsonl`, write the image field as an array of paths, and manually place the corresponding number of 〈image〉 placeholders in the question (we recommend the "Image-1: 〈image〉 newline Image-2: 〈image〉 ..." template). Example: {"id":"cmp","image":["a.jpg","b.jpg"],"question":"Image-1: 〈image〉 newline Image-2: 〈image〉 newline Compare them."} (in actual use, 〈image〉 is the standard image placeholder tag and "newline" is the newline character \n). If you use examples/vqa/inference.py, modify `answer()` to call `load_image_native` separately for each image and then torch.cat the `pixel_values` and `grid_hw`, while making it accept image as a list—no other changes are needed.

**Q: Will the RL (reinforcement learning) code be open-sourced?**

A: Open-sourcing the RL code is on our roadmap. Stay tuned.

## 5. Deployment & Hardware

**Q: How do I run it on my local machine? Which deployment framework do you recommend?**

A: U1 can be deployed locally. Framework recommendations: ollama (easy to use, but average performance); llama.cpp (slightly more involved to set up, but the best performance); vLLM (fast and GPU-friendly). If you'd rather not deploy it yourself, you can also create an API key through the [Token Plan](https://www.sensenova.cn/token-plan) and call it directly (China only; global version coming in July).

**Q: How much VRAM is needed to generate a 2K image?**

A: Our tests on an H100 showed 35–37 GB of VRAM. See [Issue #87](https://github.com/OpenSenseNova/SenseNova-U1/issues/87) for details.

**Q: What resolution gives the best results?**

A: We currently recommend 2K, which was also the primary target resolution during internal development.

**Q: Can I deploy SenseNova-U1-8B-MoT on a 5090 32G GPU?**

A: Yes, it can be deployed. Some users have reported slow API calls and low VRAM utilization; we're looking into this and welcome reports with specific environment details via GitHub Issues to help us pinpoint the cause.

**Q: Is there an all-in-one bundle for local deployment?**

A: Please follow [the quick-start guide](https://github.com/OpenSenseNova/SenseNova-U1) to deploy.

**Q: Do you support arm64 / aarch64 architectures? Right now both the uv install and the docker image are amd64.**

A: The default dependencies are torch 2.8 + cu128, a combination that currently has no corresponding aarch64 wheel. You can try switching to version 2.7.1, or 2.9 and above—PyTorch officially provides some aarch64 wheels: https://download.pytorch.org/whl/cu128/torch/. Some community users have reported successfully installing torch 2.9.0 on aarch64 (with a warning about the version mismatch from the default). We're evaluating whether to switch the default to a version with better aarch64 support.

**Q: Is vLLM deployment supported?**

A: Yes, vLLM deployment is now supported. See [Issue #93](https://github.com/OpenSenseNova/SenseNova-U1/issues/93).

**Q: Is MLX supported?**

A: MLX is not supported at this time.

**Q: Is NVFP4 quantization supported (for example, a mixed-precision setup on DGX Spark with LightLLM + LightX2V using NVFP4 for text and full precision for vision)?**

A: NVFP4 is not supported at this time.

**Q: Do you support deployment on Ascend cards / Atlas accelerator cards / Huawei domestic chips?**

A: Not at this time—deployment is only supported on Linux + CUDA platforms. Day-0 support for domestic chips is led by the respective chip vendors, and there is no unified official support plan yet.

**Q: Is arm64 deployment of lightllm_lightx2v supported?**

A: lightllm_lightx2v does not currently support arm64; it only supports Linux + CUDA platforms.

**Q: Memory / VRAM usage is high and machines with lower specs can't run it. Is there a more resource-efficient option?**

A: The model does have fairly high memory / VRAM requirements. When resources are limited, we recommend using [the GGUF](https://github.com/OpenSenseNova/SenseNova-U1) low / balanced tiers.

**Q: GGUF balanced mode runs on 8 GB, but full mode (e.g. Q6) blows past the VRAM limit. Is that normal?**

A: Yes—full mode has high VRAM requirements, and this is expected. When resources are limited, we recommend switching to [the GGUF](https://github.com/OpenSenseNova/SenseNova-U1) low / balanced tiers.

**Q: Is there a multi-GPU (2-card / 4-card) inference script? A single card runs out of VRAM.**

A: You can use `LightLLM + LightX2V` for multi-GPU inference, which already supports multi-GPU and multi-node setups. See the [deployment guide](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md).

**Q: Is there a Python version requirement? Does it have to be 3.11, for instance?**

A: The model itself doesn't restrict the Python version. There is, however, a binding between PyTorch and Python versions, so if you can set up the environment yourself, you're free to use your existing Python version.

**Q: Why weren't the ComfyUI nodes uploaded to the official ComfyUI node Registry? How do I install them?**

A: The official ComfyUI nodes have been published and registered in [the ComfyUI node Registry](https://registry.comfy.org/zh/nodes/ComfyUI-SenseNova-U1). For installation, see [this documentation](https://github.com/OpenSenseNova/SenseNova-U1/tree/main/apps/comfyui).

**Q: After turning on `think_mode`, will it run 1024 steps and take far too long?**

A: The number of inference steps is capped at either 50 or 8 steps, and this isn't affected by `think_mode`. If you observe unusual run times in practice, feel free to report your specific environment via GitHub Issues.

**Q: When using the infographic feature, 50 steps is actually a few minutes faster than 8 steps. Is that normal?**

A: The official Studio, API, and local deployment models have all been upgraded to a new version. We recommend re-testing with the same prompt on the new model. If the issue persists, feel free to report it via [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues).

**Q: On XPU (Intel GPU), it's very slow, and even after setting dtype/device it still ends up running as f32/cpu. What can I do?**

A: The XPU-related adaptations have been updated, so we recommend pulling the latest code and trying again. Also, the infographic model may throw a cast error when using f16; switching to bf16 can help. If problems persist, feel free to report them via [GitHub Issues](https://github.com/OpenSenseNova/SenseNova-U1/issues).

## 6. Prompts

**Q: How should I write prompts? The same prompt produces quite different results across platforms.**

A: You can refer to the example outputs and prompt structures in the [Prompt Gallery](https://github.com/OpenSenseNova/SenseNova-Skills/blob/main/docs/sn-infographic-examples_CN.md). We also recommend enabling thinking mode so the model reasons about your request before generating—this gives more consistent results.

**Q: Can prompt enhancement use a local model, or does it have to connect to the API?**

A: Local models are supported. See the [documentation](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/prompt_enhancement.md).
For the prompt template, see [this file](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/src/sensenova_u1/prompt_enhance/templates/infographic_system.md). 
In addition, SenseNova 6.7 Flash-Lite is currently free and can be used for prompt enhancement, please refer to the [SenseNova Token Plan](https://platform.sensenova.cn/token-plan).

## 7. Common Errors

**Q: When running with the lightx2v image, NeoChatTokenizer fails to `load / self.conversation_module` is None because conversation.py is missing from the model directory. How do I fix this?**

A: This has been fixed—please use the latest branch code. See [Issue #170](https://github.com/OpenSenseNova/SenseNova-U1/issues/170) for details, and for deployment, see [this documentation](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md).

**Q: Conversations work fine inside the image container, but text-to-image throws NoneType object has no attribute get_conv_template. How do I fix this?**

A: This has been fixed—please use the latest branch code and refer to [the deployment guide](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/deployment.md).

