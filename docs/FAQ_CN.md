# 常见问题

<p align="center"><a href="./FAQ.md">English</a> | <strong>简体中文</strong></p>

安装和运行命令见[模型指南与案例](../README_CN.md#模型指南与案例)；U1 服务配置见[部署指南](deployment_CN.md)。

**U1.5 单卡显存不足时怎么办？**

先用 `--vram_mode balanced`；仍不够时可试 `low`。两种模式都会占用系统内存：U1.5 的 RTX 5090 测试中，进程 CPU RSS 峰值约 52.2 GiB。可先按 64 GB 系统内存规划，这不是实测最低要求。参数见[任务示例](../examples/README_CN.md#单卡显存模式)，测试范围见[U1.5 显存与耗时实测](u1.5_inference_CN.md)。

**视觉问答脚本能一次输入多张图片吗？**

目前的 [VQA 脚本](../examples/vqa/inference.py)每条样本只接收一张图片；在 JSONL 的 `image` 字段填路径数组无法直接运行。多图参考编辑和图文交错生成是另外的任务，命令见[任务示例](../examples/README_CN.md)。

**开启 `--think` 会把图像生成改成 1024 步吗？**

不会。图像生成步数由 `--num_steps` 控制，默认是 50；1024 不是扩散步数。思考阶段可能额外生成文字，因此总耗时仍可能增加。
