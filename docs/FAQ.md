# Frequently asked questions

<p align="center"><strong>English</strong> | <a href="./FAQ_CN.md">简体中文</a></p>

For setup and commands, see the [model guides and showcases](../README.md#model-guides-and-showcases). For U1 serving, see the [deployment guide](deployment.md).

**What if U1.5 runs out of VRAM on a single GPU?**

Start with `--vram_mode balanced`; try `low` if it still does not fit. Both modes use host RAM: process CPU RSS peaked around 52.2 GiB in the RTX 5090 U1.5 runs. Plan for 64 GB of system RAM as a starting point, not a measured minimum. See the [examples](../examples/README.md#single-gpu-memory-modes) for the options and [U1.5 memory and timing](u1.5_inference.md) for the tested scope.

**Can the visual question answering script take multiple images at once?**

The current [VQA script](../examples/vqa/inference.py) accepts one image per sample. A list of paths in the JSONL `image` field will not run as written. Multi-image reference editing and interleaved generation are separate tasks; see the [examples](../examples/README.md).

**Does `--think` change image generation to 1024 steps?**

No. `--num_steps` controls image generation steps and defaults to 50; 1024 is not the diffusion step count. Thinking may generate extra text, so total runtime can still increase.
