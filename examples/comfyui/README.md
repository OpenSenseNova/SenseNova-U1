# Native SenseNova workflows for ComfyUI

These workflows use built-in nodes and require [ComfyUI 0.46.0 or later](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.46.0).

| Workflow | Settings |
| --- | --- |
| [U1.5 text-to-image](comfyui-sensenova-u1.5-t2i-8step.json) | 8 steps with LoRA |
| [U1.5 image editing](comfyui-sensenova-u1.5-edit-8step.json) | 8 steps with LoRA and a reference image |
| [U1.5 interleaved text and image generation](comfyui-sensenova-u1.5-interleave.json) | 50 steps with base weights |
| [U1.5 interleaved text and image generation with LoRA](comfyui-sensenova-u1.5-interleave-8step.json) | 8 steps with LoRA |
| [U1.6 text-to-image with Turbo](comfyui-sensenova-u1.6-t2i-turbo.json) | Switchable Turbo |
| [U1.6 image editing with Turbo](comfyui-sensenova-u1.6-edit-turbo.json) | Switchable Turbo and a reference image |
| [U1.6 image editing with Turbo and Qwen-Image-2.1 PE](comfyui-sensenova-u1.6-edit-turbo-with-pe.json) | Switchable Turbo, prompt enhancement, and a reference image resized by pixel count |
| [U1.6 self-correction with Turbo](comfyui-sensenova-u1.6-self-correction-turbo.json) | Switchable Turbo |
| [U1.6 self-correction with a reference image and Turbo](comfyui-sensenova-u1.6-self-correction-with-image-turbo.json) | Switchable Turbo and a reference image |
| [U1.6 self-correction](comfyui-sensenova-u1.6-self-correction-8step.json) | 8 steps with LoRA |
| [U1.6 self-correction with a reference image](comfyui-sensenova-u1.6-self-correction-with-image-8step.json) | 8 steps with LoRA and a reference image |
| [U1.6 text-to-image with Qwen-Image-2.1 PE](comfyui-sensenova-u1.6-t2i-8step-with-qwen21-pe.json) | 8 steps with LoRA and prompt enhancement |
| [U1.6 text-to-image with H3 PE](comfyui-sensenova-u1.6-t2i-8step-with-h3-pe.json) | 8 steps with LoRA and prompt enhancement |
| [U1.6 text-to-image with MOSS-Image PE](comfyui-sensenova-u1.6-t2i-8step-with-moss-pe.json) | 8 steps with LoRA and prompt enhancement |

Drag a JSON file into ComfyUI. Place the ComfyUI-format checkpoint in
`ComfyUI/models/checkpoints/` and select it in **Load Checkpoint**. For an
8-step workflow, place the matching LoRA in `ComfyUI/models/loras/` and select
it in **Load LoRA (Model Only)**. Upload and select a reference image in
**Load Image** for workflows that take an image input.

In the Turbo workflows, **Enable turbo** switches from the base checkpoint at
50 steps and CFG 4 (the default) to the 8-step LoRA at CFG 1. Keep the matching
LoRA available in both modes because ComfyUI validates the connected loader
when queuing the workflow. The Turbo editing example without PE and the Turbo self-correction example with a reference image
optionally scale it by a multiplier before aligning both dimensions to
multiples of 32. The aligned image determines the output dimensions.

The three text-to-image prompt enhancement (PE) workflows were verified on ComfyUI 0.46.0.
Place the PE text encoder selected in **Load CLIP** in
`ComfyUI/models/text_encoders/`. Turn off **Enable PE** to use the original
prompt.
