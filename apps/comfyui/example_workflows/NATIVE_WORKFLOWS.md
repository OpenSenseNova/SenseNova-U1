# Native SenseNova ComfyUI workflows

These workflows use the SenseNova implementation inside ComfyUI: a converted
single-file checkpoint, `CheckpointLoaderSimple`, and the standard sampling nodes.
Use the task-based Generate SenseNova Text implementation on ComfyUI branch
`gp/sensenova-clean-20261028`. Installing this repository's custom-node plugin alone
does not provide these native nodes.

## U1.5 templates

| Task | UI workflow: drag into ComfyUI | API prompt graph |
| --- | --- | --- |
| Text to image | [image-sensenova-text2image.json](image-sensenova-text2image.json) | [comfyui-sensenova-t2i.json](comfyui-sensenova-t2i.json) |
| Text to image with thinking | [image-sensenova-text2image-thinking.json](image-sensenova-text2image-thinking.json) | [comfyui-sensenova-t2i-thinking.json](comfyui-sensenova-t2i-thinking.json) |
| Image editing | [image-sensenova-image-edit.json](image-sensenova-image-edit.json) | [comfyui-sensenova-edit.json](comfyui-sensenova-edit.json) |
| Image editing with thinking | [image-sensenova-image-edit-thinking.json](image-sensenova-image-edit-thinking.json) | [comfyui-sensenova-edit-thinking.json](comfyui-sensenova-edit-thinking.json) |
| Interleave | [image-sensenova-interleave.json](image-sensenova-interleave.json) | [comfyui-sensenova-interleave.json](comfyui-sensenova-interleave.json) |

Place `SenseNova-U1.5-8B-MoT-BF16-T8.safetensors` in ComfyUI's
`models/checkpoints/` and `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors`
in `models/loras/`, or choose your matching filenames in the loader nodes.
The templates use the 8-step LoRA with 8 steps and CFG 1.
For editing, upload an image and select it in `LoadImage`; the API example expects
`nova_editor_example.webp` in ComfyUI's `input/` directory.

Both interleave templates use `base seed + iteration_index`, return the last
iteration with `EndLoop.accumulate=false`, and pass the final pixel LATENT to
`SenseNovaInterleaveDecode.samples`. Generated images are extracted from history.

## U1.6 API graphs

The seven files in [native-u16/](native-u16/) are API prompt graphs for
the task-based generation interface. The table above contains the UI templates.

| API graph | Purpose | Checkpoint |
| --- | --- | --- |
| [t2i-api.json](native-u16/t2i-api.json) | Text to image | step1200 |
| [edit-api.json](native-u16/edit-api.json) | Image editing | step1200 |
| [interleave-api.json](native-u16/interleave-api.json) | U1.5-style interleave protocol diagnostic | step1200 |
| [interleave-u15-lora-api.json](native-u16/interleave-u15-lora-api.json) | Same diagnostic with the U1.5 8-step LoRA | step1200 |
| [self-correction-api.json](native-u16/self-correction-api.json) | Generate and optionally self-correct | step400 EMA |
| [self-correction-image-api.json](native-u16/self-correction-image-api.json) | Self-correction with an external reference image | step400 EMA |
| [self-correction-reference-api.json](native-u16/self-correction-reference-api.json) | Self-correction with SenseNova-U1 inference alignment | step400 EMA |

Expected filenames in `models/checkpoints/`:

- step1200: `SenseNova-U1.6-8B-MoT-BF16-step1200.safetensors`
- step400 EMA: `SenseNova-U1.6-8B-MoT-BF16-selfcorr-step400-ema.safetensors`

`edit-api.json` expects `input/nova_editor_example.webp`.
`self-correction-image-api.json` expects `input/self-correction-input.png`;
provide your initial image and update the prompt to describe its requirements.
These input images and model weights are not bundled. Filenames can be changed
in the JSON. Image conditioning does not guarantee that the model returns only
one new image.

For self-correction, use ordinary `CLIPTextEncode` nodes and select
`task=self-correction` in Generate. This selects the image system prompt and
non-thinking interleave internally. Normal outputs are one image, or
image → correction text → second image.

Ordinary graphs use ComfyUI sampling. The alignment graph uses
`SenseNovaU1InferenceAlignment` with `SamplerCustomAdvanced`, the same base seed
for every iteration, and incoming history/previous samples to reproduce the
SenseNova-U1 random stream. Exact pixels also require matching input, model,
settings, device, and attention backend. This graph has no external input image;
“reference” in its filename means inference alignment.

Submit an API graph to your running ComfyUI service, for example from the
SenseNova-U1 repository root:

```bash
python - <<'PY'
import json
import urllib.request
from pathlib import Path

path = Path("apps/comfyui/example_workflows/native-u16/self-correction-api.json")
request = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": json.loads(path.read_text())}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
PY
```

Set the URL to your ComfyUI service. The response contains a `prompt_id`; use
ComfyUI's history/output view to inspect the result.

## Node names

| Display name | ID |
| --- | --- |
| CLIP Text Encode (Prompt) | `CLIPTextEncode` |
| Generate SenseNova Text | `TextGenerateSenseNova` |
| ModelSamplingSenseNova | `ModelSamplingSenseNova` |
| SenseNova Interleave Decode | `SenseNovaInterleaveDecode` |
| SenseNova-U1 Inference Alignment | `SenseNovaU1InferenceAlignment` |

Ordinary T2I and editing connect native `CLIPTextEncode` conditioning directly to
`KSampler`; they do not use Generate. The thinking variants select
`task=image thinking`. Interleave selects `task=interleave`, with its nested
thinking toggle. Self-correction selects `task=self-correction` and has no
thinking toggle. The task also selects the prompt template for both positive
and negative conditioning.

For API prompts, the dynamic inputs use `task`, `task.thinking`,
`task.negative`, and `task.samples`; the token budget is `max_tokens`.
`task.thinking` is used only for interleave. Negative conditioning and previous
samples are used by interleave/self-correction; image thinking leaves the
negative connected directly to the sampler.

Use these templates with the task-based node schema documented above.
