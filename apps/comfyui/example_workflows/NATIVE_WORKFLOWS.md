# SenseNova ComfyUI workflows

Choose the workflow family that matches your ComfyUI installation:

| Family | Required implementation | Model loading |
| --- | --- | --- |
| Custom nodes | This repository's [`apps/comfyui`](../README.md) plugin | `SenseNova U1 Model Loader` / `SenseNova U1 Local Loader`, or the SenseNova image API |
| Native U1.5 | ComfyUI with [PR #16032](https://github.com/Comfy-Org/ComfyUI/pull/16032) | Converted single-file checkpoint with `CheckpointLoaderSimple` |
| Native U1.6 | The same native nodes, plus a compatible U1.6 checkpoint | Converted single-file checkpoint with `CheckpointLoaderSimple` |

Installing the custom-node plugin does not install the native nodes. The native
examples use `CLIPTextEncode`, standard sampling nodes, and the task-based
`TextGenerateSenseNova` interface from #16032. Its source branch is
[`GACLove/ComfyUI:gp/neo-unify-interleave`](https://github.com/GACLove/ComfyUI/tree/gp/neo-unify-interleave).

## File names and formats

- All examples use `image-sensenova-<task>.json` names with lowercase, hyphen-separated words.
- Custom-node examples include `custom`: `image-sensenova-custom-<task>.json`.
- A final `-api.json` suffix identifies a ComfyUI API prompt graph. Other JSON files are UI workflows that can be dragged into ComfyUI.
- Native U1.5 examples live in this directory; native U1.6 examples live in `native-u16/`.
- `with-image` means an external reference image. `thinking` means image generation or editing with a preceding text-generation stage.

The `custom-api-text2image` example calls the SenseNova image API; it is a UI
workflow, not a ComfyUI API prompt graph.

## Custom-node workflows

Install the plugin using its [installation guide](../README.md#install), then
drag one of these UI workflows into ComfyUI.

| Task | UI workflow | Inputs |
| --- | --- | --- |
| Local text to image | [Text to image](image-sensenova-custom-text2image.json) | U1.5 model; optional 8-step LoRA |
| Local image editing | [Image editing](image-sensenova-custom-image-edit.json) | U1.5 model and uploaded reference images |
| Local interleave | [Interleave](image-sensenova-custom-interleave.json) | Local/Hugging Face model supporting interleave |
| Infographic generation and editing | [Infographic](image-sensenova-custom-infographic-text2image-edit.json) | Infographic V3 model; uploaded image for the edit path |
| SenseNova API text to image | [API image generation](image-sensenova-custom-api-text2image.json) | API credentials; no local image model required |

Text-to-image and editing use `SenseNovaU1ModelLoader`. Interleave and the
infographic example use the plugin's supported `SenseNovaU1LocalLoader` interface.
Choose installed weights in the loader or set its Hugging Face model path, and
select your own files in `LoadImage`. These workflows use the repository's local
generation methods rather than native `KSampler`. API nodes, including the
optional prompt builder in the infographic example, require the
[API environment settings](../README.md#api-environment).

## Native U1.5 workflows — PR #16032

| Task | UI workflow | ComfyUI API prompt |
| --- | --- | --- |
| Text to image | [Text to image](image-sensenova-text2image.json) | [API](image-sensenova-text2image-api.json) |
| Text to image with thinking | [Text to image with thinking](image-sensenova-text2image-thinking.json) | [API](image-sensenova-text2image-thinking-api.json) |
| Image editing | [Image editing](image-sensenova-image-edit.json) | [API](image-sensenova-image-edit-api.json) |
| Image editing with thinking | [Image editing with thinking](image-sensenova-image-edit-thinking.json) | [API](image-sensenova-image-edit-thinking-api.json) |
| Interleave | [Interleave](image-sensenova-interleave.json) | [API](image-sensenova-interleave-api.json) |

Download the converted [U1.5 checkpoint and 8-step LoRA](https://huggingface.co/t8star/SenseNova-U1.5-Comfy)
and place the files under ComfyUI:

```text
models/checkpoints/SenseNova-U1.5-8B-MoT-BF16-T8.safetensors
models/loras/SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors
```

Select your matching filenames in the loader nodes. The templates use the
8-step LoRA with **8 steps and CFG 1**. Editing uses an uploaded reference in
`LoadImage`; API graphs expect `input/nova_editor_example.webp`.

The interleave example enables thinking. To use non-thinking interleave, disable
the nested `thinking` toggle under `task=interleave`; in an API graph, set
`task.thinking` to `false`. Reference images can be supplied through the native
`HiDreamO1ReferenceImages` node, connecting its positive/negative outputs to the
initial loop state in place of the text-only conditioning.

## Native U1.6 workflows

These are ComfyUI API prompt graphs using the same native nodes as #16032.
U1.6 weights and input images are not bundled.

| Task | ComfyUI API prompt | Checkpoint | Default settings |
| --- | --- | --- | --- |
| Text to image | [API](native-u16/image-sensenova-text2image-api.json) | step1200 | 1024 × 1024, 30 steps, CFG 4 |
| Image editing | [API](native-u16/image-sensenova-image-edit-api.json) | step1200 | 832 × 1248, 30 steps, CFG 1 |
| Generate and optionally self-correct | [API](native-u16/image-sensenova-self-correction-api.json) | step400 EMA | 2048 × 2048, 50 steps, CFG 4 |
| Self-correct with an external reference image | [API](native-u16/image-sensenova-self-correction-with-image-api.json) | step400 EMA | 2048 × 2048, 50 steps, CFG 4 |

Expected checkpoint filenames in `models/checkpoints/`:

- step1200: `SenseNova-U1.6-8B-MoT-BF16-step1200.safetensors`
- step400 EMA: `SenseNova-U1.6-8B-MoT-BF16-selfcorr-step400-ema.safetensors`

All four examples use Euler sampling, the simple scheduler, and shift 3.
T2I and self-correction use base seed 42; editing uses seed 45.
Editing expects `input/nova_editor_example.webp`. Self-correction with
an external reference expects `input/self-correction-input.png`; provide your
image and update the prompt to describe its intended content. Filenames can be
changed in the JSON.

Self-correction selects `task=self-correction`: the image-generation system
prompt and non-thinking interleave. Normal output is **one image**, or
**image → correction text → second image**. The checkpoint decides whether to
request another image. A reference image is input conditioning and is excluded
from the decoded generated images; supplying it does not guarantee exactly one
output image.

These examples use ordinary ComfyUI sampling. Official-inference alignment is
maintained separately in the
[frozen self-check branch](https://github.com/GACLove/ComfyUI/tree/gp/sensenova-u1-alignment).

## Native node connections

| Display name | Node ID | Role |
| --- | --- | --- |
| CLIP Text Encode (Prompt) | `CLIPTextEncode` | Encodes positive and negative prompts |
| Generate SenseNova Text | `TextGenerateSenseNova` | Generates reasoning or the next interleaved text stage and signals when an image is requested |
| ModelSamplingSenseNova | `ModelSamplingSenseNova` | Sets the model's timestep shift for native sampling |
| SenseNova Interleave Decode | `SenseNovaInterleaveDecode` | Extracts generated text, images, and thinking from the final history |

Ordinary T2I/edit connect encoded conditioning directly to `KSampler` and decode
with `VAEDecode`; they do not require Generate. Thinking variants insert Generate
with `task=image thinking`. Interleave selects `task=interleave`, with a thinking
toggle. Self-correction selects `task=self-correction`, with no thinking toggle.
Text generation requires a checkpoint containing `lm_head`; ordinary image-only
sampling can use checkpoints without it.

The native interleave and self-correction examples:

1. Feed the current positive/negative conditioning and previous samples to Generate.
2. Run `KSampler` only when `image_requested` is true, using `base seed + iteration_index`.
3. Return the new state through `EndLoop` with `accumulate=false`.
4. Connect the final conditioning and sampler output to `SenseNovaInterleaveDecode`, including its `samples` input.

For API graphs, dynamic inputs are named `task`, `task.thinking`, `task.negative`,
and `task.samples`; the token budget is `max_tokens`. `task.thinking` applies only
to interleave. Negative conditioning and previous samples belong to the
interleave/self-correction tasks; image thinking keeps negative conditioning
connected directly to the sampler.

## Submit a ComfyUI API graph

From the SenseNova-U1 repository root:

```bash
python - <<'PY'
import json
import urllib.request
from pathlib import Path

path = Path("apps/comfyui/example_workflows/native-u16/image-sensenova-self-correction-api.json")
request = urllib.request.Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": json.loads(path.read_text())}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
PY
```

Set the URL to your ComfyUI service. The response contains a `prompt_id`; inspect
the result in ComfyUI's history/output view.
