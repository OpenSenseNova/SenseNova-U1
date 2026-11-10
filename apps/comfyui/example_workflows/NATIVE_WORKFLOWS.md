# Native SenseNova ComfyUI workflows

These workflows use the SenseNova implementation inside ComfyUI: a converted
single-file checkpoint, `CheckpointLoaderSimple`, and the standard sampling nodes.
Use ComfyUI branch `gp/sensenova-clean-20261028` at commit `0e13670d` or a revision
containing those changes. Installing this repository's custom-node plugin alone
does not provide these native nodes.

## U1.5 templates

| Task | UI workflow: drag into ComfyUI | API prompt graph |
| --- | --- | --- |
| Text to image | [image-sensenova-text2image.json](image-sensenova-text2image.json) | [comfyui-sensenova-t2i.json](comfyui-sensenova-t2i.json) |
| Image editing | [image-sensenova-image-edit.json](image-sensenova-image-edit.json) | [comfyui-sensenova-edit.json](comfyui-sensenova-edit.json) |
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

The seven files in [native-u16/](native-u16/) are copied unchanged from ComfyUI
commit `0e13670d`, under `script_examples/sensenova_u16/`. These are API prompt
graphs; the table above contains the UI templates.

| API graph | Purpose | Checkpoint |
| --- | --- | --- |
| [t2i-api.json](native-u16/t2i-api.json) | Text to image | step1200 |
| [edit-api.json](native-u16/edit-api.json) | Image editing | step1200 |
| [interleave-api.json](native-u16/interleave-api.json) | Historical U1.5-style interleave protocol diagnostic | step1200 |
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

For self-correction, both encoders use `mode=image`, `thinking=false`, and
Generate uses `generation_mode=interleave`. Normal outputs are one image, or
image → correction text → second image. The older three-image diagnostic is
not a self-correction acceptance criterion.

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
ComfyUI's history/output view to inspect the result. The maintained replay script
and numerical comparison notes remain in the ComfyUI repository under
`script_examples/sensenova_u16/`.

## Node names

| Display name | ID |
| --- | --- |
| Text Encode SenseNova | `TextEncodeSenseNova` |
| Generate SenseNova Text | `TextGenerateSenseNova` |
| ModelSamplingSenseNova | `ModelSamplingSenseNova` |
| SenseNova Interleave Decode | `SenseNovaInterleaveDecode` |
| SenseNova-U1 Inference Alignment | `SenseNovaU1InferenceAlignment` |

Encoder selects the prompt protocol and thinking prefix. Generate produces text
and the next image request; a sampler generates the pixels. Old IDs can be
migrated using ComfyUI's node replacement mechanism, but all native templates
here use the final IDs and explicitly set `generation_mode`.
