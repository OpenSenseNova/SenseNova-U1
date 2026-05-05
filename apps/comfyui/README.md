# SenseNova-U1 for ComfyUI

ComfyUI custom nodes for SenseNova-U1 API and local inference.

## Nodes

- `SenseNova Image Generate`: calls the U1-Fast image API.
- `SenseNova Chat`, `SenseNova Vision URL`, `SenseNova Vision Image`: utility API nodes.
- `SenseNova Prompt Builder`: rewrites raw ideas into image-generation prompts.
- `SenseNova U1 Local Loader`: loads a local or HuggingFace SenseNova-U1 checkpoint.
- `SenseNova U1 Local Text to Image`: runs local `t2i_generate`.
- `SenseNova U1 Local Image Edit`: runs local `it2i_generate`.
- `SenseNova U1 Local Interleave`: runs local `interleave_gen`.
- `SenseNova Interleave Preview`: renders ordered interleaved text / image results.

## Install

From the SenseNova-U1 repository:

```bash
python apps/comfyui/install.py --comfyui /path/to/ComfyUI
```

By default this creates:

```text
/path/to/ComfyUI/custom_nodes/ComfyUI-SenseNova-U1 -> /path/to/SenseNova-U1/apps/comfyui
```

Install the lightweight ComfyUI app dependencies in the Python environment used by ComfyUI:

```bash
python -m pip install -r apps/comfyui/requirements.txt
```

For local inference, make sure the SenseNova-U1 runtime dependencies are also installed in the
same environment. When using this app from the main SenseNova-U1 checkout, the loader can discover
`src/` automatically. You can override it if needed:

```bash
python -m pip install -e .
export SENSENOVA_U1_SRC="/path/to/SenseNova-U1/src"
```

Restart ComfyUI after installation.

## Workflows

Example workflows live in `workflows/`:

- `api_u1_fast_t2i.json`: API U1-Fast text-to-image.
- `local_t2i.json`: local SenseNova-U1 text-to-image.
- `local_editing.json`: local SenseNova-U1 image editing.
- `local_interleave.json`: local SenseNova-U1 interleaved generation.

Drag a workflow JSON into ComfyUI, then update `model_path`, `device`, `device_map`, and prompt
settings as needed. For a smoke test, set `num_steps` to `1` or `2` before returning to the
recommended `50`.

## API Environment

API nodes read credentials from environment variables or `.env`:

```bash
export SN_API_KEY="your-api-token"
export SN_BASE_URL="https://token.sensenova.cn/v1"
```

Tokens are not exposed as node inputs, so they are not saved into ComfyUI workflows.

## Notes On Samplers

Local U1 generation uses the sampling loop implemented by `t2i_generate`, `it2i_generate`, and
`interleave_gen`. It does not directly plug into ComfyUI's `KSampler` / latent model interface.
You can still reuse ComfyUI image IO and post-processing nodes around these U1 nodes.
