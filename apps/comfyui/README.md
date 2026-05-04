# ComfyUI SenseNova U1

ComfyUI custom nodes for SenseNova chat, vision, prompt building, and image generation APIs.

[中文文档](README.zh-CN.md)

## Nodes

- `SenseNova Chat`: sends text to a SenseNova chat model and returns text, usage JSON, and raw JSON.
- `SenseNova Prompt Builder`: uses an editable `system_prompt` defaulted from `prompts/builder_prompt.txt` to turn raw ideas, source text, or failed prompt feedback into an image generation prompt.
- `SenseNova Image Generate`: sends a prompt to the SenseNova image generation API and returns ComfyUI `IMAGE`, image base64, image URL, raw JSON, and a short tensor debug summary.
- `SenseNova Vision URL`: sends an image URL and prompt to the vision chat API and returns text, usage JSON, and raw JSON.
- `SenseNova Vision Image`: converts a ComfyUI `IMAGE` to a PNG base64 data URL, sends it to the vision chat API, and returns text, usage JSON, and raw JSON.

## Installation

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/OpenSenseNova/ComfyUI-SenseNova-U1.git
cd ComfyUI-SenseNova-U1
pip install -r requirements.txt
```

Restart ComfyUI after installing the dependencies.

## Token Plan And Environment Variables

This project currently uses the SenseNova token/API service. API tokens are read only from environment variables or a local `.env` file. They are not exposed as node inputs and should not be saved into ComfyUI workflows.

Before using these nodes:

1. Register or sign in on the SenseNova platform: https://platform.sensenova.cn/
2. Enable the required API service and models for your account.
3. Get your API key/token from the platform.
4. Read the SenseNova documentation if you need model, billing, or API details: https://platform.sensenova.cn/docs

Set your token before starting ComfyUI:

```bash
export SN_API_KEY="your-api-token"
```

The default API base URL is:

```text
https://token.sensenova.cn/v1
```

Override it if needed:

```bash
export SN_BASE_URL="https://token.sensenova.cn/v1"
```

For local development, you can also create a `.env` file:

```text
SN_API_KEY=your-api-token
SN_BASE_URL=https://token.sensenova.cn/v1
```

Do not commit `.env`.

## Supported Models

Chat:

- `sensenova-6.7-flash-lite`
- `deepseek-v4`

Vision:

- `sensenova-6.7-flash-lite`

Image:

- `sensenova-u1-fast`

`SenseNova Image Generate` always uses `n=1`. In the node UI, sizes are displayed as `widthxheight|aspect_ratio`, for example `2752x1536|16:9`. The API request only sends the `widthxheight` part.

## Prompt Builder

`prompts/builder_prompt.txt` is the default system prompt for both builder and refine workflows.

Use `SenseNova Prompt Builder` when you want a dedicated prompt-preparation node. Put the raw idea, source material, or failed prompt feedback into `prompt`, keep the default `system_prompt` or edit it, then connect the Builder `prompt` output to `SenseNova Image Generate.prompt`.

The lower-level `SenseNova Chat` node can do the same thing if you paste the same template into `SenseNova Chat.system_prompt`.

## Screenshot

`SenseNova Prompt Builder` can be connected directly to `SenseNova Image Generate`, whose `images` output connects to ComfyUI `Preview Image`.

![SenseNova Prompt Builder to Image Generate workflow](docs/demo.png)

## Dependency Policy

- `requirements.txt` lists only direct runtime dependencies and does not pin versions, so it is less likely to force upgrades or downgrades in an existing ComfyUI environment.
- `pyproject.toml` is the source for project metadata, development dependencies, and ruff.
- `torch` is not listed as a dependency because ComfyUI already provides PyTorch.
- Python requirement: `>=3.10`.

## Local Inference Placeholder

`deps/` is reserved for future local inference integrations. The current release does not include a local inference backend. Do not commit model weights, virtual environments, generated caches, or other large binary artifacts.

## Development

This project uses `uv` for development:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
```

`requirements.txt` is maintained manually as a broad compatibility file for ComfyUI users.

## License

MIT
