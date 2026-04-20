# Prompt Enhancement for SenseNova-U1

> Short user prompts — especially for **infographic** generation —
> often under-constrain the image model. Running the raw prompt through a
> strong LLM enhancer first consistently lifts structure, typography,
> information density, and "brief-readability" of the final image. This
> document describes how to turn it on, which upstream LLMs we recommend,
> and what the tradeoffs look like.

## 1. When to use

Use `--enhance` when:

- The user prompt is short or only names a topic (e.g. `"A chart about AI hardware in 2026"`).
- You are generating for demo / deck / poster use and can afford one extra
  LLM round-trip before the T2I call.

Skip `--enhance` when:

- The user already supplies a long, structured, production-ready prompt.
- Latency or third-party API cost is the primary concern.


## 2. How it works

```
user prompt ──► LLM (system prompt = infographic expander) ──► expanded prompt ──► SenseNova-U1
```

Upstream system prompt: [SenseNova-Skills / u1-infographic](https://github.com/OpenSenseNova/SenseNova-Skills/blob/main/skills/u1-infographic/references/prompts-expand-system.md).

## 3. Configuration

All configuration is environment-variable based so the same script can
switch backends without code changes.

| Env var | Default | Purpose |
| :------ | :------ | :------ |
| `U1_ENHANCE_BACKEND`  | `chat_completions` | `chat_completions` (OpenAI-compatible) or `anthropic` |
| `U1_ENHANCE_ENDPOINT` | Gemini OpenAI-compat URL | Full `/chat/completions` or `/v1/messages` URL |
| `U1_ENHANCE_MODEL`    | `gemini-3.1-pro`   | Model name string sent in the request body |
| `U1_ENHANCE_API_KEY`  | _unset_            | Bearer token (required) |

Then just add `--enhance` to your `examples/t2i/inference.py` command line.
Add `--print_enhance` to echo the original + enhanced prompt for
debugging.

### 3.1 Recommended backends

| Model | Backend | Endpoint template | Notes |
| :---- | :------ | :---------------- | :---- |
| **Gemini 3.1 Pro** (default) | `chat_completions` | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | Best overall infographic quality in our internal bench. Excellent at structured / hierarchical content. |
| SenseNova Agentic model | `chat_completions` | _(will be released soon)_ | Comparable to Gemini 3.1 Pro on zh content, cheaper per-token, preferred for production. |
| Anthropic Claude (Sonnet/Opus) | `anthropic`        | `https://api.anthropic.com/v1/messages` | Strong typography discipline, slightly less "information-dense" out of the box. |
| Kimi 2.5                      | `chat_completions` | `https://api.moonshot.cn/v1/chat/completions` | Good Chinese enhancements, weaker for English-dense infographics in our runs. |

## 4. Qualitative comparison (TODO – fill after release benchmarks)

> The table below will be populated with side-by-side samples from the same
> handful of base prompts, rendered at `2048×2048` with identical sampler
> knobs. PRs with new backends welcome.

| Base prompt | No enhance | Gemini 3.1 Pro | Internal Agentic | Claude | Kimi 2.5 |
| :---------- | :--------- | :------------- | :--------------- | :----- | :------- |
| _(infographic topic 1)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _(infographic topic 2)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _(infographic topic 3)_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
