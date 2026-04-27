# Visual Understanding Evaluation

Reproduction guide for SenseNova-U1 on visual understanding benchmarks.

The pipeline is built on top of [EvalScope](https://github.com/OpenSenseNova/evalscope/tree/neo) (Native backend). EvalScope calls the model through an OpenAI-compatible HTTP endpoint and, for open-ended benchmarks, scores the predictions with an LLM judge.

Reference config and launcher live under `evaluation/understanding/`:

- `evaluation/understanding/config.yaml` — evaluation configuration
- `evaluation/understanding/es.py` — single-entry launcher

## 1. Overview

```
┌──────────────┐     OpenAI-compatible     ┌─────────────┐
│  es.py       │ ───── HTTP requests ────▶ │ Model API   │
│  (EvalScope) │                           │ (lightllm)  │
└──────┬───────┘ ◀──── generations ─────── └─────────────┘
       │
       ▼
   results/                 (predictions, judge scores, aggregated metrics)
```

1. Deploy SenseNova-U1 behind an OpenAI-compatible endpoint (the reference setup uses lightllm).
2. Fill in `config.yaml` with the endpoint, model name, datasets, and generation parameters.
3. Run `python es.py` — it calls `evalscope.run.run_task(task_cfg="config.yaml")`, which loops over the datasets, issues requests in parallel, and writes predictions plus scores under `results/`.

## 2. Launcher

`evaluation/understanding/es.py` is deliberately tiny:

```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```

Everything else is driven by `config.yaml`.

## 3. Benchmarks

The reference run evaluates on:

- `mmmu_pro`
- `mmlu_pro`
- `mm_bench`
- `ai2d`
- `math_vista`
- `ifeval`

Add or remove items under `datasets:` to extend the evaluation.

## 4. Main Generation Parameters

These are the parameters the model is sampled with. They live under `generation_config:` in `config.yaml` and are forwarded to the OpenAI-compatible API.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `stream` | `false` | Return the full response in one shot; simpler to score and log. |
| `temperature` | `0.6` | Sampling temperature — recommended setting for thinking-enabled models. |
| `top_p` | `0.95` | Nucleus sampling cutoff; used together with `top_k`. |
| `max_tokens` | `32768` | Upper bound on generated tokens per sample. Large because `<think>…</think>` traces can be long. |
| `timeout` | `300` | Per-request timeout in seconds. |
| `extra_body.top_k` | `20` | Restrict sampling to the top-20 tokens at each step. |
| `extra_body.repetition_penalty` | `1.05` | Mild penalty to suppress loops in long reasoning traces. |
| `extra_body.chat_template_kwargs.enable_thinking` | `true` | Let the chat template emit a `<think>…</think>` section before the final answer. |

Post-processing on the prediction:

- `dataset_args.remove_until: </think>` — everything up to and including the closing `</think>` tag is stripped before grading, so only the final answer is scored.
- `ignore_errors: true` — transient single-sample API failures do not abort the whole run.

## 5. Judge Model

Open-ended benchmarks are scored by an LLM judge.

| Field | Value |
| --- | --- |
| `judge_worker_num` | `64` (parallel judge calls) |
| `judge_model_args.model_id` | `gpt-4o-mini-2024-07-18` |
| `judge_model_args.api_key` | *(fill in)* |
| `judge_model_args.api_url` | *(fill in — OpenAI-compatible judge endpoint)* |
| `judge_model_args.generation_config.max_tokens` | `4096` |
| `judge_model_args.generation_config.timeout` | `300` |

The reference `config.yaml` leaves the judge `api_key` / `api_url` blank; fill them before running judge-dependent tasks.

## 6. Runtime Settings

| Field | Value | Meaning |
| --- | --- | --- |
| `eval_backend` | `Native` | EvalScope native backend. |
| `eval_type` | `openai_api` | Drive the model through an OpenAI-compatible endpoint. |
| `eval_batch_size` | `64` | In-flight concurrent requests sent to the model server. |
| `api_url` | `http://<host>:8000/v1/` | OpenAI-compatible serving endpoint (lightllm in the reference setup). |
| `model` | `SenseNova-U1` | Model name as exposed by the serving endpoint. |
| `use_cache` | `results/` | Reuse previously generated answers — supports resume / retry. |
| `work_dir` | `results/` | Output root for predictions, judgments, and scores. |
| `no_timestamp` | `true` | Write into a stable directory (plays well with `use_cache`). |

## 7. Reference `config.yaml`

```yaml
eval_backend: Native
eval_type: openai_api
eval_batch_size: 64
api_url: http://<host>:8000/v1/   # lightllm deployment
model: SenseNova-U1
datasets:
  - mmmu_pro
  - mmlu_pro
  - mm_bench
  - ai2d
  - math_vista
  - ifeval
dataset_args:
  remove_until: </think>
ignore_errors: true
generation_config:
  stream: false
  temperature: 0.6
  timeout: 300
  max_tokens: 32768
  top_p: 0.95
  extra_body:
    top_k: 20
    repetition_penalty: 1.05
    chat_template_kwargs:
      enable_thinking: true

judge_worker_num: 64
judge_model_args:
  api_key: ""
  api_url: ""
  model_id: gpt-4o-mini-2024-07-18
  generation_config:
    max_tokens: 4096
    timeout: 300
use_cache: results/
work_dir: results/
no_timestamp: true
```

## 8. Running the Evaluation

1. Deploy SenseNova-U1 on an OpenAI-compatible endpoint and confirm connectivity:

   ```bash
   curl -sSf -m 5 "$api_url"
   ```

2. Edit `evaluation/understanding/config.yaml`: set `api_url`, `model`, and the judge `api_key` / `api_url` if needed.
3. Launch:

   ```bash
   cd evaluation/understanding
   python es.py
   ```

Predictions, judge outputs, and final scores are written under `results/`. Because `use_cache: results/` and `no_timestamp: true` are set, rerunning the command skips already-answered samples, so interrupting and resuming is safe.
