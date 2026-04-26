# 视觉理解评测

SenseNova-U1 在视觉理解 benchmark 上的复现指南。

评测流程基于 [EvalScope](https://github.com/OpenSenseNova/evalscope/tree/neo) 的 Native 后端：EvalScope 通过 OpenAI 兼容接口调用被测模型，并在开放性任务上使用 LLM judge 进行打分。

参考配置与启动脚本位于 `evaluation/understanding/`：

- `evaluation/understanding/config.yaml` — 评测配置
- `evaluation/understanding/es.py` — 一键启动脚本

## 1. 总体流程

```
┌──────────────┐    OpenAI 兼容 HTTP    ┌─────────────┐
│  es.py       │ ─────── 请求 ──────▶ │  模型服务   │
│ (EvalScope)  │                      │ (lightllm)  │
└──────┬───────┘ ◀─────── 回复 ─────── └─────────────┘
       │
       ▼
   results/                 （预测、judge 打分、聚合指标）
```

1. 将 SenseNova-U1 部署为 OpenAI 兼容接口（参考部署使用 lightllm）。
2. 在 `config.yaml` 中填写接口、模型名、数据集与生成参数。
3. 运行 `python es.py`，内部调用 `evalscope.run.run_task(task_cfg="config.yaml")`，按 `datasets:` 顺序并发下发请求，并把预测与分数写入 `results/`。

## 2. 启动脚本

`evaluation/understanding/es.py` 非常简单：

```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```

其余行为全部由 `config.yaml` 决定。

## 3. 评测基准

参考配置中评测以下 benchmark：

- `mmmu_pro`
- `mmlu_pro`
- `mm_bench`
- `ai2d`
- `math_vista`
- `ifeval`

如需扩展，直接在 `datasets:` 下增减条目即可。

## 4. 主要生成参数

下列参数控制被测模型的采样行为，写在 `config.yaml` 的 `generation_config:` 下，会被转发到 OpenAI 兼容接口。

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `stream` | `false` | 一次性返回完整回复，便于打分与日志记录。 |
| `temperature` | `0.6` | 采样温度，推理（thinking）型模型的推荐值。 |
| `top_p` | `0.95` | Nucleus 采样阈值，与 `top_k` 配合使用。 |
| `max_tokens` | `32768` | 单条样本生成上限；因 `<think>…</think>` 推理段较长，需设较大值。 |
| `timeout` | `300` | 单条请求超时时间（秒）。 |
| `extra_body.top_k` | `20` | 每步从概率最高的 20 个 token 中采样。 |
| `extra_body.repetition_penalty` | `1.05` | 轻量重复惩罚，抑制长链式推理中的循环。 |
| `extra_body.chat_template_kwargs.enable_thinking` | `true` | 让 chat template 输出 `<think>…</think>` 推理段，再给出最终答案。 |

预测后处理：

- `dataset_args.remove_until: </think>`：打分前截掉直到 `</think>` 为止的内容，仅对最终答案评分。
- `ignore_errors: true`：单条样本的偶发 API 错误不会中断整体评测。

## 5. Judge 模型

开放性 benchmark 使用 LLM judge 打分。

| 字段 | 取值 |
| --- | --- |
| `judge_worker_num` | `64`（judge 并发数） |
| `judge_model_args.model_id` | `gpt-4o-mini-2024-07-18` |
| `judge_model_args.api_key` | *（需填写）* |
| `judge_model_args.api_url` | *（需填写 —— OpenAI 兼容 judge 接口）* |
| `judge_model_args.generation_config.max_tokens` | `4096` |
| `judge_model_args.generation_config.timeout` | `300` |

参考 `config.yaml` 中 judge 的 `api_key` / `api_url` 为空，运行依赖 judge 的任务前请补齐。

## 6. 运行时配置

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `eval_backend` | `Native` | EvalScope 原生后端。 |
| `eval_type` | `openai_api` | 通过 OpenAI 兼容接口驱动模型。 |
| `eval_batch_size` | `64` | 发给模型服务的并发请求数。 |
| `api_url` | `http://<host>:8000/v1/` | OpenAI 兼容服务地址（参考部署为 lightllm）。 |
| `model` | `SenseNova-U1` | 模型服务端暴露的模型名。 |
| `use_cache` | `results/` | 复用已有结果，支持断点续评。 |
| `work_dir` | `results/` | 预测、评判、分数的输出根目录。 |
| `no_timestamp` | `true` | 写入固定目录而非带时间戳的目录（便于与 `use_cache` 联动）。 |

## 7. 参考 `config.yaml`

```yaml
eval_backend: Native
eval_type: openai_api
eval_batch_size: 64
api_url: http://<host>:8000/v1/   # lightllm 部署
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

## 8. 执行评测

1. 将 SenseNova-U1 部署为 OpenAI 兼容接口，并确认连通性：

   ```bash
   curl -sSf -m 5 "$api_url"
   ```

2. 编辑 `evaluation/understanding/config.yaml`：设置 `api_url`、`model`，如需 judge 再补齐 `api_key` / `api_url`。
3. 启动：

   ```bash
   cd evaluation/understanding
   python es.py
   ```

预测、judge 结果与最终分数都会写入 `results/`。由于启用了 `use_cache: results/` 与 `no_timestamp: true`，重跑会跳过已完成样本，可以安全地中断再恢复。
