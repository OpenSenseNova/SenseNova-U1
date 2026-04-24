# Image Generation Evaluation

Reproduction scripts for SenseNova-U1 on image generation benchmarks. Each
benchmark lives in its own subfolder under [`evaluation/gen/`](../gen/) and
ships with a generation script, an evaluation script, and a shell launcher
wiring them together:

```
evaluation/gen/
├── bizgeneval/              # BizGenEval — business / infographic prompts
│   ├── gen_images_bizgeneval.py
│   ├── eval_images_bizgeneval.py
│   ├── run_bizgeneval.sh
│   └── data/test.jsonl
├── igenbench/               # IGenBench — general-purpose T2I benchmark
│   ├── gen_images_igenbench.py
│   ├── eval_images_igenbench.py
│   └── run_igenbench.sh
├── longtext/                # LongText — long-text rendering (en / zh)
│   ├── gen_images_longtext.py
│   ├── eval_images_longtext.py
│   ├── run_longtext.sh
│   └── data/{text_prompts.jsonl,text_prompts_zh.jsonl}
├── cvtg/                    # CVTG-2K — complex visual text generation
│   ├── eval_cvtg.py
│   └── run_cvtgeval.sh
└── tiif/                    # TIIF-Bench — text-image instruction following
    ├── eval_tiif.py
    └── run_tiifeval.sh
```

Every benchmark follows the same two-stage flow: **generate images**, then
**evaluate them** (usually against an OpenAI-compatible judge model). The
shell launchers chain both stages, so the typical entry point is just:

```bash
bash evaluation/gen/<bench>/run_<bench>.sh
```

Edit the variables at the top of each launcher (model path, API key / base,
judge model, output dirs) before running.

## BizGenEval

Infographic / business-style prompts. Images are judged by an
OpenAI-compatible VLM (Gemini 3 Pro by default).

End-to-end:

```bash
bash evaluation/gen/bizgeneval/run_bizgeneval.sh
```

Or run the two stages manually:

```bash
# 1) Generate
python evaluation/gen/bizgeneval/gen_images_bizgeneval.py \
  --model-path sensenova/SenseNova-U1-8B-MoT-SFT \
  --output-dir outputs/sensenova/bizgeneval \
  --cfg-scale 4.0 --cfg-norm none --timestep-shift 3.0 --num-steps 50

# 2) Judge
python evaluation/gen/bizgeneval/eval_images_bizgeneval.py \
  --image-dir outputs/sensenova/bizgeneval \
  --output-dir outputs/sensenova/bizgeneval_eval \
  --api-base  http://your-api-base/v1 \
  --api-key   sk-... \
  --judge-model gemini-3-pro-preview \
  --concurrency 8
```

Prompts are loaded from [`bizgeneval/data/test.jsonl`](../gen/bizgeneval/data/test.jsonl).
The summary (per-item scores + aggregate) is written under `--output-dir`.

## IGenBench

General-purpose T2I benchmark with direct image-question judging.

```bash
bash evaluation/gen/igenbench/run_igenbench.sh
```

Manual:

```bash
python evaluation/gen/igenbench/gen_images_igenbench.py \
  --model-path sensenova/SenseNova-U1-8B-MoT-SFT \
  --output-dir outputs/sensenova/igenbench \
  --cfg-scale 4.0 --cfg-norm none --timestep-shift 3.0 --num-steps 50

python evaluation/gen/igenbench/eval_images_igenbench.py \
  --image-dir outputs/sensenova/igenbench \
  --output-dir outputs/sensenova/igenbench_eval \
  --api-base  http://your-api-base/v1 \
  --api-key   sk-... \
  --judge-model gemini-3-pro-preview \
  --concurrency 128
```

Set `--gen-model-name` to tag the judgments with a custom identifier (useful
when comparing multiple generators under the same `--output-dir`).

## LongText

Long-text rendering benchmark, run separately for English (`--lang en`) and
Chinese (`--lang zh`). The launcher executes both passes back to back:

```bash
bash evaluation/gen/longtext/run_longtext.sh
```

Manual (single language):

```bash
python evaluation/gen/longtext/gen_images_longtext.py \
  --model-path sensenova/SenseNova-U1-8B-MoT-SFT \
  --output-dir outputs/longtext/en \
  --lang en \
  --cfg-scale 4.0 --cfg-norm none --timestep-shift 3.0 --num-steps 50

python evaluation/gen/longtext/eval_images_longtext.py \
  --image-dir  outputs/longtext/en \
  --output-dir outputs/longtext/en_eval \
  --mode en
```

Evaluation runs OCR + text-match locally, so no judge API is required.
Prompts live in [`longtext/data/`](../gen/longtext/data/) (`text_prompts.jsonl`
for `en`, `text_prompts_zh.jsonl` for `zh`).

## CVTG-2K

Complex visual text generation at 2K resolution, evaluated with the
TextCrafter metrics suite (PaddleOCR + unified metrics). The launcher
covers multi-GPU and multi-node sharding, plus an optional eval stage.

```bash
BENCHMARK_ROOT=/path/to/CVTG-2K \
TEXTCRAFTER_ROOT=/path/to/TextCrafter_Eval \
  bash evaluation/gen/cvtg/run_cvtgeval.sh
```

Common overrides (set as env vars before the launcher):

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `MODEL_PATH` | `sensenova/SenseNova-U1-8B-MoT-SFT` | Local checkpoint path or HF model id |
| `BENCHMARK_ROOT` | — (required) | CVTG-2K dataset root |
| `OUTPUT_DIR` | `<repo>/outputs/sensenova/cvtg` | Generated-image + results dir |
| `TEXTCRAFTER_ROOT` | — (required when `RUN_EVAL=1`) | Upstream TextCrafter eval source |
| `PADDLEOCR_SOURCE_DIR` | — | Pre-downloaded PaddleOCR cache (copied to `$HOME/.paddleocr` if missing) |
| `IMAGE_SIZE` / `CFG_SCALE` / `TIMESTEP_SHIFT` / `NUM_STEPS` | `2048` / `7.0` / `1.0` / `50` | Sampling config |
| `CVTG_SUBSETS` / `CVTG_AREAS` | `CVTG,CVTG-Style` / `2,3,4,5` | Which splits to run |
| `LAUNCH_MODE` | `device_map_multi` | `device_map`, `device_map_multi`, or `ddp` |
| `GPUS` / `CUDA_VISIBLE_DEVICES` / `GPUS_PER_WORKER` | `8` / `0..7` / `2` | GPU layout |
| `NUM_NODES` / `NODE_RANK` | `1` / `0` | Multi-node sharding |
| `RUN_GENERATION` / `RUN_EVAL` | `1` / `1` (auto-`0` on non-rank-0 nodes) | Stage toggles |

Example — 8-GPU single-node run, generation only:

```bash
RUN_EVAL=0 GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BENCHMARK_ROOT=/path/to/CVTG-2K \
  bash evaluation/gen/cvtg/run_cvtgeval.sh
```

Generated images land under `$OUTPUT_DIR/<subset>/<area>/<key>.png`, and the
aggregated metrics are written to `$OUTPUT_DIR/CVTG_results.json`.

## TIIF-Bench

Text-image instruction following benchmark, evaluated with a GPT-4o-class
judge via `eval/eval_with_vlm_mp.py` from the upstream TIIF-Bench repo.

```bash
TIIF_BENCH_ROOT=/path/to/TIIF-Bench API_KEY=sk-... \
  bash evaluation/gen/tiif/run_tiifeval.sh
```

Required / common overrides:

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `MODEL_PATH` | `sensenova/SenseNova-U1-8B-MoT-SFT` | Local checkpoint path or HF model id |
| `TIIF_BENCH_ROOT` | — (required) | Upstream TIIF-Bench repo |
| `OUTPUT_DIR` | `<repo>/outputs/sensenova/tiif` | Generated-image + results dir |
| `TIIFBENCH_SPLIT` | `testmini` | Which split to run (`testmini` / `test`) |
| `TIIFBENCH_EVAL_MODEL` | `gpt-4o` | Judge model |
| `API_KEY` (+ optional `TIIFBENCH_AZURE_ENDPOINT` / `TIIFBENCH_API_VERSION`) | — | Judge API credentials |
| `IMAGE_SIZE` / `CFG_SCALE` / `CFG_NORM` / `TIMESTEP_SHIFT` / `NUM_STEPS` | `1024` / `4.0` / `global` / `3.0` / `50` | Sampling config |
| `GPUS` / `CUDA_VISIBLE_DEVICES` | `8` / `0..7` | GPU layout (generation uses `torchrun`) |
| `NUM_NODES` / `NODE_RANK` | `1` / `0` | Multi-node sharding (eval runs only on node 0) |
| `RUN_GENERATION` / `RUN_EVAL` | `1` / `1` | Stage toggles |

Example — single-node generation + eval against an Azure OpenAI endpoint:

```bash
API_KEY=sk-... \
TIIFBENCH_AZURE_ENDPOINT=https://your-endpoint.openai.azure.com \
TIIF_BENCH_ROOT=/path/to/TIIF-Bench \
MODEL_PATH=/path/to/checkpoint \
  bash evaluation/gen/tiif/run_tiifeval.sh
```

Per-question judgments are written to `$OUTPUT_DIR/tiifbench-<split>_results/eval_json/`,
with a dimension-level summary in `result_summary_dimension.txt` next to it.

## Tips

- **Sampling config.** Defaults mirror the values used in the SenseNova-U1
  tech report. CVTG-2K in particular expects 2048-pixel outputs — lower
  resolutions will not be comparable.
- **Judge APIs.** All API-based evaluators accept any OpenAI-compatible
  endpoint — point them at SenseNova, Gemini (OpenAI-compat), Azure OpenAI,
  or a local vLLM / sglang server as needed.
- **Sharding.** `run_cvtgeval.sh` and `run_tiifeval.sh` accept
  `NUM_NODES` / `NODE_RANK` for multi-node generation; the remaining
  benchmarks are single-process and scale by running multiple invocations
  against disjoint `--output-dir`s.
- **Re-evaluation.** `eval_images_bizgeneval.py` / `eval_images_igenbench.py`
  skip items whose judgments already exist in `--output-dir`. Pass
  `--force-rerun` to ignore the cache.
