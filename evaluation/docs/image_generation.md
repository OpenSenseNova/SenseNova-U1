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
│   ├── run_igenbench.sh
│   └── data/*.json
├── longtext/                # LongText — long-text rendering (en / zh)
│   ├── gen_images_longtext.py
│   ├── eval_images_longtext.py
│   ├── run_longtext.sh
│   └── data/{text_prompts.jsonl,text_prompts_zh.jsonl}
├── cvtg/                    # CVTG-2K — complex visual text generation
│   ├── eval_cvtg.py
│   ├── unified_metrics_eval.py
│   ├── sa_0_4_vit_l_14_linear.pth
│   ├── run_cvtgeval.sh
│   └── data/{CVTG,CVTG-Style}/{2..5}{,_combined}.json
└── tiif/                    # TIIF-Bench — text-image instruction following
    ├── eval_tiif.py
    ├── run_tiifeval.sh
    ├── eval/{eval_with_vlm_mp,summary_results,summary_dimension_results}.py
    └── data/{testmini,test}{_prompts,_eval_prompts}/*.jsonl
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

Prepare the IGenBench metadata from
[`Brookseeworld/IGenBench-Dataset`](https://huggingface.co/datasets/Brookseeworld/IGenBench-Dataset/tree/main)
and place the per-item JSON files under
[`igenbench/data/`](../gen/igenbench/data/). The scripts read those JSON files
directly for both generation prompts and evaluation questions.

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
in-tree [`unified_metrics_eval.py`](../gen/cvtg/unified_metrics_eval.py)
script (PaddleOCR-based word accuracy + unified visual-text metrics).
Generation runs as a single Python process with the model sharded across
visible GPUs via HuggingFace `device_map`.

```bash
bash evaluation/gen/cvtg/run_cvtgeval.sh
```

Prepare the CVTG-2K data from
[`dnkdnk/CVTG-2K`](https://huggingface.co/datasets/dnkdnk/CVTG-2K)
and place it under [`cvtg/data/`](../gen/cvtg/data/). The LAION
aesthetic-predictor head
[`sa_0_4_vit_l_14_linear.pth`](../gen/cvtg/sa_0_4_vit_l_14_linear.pth)
sits next to the eval script.

Common overrides (set as env vars before the launcher):

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `MODEL_PATH` | `sensenova/SenseNova-U1-8B-MoT-SFT` | Local checkpoint path or HF model id |
| `BENCHMARK_ROOT` | `evaluation/gen/cvtg/data` | CVTG-2K dataset root |
| `OUTPUT_DIR` | `<repo>/outputs/sensenova/cvtg` | Generated-image + results dir |
| `PADDLEOCR_SOURCE_DIR` | — | Pre-downloaded PaddleOCR cache (copied to `$HOME/.paddleocr` if missing) |
| `IMAGE_SIZE` / `CFG_SCALE` / `TIMESTEP_SHIFT` / `NUM_STEPS` | `2048` / `7.0` / `1.0` / `50` | Sampling config |
| `SAVE_SIZE` | unset (= `IMAGE_SIZE`) | Downsample with LANCZOS to this resolution before writing PNGs. Set to `1024` to use the "generate at 2048, evaluate at 1024" protocol. |
| `CVTG_SUBSETS` / `CVTG_AREAS` | `CVTG,CVTG-Style` / `2,3,4,5` | Which splits to run |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | GPUs available for model sharding |
| `DEVICE_MAP` / `MAX_MEMORY_PER_GPU_GB` | `auto` / `70` | HF `device_map` strategy and per-GPU memory cap |
| `RUN_GENERATION` / `RUN_EVAL` | `1` / `1` | Stage toggles |

Example — generation only:

```bash
RUN_EVAL=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash evaluation/gen/cvtg/run_cvtgeval.sh
```

Generated images land under `$OUTPUT_DIR/<subset>/<area>/<key>.png`, and the
aggregated metrics are written to `$OUTPUT_DIR/CVTG_results.json`. Re-runs
skip samples whose output PNG already exists, so an interrupted run can be
resumed by simply re-invoking the launcher.

## TIIF-Bench

Text-image instruction following benchmark, evaluated with a GPT-4o-class
judge via the in-tree
[`eval/eval_with_vlm_mp.py`](../gen/tiif/eval/eval_with_vlm_mp.py).

```bash
API_KEY=sk-... \
  bash evaluation/gen/tiif/run_tiifeval.sh
```

Prepare the TIIF-Bench data from
[`A113N-W3I/TIIF-Bench`](https://github.com/A113N-W3I/TIIF-Bench)
and place the prompts under
[`tiif/data/`](../gen/tiif/data/). The three eval helper scripts live under
[`tiif/eval/`](../gen/tiif/eval/).

Required / common overrides:

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `MODEL_PATH` | `sensenova/SenseNova-U1-8B-MoT-SFT` | Local checkpoint path or HF model id |
| `OUTPUT_DIR` | `<repo>/outputs/sensenova/tiif` | Generated-image + results dir |
| `TIIFBENCH_SPLIT` | `testmini` | Which split to run (`testmini` / `test`) |
| `TIIFBENCH_EVAL_MODEL` | `gpt-4o` | Judge model |
| `API_KEY` (+ optional `TIIFBENCH_AZURE_ENDPOINT` / `TIIFBENCH_API_VERSION`) | — | Judge API credentials |
| `IMAGE_SIZE` / `CFG_SCALE` / `CFG_NORM` / `TIMESTEP_SHIFT` / `NUM_STEPS` | `1024` / `4.0` / `global` / `3.0` / `50` | Sampling config |
| `SAVE_SIZE` | unset (= `IMAGE_SIZE`) | Downsample with LANCZOS to this resolution before writing PNGs. Set to `1024` (with `IMAGE_SIZE=2048`) to use the "generate at 2048, evaluate at 1024" protocol. |
| `GPUS` / `CUDA_VISIBLE_DEVICES` | `8` / `0..7` | GPU layout (generation uses `torchrun`) |
| `NUM_NODES` / `NODE_RANK` | `1` / `0` | Multi-node sharding (eval runs only on node 0) |
| `RUN_GENERATION` / `RUN_EVAL` | `1` / `1` | Stage toggles |

Example — single-node generation + eval against an Azure OpenAI endpoint:

```bash
API_KEY=sk-... \
TIIFBENCH_AZURE_ENDPOINT=https://your-endpoint.openai.azure.com \
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
- **Multi-GPU.** `run_tiifeval.sh` uses DDP (`torchrun --nproc_per_node`)
  with one full model replica per GPU. `run_cvtgeval.sh` instead shards a
  single model across GPUs via HF `device_map="auto"` — preferred when one
  GPU cannot hold the whole model. To scale further, run multiple
  invocations against disjoint `--output_dir`s and merge the results.
- **Re-evaluation.** `eval_images_bizgeneval.py` / `eval_images_igenbench.py`
  skip items whose judgments already exist in `--output-dir`. Pass
  `--force-rerun` to ignore the cache.
