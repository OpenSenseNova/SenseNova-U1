# Parameter Breakdown

`SenseNova-U1-8B-MoT` contains roughly **8B understanding parameters** and
**8B generation parameters**. To avoid confusion caused by the naming and to
present the architecture more accurately, we provide a small inspection
script that parses parameter names of the loaded checkpoint and reports a
detailed parameter breakdown.

## Run the script

```bash
python scripts/inspect_model_params.py \
    --model_path sensenova/SenseNova-U1-8B-MoT
```

Useful argments:

- `--dtype {float32,float16,bfloat16}` (default: `bfloat16`) — load dtype. It
  does **not** affect parameter counts; it only affects the reported `memory`
  column, since each `bf16/fp16` element occupies 2 bytes versus 4 bytes for
  `fp32`.
- `--show_groups <name1,name2>` (default: `shared`) — list member parameters
  of the specified groups. Use `all` for every group, or an empty string to
  disable.
- `--custom_groups_json <path>` — override the default grouping rules with a
  JSON file of the form `{"group_name": ["prefix1", "prefix2"]}`.

## Example output

```text
Model: sensenova/SenseNova-U1-8B-MoT
Load dtype:   bfloat16
Total params: 17.552B
Total memory: 35.105GB (bfloat16)
---------------------------------------------------------------------
group                              params memory (bfloat16)      ratio
---------------------------------------------------------------------
generation_transformer             8.186B         16.373GB     46.64%
understanding_transformer          8.121B         16.243GB     46.27%
shared                             1.245B          2.489GB      7.09%
---------------------------------------------------------------------
Pathway breakdown (shared counted in both):
---------------------------------------------------------------------
pathway                            params memory (bfloat16)      ratio
---------------------------------------------------------------------
understanding pathway              9.366B         18.732GB     53.36%
generation pathway                 9.431B         18.862GB     53.73%

---------------------------------------------------------------------
Members of group 'shared' (2 params, 1.245B total, 2.489GB @ bfloat16)
---------------------------------------------------------------------
param name                                                  numel    dtype
---------------------------------------------------------------------
language_model.model.embed_tokens.weight                 622.330M bfloat16
language_model.lm_head.weight                            622.330M bfloat16
```

## How to read it

### 1. Parameters (mutually exclusive, sums to 100%)

Each parameter is counted exactly once and assigned to one of three groups
based on its module path:

- `understanding_transformer` ≈ **8.12B (46%)** — vision und.
  (`vision_model.*`) plus the LLM expert without `_mot_gen` suffix
  (`language_model.*` minus the generation expert and the shared text I/O).
- `generation_transformer` ≈ **8.19B (47%)** — generation-side modules
  (`fm_modules.*`: vision gen., flow-matching head, timestep / noise
  embedders) plus the LLM expert with `_mot_gen` suffix
  (`language_model.*` containing `_mot_gen`).
- `shared` ≈ **1.25B (7%)** — text-token I/O reused by both pathways:
  `language_model.model.embed_tokens` and `language_model.lm_head`.

### 2. Pathway coverage (forward activations, ratios sum to >100%)

A *pathway* sums the parameters that are actually traversed during the forward pass of one task.
Because both tasks reuse the `shared` group, the ratios overlap and add up to more than 100%.

- **Understanding pathway** ≈ `understanding_transformer + shared` ≈ **9.37B (53%)**.
  Image goes through `vision_model` → tokens go through `embed_tokens` →
  the LLM runs on the `non-_mot_gen` expert → `lm_head` produces text logits.

- **Generation pathway** (single-turn thinking interleave) ≈
  `generation_transformer + shared` ≈ **9.43B (54%)**.
  The condition image goes through `fm_modules.vision_model_mot_gen`, while
  the text prompt goes through `embed_tokens` → the LLM runs on the
  `_mot_gen` expert → text reasoning is produced via `lm_head` and
  the image is decoded via `fm_modules.fm_head`.

### Why `embed_tokens` and `lm_head` are "shared", not "understanding-only"

`embed_tokens` is needed by every text token and is therefore obviously
shared. `lm_head` is also exercised by the generation pathway in some scenarios,
e.g., t2i-reasoning runs a thinking phase that emits text tokens **before** any image token is produced,
so `lm_head` is on the critical path of both pathways — hence the "shared" label.
