# Examples

Reference inference scripts for SenseNova-U1. Every script here is intentionally
self-contained — on top of the `sensenova_u1` package itself it only pulls in
`torch`, `transformers`, `pillow`, `numpy` (and optionally `tqdm` /
`flash-attn`).

Each task lives in its own subfolder with a matching `data/` directory of
sample inputs:

```
examples/
├── README.md
├── t2i/                       # text-to-image
│   ├── inference.py
│   └── data/
│       └── samples.jsonl
├── editing/                   # image editing (it2i)
│   ├── inference.py
│   └── data/
│       └── samples.jsonl
├── interleave/                # interleaved generation
│   ├── inference.py
│   └── data/
│       └── samples.jsonl
└── vqa/                       # visual understanding / VQA
    ├── inference.py
    └── data/
        └── questions.jsonl
```

## Text-to-Image

Single prompt:

```bash
python examples/t2i/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "一个咖啡店门口有一个黑板，上面写着日日新咖啡，2元一杯，旁边有个霓虹灯，写着商汤科技，旁边有个海报，海报上面是一只小浣熊，海报下方写着SenseNova newbee。" \
  --width 2048 --height 2048 \
  --output out.png \
  --profile
```

Batched prompts from a JSONL file (each line must contain a `prompt`;
`width` / `height` / `seed` are optional):

```bash
python examples/t2i/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --profile
```

See [`t2i/data/samples.jsonl`](./t2i/data/samples.jsonl) for a tiny starter
file. Supported resolution buckets and the full CLI flag list live in the
top-level [README](../README.md#text-to-image).

## Image Editing (it2i)

Single edit:

```bash
python examples/editing/inference.py \
  --model_path OpenSenseNova/SenseNova-U1-Mini \
  --prompt "Turn the background into a starry night sky." \
  --image path/to/input.jpg \
  --output edited.png \
  --profile
```

Batched edits from a JSONL file (each line must contain a `prompt` and
`image` path; `seed` / `type` are optional; `image` can also be a list of
paths to pass multiple reference images):

```bash
python examples/editing/inference.py \
    --model_path OpenSenseNova/SenseNova-U1-Mini \
    --jsonl examples/editing/data/samples.jsonl \
    --output_dir outputs/editing/ \
    --profile
```

Output resolution is decoupled from the input and has two modes:

- **Auto (default)**: omit `--width / --height` and the output tracks the
  input via `smart_resize` — aspect ratio preserved, total pixels clamped
  to `[--min_pixels, --max_pixels]`. This is the natural "edit in place" default.
- **Explicit**: pass `--width W --height H` (both multiples of 32, the
  image-token grid factor). Useful for re-aspecting / resizing during the
  edit; **2048 × 2048** is a good general-purpose choice and matches the
  t2i recommendation.

JSONL mode additionally honors per-sample `width` + `height` fields when
both are present; they override the CLI default for that line.

CFG defaults: `--cfg_scale 4.0` (text guidance), `--img_cfg_scale 1.0` (image CFG **off** by default).
