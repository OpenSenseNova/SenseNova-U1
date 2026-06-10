# LoRA fine-tuning (style adapters) for SenseNova-U1 8B

A self-contained recipe for training a small **style LoRA** on top of the
released `SenseNova-U1-8B-MoT-SFT` checkpoint — e.g. Pixar, Studio Ghibli, or
any style you can collect a few hundred example images of.

Only the **flow-matching generation branch** (`fm_modules.*`) is adapted —
everything else (the LLM, the ViT, the MLP adapter) stays frozen. That keeps
the trainable parameter count to a few million, fits comfortably on a
single 8 × 80 GB node, and produces a checkpoint that is tens of MB rather
than tens of GB.

All commands below are run from the `training/` directory.

---

## 1. Prepare the data

Drop your style images into a folder — anything Pillow can open
(`.png/.jpg/.jpeg/.webp/.bmp/.tiff`) is fine.

Optional: drop a sidecar `image.txt` next to each `image.png` containing a
caption — the [Kohya/Civitai](https://github.com/kohya-ss/sd-scripts)
convention. If a sidecar is missing the script can auto-caption with BLIP.
(For Civitai-sourced sets, the curated example-image prompts make good
sidecar captions.)

```bash
# minimum invocation — captions come from the trigger word + sidecar .txt
python tools/prepare_lora_dataset.py \
    --image_dir   /path/to/pixar_images \
    --output_dir  data/pixar_lora \
    --trigger_word "in pixar style" \
    --dataset_name pixar_style \
    --repeat_time 20

# with BLIP auto-captions (recommended for sets without sidecars)
pip install transformers safetensors pillow
python tools/prepare_lora_dataset.py \
    --image_dir   /path/to/pixar_images \
    --output_dir  data/pixar_lora \
    --trigger_word "in pixar style" \
    --auto_caption blip \
    --dataset_name pixar_style \
    --repeat_time 20
```

The script writes:

- `data/pixar_lora/images/` — symlinks (or copies) of the source images.
- `data/pixar_lora/annotations.jsonl` — one JSON object per sample.
- `data/pixar_lora/pixar_style_meta.json` — top-level meta JSON that you point `mm_data_path` at.

A reasonable `repeat_time` for a 100-image set is **20**; for 500 images, **5**.

---

## 2. Launch training

The launcher is a near-copy of `8B.sh` with the LoRA knobs flipped on and the
freeze flags set so only adapters update.

```bash
# point at your data + base checkpoint, then run
export MODEL_NAME_OR_PATH=/path/to/SenseNova-U1-8B-MoT-SFT
export VOCAB_FILE=/path/to/qwen3/tokenizer
export TOKENIZER_PATH=/path/to/qwen3/tokenizer
export mm_data_path=data/pixar_lora/pixar_style_meta.json

bash shell/train_u1/8B_lora.sh
```

Override anything else from the command line — these are the knobs most
worth tweaking:

| Env var               | Default       | Notes |
|-----------------------|---------------|-------|
| `lora_r`              | `16`          | Rank. 8–32 covers most style fine-tunes. |
| `lora_alpha`          | `32`          | Effective scale = `alpha / r`. Keep the ratio ≈ 2 unless you know why. |
| `lora_dropout`        | `0.0`         | Set to 0.05 if you see overfitting on a small set. |
| `lora_target_prefixes`| `fm_modules.` | Comma-separated. Add `language_model.layers.` to also adapt the LLM — much more capacity, much slower. |
| `lr`                  | `1e-4`        | LoRA is more LR-tolerant than full fine-tunes — 1e-4 to 5e-4 are all reasonable. |
| `total_steps`         | `5000`        | For ~100 images × repeat_time=20, 3-5k steps usually converges. |

A successful run prints, soon after start-up:

```
[LoRA] wrapped N Linear layers; trainable=4,xxx,xxx / total=8,xxx,xxx,xxx (0.0xx%)
```

If `trainable=0`, double-check `lora_target_prefixes` and the
`target_leaf_names` list in the config — your model variant might use a
different naming convention.

Checkpoints land in `RUN/$JOB_NAME/<timestamp>/<step>/lora_state.pt` (tens
of MB each) instead of the usual multi-shard model dump.

---

## 3. Export to HuggingFace / PEFT format

```bash
python tools/export_lora_to_hf.py \
    --src RUN/sensenovau1_8b_lora_pixar/<timestamp>/<step>/lora_state.pt \
    --tgt outputs/pixar_lora_hf \
    --base_model_name_or_path SenseNova/SenseNova-U1-8B-MoT-SFT
```

Produces `outputs/pixar_lora_hf/{adapter_model.safetensors, adapter_config.json}`,
loadable with:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "outputs/pixar_lora_hf")
```

The adapter's `target_modules` use the training-side module paths; if the
inference-side modeling file nests `fm_modules` differently, adjust the paths
in `adapter_config.json` accordingly.

---

## 4. Limitations / gotchas

- The shipped LoRA config only touches `fm_modules.*` — the understanding
  branch (CE-supervised) is fully frozen, so don't expect this LoRA to change
  captioning behavior.
- Under ISP weight-parallel (`wp_size=8`) the base linears are sharded, but
  each LoRA branch is built at the *full* layer dimensions and replicated on
  every rank (its gradients are reduced through the existing replica-param
  machinery). Cumulative HBM is `wp_size ×` per adapter — still negligible
  at r=16.
- Resuming (`auto_resume=true` or pointing `load_ckpt_folder` at a LoRA
  checkpoint) restores `lora_state.pt` instead of full model shards — the
  run must use the same `lora_*` settings, otherwise adapter shapes won't
  match.
- EMA is disabled under LoRA (it would shadow-copy and average the frozen
  base for no benefit).
