# SenseNova-U1

> Training code for **SenseNova-U1**.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Repository layout

```
.
├── train_sensenovau1.py           # entry point
├── configs/sensenovavl_qwen3_gen/ # 8B + A3B training configs (env-var driven)
├── shell/                         # torchrun launchers
│   └── train_u1/{8B,A3B}.sh       # per-model env-var presets + torchrun call
├── sensenovalm/                   # training framework (derived from InternEvo)
│   ├── core/                      # trainer, scheduler, parallel context
│   ├── checkpoint/                # save/resume, HF-format conversion
│   ├── solver/optimizer/          # HybridZeroOptimizer (ZeRO-1)
│   ├── model/                     # Qwen3MoeMoT, MTP, MoE primitives
│   ├── initialize/                # distributed-env initialization
│   └── ...
├── sensenovavl/                   # multimodal model + data pipeline
│   ├── data/                      # streaming + packing + conv templates
│   ├── model/sensenovavl_moe_chat # SenseNovaVLChatMoTModel + InternViT
│   └── train/                     # get_model(), pipeline wiring
└── data/                          # sample dataset — downloaded separately (see "Data")
```

---

## Quickstart

### Requirements

- Python 3.10+
- PyTorch 2.5+ — full list in [`requirements.txt`](requirements.txt)

**Minimum hardware (as shipped — `shell/train_u1/{8B,A3B}.sh` defaults):**

| Model | GPUs (minimum) | HBM / GPU | Why |
|---|---|---|---|
| `8B.sh` (dense) | **1 × 8** | 80 GB | `wp=8 × tp=1 × pp=1 = 8` ranks; `seq_len=28672`, `num_imgs=144` |
| `A3B.sh` (MoE)  | **2 × 8** | 80 GB | `wp=8 × tp=2 × pp=1 = 16` ranks required by ISP; same seq budget |

To fit on smaller setups, reduce `seq_len` / `num_imgs` (memory) and / or
`wp_size` / `tp_size` (topology) at the top of the launcher.

```bash
# install PyTorch matching your CUDA from https://pytorch.org first, then:
pip install -r requirements.txt
```

> The repo ships training code only.

### Configure your environment

The launchers are plain `torchrun` wrappers — edit `shell/train_u1/{8B,A3B}.sh`
and set at least:

| Variable | Purpose |
|---|---|
| `MODEL_NAME_OR_PATH` | HF weights to initialize from (or `LLM_PATH` + `VIT_PATH` + `MLP_PATH` for component-wise loading) |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Qwen3 tokenizer directory (same path is fine for both) |
| `mm_data_path` | dataset meta JSON (see `data/sample/sample_data_meta.json` for the schema) |
| `JOB_NAME` | unique name for the run; outputs land under `RUN/$JOB_NAME/` |

Distributed topology is also env-var driven (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `NPROC_PER_NODE` | 8 | GPUs per node |
| `NNODES` | 1 | Number of nodes |
| `NODE_RANK` | 0 | This node's rank in `[0, NNODES)` |
| `MASTER_ADDR` | 127.0.0.1 | Rendezvous host (rank-0 node's IP) |
| `MASTER_PORT` | 29500 | Rendezvous port |

All other knobs (`seq_len`, `total_steps`, `lr`, parallelism sizes,
flow-matching schedule, CFG drop probabilities, ...) are also env-var
overrideable. See [`configs/sensenovavl_qwen3_gen/`][configs] for the full
list of consumers.

[configs]: configs/sensenovavl_qwen3_gen/

### Launch a training run

```bash
# 8B smoke test — single node, 8 GPUs.
bash shell/train_u1/8B.sh

# A3B requires 2 nodes (wp × tp = 16 ranks under ISP) — run on each node,
# replacing NODE_RANK / MASTER_ADDR:
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash shell/train_u1/A3B.sh
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash shell/train_u1/A3B.sh
```

Outputs are written to `RUN/$JOB_NAME/$TIMESTAMP/{logs,tensorboards,shell}/`.
A successful run prints `step=0 loss=…` after model load and dataset
warm-up (typically 3–5 minutes for A3B on 16 H800s).

---

## Tasks supported

| Task | Notes |
|---|---|
| `mm_t2i` |  Text-to-image generation |
| `mm_it2i` |  Image editing (single- or multi-input) |
| `mm_interleave_gen` | Interleaved text + image generation |
| `mm_interleaved` | Interleaved text/image understanding |
| `multimodal` (OCR, VQA, …) |  Generic multimodal understanding |

Loss is automatically bucketed per task for logging, so you can monitor
each capability separately on TensorBoard / W&B.

---

## Configuration deep-dive

- **Sequence packing.** All samples are packed into a single fixed-length
  sequence (`seq_len`) per micro-batch. Per-document attention is enforced
  by FlexAttention block masks. The packing budget is
  `max_packed_tokens = seq_len`, `num_images_expected = num_imgs`.
- **Parallelism.** `parallel.tensor.mode='isp'` enables Intern Sequence
  Parallel; combined with `weight.size=wp_size` (FSDP-like weight
  sharding) and `zero1.size=-1` (full DP ZeRO-1). Pipeline parallel is
  disabled (`pp_size=1`) in shipped configs.
- **Flow-matching schedule.** `time_schedule={standard,dynamic}` with
  `time_shift_type={exponential,linear}`; `P_mean`, `P_std` control
  the logit-normal timestep sampling. `noise_scale_mode={fixed,resolution,
  dynamic}` adapts the noise floor to the image's token count.
- **CFG drop.** Classifier-free-guidance training drop is governed by
  `cfg_txt_uncond_drop_prob`, `cfg_img_uncond_drop_prob`,
  `cfg_txtimg_uncond_drop_prob`, with `cfg_is_uncond_drop_independent`
  toggling independent vs. mutually-exclusive draws.

For the exact wire format of the chat template, see
`sensenovavl/data/dataset.py::preprocess_sensenovalm_v3_mm_chat`.

---

## Data

### Sample dataset (smoke test)

A tiny illustrative meta JSON, jsonl annotations, and the matching
image / video assets are released as a separate Hugging Face dataset:

**[`SenseNova/SenseNova-U1-Training-Sample`](https://huggingface.co/datasets/SenseNova/SenseNova-U1-Training-Sample)** (~680 MB)

Download it into `training/data/` before launching a run:

```bash
huggingface-cli download \
    SenseNova/SenseNova-U1-Training-Sample \
    --repo-type dataset \
    --local-dir training/data
```

The unpacked layout matches the paths in
`training/data/sample/sample_data_meta.json` — no path rewriting needed,
the shipped `shell/train_u1/{8B,A3B}.sh` will pick it up directly. The
sample is sized to exercise every task type (see the table above), **not**
to produce a usable model.

### Real training

For real training, prepare your own data in the same schema:

1. Write jsonl files (one sample per line; each sample is a
   `{conversations, image, ...}` dict; see the sample jsonls in the HF
   dataset for the exact shape).
2. Build a meta JSON listing your datasets with `root`, `annotation`,
   `repeat_time`, and `task` fields (modeled after
   `sample/sample_data_meta.json` in the HF dataset).
3. Set `mm_data_path` to that meta JSON in your shell script.

A standalone data-prep guide is on the TODO list; PRs welcome.

---

## Project status

This repository was extracted from a larger internal codebase and reduced
to the training-only surface area. 

What's in scope:

- Distributed training of SenseNova-U1 (8B dense and 38B-A3B MoE)
- Mixed task training across the five `type_id` categories above
- Streaming-resumable data loading with checkpoint state
- HF-format checkpoint export via [`tools/revert2hf.py`](tools/revert2hf.py)
  (see the next section).

---

## Checkpoint conversion (internevo → HuggingFace)

Training writes checkpoints in the internevo shard layout (`model_wp{N}_pp0.pt`
plus optional `model_moe_layer{L}_expert{E}_wp{W}.pt` for MoE). To consume one
of these from `transformers` / `safetensors`, convert with:

```bash
python tools/revert2hf.py \
    --src /path/to/RUN/<job>/<step> \
    --tgt /path/to/output/hf_dir \
    --extras-from /path/to/reference-hf-model    # optional: copies config.json + tokenizer
```

The same command works for both **dense** (8B) and **MoE** (A3B) checkpoints —
MoE shards are auto-detected and the MoT U/G dual-branch experts are routed to
`mlp.experts.*` and `mlp_mot_gen.experts.*` respectively. Output layout matches
the publicly-released checkpoints:

- `model-{NNNNN}-of-00016.safetensors` — dense (vit, mlp1, fm_modules, llm dense weights, MoE gates)
- `moemodel-{NNNNN}-of-{N}.safetensors` — one per LLM MoE layer (both branches)
- `model.safetensors.index.json` — combined index
- `config.json` + tokenizer files — copied from `--extras-from` if provided

The tool streams shards via `torch.load(mmap=True)` and processes one output
slice at a time, so it runs comfortably inside ~32GB cgroup limits even for >100GB MoE checkpoints.

---

## License

This project is released under the **Apache License, Version 2.0** — see
[`LICENSE`](../LICENSE) for the full text.

Some files in [`sensenovavl/model/sensenovavl_moe_chat/`](sensenovavl/model/sensenovavl_moe_chat/)
are **derived from [InternVL](https://github.com/OpenGVLab/InternVL)** (Copyright
(c) 2023 OpenGVLab) and retain their original **MIT license**. Each such file
carries a header attributing the upstream copyright; modifications made in this
repository are licensed under Apache-2.0. The MIT terms apply to the original
portions; the Apache-2.0 terms apply to this distribution as a whole.

---

## Acknowledgements

SenseNova-U1's training code stands on the shoulders of several outstanding
open-source projects:

- [**InternVL**](https://github.com/OpenGVLab/InternVL) (OpenGVLab, MIT) —
  vision encoder design and reference implementation.
- [**InternEvo**](https://github.com/InternLM/InternEvo) (OpenGVLab,
  Apache-2.0) — ZeRO + sequence-parallel training-framework backbone.
- [**ColossalAI**](https://github.com/hpcaitech/ColossalAI) (HPC-AI Tech,
  Apache-2.0) — engine / scheduler / parallel-context primitives.
- [**DeepSpeed**](https://github.com/microsoft/DeepSpeed) (Microsoft,
  Apache-2.0) — MoE (gshard / dropless) and ZeRO partitioning patterns.
- [**FastChat**](https://github.com/lm-sys/FastChat) (lm-sys, Apache-2.0)
  — chat template infrastructure.
- [**Flash-Attention**](https://github.com/Dao-AILab/flash-attention)
  (Tri Dao, BSD-3-Clause) — fused attention, RMSNorm, MHA, rotary, and
  cross-entropy reference implementations.
- [**Ring-Flash-Attention**](https://github.com/zhuzilin/ring-flash-attention)
  (zhuzilin, MIT) — zigzag ring attention with sliding window.
- [**NVIDIA Apex**](https://github.com/NVIDIA/apex) (NVIDIA,
  BSD-3-Clause) — fused RMSNorm reference.
- [**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM) (NVIDIA) —
  pipeline-parallel P2P comm, distributed cross-entropy, timer utilities.
- [**OpenMMLab**](https://github.com/open-mmlab) (Apache-2.0) — selected
  general utility patterns.
- [**HuggingFace Transformers**](https://github.com/huggingface/transformers)
  (HuggingFace, Apache-2.0) — `PreTrainedModel` / `PretrainedConfig`
  base classes and select dataset / sampler utilities.
- [**timm**](https://github.com/huggingface/pytorch-image-models)
  (Ross Wightman, Apache-2.0) — `DropPath` and vision-encoder building
  blocks (external dependency).
- [**deeplink_ext**](https://github.com/DeepLink-org/deeplink_ext)
  (DeepLink-org) — fused attention / norm / rotary kernels used as an
  external dependency.

Per-file copyrights and licenses are listed in [`NOTICE`](NOTICE) and
in the headers of individual source files. Thanks to the broader Qwen,
PyTorch, and diffusion-modeling communities whose ideas influenced this
work.

---

## Citation

If this project is helpful for your research, please consider **star** ⭐ and **citation** 📝 :

```bibtex
@misc{sensenova2026neounify,
  title        = {NEO-unify: Building Native Multimodal Unified Models End to End},
  author       = {SenseNova},
  journal      = {Hugging Face blog},
  url          = {https://huggingface.co/blog/sensenova/neo-unify},
  year         = {2026}
}

@article{sensenova2026sensenovau1,
  title        = {SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture},
  author       = {Diao, Haiwen and Wu, Penghao and Deng, Hanming and Wang, Jiahao and Bai, Shihao and Wu, Silei and Fan, Weichen and Ye, Wenjie and Tong, Wenwen and Fan, Xiangyu and others},
  journal      = {arXiv preprint arXiv:2605.12500},
  year         = {2026}
}
```
