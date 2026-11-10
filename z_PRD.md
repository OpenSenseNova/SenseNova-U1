# PRD: Apple Silicon (MPS) Inference for SenseNova-U1.5-8B-MoT

Status: Draft for handoff
Repo: `/Users/ianscrivener/zCode_26_11/SenseNova-U1`
Branch: `mps-port` (created from `45fbbc5`, no commits yet)
Model: `sensenova/SenseNova-U1.5-8B-MoT` (Hugging Face)
Owner: Ian Scrivener

---

## 1. Summary

Make `sensenova/SenseNova-U1.5-8B-MoT` run inference (text-to-image first, then the other supported inference tasks) on Apple Silicon Macs using PyTorch's MPS backend, instead of CUDA only. Training is out of scope.

The first milestone is a 64×64 text-to-image image that looks correct. Full-size generation needs explicit approval from the user before it is run.

## 2. Background and Why This Matters

- The original spike used the wrong repo. `SenseNova-Vision` does not contain the NEO architecture that U1.5 needs (`NEOChatModel` / `neo_chat`). That work is abandoned. Its files are not part of this PRD.
- `SenseNova-U1` is the correct CUDA codebase. It already contains the NEO code under `src/sensenova_u1/models/neo_unify/`, and its `config.json` `auto_map` points at `configuration_neo_chat.py` and `modeling_neo_chat.py`, which exist in this repo.
- The first 64×64 MPS attempt on U1.5 crashed the machine (36 GB unified memory). The cause is almost certainly the checkpoint size, not the model code (see §6).

## 3. Goals

1. **G1**: A documented, reproducible way to install and run U1.5 inference on macOS arm64 with `uv`.
2. **G2**: A 64×64 text-to-image run on MPS that produces a visually sensible image.
3. **G3**: Memory-safe loading on a 36 GB Mac. The machine must not crash.
4. **G4**: No regression on the CUDA path. Existing behaviour on CUDA/XPU stays the same.
5. **G5**: After G2 is approved, a larger-resolution run (for example 512 or 1024 px) that the user has signed off on.

## 4. Non-Goals

- Training, fine-tuning, or any gradient-based work.
- Porting the SenseNova-Vision (`bagel`) code.
- Performance parity with CUDA. Correct output comes first; speed is secondary.
- Quantization (nf4/int8) on MPS. bitsandbytes is CUDA-only. Quantization is out of scope unless the user asks for it.
- Windows, Linux, or Intel Mac support beyond not breaking them.

## 5. Constraints and Operating Rules

These come from the user and are binding.

| Rule | Detail |
|---|---|
| Package manager | Use `uv` only. Do not use pip or conda directly. |
| Venv | Delete and recreate `.venv` with `uv venv`. Do not reuse a venv built for another platform. |
| Python | 3.11 (CPython 3.11.13 was used successfully). |
| Mode | Inference only. No training code paths should be exercised or changed beyond what is needed to import cleanly. |
| Memory | Swap growth is accepted. The user will close other apps. Allowed system memory use is up to 95%. |
| **Memory watchdog (mandatory)** | Any test or model load must run under a watchdog. The watchdog must kill the specific test PID (`kill <PID>`, never `pkill`/`killall`) if system memory use goes above 95%. Check with `psutil.virtual_memory().percent` (available in `.venv`) or `memory_pressure`. This rule was added after a crash. |
| Full-size tests | Ask the user before running any test above 64×64. 64×64 may run without asking. |
| Iteration | Keep iterating until good 64×64 images are produced, unless blocked. |
| Commits | Commit work on `mps-port` as it reaches milestones (see §11). Include the Co-authored-by trailer. |
| Docs | Update docs that describe install or device behaviour. Do not create planning markdown other than this PRD. |

## 6. Current State (as of handoff)

### 6.1 Code

- `src/sensenova_u1/utils/accel.py` (modified, uncommitted at time of writing):
  - `best_available_device()` now returns `cuda → xpu → mps → cpu`.
  - `is_available("mps")`, `empty_cache`, `synchronize`, `manual_seed_all` have MPS branches.
  - `SUPPORTED_DEVICE_TYPES` is still `("cuda", "xpu")`. This is probably an oversight and needs a decision (§8).
  - Syntax checked. Not yet run on MPS.
- No other tracked file is modified. `pyproject.toml` and `requirements.txt` are unchanged.

### 6.2 Environment

- Host: Apple Silicon arm64, macOS 27.0.1, 36 GB unified memory.
- `uv` 0.11.21.
- `.venv/` (CPython 3.11.13): torch 2.8.0, torchvision 0.23.0, transformers 5.22.2, plus the rest of `requirements.txt` with the cu128 `--extra-index-url` line removed.
- MPS was confirmed available in this venv.
- The package is not installed. Run with `PYTHONPATH=src`.

### 6.3 Checkpoint

- Path: `/Volumes/Crucial500Gb/HUGGINGFACE_HUB_ACTIVE/models--sensenova--SenseNova-U1.5-8B-MoT/snapshots/a2462cc1b549bace2c0896143bbbcbb3fa217c76`
- The external volume must be mounted.
- Layout: `model.safetensors.index.json` plus 8 shard symlinks. The shards resolve to full files. Seven `*.incomplete` blobs remain in the cache, so the download is not fully clean. Verify before relying on it.
- Index `total_size` is about 35 GB, which implies fp32 weights.
- Missing from the snapshot: `ema.safetensors`, `ae.safetensors`, `llm_config.json`, `vit_config.json`, `generation_config.json`, `tokenizer.json`. The loader must handle this (see §7.2).
- `config.json`: `architectures` `NEOChatModel`, `model_type` `neo_chat`, `auto_map` to `configuration_neo_chat.py` / `modeling_neo_chat.py`. The LLM config is nested under `llm_config` (Qwen3ForCausalLM, 42 layers, hidden 4096, 32 heads, 8 KV heads).

### 6.4 Crash

- A 64×64 bf16 run crashed the whole machine with no watchdog running.
- Working theory: the loader materializes most of the ~35 GB fp32 checkpoint before casting to bf16, which exceeds 36 GB of unified memory.
- This theory is not confirmed. Confirming it is the first task.

## 7. Requirements

### 7.1 Functional

- **FR1 Device**: `--device mps` works. The default device resolves to MPS on Apple Silicon when CUDA is absent.
- **FR2 dtype**: bf16 or fp16 on MPS. fp32 must be avoided for the full model because of memory. Choose bf16 if correct on MPS, and fall back to fp16 if bf16 kernels are missing or produce NaNs.
- **FR3 Loading**: load the U1.5 snapshot without reading the whole checkpoint into memory at once. Options include per-shard streaming with cast to the target dtype, `low_cpu_mem_usage`, or a device-direct load. Choose the approach that fits in memory.
- **FR4 Missing files**: handle the missing `ema`, `ae`, `llm_config.json`, and `vit_config.json` files. Either load the config from the nested `config.json` section, or document and implement the fallback.
- **FR5 Attention**: the NEO path must not call CUDA-only or flash-attention-only kernels on MPS. Use SDPA or a documented fallback. Check that `flex_attention` is not required in the default T2I path.
- **FR6 CFG**: classifier-free guidance batching is serial on MPS (see `cfg_batching.py`, about line 72). This is acceptable. Document the speed cost.
- **FR7 Autocast**: on MPS, autocast maps to the cpu device (`modeling_neo_chat.py`, about lines 99–104). Verify that dtype handling does not silently produce fp32 on MPS.
- **FR8 Offload**: `layer_offload.py` is CUDA-only. It is bypassed in `full` mode. It must not be selected on MPS.
- **FR9 CLI**: `examples/t2i/inference.py` runs end to end with `--model_path`, `--width`, `--height`, `--num_steps`, `--dtype`, `--device`. Defaults must not force a 2048×2048 run on a laptop. Consider changing the default or documenting the override.

### 7.2 Non-Functional

- **NFR1 Memory**: peak system memory during a 64×64 run stays below 95%. The watchdog enforces this.
- **NFR2 Reproducibility**: a fresh clone can run the documented install steps using `uv` and produce a 64×64 image.
- **NFR3 No CUDA regression**: CUDA-only code paths and their tests behave as before.
- **NFR4 Observability**: log device, dtype, and memory before and after the load, and per-step timing during generation.

## 8. Open Decisions (resolve with the user before implementing)

1. **Loading strategy**: streaming per-shard bf16 load (recommended), or a static loader check first and then the load?
2. **`SUPPORTED_DEVICE_TYPES`**: add `"mps"`, or keep it for the device-type registry only?
3. **Default resolution in the CLI**: keep 2048, or change the macOS default to something smaller?
4. **Dependency path**: add a macOS-specific requirements file, or document a manual `grep -v` step? The `pyproject.toml` cu128 pin breaks plain `uv sync` on macOS.
5. **bf16 quality**: if bf16 on MPS gives NaNs or black images, is fp16 acceptable?

## 9. Plan

### Phase 0: Safety and baseline (no model load)

- Start a memory watchdog script that polls `psutil.virtual_memory().percent` and kills a named PID above 95%. Keep it running for every later phase.
- Verify the `.venv` imports: `PYTHONPATH=src .venv/bin/python -c "import sensenova_u1, torch; print(torch.backends.mps.is_available())"`.
- Verify the snapshot is complete. Check each shard's size against the index, and that the `.incomplete` blobs are not needed. Re-download if needed.

Exit: imports OK, watchdog works, shard sizes checked.

### Phase 1: Static loader analysis (no weights loaded)

- Trace `checkpoint_loading.py` (`resolve_model_artifact` ~57, `_infer_sensenova_resources` ~127, `_default_device` ~140, `_has_complete_model_weights` ~263, `load_model_and_tokenizer` ~276).
- Confirm the `auto_map` and nested `llm_config` paths resolve.
- List every reference to `ema.safetensors`, `ae.safetensors`, `llm_config.json`, and `vit_config.json`.
- Write down the exact failure for each missing file.

Exit: a written list of loader changes needed. Share with the user.

### Phase 2: Memory-safe load

- Implement streaming load (per shard, cast to bf16, free the fp32 tensor before the next shard).
- Run under the watchdog. Log memory after each shard.
- Stop immediately if memory approaches 95%. Do not retry without reducing the load footprint.

Exit: model object built on MPS without crashing. Peak memory recorded.

### Phase 3: 64×64 text-to-image on MPS

- Run `examples/t2i/inference.py` with `--device mps --width 64 --height 64 --num_steps 4` (fewer steps first, then more).
- Check the output is not NaN or black. Look at the image.
- Iterate on dtype, attention, or sampling as needed until the output looks sensible.

Exit: good 64×64 image saved under `/tmp` or a repo-ignored output dir. Commit.

### Phase 4: Approval gate

- Stop and ask the user before running any test above 64×64. Give the expected memory, time, and output.

### Phase 5: Other inference tasks and CUDA regression

- Check the other inference entry points that the user cares about (understanding, I2I/editing if in scope). Use 64×64 or smaller first.
- Run existing tests on the CUDA-agnostic parts. Confirm no change in CUDA behaviour (code review of diffs, plus tests that run without a GPU).

Exit: CUDA paths unchanged. Test results summarised.

### Phase 6: Documentation and cleanup

- Add a macOS install section to the README (uv steps, the requirements workaround, the external-volume checkpoint note, memory warning).
- Remove temporary files and `/tmp` outputs that are not needed. Do not commit `.venv`.

## 10. Acceptance Criteria

- **AC1**: `uv`-only install on a clean macOS arm64 machine documented and reproducible.
- **AC2**: `--device mps --width 64 --height 64` produces a non-black, non-NaN image that a human reviewer accepts as a sensible rendering of the prompt.
- **AC3**: During AC2 the system memory never exceeds 95%, as shown by watchdog logs.
- **AC4**: No CUDA behaviour change. Diff review shows CUDA branches untouched except for guard additions.
- **AC5**: Full-size run is attempted only after explicit user approval.
- **AC6**: Commits on `mps-port` have clear messages and the Co-authored-by trailer.

## 11. Commit Plan

1. `accel: add MPS device selection` (current `accel.py` change).
2. `loader: streaming bf16 load for missing ema/ae configs` (after Phase 2).
3. `mps: fix attention and dtype on NEO path` (after Phase 3).
4. `docs: macOS install notes` (Phase 6).

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Machine crash from memory | High without guard | Severe | Watchdog in every phase; streaming load; 64×64 first |
| Checkpoint incomplete (`.incomplete` blobs) | Medium | Blocks load | Verify sizes; re-download |
| bf16 unsupported or NaN on MPS | Medium | Bad output | fp16 fallback; compare against CPU for a tiny case |
| transformers 5.x incompatible with NEO code | Medium | Blocks load | Pin to a compatible version if needed (check `pyproject.toml` range `>=4.57.1,<6`) |
| `flex_attention` required on MPS | Unknown | Blocks forward | Use SDPA fallback |
| Slow generation (serial CFG) | High | Usability | Document; reduce steps for tests |
| Missing `ema`/`ae` files | Certain | Blocks load | Loader fallback (Phase 1) |

## 13. Validation Commands

Watchdog (run first, keep it alive):

```bash
# Use a specific PID for the test process. Never use pkill or killall.
.venv/bin/python -c "import psutil; print(psutil.virtual_memory().percent)"
```

Import check:

```bash
PYTHONPATH=src .venv/bin/python -c "import sensenova_u1, torch; print(torch.__version__, torch.backends.mps.is_available())"
```

64×64 test (only under the watchdog, only with the PID watched):

```bash
PYTHONPATH=src .venv/bin/python examples/t2i/inference.py \
  --model_path "/Volumes/Crucial500Gb/HUGGINGFACE_HUB_ACTIVE/models--sensenova--SenseNova-U1.5-8B-MoT/snapshots/a2462cc1b549bace2c0896143bbbcbb3fa217c76" \
  --device mps --dtype bf16 --width 64 --height 64 --num_steps 4 \
  --output /tmp/u1out/t64.png
```

Check `--output` against the actual argument name in `examples/t2i/inference.py` before running.

## 14. Key Files

- `src/sensenova_u1/utils/accel.py`: device selection (modified).
- `src/sensenova_u1/utils/checkpoint_loading.py`: loader (main change area).
- `src/sensenova_u1/models/neo_unify/modeling_neo_chat.py`: NEO model. `t2i_generate` ~1715. Autocast ~99–104.
- `src/sensenova_u1/models/neo_unify/cfg_batching.py`: device gate ~72.
- `src/sensenova_u1/models/neo_unify/layer_offload.py`: CUDA-only, bypass on MPS.
- `examples/t2i/inference.py`: CLI. `SenseNovaU1T2I.__init__` ~83, `parse_args` ~198. Defaults: 2048 px, 50 steps, bf16.
- `pyproject.toml` (about lines 102–104) and `requirements.txt`: cu128 pins that break macOS installs.

## 15. Reference Notes for the Implementing Agent

- Do not use the SenseNova-Vision repo. Its MPS work is not relevant to U1.5.
- Do not run anything that loads the full checkpoint without the watchdog.
- Do not run full-size generation without user approval.
- The T2I wrapper defaults are `cfg_scale=4.0`, `timestep_shift=3.0`, `cfg_batching=True`. `t2i_generate` defaults are `image_size=(256,256)`, `num_steps=30`, `cfg_scale=1`. Confirm which applies in the CLI path.
- Assumptions not yet verified: the loader accepts the U1.5 layout unchanged; the default T2I path avoids CUDA-only calls; `SUPPORTED_RESOLUTIONS` accepts 64; the test suite runs on macOS.
