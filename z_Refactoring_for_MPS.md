# Refactoring SenseNova-U1 for Apple Silicon (MPS)

Status: **in progress, blocked on memory-safe loading of the checkpoint.** One small code change is in place. No MPS generation has completed. The 64×64 test crashed the machine. Full details are in `z_PRD.md` and `z_crash_report.md`.

Branch: `mps-port` (from `45fbbc5`). No commits yet.

## 1. Scope and estimate

| Scope | Estimate | Notes |
|---|---|---|
| Inference on MPS, bf16 | 2–4 days | Device plumbing, loader memory fix, validation on Apple Silicon |
| Training on MPS | Out of scope | Depends on CUDA-only paths |

The model is `sensenova/SenseNova-U1.5-8B-MoT`, which uses the NEO architecture in `src/sensenova_u1/models/neo_unify/`. This is the code path the U1 repo already supports, unlike the abandoned SenseNova-Vision spike.

## 2. Environment

- Host: Apple Silicon arm64 Mac Studio, macOS 27.0.1, 36 GB unified memory.
- `uv` 0.11.21. The venv is `.venv/` with CPython 3.11.13.
- Installed: torch 2.8.0, torchvision 0.23.0, transformers 5.22.2, plus the other `requirements.txt` packages.
- `--extra-index-url` (cu128) was stripped from the install. `torch.backends.mps.is_available()` returns True.
- Run with `PYTHONPATH=src` (`sensenova_u1` is not installed as a package).

## 3. Packaging blockers (not yet changed)

- `pyproject.toml` pins torch 2.8.0 and torchvision 0.23.0 to an explicit cu128 index (about lines 102–104). A plain `uv sync` fails on macOS.
- `requirements.txt` has an `--extra-index-url` cu128 line.
- `setup.sh` is CUDA-only (`nvcc`, conda, `flash-attn`).

Decision needed: add a macOS path (a marker-gated index, or a documented install command). Do not change the CUDA path.

## 4. Code changes (working tree)

| File | Change | Verified |
|---|---|---|
| `src/sensenova_u1/utils/accel.py` | `best_available_device()` now returns cuda → xpu → mps → cpu. `is_available("mps")`, `empty_cache`, `synchronize`, and `manual_seed_all` handle MPS. `INFERENCE_DEVICE_TYPES` removed (unused). `SUPPORTED_DEVICE_TYPES` still `("cuda", "xpu")`. | Syntax check only |

## 5. Known MPS issues to address

- `examples/t2i/inference.py`: `--width`/`--height` default to 2048. `--num_steps` defaults to 50. `--dtype` defaults to bf16.
- `src/sensenova_u1/models/neo_unify/modeling_neo_chat.py` (about lines 99–104): autocast on MPS maps to CPU. This may cause a silent fp32 path.
- `src/sensenova_u1/models/neo_unify/cfg_batching.py` (about line 72): CFG batching runs serially on MPS. This is a speed cost, not a correctness issue.
- `src/sensenova_u1/models/neo_unify/layer_offload.py`: CUDA-only. Bypassed in the default `full` mode.
- `src/sensenova_u1/utils/checkpoint_loading.py`: loader. `_has_complete_model_weights` (~263) and `load_model_and_tokenizer` (~276). This is the likely memory hotspot.
- `SUPPORTED_RESOLUTIONS`: unverified that 64×64 is accepted.

## 6. Crash and memory

- A 64×64, 4-step bf16 run pushed the machine into a watchdog timeout and it rebooted. No watchdog was running.
- The panic log reports `watchdog timeout: no checkins from watchdogd in 91 seconds`.
- Suspected cause (unconfirmed): the checkpoint is about 35 GB in fp32 shards. The loader may materialize them before the bf16 cast, on a 36 GB machine.
- Rule: every test runs under a watchdog that kills the process by PID above 95% system memory. `pkill` and `killall` are not used.

## 7. Next steps

1. Choose a memory-safe load path: per-shard streaming to bf16, or a device map. Measure peak RSS under the watchdog.
2. Verify the checkpoint loads with the U1 loader. Check that `ema`, `ae`, `llm_config.json`, and `vit_config.json` are not needed.
3. Run a 64×64, 4-step MPS generation under the watchdog. Inspect the output.
4. Ask before any test larger than 64×64.
5. Run the existing tests and confirm the CUDA path is unchanged.
6. Add a macOS install path and README section.

## 8. Open decisions

See `z_PRD.md` §8: loading strategy, `SUPPORTED_DEVICE_TYPES`, default resolution, dependency path, and bf16 vs fp16.
