# Crash report: 64×64 MPS test on SenseNova-U1.5-8B-MoT

Status: **cause partly confirmed.** Kernel panic type confirmed. The process that drove memory to the limit is not confirmed.

## 1. Summary

During a 64×64 text-to-image test of `sensenova/SenseNova-U1.5-8B-MoT` on the MPS device, the Mac Studio became unresponsive and rebooted. No memory watchdog was running. The kernel panic is a **watchdog timeout** (`watchdogd` made no check-in for 91 s), which is the usual result of a system hang under extreme memory pressure and swap thrash.

## 2. Host

| Item | Value |
|---|---|
| Machine | Mac Studio (Mac16,9), Apple M4 |
| Memory | 36 GB unified (`hw.memsize` = 38654705664) |
| OS | macOS 27.0.1 (26A434) |
| Kernel | Darwin 27.0.0 xnu-13432.1.9, RELEASE_ARM64_T6041 |

## 3. Test that was running

Repo: `/Users/ianscrivener/zCode_26_11/SenseNova-U1`, branch `mps-port`.

```
PYTHONPATH=src .venv/bin/python examples/t2i/inference.py \
  --model_path /Volumes/Crucial500Gb/HUGGINGFACE_HUB_ACTIVE/models--sensenova--SenseNova-U1.5-8B-MoT/snapshots/a2462cc1b549bace2c0896143bbbcbb3fa217c76 \
  --width 64 --height 64 --num_steps 4 --output /tmp/u1out/t64.png
```

Settings: dtype bf16 (default), device `mps` (auto), torch 2.8.0, transformers 5.22.2.

The checkpoint is about 35 GB (fp32 shards, per the safetensors index `total_size`). On disk it was incomplete at the time, with about 20 GB present and 7 `*.incomplete` blobs. The loader is not known to stream shards.

## 4. Timeline (local time, 2026-11-10)

| Time | Event |
|---|---|
| 03:25 | Unrelated `python3.14` diagnostic file written (`python3.14_2026-11-10-032502_MacStudioM4-2.diag`). Not yet examined. |
| ~08:45 | Jetsam event (`JetsamEvent-2026-11-10-084501.ips`). Snapshot taken during the test window. |
| 09:13 | Reboot (`last reboot`). |
| 09:25 | Panic record written (`panic-full-2026-11-10-092530.0002.panic`). |

The exact start time of the test is not confirmed. The Jetsam snapshot and the reboot both fall in the window where the test ran.

## 5. Evidence

### 5.1 Panic record (`/Library/Logs/DiagnosticReports/.contents.panic`)

```
panic(cpu 9 caller ...): watchdog timeout: no checkins from watchdogd in 91 seconds
```

Interpretation: the kernel was alive but stalled for 91 s. This is consistent with memory thrash or compressor saturation. It is not a normal OOM kill.

### 5.2 Jetsam event at 08:45 (`JetsamEvent-2026-11-10-084501.ips`)

| Metric | Value |
|---|---|
| Anonymous pages | 1,018,939 (about 16.7 GB at 16 KB pages) |
| Free pages | 86,108 (about 1.4 GB) |
| Compressor | about 353k pages |
| Zone map | about 1.1 GB |
| Largest process | `ZippyCrispDictation`, about 197k pages (about 3.2 GB) |
| Other top processes | WindowServer, Code Helper, Firefox GPU Helper, plugin-container |

**No `python` or U1 process appears in the top list.** The Jetsam snapshot does not identify the test process. Its memory use may have been in the swap/compressed pool, which Jetsam does not list.

## 6. Suspected cause (unconfirmed)

1. **Loader materializes fp32 shards before casting to bf16.** About 35 GB of fp32 weights on a 36 GB machine, with other apps (Code, Firefox, WindowServer) already using several GB. Peak resident memory would exceed physical RAM, and swap would thrash until the watchdog fired.
2. **No memory guard.** Nothing stopped the process before the system hung.
3. Other applications were running and consuming memory (see §5.2), which reduced headroom further.

To confirm: run the loader alone under a watchdog, with the process RSS logged per shard. The result will show whether the fp32 materialization is the cause.

## 7. Rule adopted

Saved as a user-scoped memory: **every test run must have a watchdog that terminates the test when system memory exceeds 95%.** Watchdog kills by PID only (`pkill`/`killall` are not allowed in this environment). Monitor with `psutil.virtual_memory().percent`.

Additional rules for the next attempt:

- Ask the user before any test larger than 64×64.
- Do not load the fp32 checkpoint without a streaming/casting plan (see `z_PRD.md` §8).
- Close other heavy applications before a load test.

## 8. Open items

- [ ] Read `python3.14_2026-11-10-032502_MacStudioM4-2.diag` to check whether it relates to this test.
- [ ] Confirm the snapshot is complete against its index before any load test.
- [ ] Measure per-shard peak RSS with a watchdog, loading one shard at a time.
- [ ] Decide the memory-safe loading strategy with the user (`z_PRD.md` §8).
