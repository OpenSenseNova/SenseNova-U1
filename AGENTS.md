# SenseNova-U1 agent instructions

This file is the repository-level entry point for agents working on code or documentation. Follow the linked canonical guides instead of creating a second policy.

## Start here

- For an official model/checkpoint release, read [the model release guide](docs/contributing/model_release.md) before editing.
- For the shared Python runtime or CLI examples, read [installation](docs/installation.md) and [examples](examples/README.md).
- For the complete weight and adapter inventory, use [the model catalog](docs/models.md).
- Check the matching `_CN.md` page whenever an English documentation page changes.

## Documentation ownership

| Area | Canonical location |
| --- | --- |
| Homepage overview, quick start, navigation, and release notes from the most recent three months | `README.md`, `README_CN.md` |
| All official weights, adapters, SFT releases, and community integrations | `docs/models.md`, `docs/models_CN.md` |
| A release's purpose, supported tasks, runnable commands, settings, evidence, and limitations | `docs/<release>_model.md`, `_CN.md` |
| Inputs, prompts, outputs, checkpoint/adapter provenance, and prompt-enhancement status | `docs/<release>_showcases.md`, `_CN.md` |
| Long comparison tables and release evaluation records | `docs/evaluations/` |
| Measured memory, timing, and runtime behavior | `docs/*_inference.md`, `_CN.md` |
| Shared environment and reusable CLI options | `docs/installation*.md`, `examples/README*.md` |
| Serving or deployment frameworks | `docs/deployment*.md`, `docs/inference_infra*.md` |

Keep the homepage concise. Link to canonical pages instead of copying the full model catalog, long benchmark tables, or complete parameter references.

## Evidence and compatibility

- The official model card is the source of truth for weight names, revisions, base/adapter relationships, and license terms.
- Verify text-to-image, editing, interleaved generation, and visual Q&A separately. Do not infer task support from a family name.
- Run commands presented as verified on the current repository revision. If not run, say so.
- Every performance or memory claim records checkpoint and adapter revisions, repository commit, test date, GPU and driver, key dependency versions, task parameters, run count, and metric definition.
- Keep release benchmark records distinct from current runtime measurements. If settings or revisions are incomplete, label them as release records.
- Do not turn PyTorch allocated/reserved peaks or allocator-cap experiments into minimum-GPU requirements. State tested scope and leave headroom for CUDA and other allocations.
- When pages are linked in a copy/paste workflow, keep checkpoint, task, memory mode, and tested scope consistent; otherwise mark the version boundary explicitly.

## Update and synchronization

When adding or changing an official release:

1. Add the weight and every adapter, including its matching base, to both model catalogs.
2. Add the paired model guide and showcase to the homepage table when both exist.
3. Keep `README.md` and `README_CN.md` factually aligned. Keep News from the most recent three months on the homepage and move older entries to `docs/news*.md`.
4. Preserve published paths where possible. Before deleting or renaming one, search for inbound links; update them in the same change or leave a short migration page.
5. Keep community conversions and integrations separate from official releases.
6. Put alternate serving frameworks on their deployment pages and name the compatible checkpoint.

## Validation checklist

Before proposing documentation changes:

- Run `git diff --check`.
- Verify local relative links, image paths, and heading anchors in both languages.
- Compare English and Chinese model names, tasks, commands, limitations, and dated release notes.
- Run the relevant CLI smoke test when hardware and weights are available; otherwise record that GPU execution was not performed.
- Run `uv run pre-commit run --all-files` when the environment permits. Documentation-only changes do not require GPU tests.
- Trace claims to a model card, paper, evaluation record, runtime measurement, or an explicitly labeled observation.

When a rule changes, update [the canonical model release guide](docs/contributing/model_release.md) first and keep this file as the discovery and checklist layer.
