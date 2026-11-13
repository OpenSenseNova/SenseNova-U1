# Add a model release

<p align="center"><strong>English</strong> | <a href="./model_release_CN.md">简体中文</a></p>

Use this process for an official checkpoint that this repository supports directly. The [model catalog](../models.md) can also list SFT weights and community conversions with their model cards; the [project README](../../README.md#model-guides-and-showcases) lists releases that have both a guide and a showcase.

## 1. Confirm the checkpoint and tasks

Read the official model card and record the weight repository, revision, release stage, and any required base checkpoint or adapter. Check which tasks the checkpoint supports against the repository's [inference scripts](../../examples/README.md): text-to-image, editing, interleaved generation, and visual Q&A. Do not infer support for every task from the model family name.

Check whether the model uses the shared Python runtime or a separate serving framework. Test the command you plan to publish with the stated checkpoint; if it has not been run on this repository revision, say so in the guide.

## 2. Write the guide and showcase

| File in `docs/` | Include |
| --- | --- |
| `<release>_model.md` | Model purpose, official weights and compatible adapters, supported tasks, short runnable commands for distinct workflows, model-specific settings, a few sourced results useful for choosing a release, and known limitations. Link to the showcase. |
| `<release>_showcases.md` | Examples for the tasks actually shown. Include the checkpoint and adapter used, input images, prompts, outputs, and whether prompt enhancement was applied. Link back to the guide. |

Add `_CN.md` counterparts. A related series can share a pair when the versions use the same workflow, as the [U1 infographic series](../u1_infographic_model.md) does. Usually a LoRA belongs in its base model guide; give it separate pages only when its workflow and examples justify them.

The original U1 showcase keeps its published `docs/showcases.md` and `docs/showcases_CN.md` paths. Use the versioned filename pattern above for new pages.

## 3. Put supporting material in the right place

- Link to [installation](../installation.md) for the shared Python environment and to [examples](../../examples/README.md) for reusable CLI options. Keep only short commands needed to start each distinct workflow in the model guide.
- The model card is the source for weights, revisions, and license terms. Papers and technical reports carry architecture, training, and full experimental analysis. Summarize only what helps readers choose a model and link to the originals.
- Keep a few key release results in the model guide. Put long comparison tables in `docs/evaluations/`, as with the [Preview release results](../evaluations/u1.5_preview_release.md) and [infographic release results](../evaluations/u1_infographic_release.md). If full evaluation settings or checkpoint revisions are missing, label the tables as release records rather than current reproducible results.
- Keep memory and timing measurements with the tested release. Record checkpoint and adapter revisions, code commit, test date, GPU and driver, key library versions, task settings, number of runs, and the meaning of each metric. Put long records on a separate page, as with [U1.5 inference](../u1.5_inference.md). Remeasure after a relevant weight or runtime change before presenting numbers as current. Do not publish measurements that lack enough provenance to interpret or reproduce.
- Prompt enhancement is optional. Link its instructions separately, as in the [U1.5 prompt enhancement guide](../u1.5_best_practices.md), and identify any enhanced examples in the showcase.
- For another serving framework, document its setup separately and name the compatible checkpoint. The [U1 LightLLM + LightX2V deployment](../deployment.md) and [architecture notes](../inference_infra.md) are one such path.

## 4. Update entry points and check the links

1. Add official weights and adapters to the [model catalog](../models.md), including each adapter's base checkpoint. Keep community weights and tools in the community section.
2. Add the guide and showcase together to the [model guides and showcases](../../README.md#model-guides-and-showcases) table.
3. Update the [project README](../../README.md) when a release changes the featured model or merits a News item.
4. Check the English and Chinese pages, links, image paths, and heading anchors. Preserve paths already published on the main branch where possible.
