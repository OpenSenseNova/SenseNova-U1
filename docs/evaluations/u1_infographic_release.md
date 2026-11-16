# SenseNova-U1 Infographic V1–V3

<p align="center"><strong>English</strong> | <a href="./u1_infographic_release_CN.md">简体中文</a></p>

These are the training notes and comparison tables from the V1–V3 releases. See the [model guide](../u1_infographic_model.md) for weights and commands.

## Release details

### SenseNova-U1-8B-MoT-Infographic-V3

**SenseNova-U1-8B-MoT-Infographic-V2** and **SenseNova-U1-8B-MoT-Infographic-V3** are independently trained from the same base model and optimized with different priorities. V2 primarily focuses on infographic text-to-image generation, while V3 targets an integrated infographic generation-and-editing workflow, preserving text-to-image (T2I) capability while significantly strengthening image editing (IT2I) capability.

To balance generation and editing capabilities, V3 restarts training from the MT stage with diverse infographic editing task data and jointly trains T2I and image-editing tasks at a reasonable ratio, followed by SFT and RL stages.

- **Model Performance:** V3 reaches an Overall Total of 50.23 on Qwen-Image-Bench, improving by 2.23 over V2's 48.00. It also achieves an Average of 5.89 on WeEdit and an overall score of 7.89 on GEdit-Bench, ranking first among the open-source models included in both evaluations.
- **Generation Quality:** While retaining infographic T2I capability, V3 adds IT2I and supports either region-marked precise changes or natural-language-only editing. It handles localized text edits, local content insertion/removal/replacement, global style editing, and layout editing while preserving unedited regions whenever possible.

### SenseNova-U1-8B-MoT-Infographic-V2

V2 upgrades the previous release by incorporating more high-quality synthetic and real data during MT/SFT and rebalancing both the distribution within infographic samples and their ratio to other data. This targets small-text rendering, complex dense layouts, and high-aesthetic generation. During RL, DPO improves overall visual quality; during GRPO, further reward-recipe optimization reduces unintended black or overly dark backgrounds.

- **Model Performance:** BizGenEval hard/easy improves from 46.6 / 65.4 to 50.3 / 67.9 (+3.7 / +2.5), while IGenBench Q-ACC/I-ACC improves from 69.5 / 17.0 to 71.4 / 18.3 (+1.9 / +1.3). OneIG(EN/ZH) reaches 55.4 / 53.5, roughly matching the previous release.
- **Generation Quality:** Text edges are sharper, high-information-density layouts are more stable, and poster, dashboard, and report-style infographics look more polished. The release also fixes the black-background issue.

### SenseNova-U1-8B-MoT-Infographic

The first infographic-specialized release extends MT training from **SenseNova-U1-8B-MoT**, adjusts the data ratio between understanding and generation tasks during MT/SFT, and further optimizes the reward recipe during RL to reduce unintended black backgrounds.

- **Model Performance:** BizGenEval hard/easy improves from 39.8 / 61.1 to 46.6 / 65.4 (+6.8 / +4.3), while IGenBench Q-ACC/I-ACC improves from 51.3 / 4.2 to 69.5 / 17.0 (+18.2 / +12.8), with robust visual understanding retained.
- **Generation Quality:** Supports complex infographics across 100+ styles and layouts, improving visual aesthetics and text rendering in dense small-text scenarios such as arXiv-style pages.

## Benchmark Results

### WeEdit (Image Editing)

| Model | Instruction Adherence&nbsp;↑ | Text Clarity&nbsp;↑ | Background Preservation&nbsp;↑ | Average&nbsp;↑ |
| :--- | ---: | ---: | ---: | ---: |
| ***Closed-source Models*** | | | | |
| Nano-Banana-Pro | **8.58** | **9.10** | **8.85** | **8.84** |
| Seedream 4.5 | 6.29 | 7.66 | 6.38 | 6.78 |
| Gemini-2.5-Flash-Image | 3.92 | 7.14 | 7.80 | 6.29 |
| Qwen-Image-2.0 | 5.04 | 6.08 | 5.68 | 5.60 |
| ***Open-source Models*** | | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | **5.67** | 5.94 | 6.06 | **5.89** |
| FireRed-Image-Edit | 4.15 | **6.33** | **7.14** | 5.87 |
| HY-Image-3-Instruct | 4.16 | 5.99 | 7.03 | 5.73 |
| Qwen-Image-Edit-2509 | 3.49 | 5.84 | 6.80 | 5.38 |
| LongCat-Image-Edit | 2.78 | 4.36 | 7.03 | 4.72 |
| Qwen-Image-Edit-2511 | 3.18 | 3.93 | 4.63 | 3.91 |
| BAGEL | 1.97 | 4.01 | 5.75 | 3.91 |

<sub>Note: Higher is better. Group-best scores are bolded. Average is the arithmetic mean of the three metrics. V3 ranks first by Average among the open-source models included in this evaluation.</sub>

### GEdit-Bench (Image Editing)

| Model | EN_G_SC&nbsp;↑ | EN_G_PQ&nbsp;↑ | EN_G_O&nbsp;↑ |
| :--- | ---: | ---: | ---: |
| ***Closed-source Models*** | | | |
| Qwen-Image-2.0 | **9.02** | 8.02 | **8.37** |
| UniWorld-V2 | 8.39 | 8.02 | 7.83 |
| Seedream 4.5 | 8.27 | 8.17 | 7.82 |
| Nano-Banana-Pro | 8.10 | 8.34 | 7.74 |
| Seedream 4.0 | 8.14 | 8.12 | 7.70 |
| Nano-Banana | 7.40 | **8.45** | 7.29 |
| ***Open-source Models*** | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | **8.65** | 7.58 | **7.89** |
| LongCat-Image-Edit | 8.18 | **8.00** | 7.64 |
| Qwen-Image-Edit-2511 | 8.00 | 7.86 | 7.56 |
| Qwen-Image-Edit-2509 | 8.15 | 7.86 | 7.54 |
| Step1X-Edit | 7.66 | 7.35 | 6.97 |
| BAGEL | 7.36 | 6.83 | 6.52 |
| OmniGen2 | 7.16 | 6.77 | 6.41 |
| UniWorld-v1 | 4.93 | 7.43 | 4.85 |
| AnyEdit | 3.18 | 5.82 | 3.21 |

<sub>Note: Higher is better. Group-best scores are bolded. V3 delivers the best overall performance among the open-source models.</sub>

### Qwen-Image-Bench (Text-to-Image)

| Model | Quality&nbsp;↑ | Aesthetics&nbsp;↑ | Alignment&nbsp;↑ | Real-world Fidelity&nbsp;↑ | Creative Generation&nbsp;↑ | Overall Total&nbsp;↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-Image-2 | **59.09** | **68.48** | **65.78** | **59.40** | **75.34** | **64.58** |
| Nano-Banana-Pro | 55.30 | 61.38 | 60.30 | 55.91 | 64.54 | 57.84 |
| Qwen-Image-2.0 | 55.16 | 60.36 | 57.86 | 53.06 | 63.59 | 57.84 |
| HunyuanImage-3.0 | 50.76 | 54.66 | 53.16 | 45.33 | 48.33 | 50.81 |
| **SenseNova-U1-8B-MoT-Infographic-V3** | 49.53 | 52.82 | 52.00 | 44.32 | 48.34 | 50.23 |
| SenseNova-U1-8B-MoT-Infographic-V2 | 48.15 | 49.60 | 50.08 | 43.68 | 44.65 | 48.00 |
| SenseNova-U1-8B-MoT-Infographic | 47.12 | 48.15 | 48.90 | 43.35 | 45.40 | 47.11 |
| HiDream-O1 | 44.20 | 45.35 | 43.74 | 40.28 | 36.24 | 42.84 |

<sub>Note: Higher is better. Column-best scores and the V3 model name are bolded. V3 reaches an Overall Total of 50.23, improving by 2.23 over V2's 48.00.</sub>

### Historical Infographic Generation Benchmarks

| Model | BizGenEval Avg. (hard / easy)&nbsp;↑ | IGenBench Q-ACC&nbsp;↑ | IGenBench I-ACC&nbsp;↑ | OneIG(EN)&nbsp;↑ | OneIG(ZH)&nbsp;↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Commercial Models*** | | | | | |
| Nano-Banana-Pro | 76.7 / 93.7 | 90.6 | 48.8 | 58.1 | 56.8 |
| Nano-Banana-2.0 | 68.5 / 92.5 | 85.6 | 34.4 | 54.0 | 54.9 |
| GPT-Image-1.5 | 35.9 / 81.6 | 55.0 | 12.0 | - | - |
| Qwen-Image-2.0 | 45.5 / 65.8 | 50.0 | 3.0 | 54.1 | 50.9 |
| Seedream-4.5 | 30.1 / 66.2 | 61.0 | 6.0 | 56.4 | 55.0 |
| ***Open-source Models*** | | | | | |
| **SenseNova-U1-8B-MoT-Infographic-V3** | 49.5 / **68.5** | 66.3 | 13.2 | 54.6 | 52.4 |
| **SenseNova-U1-8B-MoT-Infographic-V2** | **50.3** / 67.9 | **71.4** | **18.3** | 55.4 | 53.5 |
| **SenseNova-U1-8B-MoT-Infographic** | 46.6 / 65.4 | 69.5 | 17.0 | **55.6** | 53.3 |
| **SenseNova-U1-8B-MoT** | 39.8 / 61.1 | 51.3 | 4.2 | 54.5 | 53.8 |
| Z-Image | 8.2 / 43.8 | 30.0 | 1.0 | 54.6 | 53.5 |
| Qwen-Image-2512 | 6.3 / 41.0 | 32.2 | 1.0 | 53.0 | 51.5 |
| Qwen-Image | 2.8 / 23.8 | 36.0 | 0.0 | 53.9 | 54.8 |
| Bagel | 2.0 / 3.7 | 4.9 | 0.0 | 36.1 | 37.0 |

<sub>Note: V3 is the current release and is pinned to the first row of the open-source section. IGenBench scores are reported as percentages; the remaining commercial and open-source models are sorted separately by the average of BizGenEval hard/easy and IGenBench Q-ACC/I-ACC. OneIG is included as a general generation reference.</sub>

V2 and V3 are independently trained from the same base model with different task and data mixtures. V3 jointly trains infographic text-to-image generation and image editing with a slightly lower proportion of chart-related training data than V2, resulting in lower IGenBench scores than the generation-focused V2. At the same time, V3 remains broadly comparable to V2 on BizGenEval, improves the broader Qwen-Image-Bench score from 48.00 to 50.23, and delivers a clear improvement in infographic editing capability.
