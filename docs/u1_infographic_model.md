# SenseNova-U1-8B-MoT-Infographic 📊

**SenseNova-U1-8B-MoT-Infographic** is built upon the base U1-8B-MoT model with an extended MT training phase. During both the MT and SFT stages, we optimized the data distribution between visual understanding and generation tasks. In the RL stage, we further refined the reward recipe to reduce the occurrence of unintended black backgrounds in generated infographics.

- **Benchmark Performance:** Compared with the base **SenseNova-U1-8B-MoT** model, BizGenEval hard/easy increased from **39.8 / 61.1** to **46.6 / 65.4** (**+6.8 / +4.3 points**), and IGenBench Q-ACC/I-ACC increased from **51.3 / 4.2** to **69.5 / 17.0** (**+18.2 / +12.8 points**), while maintaining robust visual understanding capabilities without substantial degradation.
- **Generation Quality:** The model produces complex infographics across 100+ styles and layouts, with improved visual aesthetics and text rendering — including dense small text such as arXiv-style pages.

## Benchmark Highlights

| Model | BizGenEval Avg. (hard / easy) ↑ | IGenBench Q-ACC↑ | IGenBench I-ACC ↑ | OneIG(EN) ↑  | OneIG(ZH) ↑  |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Commercial Models*** | |
| Nano-Banana-Pro | 76.7 / 93.7 | 90.6 | 48.8 | 58.1 | 56.8 |
| Nano-Banana-2.0 | 68.5 / 92.5 | 85.6 | 34.4 | 54.0 | 54.9 |
| GPT-Image-1.5 | 35.9 / 81.6 | 55.0 | 12.0 | - | - |
| Qwen-Image-2.0 | 45.5 / 65.8 | 50.0 | 3.0 | 54.1 | 50.9 |
| Seedream-4.5 | 30.1 / 66.2 | 61.0 | 6.0 | 56.4 | 55.0 |
| ***Open-source Models*** |  |
| **SenseNova-U1-8B-MoT-Infographic** | **46.6 / 65.4** | **69.5** | **17.0** | **55.6** | **53.3** |
| **SenseNova-U1-8B-MoT** | 39.8 / 61.1 | 51.3 | 4.2 | 54.5 | 53.8 |
| Z-Image | 8.2 / 43.8 | 30.0 | 1.0 | 54.6 | 53.5 |
| Qwen-Image-2512 | 6.3 / 41.0 | 32.2 | 1.0 | 53.0 | 51.5 |
| Qwen-Image | 2.8 / 23.8 | 36.0 | 0.0 | 53.9 | 54.8 |
| Bagel | 2.0 / 3.7 | 4.9 | 0.0 | 36.1 | 37.0 |

<sub>IGenBench scores are reported as percentages. Models are ordered by the arithmetic mean of BizGenEval hard, BizGenEval easy, IGenBench Q-ACC, and IGenBench I-ACC within the commercial and open-source groups separately. OneIG is included as a general generation reference. Full per-category results are intended for the Hugging Face model card.</sub>

## Showcases (Applications)

> ✨ **Curious to see the model in action? Explore 100 generated examples in our full 👉 [ 🖼️ Infographic Model Showcases ](./u1_infographic_showcases.md) 👈 gallery!**
