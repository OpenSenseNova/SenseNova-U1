# 模型参数分解

`SenseNova-U1-8B-MoT` 包括约 **8B 理解参数**和 **8B 生成参数**。为了避免命名带来的对参数量的误解，同时更好呈现模型的结构，我们提供了一个查看模型参数的脚本，可以根据模型中参数名称的解析，列出详细的模型参数统计。

## 运行脚本

```bash
python scripts/inspect_model_params.py \
    --model_path sensenova/SenseNova-U1-8B-MoT
```

常用参数：

- `--dtype {float32,float16,bfloat16}`（默认：`bfloat16`）—— 加载精度。它**不影响**参数数量，只影响 `memory` 一列：`bf16/fp16` 每个元素 2 字节，`fp32` 每个 4 字节。
- `--show_groups <name1,name2>`（默认：`shared`）—— 列出指定组里的具体参数。可填 `all` 列出所有组，填空字符串关闭。
- `--custom_groups_json <path>`—— 用形如 `{"group_name": ["prefix1", "prefix2"]}` 的 JSON 文件覆盖默认的分组规则。

## 输出示例

```text
Model: sensenova/SenseNova-U1-8B-MoT
Load dtype:   bfloat16
Total params: 17.552B
Total memory: 35.105GB (bfloat16)
---------------------------------------------------------------------
group                              params memory (bfloat16)      ratio
---------------------------------------------------------------------
generation_transformer             8.186B         16.373GB     46.64%
understanding_transformer          8.121B         16.243GB     46.27%
shared                             1.245B          2.489GB      7.09%
---------------------------------------------------------------------
Pathway breakdown (shared counted in both):
---------------------------------------------------------------------
pathway                            params memory (bfloat16)      ratio
---------------------------------------------------------------------
understanding pathway              9.366B         18.732GB     53.36%
generation pathway                 9.431B         18.862GB     53.73%

---------------------------------------------------------------------
Members of group 'shared' (2 params, 1.245B total, 2.489GB @ bfloat16)
---------------------------------------------------------------------
param name                                                  numel    dtype
---------------------------------------------------------------------
language_model.model.embed_tokens.weight                 622.330M bfloat16
language_model.lm_head.weight                            622.330M bfloat16
```

## 如何理解输出

### 1. 模型的参数量（互斥归属，加起来 100%）

第一段表示**模型的参数量**：每个参数在物理上只算一次，归到三组之一：

- `understanding_transformer` ≈ **8.12B (46%)**：理解侧的视觉 (`vision_model.*`) 加上 LLM 中**不带** `_mot_gen` 后缀的理解专家分支（`language_model.*` 去掉 `_mot_gen` 副本和 `shared` 文本 I/O）。
- `generation_transformer` ≈ **8.19B (47%)**：生成侧模块 (`fm_modules.*`：生成的视觉、flow-matching head、timestep / noise embedders) 加上 LLM 中**带** `_mot_gen` 后缀的生成专家分支（`language_model.*` 中包含 `_mot_gen` 的部分）。
- `shared` ≈ **1.25B (7%)**：两条 pathway 都会复用的文本 token 输入输出，即 `language_model.model.embed_tokens` 和 `language_model.lm_head`。

### 2. 模型 forward 时经过的参数量（pathway 覆盖，加起来 >100%）

第一段反映"参数如何存储"；**pathway** 反映"跑某个任务时前向实际经过的参数集合"。由于两条 pathway 都会经过 `shared` 部分，所以两者占比相加会超过 100%。

- **理解 pathway** ≈ `understanding_transformer + shared` ≈ **9.37B (53%)**。
  图像经过 `vision_model` → token 序列经过 `embed_tokens` → LLM 走非 `_mot_gen` 分支 → `lm_head` 输出 logits

- **生成 pathway**（以较复杂的单轮 thinking interleave 为例）≈ `generation_transformer + shared` ≈ **9.43B (54%)**。
  输入图像经过 `fm_modules.vision_model_mot_gen`，文本 prompt 经过 `embed_tokens` → LLM 走 `_mot_gen` 分支 → 经过 `lm_head` 输出 thinking 文本，再经过 `fm_modules.fm_head` 输出图像

### 为什么 `embed_tokens` 与 `lm_head` 算 `shared`，而不是仅属于"理解"

`embed_tokens` 是任何文本 token 的入口，自然是两条 pathway 共享的。`lm_head` 在一些情况下同样会被生成 pathway 调用，例如：t2i-reasoning 在出图前会先经过 thinking 阶段输出一段文本 token，这一阶段必须走 `lm_head`，因此它实际上也是两条 pathway 共用的关键模块——这正是把它归为 `shared` 的原因。
