# SenseNova-U1 提示词增强

> 简短的用户提示词——尤其是用于**信息图**生成时——通常会对图像模型限制不足。先将原始提示词通过强力 LLM 增强器处理，能够持续提升最终图像的结构性、排版质量、信息密度和"简报可读性"。本文档介绍如何开启该功能、我们推荐的上游 LLM，以及使用过程中需要权衡的因素。

## 1. 使用时机

在以下情况使用 `--enhance`：

- 用户提示词较短，或仅指明了一个主题（例如 `"2026年人工智能硬件图表"`）。
- 为演示文稿/幻灯片/海报生成素材，并且在调用文生图（T2I）模型之前，可以接受增加一轮额外的 LLM 交互开销。

在以下情况跳过 `--enhance`：

- 用户已提供一段长篇、结构化且可直接用于生产的提示词。
- 延迟或第三方 API 成本是首要考量因素。


## 2. 工作原理

```
user prompt ──► LLM (system prompt = infographic expander) ──► expanded prompt ──► SenseNova-U1
```

## 3. 配置

所有配置均基于环境变量，因此同一脚本无需修改代码即可切换后端。

| 环境变量 | 默认值 | 用途 |
| :------ | :------ | :------ |
| `U1_ENHANCE_BACKEND`  | `chat_completions` | `chat_completions`（兼容 OpenAI）或 `anthropic` |
| `U1_ENHANCE_ENDPOINT` | Gemini OpenAI 兼容 URL | 完整的 `/chat/completions` 或 `/v1/messages` URL |
| `U1_ENHANCE_MODEL`    | `gemini-3.1-pro`   | 请求体中发送的模型名称字符串 |
| `U1_ENHANCE_API_KEY`  | xxx            | Bearer token（必填） |

首先创建 `.env` 文件并填入四个必要参数，然后在 `examples/t2i/inference.py` 命令行中添加 `--enhance` 即可。
添加 `--print_enhance` 可显示原始提示词和增强后的提示词，便于调试。

如需使用 **SenseNova 6.7 Flash-Lite** 作为增强器，请从
[SenseNova 控制台 · Token Plan](https://platform.sensenova.cn/token-plan) 获取 API 密钥，然后设置：

```bash
U1_ENHANCE_BACKEND=chat_completions
U1_ENHANCE_ENDPOINT=https://token.sensenova.cn/v1/chat/completions
U1_ENHANCE_MODEL=sensenova-6.7-flash-lite
U1_ENHANCE_API_KEY=<your SenseNova API key>
```

### 3.1 推荐后端

| 模型 | 后端 | 端点模板 | 备注 |
| :---- | :------ | :---------------- | :---- |
| **Gemini 3.1 Pro**（默认） | `chat_completions` | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | 内部测评中信息图整体质量最佳，尤其擅长结构化、层级化内容。 |
| SenseNova 6.7 Flash-Lite | `chat_completions` | `https://token.sensenova.cn/v1/chat/completions` | 生成的中文质量接近 Gemini 3.1 Pro，单 token 成本更低，推荐用于生产环境。 |
| Anthropic Claude（Sonnet/Opus） | `anthropic`        | `https://api.anthropic.com/v1/messages` | 排版规范性强，开箱即用时信息密度略低。 |
| Kimi 2.5                      | `chat_completions` | `https://api.moonshot.cn/v1/chat/completions` | 中文增强效果好，测试显示生成以英文为主的信息图质量不高。 |
| Gemini 3.1 Flash-Lite（第三方服务） | `chat_completions` | `https://aigateway.edgecloudapp.com/v1/f194fd69361cd590f1fa136c9c90eca1/senseai` | 信息图整体质量高，生成速度快。 |
| Kimi 2.5/Qwen3.6-Plus（第三方服务） | `chat_completions` | `https://coding.dashscope.aliyuncs.com/v1/chat/completions` | 中文增强效果好，可灵活选择不同模型。 |

## 4. 效果对比

> 下表数据为使用相同基础提示词、分辨率及采样参数生成的图片。

| 基础提示词 | 无增强 | Gemini 3.1 Pro | SenseNova | Qwen3.6-Plus | Kimi 2.5 |
| :---------- | :------------- | :------------- | :------------- | :------------- | :------------- |
| 生成一副西红柿炒鸡蛋的中文教程图 | <img src="assets/showcases/prompt_enhancement/case1.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case1_gemini_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case1_sensenova_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case1_qwen_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case1_kimi_enhanced.webp" width="150"> |
| 生成一张介绍乒乓球比赛规则的图片 | <img src="assets/showcases/prompt_enhancement/case2.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case2_gemini_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case2_sensenova_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case2_qwen_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case2_kimi_enhanced.webp" width="150"> |
| Popularizing the importance of three meals a day | <img src="assets/showcases/prompt_enhancement/case3.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case3_gemini_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case3_sensenova_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case3_qwen_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case3_kimi_enhanced.webp" width="150"> |
| <details><summary>点击查看详细 Prompt</summary>这张信息图的标题是“猫咪与狗狗的终极对决”，采用了日系极致可爱与强烈色彩对比的插画风格。整体布局为左右对称的双栏对比结构，背景是带有细腻水彩纸纹理的米白色。画面通过色彩进行强烈的视觉分区，左半部分背景叠加了浅薄荷绿色的半透明波点图案，右半部分背景叠加了暖珊瑚粉色的对角线斜纹图案。长宽比为16:9。\n\n画面的正上方居中位置，使用超大号的粗体圆润无衬线字体写着主标题“猫咪与狗狗的终极对决”。主标题下方，使用稍小字号的深灰色黑体字写着副标题“毛孩子性格与生活方式指南”。在副标题的两侧，分别画着一个带有粉色肉垫的猫爪印图案和一个带有灰色指甲的狗爪印图案。\n\n在画面的正中央垂直方向，有一条由明黄色虚线构成的中轴线，将画面完美切割为左右两部分。中轴线的正中央，放置着一个带有爆炸星芒边缘的亮橙色圆形徽章，徽章内部用夸张的粗体等宽英文字母写着“VS”。\n\n画面左侧是猫咪的专属区域。顶部有一幅精美的插画：一只拥有大眼睛、脸颊红润的胖乎乎英国短毛猫，头顶带着一个小皇冠。插画下方用深绿色的粗体字写着“傲娇猫星人”。向下延伸，有三个垂直排列的信息模块。第一个模块中，画着一只蜷缩在原木高书架顶层熟睡的橘猫，旁边紧挨着文字“独立自主：每天需要16小时睡眠”。第二个模块中，画着一个印有小鱼骨头图案的浅蓝色陶瓷碗，碗里装满新鲜的生鱼片和鸡肉块，碗的右侧写着“纯肉食动物：需要高蛋白”。第三个模块中，画着一个半开的棕色纸箱，纸箱缝隙里露出一双发光的猫眼，旁边写着“暗中观察：喜欢狭小隐蔽的空间”。在左侧的最底部，有一个带边框的提示框，里面用倾斜的黑体字写着“专家提示：给猫咪充足的私人空间”。\n\n画面右侧是狗狗的专属区域。顶部有一幅生动的插画：一只吐着舌头、耳朵飞扬的金色寻回犬，脖子上戴着红色的波点项圈。插画下方用深红色的粗体字写着“热情汪星人”。向下延伸，同样有三个垂直排列的信息模块，与左侧保持完美的水平对齐。第一个模块中，画着一只前爪腾空、嘴里叼着绿色飞盘的边境牧羊犬，旁边紧挨着文字“社交达人：需要户外互动与奔跑”。第二个模块中，画着一个不锈钢宠物碗，里面装着混合了骨头形状饼干、胡萝卜丁和肉粒的狗粮，碗的左侧写着“杂食动物：营养均衡最重要”。第三个模块中，画着一只站立在后腿上、用双爪抱着人类大腿的小型贵宾犬，旁边写着“随时求抱抱：极度依赖主人的陪伴”。在右侧的最底部，有一个与左侧对称的提示框，里面用倾斜的黑体字写着“专家提示：保证每日充足的户外运动”。\n\n在画面的正下方，跨越左右两个区域，有一个淡黄色的宽大横幅。横幅内部用醒目的深藏青色粗体字写着“结论：无论性格如何，都是我们的完美伴侣！”横幅两端分别画着一颗跳动的红色爱心图案。整个画面信息密度极高，文字排版层次分明，色彩对比强烈且极具亲和力，所有元素均清晰可见且无重叠。图像的整体宽高比设定为9:16。</details> | <img src="assets/showcases/prompt_enhancement/case4.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case4_gemini_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case4_sensenova_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case4_qwen_enhanced.webp" width="150"> | <img src="assets/showcases/prompt_enhancement/case4_kimi_enhanced.webp" width="150"> |
