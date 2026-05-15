# SenseNova-U1-8B-MoT-Infographic 📊

**SenseNova-U1-8B-MoT-Infographic** 是在 U1-8B-MoT 模型基础上延长了 MT 阶段训练，并在 MT 与 SFT 阶段调整了理解和生成任务中的数据配比产生的模型。此外，我们针对文字渲染、背景和整体美观度进行了 RL 优化，使模型更好地适配 Infographic 复杂信息图的生成场景。

- **模型性能：** 在 BizGenEval hard/easy 上取得 **+6.8 / +4.3 points** 的显著提升，在 IGenBench Q-ACC/I-ACC 上取得 **+18.2 / +12.8 points** 的显著提升，同时保持稳健的视觉理解能力，无明显退化。
- **模型表现：** 模型能够生成图文交错、美观的复杂信息图，支持 100+ 种风格和布局，文字渲染能力强，能够渲染高密度小字、arXiv 论文等。

## Benchmark Highlights

| Model | BizGenEval Avg. (hard / easy) ↑ | IGenBench Q-ACC  ↑ | IGenBench I-ACC ↑ | OneIG(EN) ↑ | OneIG(ZH) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ***Commercial Models*** ||
| Nano-Banana-Pro | 76.7 / 93.7 | 90.6 | 48.8 | 58.1 | 56.8 |
| Nano-Banana-2.0 | 68.5 / 92.5 | 85.6 | 34.4 | 54.0 | 54.9 |
| GPT-Image-1.5 | 35.9 / 81.6 | 55.0 | 12.0 | - | - |
| Qwen-Image-2.0 | 45.5 / 65.8 | 50.0 | 3.0 | 54.1 | 50.9 |
| Seedream-4.5 | 30.1 / 66.2 | 61.0 | 6.0 | 56.4 | 55.0 |
| ***Open-source Models*** ||
| **SenseNova-U1-8B-MoT-Infographic** | **46.6 / 65.4** | **69.5** | **17.0** | **55.6** | 53.3 |
| SenseNova-U1-8B-MoT | 39.8 / 61.1 | 51.3 | 4.2 | 54.5 | **53.8** |
| Z-Image | 8.2 / 43.8 | 30.0 | 1.0 | 54.6 | 53.5 |
| Qwen-Image-2512 | 6.3 / 41.0 | 32.2 | 1.0 | 53.0 | 51.5 |
| Qwen-Image | 2.8 / 23.8 | 36.0 | 0.0 | 53.9 | 54.8 |
| Bagel | 2.0 / 3.7 | 4.9 | 0.0 | 36.1 | 37.0 |

<sub>IGenBench 分数以百分制展示。Commercial 与 open-source 组内模型按照 BizGenEval hard、BizGenEval easy、IGenBench Q-ACC、IGenBench I-ACC 四项算术平均值排序。OneIG 作为通用生成能力参考。完整分项结果建议放在 Hugging Face model card 中。</sub>

## 案例

以下案例展示了 Infographic 专用模型的效果，以及其在不同真实应用场景中的生成能力。

<table width="100%" style="table-layout: fixed;">
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 1：Infographic 1</b></div><img src="../all_small/028.webp" alt="信息图案例 028" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以深蓝色科技风格为背景，主题为“翼起进化：5G-A 极速时代”，突出展示5G-A网络的四大核心能力及其在智慧家庭中的应用。整体布局采用中心辐射式结构，中央是一个立体的、发光的5G-A芯片模块，周围环绕着数据流、光纤连接和抽象的科技光效，营造出高速、智能、互联的未来网络氛围。左侧与右侧分布多个半透明圆角矩形文本框，分别介绍不同技术特性，底部右下角则放置“天翼美好家”品牌标识及宣传语。

标题“翼起进化：5G-A 极速时代”位于左上角，字体较大，呈亮蓝色霓虹效果，并配有声波状装饰图案，下方附有进度条样式的横条，标注“下行速率 10Gbps”。

信息图主要包含以下五个部分：

1. **万兆下行（10Gbps Peak）**
   - 位置：左侧中部
   - 文本内容：
     &gt; 万兆下行 (10Gbps Peak)
     &gt; 这是速度的峰值。通过三载波聚合技术，实现网络能力的跃迁。在电信服务中，这是支撑超清8K与XR互动的核心基石。
   - 视觉元素：该文本框通过一条细线指向中央的5G-A芯片，强调其作为速度核心的地位。

2. **毫秒级时延（Low Latency）**
   - 位置：左下角
   - 文本内容：
     &gt; 毫秒级时延 (Low Latency)
     &gt; 通过边缘计算让反馈近乎实时。模糊的前景模拟了数据疾驰而过的速度感，引导用户关注稳定的网络核心。
   - 视觉元素：背景中可见二进制代码流与波浪形数据曲线，象征高速数据传输与低延迟响应。

3. **万物智联（Massive IoT）**
   - 位置：右上角
   - 文本内容：
     &gt; 万物智联 (Massive IoT)
     &gt; 每平方公里百万级连接数。远处的终端被柔化为数据节点，象征着5G-A网络如空气般无处不在却又干扰视线。
   - 视觉元素：文本框通过虚线箭头指向右上方悬浮的三个透明化建筑模型（代表城市或家居场景），它们被蓝色光环和数据点环绕，体现海量设备连接。

4. **通感一体化**
   - 位置：右侧中部
   - 文本内容：
     &gt; 通感一体化
     &gt; 基站即雷达。不仅传输数据，更能在焦平面外精准感知低空无人机或交通流量，实现低空经济的数字化管理。
   - 视觉元素：文本框通过虚线箭头指向一个基站塔模型，塔顶发出同心圆波纹，象征雷达探测功能；基站下方有网格化地面投影，表示感知范围。

5. **天翼美好家**
   - 位置：右下角
   - 视觉元素：包含“天翼美好家”品牌Logo（橙黄色螺旋图形+文字），以及一个白色背景的长方形文本框。
   - 文本内容：
     &gt; 天翼美好家
     &gt; 连接不只是技术，更是温度。千兆宽带+5G-A+全家流量共享，让每一个家庭成员都处于生活的C位。

此外，图像中还包含大量动态视觉元素：
- 中央5G-A芯片模块由多层电路板构成，顶部标有“5G-A”字样，四周连接多根光纤，散发蓝色光芒。
- 背景中遍布流动的二进制数字串（如“01001101...”）、数据波形、光斑和环形轨道，强化科技感与数据传输意象。
- 整体色调以深蓝为主，辅以亮蓝、白色高光和少量橙黄点缀，形成冷暖对比，突出品牌色。

该信息图通过图文结合、逻辑清晰的结构，系统展示了5G-A技术的性能优势与应用场景，尤其强调其在家庭智能化服务中的价值，传达出“技术即温度”的品牌理念。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 2：2024酒店智能化与市场趋势硬核洞察</b></div><img src="../all_small/033.webp" alt="信息图案例 033" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“不仅是睡觉，是代码重构空间”为醒目主标题，副标题为“2024酒店智能化与市场趋势硬核洞察”，整体采用撕纸拼贴风格的视觉设计，背景融合了混凝土墙面纹理、电路板图案、涂鸦文字（如“AI”、“Data”、“Experience”）以及霓虹色块，营造出强烈的科技感与街头潮流氛围。右侧以白色撕裂纸张为底，突出主标题和副标题，下方附有条形码及“UPGRADE NOW”字样，强化升级紧迫感。

信息图主体分为五个趋势模块，分别标记为TREND_1至TREND_5，每个模块包含英文主题名、中文解释、趋势描述及视觉符号，布局呈非对称网格状，通过撕裂边缘、箭头、色块等元素引导视线流动。

**TREND_1: AI CONCIERGE (数字心脏)**
- 视觉元素：左侧展示一个发光蓝色大脑，内部嵌有“AI”标志，神经网络延伸至下方城市建筑群（标有“HOTEL”），象征AI中枢控制酒店系统。
- 趋势描述：“从‘人效’到‘神效’。AI数字人替代机械前台，24h无缝交互。这不是削减成本，而是对服务边界的疯狂扩张。”
- 附加图标：右侧配有一个带Wi-Fi信号的金属骷髅头，红眼闪烁，增强赛博朋克风格。

**TREND_2: PERSONALIZATION (算法灵魂)**
- 视觉元素：彩色像素化人脸图像，由方格组成，体现数据化人格。
- 趋势描述：“极端个性化定制。利用大数据‘读心’。客人进门前，灯光、温度、枕头软度已由算法精准匹配。”
- 强调短语：“算法读心”置于绿色标签中，突出核心概念。

**TREND_3: HYPER SPACE (空间解构)**
- 视觉元素：右侧深紫色背景，巨大白色字体“酒店客房”被一把手术刀切割，象征空间重构；右下角配有VR眼镜图标。
- 趋势描述：“空间定义的彻底解构。酒店不再是旅途驿站。它是‘第三空间’、‘电竞舱’、‘沉浸式剧本杀场域’。场景即流量。”
- 数据标注：“将势超增高+150%”置于紫色渐变条中，强调增长潜力。
- 辅助文字：“毫克客房、办公室、直接场域”作为空间新形态举例。

**TREND_4: ESG HARDCORE (绿色野兽)**
- 视觉元素：绿色充电插头连接工业烟囱，烟囱旁有回收标志，象征绿色能源转型。
- 趋势描述：“硬核可持续主义。ESG不是口号是生存。低碳运营、智慧能源管理系统成为资本入场的‘入场券’。”
- 强调短语：“资本入场券”置于红色标签中，凸显商业价值。

**TREND_5: MARKET_SHOCK (市场重洗)**
- 视觉元素：一只拳头击碎写有“Traditional Hotel”的旧式标牌，碎片飞溅，象征行业颠覆；右侧为红白相间的“EXIT EVOLVE”警示牌，血迹效果增强冲击力。
- 总结内容：“拥抱智能，否则出局。存量时代，平庸即死。智能化不是选择题，而是逃离同质化内卷的唯一求生通道。”

此外，信息图中穿插多个动态视觉符号：
- 中央撕裂处有两条斜向黑条，分别写有“传统枯燥的旧式酒店”和“流光溢彩的智能场域”，形成对比。
- 左侧边缘有彩色几何色块堆叠，呼应“数据”主题。
- 整体色彩运用高对比度：荧光绿、亮紫、红、黑、白为主，配合阴影与立体字效果，强化视觉冲击。

该信息图通过混合多种图形语言——科技符号、街头涂鸦、数据可视化、比喻性插画——构建了一个关于2024年酒店业智能化转型的硬核叙事，强调技术驱动、用户体验革新、空间功能重构、可持续发展及市场竞争变革，最终指向“智能化是生存必需”的核心结论。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 3：Infographic 3</b></div><img src="../all_small/049.webp" alt="信息图案例 049" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 3 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 4：温度：能量的静止</b></div><img src="../all_small/092.webp" alt="信息图案例 092" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以水在冬季结冰、夏季融化的自然现象为核心主题，采用左右对称、冷暖色调分隔的视觉布局，将科学原理以艺术化水彩风格呈现。整体分为四个象限，围绕中心标题“水为何在冬结冰而在夏流淌？”展开，每个象限对应一个核心影响因素：温度、日照、降水和地理。视觉上，左侧以蓝色调为主，象征寒冷与冰晶，右侧以橙黄色调为主，象征阳光与流动的水体，形成鲜明对比。

中心标题“水为何在冬结冰而在夏流淌？”以大号艺术字体书写，其中“冬结冰”为深蓝，“夏流淌”为橙红，突出季节差异。标题周围点缀雪花、雨滴、冰锥等装饰元素，增强主题氛围。

左上角区域标题为“温度：能量的静止”，配图包括蓝色冰晶簇、水分子结构示意图（H₂O分子模型）及飘散的分子点。文字说明：“冬季气温降至0℃以下，水分子的热运动减缓，氢键得以稳定连接形成晶体结构。这是从无序流体向有序固体的转变。”其下方标注“核心机制：分子停滞”。

右上角区域标题为“日照：能量的注入”，配图为太阳、放射状光线及地球局部剖面图（标有10°N纬度线）。文字说明：“夏季太阳直射点北移，日照时长增加，地表接收的短波辐射远超散失热量。高能粒子撞击冰面，强行打破氢键束缚。”其下方标注“核心动力：辐射加热”。

左下角区域标题为“降水：形态的博弈”，配图包括云朵、雨滴、雪花等降水形态。文字说明：“冬季降水多以固态雪形式堆积，增加地表反射率（信噪比），进一步维持低温；夏季降雨则携带大量潜热，加速冰雪消融。”其下方标注“核心过程：相变反馈”。

右下角区域标题为“地理：环境的容器”，配图包括山脉、河流、地球网格（标有10°N和°N字样）及阴影区与向阳坡示意。文字说明：“纬度决定了基础温场，海拔则通过气压影响冰点。地形的阴影区与向阳坡创造了微气候，决定了结冰的局部边界。”其下方标注“核心变量：空间差异”。

整张信息图采用水彩手绘风格，线条柔和，色彩过渡自然，兼具科学严谨性与艺术美感。四个模块通过主题、颜色、图标和文字内容形成逻辑闭环，共同解释了水的季节性相变背后的多维驱动机制。所有文本均为中文，语言准确且富有科学表达力，未使用翻译或解释性语句。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 5：SCOUTING THE LEAGUE AVERAGE</b></div><img src="../all_small/000.webp" alt="信息图案例 000" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 5 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 6：Polyploidy in Agriculture: Total Chromosome Counts</b></div><img src="../all_small/001.webp" alt="信息图案例 001" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 6 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 7：Urban Construction Waste Outcomes</b></div><img src="../all_small/002.webp" alt="信息图案例 002" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 7 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 8：La Niñas globaler Fußabdruck</b></div><img src="../all_small/003.webp" alt="信息图案例 003" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 8 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 9：Healthcare Risk Management: Litigation Success by Error Type</b></div><img src="../all_small/004.webp" alt="信息图案例 004" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 9 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 10：Evolution of Peak Power Density in Standard Enterprises</b></div><img src="../all_small/005.webp" alt="信息图案例 005" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 10 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 11：Base Saturation of Fast Fashion Garments</b></div><img src="../all_small/006.webp" alt="信息图案例 006" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 11 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 12：SMART SAVER KIT</b></div><img src="../all_small/015.webp" alt="信息图案例 015" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>The infographic titled &quot;SMART SAVER KIT&quot; is presented on a light blue grid background resembling a cutting mat, with white dashed borders and decorative elements like percentage symbols (%) and barcode patterns along the edges. The overall design style is clean, modern, and playful, using 3D-rendered objects and speech-bubble-like labels to present information in an engaging, visually structured layout.

At the top center, the title &quot;SMART SAVER KIT&quot; is displayed in bold, black, uppercase letters on a white, ticket-shaped banner with a scalloped bottom edge.

Below the title, the infographic is organized into three horizontal rows, each containing two main features with corresponding icons and bilingual descriptions (English and Chinese). Each feature is accompanied by a stylized 3D object and a rounded rectangular label with a shadow effect, giving a layered appearance.

**Row 1:**
- **Flash Sales**: 
  - Icon: A yellow digital stopwatch labeled &quot;FLASHWATCH&quot; with &quot;TAME&quot; below the display, showing &quot;00:00&quot;.
  - Label: The word &quot;Flash&quot; is highlighted in yellow, followed by &quot;Sales&quot; in black. Description: &quot;Limited time, maximum discount.&quot; Below in Chinese: &quot;限时秒杀，折扣最大。&quot;
- **Coupons**:
  - Icon: A stack of pink, perforated coupons fanned out.
  - Label: The word &quot;Coupons&quot; is highlighted in pink. Description: &quot;Stack them for extra savings.&quot; Below in Chinese: &quot;叠加使用，省上加省。&quot;

**Row 2:**
- **Loyalty Card**:
  - Icon: A black VIP card with gold lettering &quot;VIP&quot; and a gold EMV chip on the left side, with four gray magnetic stripe lines at the bottom.
  - Label: The phrase &quot;Loyalty Card&quot; has &quot;Loyalty&quot; highlighted in gray. Description: &quot;Exclusive prices for members.&quot; Below in Chinese: &quot;会员专享，独特优惠。&quot;
- **Price Tracker**:
  - Icon: A white magnifying glass resting on a beige price tag.
  - Label: The phrase &quot;Price Tracker&quot; has &quot;Tracker&quot; highlighted in white. Description: &quot;Find the lowest historical price.&quot; Below in Chinese: &quot;追踪低价，货比三家。&quot;

**Row 3:**
- **Free Shipping**:
  - Icon: A small blue delivery truck with a beige cargo box.
  - Label: The phrase &quot;Free Shipping&quot; has &quot;Shipping&quot; highlighted in blue. Description: &quot;Zero cost delivery on all orders.&quot; Below in Chinese: &quot;全场包邮，送货上门。&quot;
- **Cashback**:
  - Icon: A green piggy bank with a golden coin slot and stitching details.
  - Label: The word &quot;Cashback&quot; is highlighted in green. Description: &quot;Get money back on every purchase.&quot; Below in Chinese: &quot;消费返现，边花边赚。&quot;

The visual elements are arranged symmetrically, with each icon placed to the left or right of its corresponding label. The color coding of the highlighted words (yellow, pink, gray, white, blue, green) matches the color of the associated icon, creating a cohesive visual connection between the text and image. The use of both English and Chinese text suggests a bilingual audience, likely targeting a market where both languages are commonly used, such as China or Chinese-speaking communities abroad.

All textual content is legible, with consistent font styles—bold sans-serif for headings and regular sans-serif for descriptions. The infographic does not contain any charts, graphs, or numerical data beyond the &quot;00:00&quot; on the stopwatch, which serves as a symbolic representation rather than actual data. The overall purpose is to inform viewers about various strategies for saving money while shopping, using clear visuals and concise bilingual explanations.</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 13：Love Unlocked: Read Between the Lines</b></div><img src="../all_small/023.webp" alt="信息图案例 023" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 13 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 14：Infographic 14</b></div><img src="../all_small/036.webp" alt="信息图案例 036" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以复古像素风格的电子游戏界面呈现，整体布局模拟计算机操作系统或控制台，背景为深蓝色电路板纹理，点缀着二进制代码和数据流，营造出科技感十足的视觉氛围。顶部中央显示“ENG_CAREER_LOAD...”加载进度条，当前进度为75%，暗示这是一个正在加载的职业生涯系统。

信息图分为四个主要模块：

1. **BASE_STATS: 教育背景属性**（左上角）
   - 显示角色头像：一名戴黄色安全帽、穿蓝色工装、手持扳手的工程师。
   - CLASS: ENGINEER
   - 学历等级映射：
     - Associate (Lv.1)
     - Bachelor (Lv.10)
     - Master (Lv.30)
     - PhD (Max Level)
   - 属性点分配：
     - 数学(Math) [ATK+5]
     - 物理(Phys) [DEF+5]
     - 代码(Code) [INT+10]
   - 图标：计算器、弹簧、芯片，分别对应数学、物理、代码属性。

2. **SKILL_TREE: 热门就业领域**（中央区域）
   - 核心节点：ENGINEERING CORE（一座古典建筑图标），作为所有技能分支的起点。
   - 分支结构：
     - **SOFTWARE ENGINEERING**（云朵与电路图标）：
       - Web Dev → AI/ML → Cybersec
       - AI/ML 节点旁有三个火焰图标，表示高热度或高竞争性。
     - **HARDWARE &amp; ELECTRONICS**（电阻器图标）：
       - Embedded Systems → VLSI Design → Robotics
       - Robotics 节点旁有两个火焰图标。
     - **CIVIL &amp; MECHANICAL**（齿轮图标）：
       - Structural Eng → Automotive → Aerospace
       - Aerospace 节点旁有一个火焰图标。
   - 所有分支通过绿色管线连接，部分管线带有双向箭头，表示技能可互通或相互影响。

3. **LOOT_TABLE: 薪资收益表**（右上角）
   - Entry Level (Lv.1-10): Bronze Coin [$$]（铜币图标）
   - Mid-Senior (Lv.11-30): Gold Bar [$$$]（金条图标）
   - Expert (Lv.31+): Diamond [$$$$]（钻石图标）
   - 底部标注：各地区工程专业的起薪中位数
     - US: ~$70k
     - CN: ~¥150k
     - EU: ~€50k

4. **BOSS_BATTLE: 就业挑战要素**（底部横幅）
   - 左侧：AUTOMATED AI
     - BOSS: AI取代 (Automated AI)
     - Threat Level: HIGH（红色字体）
     - 图标：一个红色眼睛、机械臂的机器人。
   - 中间：MARKET SATURATION
     - BOSS: 市场饱和 (Market Saturation)
     - Threat Level: MEDIUM（橙色字体）
     - 图标：一大群像素化人群。
   - 右侧：通关道具
     - 持续学习 [INT Boost]（紫色书本图标）
     - 软技能 [CHA Boost]（握手图标）

整体设计采用游戏化的术语（如“技能树”、“BOSS战”、“通关道具”、“属性点”、“掉落表”），将工程师的职业发展过程比喻为RPG游戏中的成长与挑战。颜色编码清晰：绿色用于技能路径，红色用于高威胁，橙色用于中等威胁，金币、金条、钻石象征不同级别的收入奖励。界面元素包含窗口边框、最小化/关闭按钮、状态指示灯等，强化了“系统加载”的主题。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 15：2024 Updates &amp; Reminders: Driving License Management &amp; Traffic Violation Processing</b></div><img src="../all_small/044.webp" alt="信息图案例 044" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 15 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 16：Mid-Autumn Festival: Healthy Mooncake Consumption &amp; Safety Guide</b></div><img src="../all_small/051.webp" alt="信息图案例 051" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 16 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 17：Core Pathways, Impacts and Future Priorities of Modern Hanfu Inheritance</b></div><img src="../all_small/054.webp" alt="信息图案例 054" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 17 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 18：What is Cyclical Unemployment?</b></div><img src="../all_small/061.webp" alt="信息图案例 061" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 18 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 19：Infographic 19</b></div><img src="../all_small/062.webp" alt="信息图案例 062" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“中外关系与国际交流活动归纳总结”，采用中国风与现代科技融合的插画风格，整体布局对称，以一座横跨水面的石拱桥作为视觉中心，象征连接与沟通。背景融合传统山水意境与现代都市天际线（如上海东方明珠塔、金茂大厦等），体现古今交融、东西对话的主题。画面左右两侧分别站立一位身着传统汉服的中国人物和一位身着西方古典服饰的外国人物，代表不同文明的交流与互动。

信息图通过多个装饰性对话框、卷轴和图标，系统呈现中外关系的四大核心维度：

1. **政治互信与战略沟通**（左侧红色云纹边框对话框）：
   - 文本内容：“高层互访频繁，建立多层次对话机制，深化全面战略伙伴关系。”
   - 视觉元素：位于左侧，背景有传统楼阁、山峦、松树，下方基座刻有龙纹图案，旁边立有一块刻有“诚信”二字的石碑。
   - 附加标语（下方红色对话框）：“核心利益相互尊重，求同存异。”

2. **经贸合作与互利共赢**（右侧蓝色齿轮与罗盘边框对话框）：
   - 文本内容：“共建‘一带一路’，推动贸易自由化便利化，加强产业链供应链合作。”
   - 视觉元素：位于右侧，背景有西式穹顶建筑、现代摩天大楼，下方基座呈电路板纹理，象征科技与经济。
   - 附加标语（下方蓝色对话框）：“开放市场，共享发展机遇。”

3. **文明互鉴与人文交流**（中央蓝色地球仪对话框）：
   - 文本内容：“举办文化年、旅游年，加强教育、科技、媒体、智库、青年等领域交流。”
   - 视觉元素：位于桥上方，以地球仪为核心，周围环绕科技线条，突出全球化与互联互通。

4. **总体目标与理念**（顶部金色卷轴）：
   - 文本内容：“构建人类命运共同体，携手应对全球性挑战。”

5. **核心价值与成果**（桥下中央徽章）：
   - 图标：一本打开的书，两侧饰以橄榄枝。
   - 文本内容：“促进民心相通，增进相互理解。”

6. **总结部分**（底部卷轴）：
   - 文本内容：“总结：中外关系在曲折中前进，国际交流活动日益丰富，为世界和平与发展作出重要贡献。”

整体设计运用了丰富的中国传统符号（如祥云、龙纹、亭台楼阁）与现代科技元素（如电路板、数据流、地球网格）相结合，色彩以蓝、金、红为主调，营造庄重、和谐、前瞻的氛围。所有文本均为简体中文，无英文或其他语言。信息结构清晰，逻辑层次分明，从具体领域到总体目标再到最终总结，形成完整叙事闭环。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 20：SEEK &amp; FIND: HIDDEN ROVER SECRETS,</b></div><img src="../all_small/063.webp" alt="信息图案例 063" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 20 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 21：Infographic 21</b></div><img src="../all_small/065.webp" alt="信息图案例 065" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以一个木质边框的软木公告板为背景，呈现了一则关于电力设施检修导致的停电通知。整体设计采用手绘风格的便签纸、照片和插图拼贴形式，通过不同颜色的便签纸（黄色、蓝色、白色、橙色、绿色）和图钉固定在软木板上，营造出社区公告栏的真实感。背景是老旧居民区街道场景，有行人、小吃摊、电线杆和窗台上的猫，增强了生活气息。

中央最醒目的是一张撕边白纸，用粗黑字体写着标题“停电通知！”，下方配有红色下划线，正文说明：“为了保障您的用电安全，我们将进行必要的电力设施检修。”

左上角黄色便签纸标明停电时间：“停电时间：2024年5月20日（周一）上午8:30 - 下午5:30”，并配有钟表和日历图标。

右上角蓝色便签纸标注影响范围：“影响范围：幸福社区、阳光小区、建设路沿线及周边商铺。”，旁边附有一幅手绘建筑平面图示意区域。

左侧中部是一张照片，显示两名身穿蓝色工作服、头戴黄色安全帽的电力工人正在电杆上作业，配有一个对话气泡：“检修中，请谅解！”

左下角白色便签纸提供生活提示：“生活提示：提前储备饮用水和食物。冰箱内食物请妥善保管，减少开启次数。”，并配有冰箱、水瓶和儿童打开冰箱的插图。

中间偏下绿色便签纸列出注意事项，共三条：
1. 请提前备好应急照明工具（手电筒、蜡烛）。
2. 尽量拔掉家中电器插头，避免来电瞬间损坏。
3. 停电期间，请勿乘坐电梯。
此部分配有手电筒、充电宝等应急工具图标，以及一人手持手电筒在黑暗中行走的插图。

右侧橙色便签纸提供服务热线：“如有疑问，请致电电力服务热线：95598”，并配有电话和对话气泡图标。

右下角一张插图描绘两位居民在栅栏旁交谈，其中一人说：“请大家相互转告，谢谢配合！”

右上角还有一个小方框插图，展示一栋深色建筑轮廓，窗户亮着灯，旁边有一个大问号，可能象征用户疑问或未覆盖区域。

所有便签纸均用彩色图钉（红、绿、蓝）或胶带固定，部分边缘有撕裂效果，增强真实感。整体布局层次分明，信息分区清晰，视觉引导自然，便于居民快速获取关键信息。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 22：VISION VOGUE | THE EYE ISSUE,</b></div><img src="../all_small/066.webp" alt="信息图案例 066" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 22 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 23：Infographic 23</b></div><img src="../all_small/069.webp" alt="信息图案例 069" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以赛博朋克（Cyberpunk）科幻风格呈现“猪肝烹饪方法与食谱分享”，整体采用深色背景搭配霓虹蓝、橙、绿等高对比度发光元素，营造出未来科技厨房的视觉氛围。画面中布满电路板纹理、机械臂、透明容器、数据显示屏等科技元素，强化了“赛博”主题。信息结构清晰，分为三个主要部分：上部为“烹饪前处理：净化与激活”，中部为“核心烹饪技法：赛博爆炒与低温慢煮”，下部为“精选赛博食谱：未来风味指南”。

**标题与布局：**
主标题“猪肝烹饪方法与食谱分享”位于顶部中央，以粗体蓝色霓虹字体显示，置于带有金属边框的矩形面板内。下方通过三条水平发光管道将内容划分为三大部分，每部分均有独立的副标题和对应图标。

---

**第一部分：烹饪前处理（上）**

此部分介绍猪肝的预处理步骤，共分三步，从左至右排列：

1. **切片与冲洗**  
   - 图标：左侧为一个发光的切割工具图标，右侧为机械臂喷水冲洗猪肝的场景。  
   - 文字说明：“去除血水，切薄片。”  
   - 视觉元素：两块生猪肝放置在金属网格托盘上，上方有蓝色光束模拟冲洗过程。

2. **浸泡去腥**  
   - 图标：一个透明玻璃罐，内部液体呈绿色并冒泡，底部有数字显示屏显示“103G”和计时器图标。  
   - 文字说明：“料酒/牛奶浸泡20分钟。”  
   - 视觉元素：罐体连接管线，暗示自动化处理流程。

3. **腌制上浆**  
   - 图标：一个玻璃碗中盛放猪肝片，一支注射器正注入橙色液体。  
   - 文字说明：“淀粉、生抽、胡椒粉抓匀。”  
   - 视觉元素：碗内食材被灯光照亮，突出“上浆”动作。

---

**第二部分：核心烹饪技法（中）**

此部分展示两种核心烹饪方式，左右对称布局：

- **【技法一：极速爆炒】**  
  - 图标：机械臂持锅在火焰上翻炒，锅中飞溅着红绿辣椒与猪肝，伴有闪电特效。  
  - 文字说明：“大火快炒，锁住水分，口感嫩滑。适合：爆炒猪肝。”  
  - 背景：橙红色火焰与电弧效果，强调高温快速。

- **【技法二：恒温慢煮】**  
  - 图标：一个透明真空低温烹调机（Sous-vide），内部有加热盘，右侧数字屏显示“65℃”。  
  - 文字说明：“低温长时，软糯入味，保留营养。适合：卤猪肝、猪肝汤。”  
  - 背景：冷色调蓝光，体现精准控温。

---

**第三部分：精选赛博食谱（下）**

此部分提供两款具体菜谱，左右并列，分别用橙色和蓝色边框区分：

- **食谱 A：霓虹尖椒爆猪肝**  
  - 图标：一道成品菜图，盘中猪肝片与青红尖椒混合，边缘有霓虹光晕。  
  - 食材列表：  
    “猪肝(已处理)，尖椒，蒜末，姜片，赛博酱汁(生抽，老抽，糖，醋)。”  
  - 流程说明：  
    “热油爆香料头 -&gt; 下猪肝滑炒变色 -&gt; 倒入酱汁与尖椒 -&gt; 大火收汁出锅。”  
  - 底部标签：“DOWNLOAD COMPLETE”

- **食谱 B：量子枸杞猪肝汤**  
  - 图标：一碗热气腾腾的汤，内含猪肝片、枸杞、菠菜，汤面漂浮蒸汽。  
  - 食材列表：  
    “猪肝片，枸杞，菠菜，姜丝，能量清汤(鸡汤或骨汤)，盐。”  
  - 流程说明：  
    “清汤煮沸加姜丝 -&gt; 下猪肝片与枸杞 -&gt; 煮至猪肝熟透(约2分钟) -&gt; 加菠菜烫软调味。”  
  - 底部标签：“DOWNLOAD COMPLETE”

---

**整体视觉与设计细节：**
- 背景为雨夜都市街景，远处可见霓虹招牌与高楼轮廓，增强赛博朋克氛围。
- 所有文字均使用无衬线字体，配合发光边框或阴影，提升可读性。
- 数据编码方式包括：时间（20分钟）、温度（65℃）、重量（103G）等数值明确标注。
- 动作流程通过箭头、图标和分步文字清晰表达，便于用户理解操作顺序。
- 整体设计融合了现代烹饪科学与未来主义美学，使传统食材的烹饪方法呈现出高科技感。

该信息图不仅传递实用烹饪知识，更通过视觉叙事构建了一个“未来厨房”的沉浸式体验。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 24：植物茎尖冷冻保存方案：分步指南与科学原理</b></div><img src="../all_small/079.webp" alt="信息图案例 079" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以深蓝色科技风格为背景，标题为“植物茎尖冷冻保存方案：分步指南与科学原理”，采用现代感十足的渐变蓝底、发光边框和网络状纹理设计，右上角点缀一个半透明的细胞结构图案，整体布局清晰、逻辑严谨，分为三个主要模块：左侧“知识模块（一）：基础与目的”，中间“核心流程（7步）”，右侧“知识模块（二）：关键试剂与环境”，底部为“结论”部分。

**标题与整体布局：**
主标题位于顶部中央，白色粗体字，下方有一条发光蓝色横线。内容区域被划分为三列，左右两列为知识模块，中间为核心流程图。底部为总结性结论栏，使用箭头形蓝色标签突出“结论”二字。

---

**知识模块（一）：基础与目的**

此模块位于左侧，包含两个子部分：

1. **冷冻保存定义与应用**
   - 图标：左上为大脑与地球组合图标，右下为DNA双螺旋与幼苗图标。
   - 文本内容：
     - 脱毒（配禁止病毒图标）
     - 遗传保存（配DNA图标）
     - 微繁殖（配三株幼苗图标）
   - 底部说明文字：“利用超低温技术长期保存生物材料活性，防止变异。”

2. **茎尖培养目的**
   - 图标：显微镜图标。
   - 文本内容：
     - 获取无菌、分生能力强的材料
     - 为冷冻保存提供标准化样本

---

**核心流程（7步）**

位于中心区域，以横向流程图形式展示，每一步配有插图、步骤编号、标题和简要说明，通过蓝色箭头连接，形成清晰的顺序流。

1. **茎尖培养**
   - 插图：培养皿中的一株绿色幼苗。
   - 说明：无菌条件下获取。

2. **DMSO预培养**
   - 插图：锥形瓶内含液体。
   - 说明：低浓度处理，提高耐受。

3. **引入冷冻保护剂**
   - 插图：滴管向小瓶中滴加液体，旁边有冒热气的容器。
   - 说明：高浓度保护，防止冰晶。

4. （未编号，但为流程衔接）→ 下一步为液氮储存前的准备，图示未直接标注，但由箭头引导至第5步。

5. **液氮储存**
   - 插图：标有“LN₂”的液氮罐。
   - 说明：长期低温保存。

6. **解冻过程**
   - 插图：试管置于水浴锅中，温度标注为“37-40°C”。
   - 说明：快速复温，避免重结晶。

7. **恢复与培养**
   - 插图：盆栽中的幼苗，带有根系。
   - 说明：转移至新鲜培养基，监测生长。

---

**知识模块（二）：关键试剂与环境**

位于右侧，包含三个子部分：

1. **DMSO作用（渗透保护）**
   - 插图：细胞结构示意图，显示分子渗透入细胞。
   - 文本内容：
     - 渗透入细胞
     - 降低冰点
     - 减少胞内冰晶形成
   - 补充说明：“二甲基亚砜，一种常用的渗透性冷冻保护剂。”

2. **液氮特性（-196°C，安全高效）**
   - 插图：温度计显示“-196°C”，旁有盾牌图标（带对勾）。
   - 文本内容：
     - 极低温度
     - 化学惰性
     - 成本相对低廉
   - 补充说明：“提供稳定、安全的超低温环境。”

3. **半固体培养基（营养+物理支撑）**
   - 插图：烧瓶内含彩色颗粒液体。
   - 文本内容：
     - 提供必要营养成分
     - 给予物理支撑
     - 维持适宜pH和渗透压

---

**结论**

位于底部，蓝色长条标签内，白色文字：
“茎尖冷冻保存整合生物原理与技术，需掌握DMSO、液氮及组织培养，应用前景广阔。”

---

**视觉与数据编码方法：**
- 使用统一的蓝色调和发光效果增强科技感。
- 所有步骤均用数字编号并配以直观插图，便于理解。
- 关键术语如“DMSO”、“LN₂”、“37-40°C”、“-196°C”等均明确标注。
- 图标用于辅助说明概念（如病毒、DNA、显微镜、温度计等），提升可读性。
- 流程图采用标准箭头连接，体现操作顺序。
- 文字排版清晰，层级分明，重点突出。

该信息图完整呈现了植物茎尖冷冻保存的技术路径、科学原理与实际应用，兼具教育性和实用性，适合科研人员或学生学习参考。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 25：Infographic 25</b></div><img src="../all_small/018.webp" alt="信息图案例 018" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这是一张以温暖手绘卡通风格呈现的社区治理主题信息图，整体采用米白色背景搭配彩色圆点网格边框，视觉上充满亲和力与生活气息。标题“社区治理：让爱串门！”以橙色立体艺术字置于顶部中央，字体活泼可爱，右侧配有一个手持公文包、头戴蓝色发卡的女社工卡通形象，面带微笑挥手致意。

图像主体内容分为四个主要部分：左上角为社区全景插画，展示蜿蜒小路连接着几栋温馨房屋，居民们在树下交流，蜜蜂飞舞，象征和谐邻里关系；中央偏右是一个拟人化的房子角色，张开双臂欢迎居民，旁边配有对话气泡解释“什么是社区治理？”——“以居民为中心，通过专业社会工作方法，多方参与共同解决社区问题。核心理念：‘助人自助’。通过激发内生动力，将钢筋水泥的楼宇转化成有温度的共同体。它是现代城市最有温度的微型实验！”

右侧竖立一张用回形针固定的“社工的沟通锦囊”纸张，内容包括：
- 专业关系是基石。
- 运用“同理心”去倾听居民的抱怨，在回应时多用“我们”。
- 如果遇到冲突，学会“中立调解”：理解情绪，关注利益。
- 温柔且坚定地推动共识。
上方配有两个耳朵和一颗红心的图标，强调倾听与共情。

下方横向排列四个步骤流程，通过箭头连接，形成清晰的逻辑链条：

**Step 1: 走访入户听民声**
插图描绘一位社工手持放大镜查看“Resident message board”（居民留言板），上面贴有多张便签。文字说明：“带着笑脸去串门。通过入户访谈、问卷调查，摸排社区痛点（如停车难、适老化改造等）。切记：社工要长出一双‘发现资源’的眼睛，不只是看问题，更要看潜力。”

**Step 2: 民主协商定方案**
插图展示几位居民围坐圆桌讨论，桌上摆放文件和灯泡图标（代表创意）。文字说明：“召开居民议事会。不是社工说了算，而是居民共同商量。通过罗伯特议事规则或社区漫步法，让每个人都有发声机会。保护‘沉默少数’的参与权！”

**Step 3: 五社联动聚力量**
插图呈现三只手紧握在一起，分别标有“政府”、“物业”、“志愿者”字样，周围有金色光芒射出。文字说明：“叮！跨界合作开启。整合社区、社会组织、社工、志愿者和慈善资源。就像拼图一样，把多方优势组合起来。绝对不要单打独斗！”

**Step 4: 成效评估与赋能**
插图描绘一棵结满笑脸果实的大树下，孩子们在彩旗装饰下欢快玩耍。文字说明：“项目结束后，进行总结反馈。不仅要看环境变美了没，更要看居民组织起来了没。培育社区骨干，让治理从‘项目驱动’转向‘自发运转’。”

整张信息图通过生动的插画、清晰的步骤分解和实用的文字指导，系统阐释了社区治理的核心理念与操作路径，强调居民参与、多方协作和持续赋能，传递出“让爱串门”的温情主旨。左侧边缘还点缀有小蜜蜂、情侣贴纸、格纹图案等装饰元素，增强趣味性与设计感。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 26：Core Impacts of Urbanization Across Key Sectors</b></div><img src="../all_small/046.webp" alt="信息图案例 046" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 26 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 27：POPULAR EDIBLE PREPARATIONS OF PUMPKIN</b></div><img src="../all_small/057.webp" alt="信息图案例 057" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 27 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 28：Infographic 28</b></div><img src="../all_small/072.webp" alt="信息图案例 072" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以复古像素风格的街机游戏界面为设计主题，整体视觉呈现为一个充满科幻感的电子游戏屏幕，背景是深邃宇宙中的星云与光束，中央是一个巨大的彩色漩涡，象征着知识探索的核心。标题“心理学与社会学：知识普及大冒险”以醒目的蓝紫色霓虹字体置于顶部，具有强烈的视觉冲击力。

画面采用对称式布局，左右两侧分别代表“玩家1：心理学”和“玩家2：社会学”，各有一名像素风格的角色形象。左侧心理学角色头盔内有大脑图案，周围环绕齿轮与对话气泡；右侧社会学角色头盔内有群体图标，背景是城市轮廓与网络连接图。两者通过箭头路径向中心汇聚，象征学科融合。

每个玩家路径包含三个关卡，以阶梯状平台展示：

**玩家1：心理学**
- 第一关：自我认知迷宫  
  描述：“了解内心，发现潜能”  
  视觉元素：人物站在三面镜子前，镜中反射不同表情，象征自我探索。
- 第二关：情绪管理挑战  
  描述：“调节情绪，保持平衡”  
  视觉元素：人物在雨中行走，周围有愤怒、悲伤、快乐等表情符号，象征情绪波动。
- 第三关：行为改变引擎  
  描述：“塑造习惯，实现目标”  
  视觉元素：人物奔跑于障碍赛道，前方有旗帜与砖块，象征目标达成。

**玩家2：社会学**
- 第一关：社会结构拼图  
  描述：“分析层级，理解规则”  
  视觉元素：人物在积木堆旁，积木颜色各异，象征社会阶层与结构。
- 第二关：文化影响波动  
  描述：“洞察趋势，适应环境”  
  视觉元素：人物冲浪于波浪之上，波浪中嵌入社交媒体、购物车、视频播放等图标，象征文化潮流。
- 第三关：群体互动竞技场  
  描述：“优化合作，解决冲突”  
  视觉元素：多人互动场景，有人交谈、握手、争论，体现群体动态。

画面中央是终极BOSS战区域，名为“BOSS战：现实应用大融合”。BOSS为一只三头紫色怪兽，每头分别标注“压力”、“偏见”、“沟通障碍”、“不平等”、“信贷”，象征现实中的多重挑战。两个玩家角色从两侧发射激光攻击BOSS，下方标语为“合作攻击！应用知识，共创未来！”

底部显示游戏结束界面，类似经典街机“GAME OVER”屏幕，文字内容包括：
- “GAME OVER”
- “通关：更美好的社会与人生”
- “SCORE: ∞”
- “INSERT COIN TO CONTINUE LEARNING”

整个信息图通过游戏化叙事将心理学与社会学的知识点转化为可体验的关卡挑战，强调跨学科合作应对现实问题，最终导向个人成长与社会进步。色彩以深蓝、紫色为主调，搭配霓虹绿、橙色高光，营造出80年代街机游戏的怀旧氛围，同时融入现代科技感。所有文本均为中文，无英文翻译或注释，语言风格活泼且富有激励性。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 29：R-SQUARED EXPLAINED</b></div><img src="../all_small/078.webp" alt="信息图案例 078" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 29 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 30：AMERICAN OPTIONS: FLEXIBILITY &amp; STRATEGY IN FINANCE</b></div><img src="../all_small/082.webp" alt="信息图案例 082" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 30 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 31：核心定义</b></div><img src="../all_small/007.webp" alt="信息图案例 007" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“徒手攀岩小知识”为主题，采用卡通插画风格，色彩柔和，背景为淡紫色渐变与白色拼接，点缀有云朵、蜜蜂、瓢虫、花朵、蝴蝶等自然元素，整体设计温馨、童趣，适合大众尤其是青少年阅读。

标题“徒手攀岩小知识”位于左上角，置于一个云朵形状的白色边框内，字体为深棕色，粗体，旁边配有戴黄色头盔的小男孩攀岩形象和一只橘白相间的猫咪，上方还有蜜蜂飞舞，营造出轻松活泼的氛围。

信息图内容分为五个主要模块，通过虚线箭头连接，形成环状或流程式布局：

1. **核心定义**（左侧中部）：
   - 标题为“核心定义”，置于浅黄色圆角矩形背景中。
   - 文字说明：“不使用绳索和保护装备进行攀登。”
   - 配图：一位穿黄色上衣、紫色裤子的女孩正在攀爬岩石，周围环绕星星、花朵装饰，突出“徒手”的概念。

2. **前期准备**（右上角）：
   - 标题为“前期准备”，置于浅黄色圆角矩形背景中。
   - 文字说明：“体能训练、心理建设、粉袋、攀岩鞋。”
   - 配图：一对男女正在做拉伸运动；下方展示一双攀岩鞋（一蓝一橙）、一个紫色粉袋、以及一个男孩正在搓镁粉的手部特写，直观展示所需准备事项。

3. **技巧与挑战**（中间偏右）：
   - 标题为“技巧与挑战”，置于浅绿色圆角矩形背景中，右侧用胶带贴纸效果装饰。
   - 图标部分包含四个圆形图标，分别代表：
     - 动态移动（人形跳跃）
     - 抓点技巧（手抓握动作）
     - 平衡控制（人形单腿平衡）
     - 心理压力（人形攀岩时紧张状态）
   - 文字说明：“动态移动、抓点技巧、平衡控制、心理压力。”

4. **安全注意**（左下角）：
   - 标题为“安全注意”，置于浅米色圆角矩形背景中。
   - 文字说明：“了解路线、天气状况、量力而行、自我评估。”
   - 配图：左侧为小男孩攀岩后坠落至蓝色垫子上的场景，表现安全防护；右侧为两个孩子表情担忧，一个捂眼、一个扶额，强调风险意识。

5. **精神与意义**（右下角）：
   - 标题为“精神与意义”，置于浅米色圆角矩形背景中。
   - 文字说明：“挑战自我、与自然融合、专注力提升、自由。”
   - 配图：一个男孩站在雪山之巅，张开双臂，头顶太阳和云朵，象征成就感与自由感。

各模块之间通过虚线箭头连接，形成从“核心定义”→“前期准备”→“技巧与挑战”→“安全注意”→“精神与意义”的逻辑闭环，引导读者系统理解徒手攀岩的全貌。视觉元素丰富，文字简洁明了，图文结合紧密，适合科普传播。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 32：Infographic 32</b></div><img src="../all_small/008.webp" alt="信息图案例 008" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 32 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 33：Infographic 33</b></div><img src="../all_small/009.webp" alt="信息图案例 009" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 33 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 34：Infographic 34</b></div><img src="../all_small/010.webp" alt="信息图案例 010" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以清新可爱的卡通风格呈现，整体背景为浅绿色，带有手绘波浪线、星星和彩色圆点装饰边框，营造出轻松活泼的氛围。标题“趣味跳房子：扩展玩法与关键点”采用大号艺术字体，红色与绿色搭配，下方副标题“让经典游戏更有趣、更健康、更安全！”进一步点明主题。

信息图内容分为五大模块，布局清晰，左右对称分布：

**1. 扩展玩法（变体与规则）**
位于左上角，包含三个子部分：
- **1. 多样格子阵**：展示三种不同形状的跳房子格子设计：左侧为经典竖向排列（数字1至10），中间为螺旋圆形（中心10，外圈依次为9,8,7,6,5,4,3,2,1），右侧为蛇形蜿蜒路径（数字1至16）。文字说明：“经典、螺旋、蛇形等多种形状。数字可自定义。”
- **2. 特殊动作规则**：通过四幅卡通插图展示四种跳跃动作：单脚跳、双脚跳、旋转跳、触地。文字说明：“增加单脚/双脚交替、旋转、触地等指令。”
- **3. 障碍挑战**：展示两种挑战模式：左侧为“雷区”，格子中放置炸弹图标，需跳过；右侧为“限时”，格子中放置闹钟图标，表示时间限制。文字说明：“设置雷区（跳过）、限时挑战、背身投掷。”

**2. 健康益处**
位于中上部，以四个圆形图标配文字说明的形式呈现：
- 肌肉图标：增强肌肉力量，锻炼腿部和核心肌群。
- 心脏图标：提高心肺功能，促进血液循环，增加耐力。
- 大脑图标：锻炼协调性与平衡感，改善身体控制能力。
- 两人互动图标：促进社交互动，培养团队合作与沟通。
中央有一名跳跃中的男孩卡通形象，连接各益处。

**3. 所需装备**
位于右上角，列出三项基本装备及其图片：
- 粉笔/胶带
- 投掷物（沙包/石子）
- 运动鞋
下方附注：“简单易得，安全第一。”

**4. 游戏流程（简易步骤）**
位于右中部，通过四步流程图展示游戏步骤：
1. 绘制格子阵（男孩用粉笔画格子）
2. 投掷标记物（男孩投掷小石子）
3. 跳跃前进（避开标记）（男孩跳跃过格子）
4. 返回拾取，完成回合（男孩弯腰捡起石子）
步骤间以黄色箭头连接，形成闭环。

**5. 安全提示（关键点）**
位于底部，包含三个安全建议，每项配有警示图标或卡通图示：
- 场地与装备检查：三角警示牌图标，文字说明“确保地面平整防滑，穿着合适运动鞋。”
- 避免碰撞：两个男孩相撞并有叉号图标，文字说明“保持安全距离，有序进行游戏。”
- 适度运动与休息：男孩喝水图标，文字说明“注意补水，避免过度疲劳。”

所有文本均为简体中文，排版清晰，图文结合紧密，视觉引导明确，适合儿童及家长阅读理解。信息结构完整，从玩法创新到健康价值、装备准备、操作流程再到安全注意事项，全面覆盖了趣味跳房子游戏的各个方面。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 35：Infographic 35</b></div><img src="../all_small/011.webp" alt="信息图案例 011" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这是一张以手绘插画风格呈现的《蒜香排骨》烹饪流程图解，整体采用暖色调背景（米黄色），布局清晰，图文并茂，通过四个主要步骤系统性地指导读者完成这道菜肴。标题《蒜香排骨》位于顶部中央，使用深棕色粗体书法字体，醒目突出。

整个信息图分为四个核心步骤，分别标注为“步骤1：准备食材”、“步骤2：腌制入味”、“步骤3：炸制蒜香”和“步骤4&amp;5：合味出锅”，每个步骤均配有详细的说明文字、插画示意图以及附加的“厨师秘技”和“小心得”气泡框，增强实用性和可读性。

---

**步骤1：准备食材**

- **说明文字**：
  “精选肋排，切段洗净，沥干水分；大蒜去皮剁成蒜末(大量)，姜切片，葱切段。所有调料准备就绪。”
- **插画内容**：
  木质砧板上摆放着切好的肋排块、一整头大蒜及数瓣蒜、生姜块与切片、葱段与葱花、小碗中的料酒、生抽、蚝油、糖、盐、淀粉等调料。
- **厨师秘技**：
  “选用带软骨的肋排，口感更丰富。蒜末要多，是灵魂所在，分两次使用。”
- **小心得**：
  “排骨一定要沥干或用厨房纸吸干水分，方便腌制入味。”

---

**步骤2：腌制入味**

- **说明文字**：
  “将排骨放入碗中，加入一半蒜末、姜片、料酒、生抽、蚝油、少许糖、盐和淀粉，抓匀腌制30分钟以上。”
- **插画内容**：
  一个白色瓷碗内盛放着腌制中的排骨，表面撒有蒜末、姜片，并有一勺食用油正被倒入碗中。
- **厨师秘技**：
  “最后封入少许食用油，锁住水分，炸出来更鲜嫩。腌制时间越长越入味。”
- **小心得**：
  “可以提前一晚腌制放入冰箱，第二天做更方便。”

---

**步骤3：炸制蒜香**

- **说明文字**：
  “锅中倒宽油，油温六成热下排骨，中小火慢炸至金黄熟透捞出。另起锅或留底油，小火将剩余蒜末炸至金黄酥脆。”
- **插画内容**：
  左侧小锅中炸制金黄酥脆的蒜末，右侧大锅中炸制排骨，锅底有火焰示意加热。
- **厨师秘技**：
  “炸排骨要二次复炸，第一次炸熟，第二次升高油温复炸30秒，表皮更酥脆。炸蒜末火候关键，焦了会苦。”
- **小心得**：
  “复炸时注意安全，避免油飞溅。炸好的蒜末也就是‘金蒜’，超级香！”

---

**步骤4&amp;5：合味出锅**

- **说明文字**：
  “将炸好的排骨倒入炸蒜末的锅中，快速翻炒均匀，撒上葱花和少许辣椒碎点缀，即可出锅装盘。”
- **插画内容**：
  锅中排骨与金蒜、葱花、辣椒碎混合翻炒，热气升腾，随后盛入盘中。
- **厨师秘技**：
  “出锅前大火快炒，让蒜香包裹每一块排骨，镬气十足，香气扑鼻。”
- **小心得**：
  “趁热享用，外酥里嫩，蒜香浓郁，简直停不下来！”

---

**最终成品图**：

在画面底部中央，展示了一盘堆叠整齐、色泽诱人的蒜香排骨，表面覆盖着金黄蒜末和翠绿葱花，盘边有少量酱汁，热气缭绕，极具食欲感。

---

**整体设计风格与视觉元素**：

- **风格**：手绘水彩质感，线条柔和，色彩温暖，带有生活气息。
- **布局**：从左上角开始按顺时针方向引导视线，步骤之间通过箭头连接，形成清晰的流程逻辑。
- **字体**：标题使用书法体，步骤标题为圆角矩形标签，说明文字为标准宋体，秘技与心得使用云朵状对话框，区分层级。
- **数据编码**：无数值图表，但包含明确的时间（如“30分钟以上”、“30秒”）、温度描述（“六成热”）和操作顺序。

此图不仅提供完整的菜谱，还融入专业技巧与个人经验，兼具教育性与趣味性，适合家庭烹饪爱好者参考使用。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 36：Infographic 36</b></div><img src="../all_small/012.webp" alt="信息图案例 012" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 36 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 37：CROP PHYSIOLOGY</b></div><img src="../all_small/013.webp" alt="信息图案例 013" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 37 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 38：Infographic 38</b></div><img src="../all_small/014.webp" alt="信息图案例 014" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以科技感十足的深蓝色调为背景，辅以霓虹蓝紫边框和电路板纹理装饰，整体风格现代、未来感强。标题“2024年小米及竞品最新电暖器机型参数对比”位于顶部中央，采用发光白色字体，醒目突出。

图表主体为横向三列对比表格，左侧列为主推产品“小米米家石墨烯踢脚线电暖器”，并配有金色“高亮推荐”徽章，其单元格背景为渐变金棕色，文字为金色；中间列为“戴森 AM09 无叶冷暖风扇”，右侧列为“飞利浦暖风机 3000系列”，这两列背景为深蓝色，文字为白色或浅蓝色，与主推产品形成视觉区分。

表格纵向分为六个参数类别，每个类别左侧配有图标：
- 芯片：图标为芯片形状
- 电池：图标为电池形状
- 功能：图标为方块加号和方块减号组合
- 重量：图标为秤盘
- 价格：图标为人民币符号“¥”
- 发售时间：图标为日历

各参数在三款产品中的具体数据如下：

**小米米家石墨烯踢脚线电暖器**
- 芯片：智能温控芯片
- 电池：无电池
- 功能：石墨烯速热，米家APP互联，语音控制，IPX4防水
- 重量：6.5kg
- 价格：¥599
- 发售时间：2023 Q4

**戴森 AM09 无叶冷暖风扇**
- 芯片：无芯片
- 电池：无电池
- 功能：喷射控流技术，冷暖两用，遥控器
- 重量：2.68kg
- 价格：¥3999
- 发售时间：2015

**飞利浦暖风机 3000系列**
- 芯片：无芯片
- 电池：无电池
- 功能：陶瓷发热，广角送风，过热保护
- 重量：4.1kg
- 价格：¥1299
- 发售时间：2023 Q1

信息图通过清晰的网格结构、颜色编码和图标辅助，直观呈现了三款电暖器在核心技术、智能功能、物理属性和市场定位上的差异，尤其突出了小米产品在智能互联和性价比方面的优势。所有文本均为简体中文，内容完整无遗漏。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 39：AUTO SPECS: VELOCITY</b></div><img src="../all_small/025.webp" alt="信息图案例 025" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以科技感十足的深蓝色调为背景，配以网格线和二进制代码边框，整体呈现未来派数字仪表盘风格。顶部标题为“AUTO SPECS: VELOCITY”，采用金属质感银色大字，下方有黄黑相间的警示条纹，右上角标有“TEST MODE”红色标签，表明当前处于测试模式。

图表中心是“MODULE_01: 动力心脏（中央仪表）”，一个圆形速度表式仪表盘，外圈刻度从0到8，指针指向约5.5位置，中心显示橙色数字“1025 hp”，下方标注“Turbo”。仪表盘左侧蓝色区域表示低负载，右侧红色区域表示高负载，当前指针位于红区，象征高功率输出状态。

左侧垂直排列四个图标：红色转速表、红色油箱（标有“RHNO”）、黄色电池（带闪电符号）和黄色警告三角，可能代表系统监控状态。

左上方模块“综合马力输出（BHP）”显示：
- 1025 hp
- 三电机全轮驱动
- 系统负载98%
- 弹射起步模式已激活

左下方“MODULE_02: 扭矩爆发（左侧折线图）”包含一个折线图，标题为“0-100km/h 加速分析”，横轴未标单位但可推断为时间，纵轴未标单位但可推断为速度。曲线呈陡峭上升趋势，终点处有一个白色圆点标记，对应加速完成点。旁边文字说明“扭矩爆发分析”：
- 1.99s
- G值瞬间突破1.5G
- 轮胎抓地力临界点监控中

右下方“MODULE_03: 硬件配置（右侧列表）”列出四项配置，每项前有金色连接线：
- 旗舰配置自检
- 制动：碳陶复合盘
- 空力：主动式尾翼
- 悬挂：电磁感应阻尼

在“MODULE_02”和“MODULE_03”之间，有一个独立小框“旗舰配置自检”，内容为：
- 制动：碳陶复合盘
- 动式尾翼 | 电磁感应阻尼

底部“MODULE_04: 能源效率（底部进度条）”包含一个双层进度条：
- 上层绿色条，标签“续航与能耗 (RANGE / E-EFFICIENCY)”
- 下层蓝色条，长度短于绿色条
- 文字说明：“CLTC 750km。动能回收等级：强。系统热管理：最优。”

背景中可见一辆跑车的蓝色线框模型，带有尾翼，暗示车辆设计特征。整体布局清晰分区，各模块通过标签编号区分，数据编码方式包括数字、图表、进度条和文本描述，视觉元素丰富，强调高性能与先进技术。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 40：Infographic 40</b></div><img src="../all_small/076.webp" alt="信息图案例 076" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“浅表切口手术部位感染（SSIs）与炎症”，采用清晰的流程图结构，系统性地介绍了手术部位感染的相关知识。整体布局分为顶部的“示例与症状”区域和下方四个主要模块：1. 定义与类型、2. 病理生理与风险因素、3. 炎症反应、4. 控制与影响。各模块通过箭头连接，形成逻辑递进关系，视觉风格简洁明了，使用不同颜色区分模块（蓝色、橙色、紫色、绿色），并辅以插图和图标增强理解。

在顶部“示例与症状”部分，展示了三种典型情况：
- 腹部：红肿与裂开（配图显示腹部切口发红、裂开）
- 肢体：脓性分泌物（配图显示肢体切口有黄色脓液渗出）
- 早期愈合阶段：红肿（配图显示轻微红肿）
同时列出典型体征：红肿（配红色圆点图标）、疼痛（配悲伤表情图标）、发热（配体温计图标）。

模块1：“定义与类型”（蓝色背景）：
- 图解展示皮肤分层结构，标注“浅表切口”（位于表皮与真皮层）、“深部切口”（深入皮下组织）、“器官/腔隙”（最深层，含内脏或腔隙）
- 文字说明：“涵盖浅表皮肤至深部组织及体腔。”

模块2：“病理生理与风险因素”（橙色背景）：
- 细菌因素：配图显示多种细菌形态，文字注明“例如：MRSA”
- 患者因素：配图包括人形图标、血糖滴管图标（标有“lco”）、肥胖人形图标，文字注明“糖尿病，肥胖”
- 手术因素：配图包括时钟图标、手术器械图标，文字注明“手术时间，污染”

模块3：“炎症反应”（紫色背景）：
- 血流增加 → 导致红肿/发热（配图显示血管扩张，血流方向箭头）
- 液体积聚 → 导致肿胀（配图显示组织间隙积液）
- 神经刺激 → 导致疼痛（配图显示神经元被闪电刺激）

模块4：“控制与影响”（绿色背景）：
- 术前：抗菌沐浴（配淋浴图标）
- 术中：无菌技术，抗生素（配医护人员传递药瓶图标）
- 术后：伤口护理（配敷料与消毒液图标）
- 影响：向下箭头指向三个负面结果——“增加发病率”、“延长住院”、“经济负担”（分别配绿色向下箭头、床铺图标、钱袋图标）
- 结论：“教育，依从性，监测可减少SSIs。”

所有模块之间通过彩色箭头连接，表示因果或流程关系，整体设计注重医学准确性与视觉传达效果，适用于医疗教育或患者宣教场景。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 41：Infographic 41</b></div><img src="../all_small/080.webp" alt="信息图案例 080" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“变异性，标准差与管理会计：图表与分析概览”，采用清晰的双栏布局，左侧聚焦于数据可视化与关键点说明，右侧则系统性地呈现核心概念、管理应用、案例研究及结论。整体设计风格简洁专业，使用浅蓝色背景与深色文字，辅以彩色图形元素增强可读性。

**左侧部分：变异性水平与概率分布**
- **主图表**：一个概率密度函数（PDF）曲线图，横轴标注“数据值（Data Value）”，纵轴标注“概率密度”。图中包含三条不同形态的钟形曲线，分别代表高、中、低三种变异性水平。
  - **绿色曲线**：最尖锐、最集中，表示“低变异性”，箭头指向右侧标签“风险低，预测易”。
  - **橙色曲线**：中等宽度，表示“中等变异性”，箭头指向“风险适中”。
  - **红色曲线**：最宽、最平缓，表示“高变异性”，箭头指向“风险高，预测难”。
- **下方补充框：“变异性水平关键点”**
  - 红色图标（⚠️）：“红色：高风险，不确定性大”
  - 橙色图标（⚠️）：“橙色：中等风险”
  - 绿色图标（✅）：“绿色：低风险，稳定性高”

**右侧部分：核心概念与管理应用**
- **核心概念**（三步流程图，由左至右）
  1. **变异性**：配图显示散点图与趋势线，文字说明“数据分散程度 → 影响风险与预算”。
  2. **标准差**：配图显示一把尺子测量正态分布曲线的宽度，文字说明“平均距离均值的度量 → 衡量可预测性”。
  3. **统计分布**：配图显示正态分布曲线，其中一部分被绿色填充，文字说明“正态分布（对称）&amp; 概率密度函数 (PDF) → 模型化不确定性并评估风险”。

- **管理应用**（四步流程图，从上至下再向右）
  1. **预算与预测**：图标为靶心与箭头，文字“设定现实目标”。
  2. **成本差异分析**：图标为放大镜与连接点，文字“识别偏差，控制成本”。
  3. **风险管理**：图标为盾牌与箭头，文字“量化波动，对冲策略”。
  4. **决策制定**：图标为分叉路径与对勾，文字“选择稳定供应商/投资”。

- **案例研究**（两个并列模块）
  - **电子商务**：图标为购物车与计时器，文字“减少处理时间变异性 → 提高效率”。
  - **制造业**：图标为工厂与齿轮，文字“利用标准差 → 稳定原材料成本”。

- **结论**（底部总结框）
  “理解变异性与标准差对于精确的财务分析、风险评估和竞争优势至关重要。”

**视觉与结构关系**
- 左侧的概率分布图直观展示了变异性与风险之间的反比关系，颜色编码（红=高风险，绿=低风险）贯穿全图，形成视觉一致性。
- 右侧的核心概念通过箭头串联，构成逻辑递进；管理应用部分则通过垂直与水平箭头构建了从基础到实践的流程。
- 整体信息流从理论（变异性→标准差→统计分布）过渡到应用（预算、成本、风险、决策），最后以行业案例和结论收尾，结构完整，层次分明。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 42：深渊生存：高压极寒适应机制</b></div><img src="../all_small/096.webp" alt="信息图案例 096" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以深蓝色调为主，营造出深海幽暗神秘的氛围，标题为“深渊生存：高压极寒适应机制”，字体为银白色立体字，位于图像顶部中央，具有科技感和视觉冲击力。整体布局呈对称结构，中心为一条半透明的深海鱼（类似吞噬鳗或深海鮟鱇鱼）的解剖插图，展示其内部骨骼与器官，周围环绕着发光粒子轨迹，象征生命活动与能量流动。四个主要模块分别位于左上、右上、左下、右下，每个模块均采用圆角矩形边框，带有金属质感和淡蓝色光晕，内含插图与文字说明，围绕中心鱼类形成放射状结构，强调各适应机制与核心生物的关联性。

各模块内容如下：

1. 左上模块：标题为“压力适应机制”，配图为细胞与血管网络的微观示意图，显示红细胞、血浆及高蛋白分子结构。文字内容包括：
   - 血液高蛋白低红细胞
   - 骨骼壳质极薄
   - 肌肉高压收缩
   - 旁边突出标注“高蛋白”（蓝绿色字体），强调关键特征。

2. 右上模块：标题为“温度适应策略”，配图为生物体横切面示意图，展示外层脂肪组织与内部绿色核心结构，模拟隔热层。文字内容包括：
   - 低代谢率节能
   - 脂肪蛋白隔热
   - 抵御极寒水温
   - 旁边突出标注“低代谢”（蓝绿色字体），强调核心策略。

3. 左下模块：标题为“生物发光应用”，配图为一只深海生物（如发光鱼或乌贼）的特写，其口部或触手发出明亮蓝光，周围有光点轨迹。文字内容包括：
   - 诱捕
   - 黑暗环境诱捕
   - 防御天敌威胁
   - 生存关键信号

4. 右下模块：标题为“特殊生理系统”，配图为分子结构与蛋白质链的抽象科学插图，呈现复杂化学键网络。文字内容包括：
   - 高压
   - 高压水分子处理
   - 生化过程适应
   - 极端环境生存

此外，在中心鱼类周围，有连接四个模块的虚线光带，将“高蛋白”、“低代谢”、“诱捕”、“高压”四个关键词分别指向对应的模块，形成逻辑闭环。背景中散布着大量半透明气泡状或细胞状图形，增强深海环境的真实感与沉浸感。整体设计融合了科学可视化与艺术渲染，采用三维效果、发光元素与冷色调光影，使信息传达既专业又具吸引力。所有文本均为简体中文，无英文或其他语言，符合中国科普或教育类信息图风格。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 43：SUSTAINABILITY LOG</b></div><img src="../all_small/040.webp" alt="信息图案例 040" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 43 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 44：The Evolution of Yi Yang Qianxi</b></div><img src="../all_small/041.webp" alt="信息图案例 041" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 44 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 45：用植物成长讲耐心与时间</b></div><img src="../all_small/071.webp" alt="信息图案例 071" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“用植物成长讲耐心与时间”为主题，采用温暖治愈的绘本风格，通过局部放大的微观视角，生动描绘了植物从种子到开花结果的五个成长阶段，隐喻人生中耐心与时间的价值。整体布局呈非线性网状结构，五个圆形放大镜分别聚焦于不同生长阶段，由藤蔓连接，象征生命连续不断的过程。每个放大镜内包含细腻的手绘插图，并配有编号和文字说明，形成清晰的视觉叙事流。

标题位于顶部中央，字体为深棕色手写体，背景点缀浅绿色与橙色圆点，营造柔和氛围。副标题“绘本治愈风·局部放大·温暖的微观世界”位于主标题下方，进一步说明设计风格与核心概念。

五个阶段按顺序排列如下：

① 耐心播种  
- 插图：一枚棕色种子埋入湿润土壤中，周围有微光闪烁，左侧配有一把小铲子。  
- 文字：“在静默中种下希望。耐心，是相信微小开始的力量。不要急于看见结果，给予它最初的信任与时间。”

② 时间萌芽  
- 插图：一株嫩绿幼苗破土而出，叶片上挂着水珠，背景为浅黄渐变，左侧有一个带时钟图案的绿色洒水壶。  
- 文字：“时间，是无声的催化剂。生命的破土往往在不经意间。成长是非线性的，接受它独特的节奏。”

③ 持续滋养  
- 插图：一株稍大的绿植在雨中生长，叶片宽大，水珠滴落，上方有太阳与雨滴符号，右侧有一只手托着小苗。  
- 文字：“日复一日的照料积累成变化。耐心不是静止，而是温柔的坚持。阳光、雨露、关爱，缺一不可。”

④ 等待花期  
- 插图：一朵含苞待放的花蕾，花瓣呈粉橙色，背景为淡绿色，旁边有一个沙漏，沙粒正缓缓流下。  
- 文字：“等待本身就是过程的一部分。不要催促绽放，信任自然的时间表。最美的风景，往往在漫长准备之后。”

⑤ 终见成果  
- 插图：一朵盛开的橙黄色花朵，中心有蜜蜂采蜜，背景星光点点，周围点缀着小果实。  
- 文字：“收获是耐心与时间的结晶。当花朵盛开，你会明白，所有的等待与付出都是值得的。成果并非瞬间，而是过程的圆满。”

此外，图中还穿插多个装饰性元素：如藤蔓缠绕连接各阶段、小型时钟图标、云朵与雨滴、小果实等，增强画面连贯性与童趣感。底部有一行总结性标语：“生命不息，耐心不止。与时间做朋友。”，字体纤细，居中对齐，强化主题。

整体色彩以大地色系（棕、绿、米黄）为主，辅以暖橙与浅粉，营造宁静、温暖、充满希望的视觉感受。图表类型为流程式信息图，结合插画与文字，通过“放大镜”这一视觉隐喻，引导观者关注细节与内在过程，强调耐心与时间在成长中的关键作用。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 46：Infographic 46</b></div><img src="../all_small/077.webp" alt="信息图案例 077" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以深蓝色为背景，采用清晰的流程图和模块化布局，系统地介绍了“铬酸银沉淀反应：实验演示与化学原理”。整体设计风格现代、专业，使用黄色和白色文字突出关键信息，配合实验室器皿、分子结构、齿轮等图标增强视觉表达。

标题位于顶部中央，以大号白色和黄色字体呈现：“铬酸银沉淀反应：实验演示与化学原理”。

主流程图部分从左至右展示反应过程：
- 左侧为无色硝酸银（AgNO₃）溶液，置于锥形瓶中，配有试管、烧瓶、原子模型等图标，标注“AgNO₃ (无色硝酸银)”。
- 中间为混合反应后生成的红棕色沉淀，锥形瓶内有旋转涡流和沉淀物，标注“Ag₂CrO₄↓ (红棕色铬酸银沉淀)”，并用箭头连接左右两侧，标示“混合反应”和“生成沉淀”。
- 右侧为黄色铬酸钠（Na₂CrO₄）溶液，同样置于锥形瓶中，配有分子结构、烧瓶、齿轮图标，标注“Na₂CrO₄ (黄色铬酸钠)”。
- 化学方程式位于中间下方，以醒目的黄色和白色字体显示：“2AgNO₃ + Na₂CrO₄ → Ag₂CrO₄↓ + 2NaNO₃ (旁观离子)”。
- 方程式下方有一条黄色边框的说明框，内容为：“沉淀反应原理：双置换反应，生成物溶解度超限时发生沉淀。”

下方分为三个并列的模块，每个模块均有橙色边框和标题：

1. **分析与解释**：
   - 溶解度与离子平衡（Ksp）
   - 旁观离子（净离子方程式简化）

2. **化学成分详解**：
   - 硝酸银 (Silver Nitrate)：防腐、摄影、药物制备
   - 铬酸钠 (Sodium Chromate)：染色、防腐、氧化剂

3. **相关化学概念与应用**：
   - 药物化学作用：纯化API、优化疗效、减少副作用

底部为总结性结论框，以黄色边框和白色文字强调：“结论：沉淀反应原理在药物、工业、分析化学中至关重要。”

整个信息图通过流程图、化学式、图标和分块文本，逻辑清晰地呈现了铬酸银沉淀反应的实验步骤、化学原理、成分特性及其在多个领域的应用价值。所有文字均为中文，符合要求。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 47：Infographic 47</b></div><img src="../all_small/081.webp" alt="信息图案例 081" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“第一季度财务报告分析”，采用深蓝色科技感背景，配以电路板纹理，整体布局分为六个彩色模块，呈2×3网格排列。每个模块使用不同主色调（蓝、橙、绿、紫、金、棕）区分主题，并包含图标、标题、概念说明、重要性/影响描述、图表和数据标签，视觉风格现代且专业。

---

**模块1：总收入（蓝色）**
- **标题**：1. 总收入
- **图标**：上升箭头与堆叠硬币、钱袋符号
- **概念**：商业活动总收入，衡量财务健康与增长潜力。
- **重要性**：盈利与增长的顶线指标。
- **图表类型**：折线图，X轴为月份（Jan, Feb, Mar），Y轴为金额（单位：万）
- **数据点**：
  - Jan: 36万
  - Feb: 38.2万
  - Mar: 37万
- **标注**：在Mar数据点旁有“增长潜力”标签，指向向上箭头。

---

**模块2：销售成本（橙色）**
- **标题**：2. 销售成本
- **图标**：齿轮组与带向下箭头的美元标签
- **概念**：生产直接成本（材料、人工、费用）。
- **影响**：有效管理是维持盈利关键。
- **图表类型**：环形图（Donut Chart），中心黑色，外圈分段显示成本构成
- **数据构成**（按顺时针方向）：
  - 第三方：26%（图标：文件+美元符号）
  - 内部：20%（图标：人形+加号）
  - 有效管理：35%（突出标注，无具体图标，但有对话框指向）
  - 开发：9%（图标：文档+笔）
  - 许可费用：15%（图标：文件+美元符号）
  - 交易：15%（图标：手握美元符号）
  - 客户：20%（图标：人形）
- **注释**：各部分百分比总和为130%，可能表示“有效管理”为优化目标而非实际成本占比，或存在数据重叠/分类交叉。

---

**模块3：毛利润（绿色）**
- **标题**：3. 毛利润
- **图标**：计算器+加号、上升柱状图
- **概念**：总收入减销售成本，反映运营效率。
- **实际应用**：分析毛利率评估产品盈利与定价。
- **图表类型**：折线图，X轴为月份（Jan, Feb, Mar），Y轴为金额（单位：万）
- **数据点**：
  - Jan: 31.8万
  - Feb: 34.4万
  - Mar: 44.8万
- **标注**：在Mar数据点旁有“运营效率提升”标签，指向向上箭头。

---

**模块4：运营费用（紫色）**
- **标题**：4. 运营费用
- **图标**：办公楼（标有OFFICE）、钱袋带减号
- **概念**：日常运营成本（SG&amp;A、营销）。
- **重要性**：控制费用可提升盈利与释放资源。
- **图表类型**：分组柱状图，X轴为月份（Jan, Feb, Mar），Y轴为金额（单位：万）
- **图例**：
  - 蓝色：销售、一般和行政费用
  - 紫色：市场营销费用
- **数据点**（估算值，基于柱高相对比例）：
  - Jan：
    - 销售、一般和行政费用：约12万
    - 市场营销费用：约9万
  - Feb：
    - 销售、一般和行政费用：约11万
    - 市场营销费用：约8万
  - Mar：
    - 销售、一般和行政费用：约8万
    - 市场营销费用：约6万
- **标注**：在Mar柱状图上方有“控制费用”标签，指向下降趋势。

---

**模块5：净利润（金色）**
- **标题**：5. 净利润
- **图标**：皇冠、堆叠金币、向上箭头
- **概念**：扣除所有费用后的最终利润，代表底线盈利能力。
- **结论**：持续增长是公司健康与价值标志。
- **图表类型**：折线图，X轴为月份（Jan, Feb, Mar），Y轴为金额（单位：万）
- **数据点**：
  - Jan: 8.7万
  - Feb: 10万
  - Mar: 19.7万
- **标注**：在Mar数据点旁有“底线盈利能力 公司健康标志”标签，指向发光数据点。

---

**模块6：季度总结与行动（棕色）**
- **标题**：季度总结与行动
- **图标**：指南针、靶心
- **文本内容**：
  - **总结**：总收入保持高位，毛利润显著提升，净利润强劲增长。
  - **行动建议**：继续优化销售成本结构，保持运营费用控制，关注高毛利产品线。
- **流程图**：三个圆形图标（分别对应收入、利润、目标）通过箭头连接至右侧大圆圈“持续增长与价值创造”，形成闭环反馈路径。

---

**整体设计特点**：
- 使用统一的深色背景与明亮边框，增强对比度。
- 每个模块顶部有编号和标题，结构清晰。
- 图表均配有明确的坐标轴标签和数据点数值。
- 文字排版简洁，重点信息用标签框突出。
- 配色方案与主题相关联（如绿色代表增长、紫色代表费用控制等）。
- 所有文字均为简体中文，符合中国商业报告语境。

该信息图完整呈现了企业第一季度核心财务指标的动态变化、成本结构及经营策略，兼具数据可视化与战略指导功能。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 48：Infographic 48</b></div><img src="../all_small/017.webp" alt="信息图案例 017" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 48 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 49：职称评审成分表</b></div><img src="../all_small/022.webp" alt="信息图案例 022" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“职称评审成分表”为主题，采用清新柔和的薄荷绿背景，搭配3D卡通风格的视觉元素，整体设计活泼且富有教育性。标题“职称评审成分表”以立体米白色字体呈现于木质搁板上，周围点缀着笔、笔记本、回形针等文具图标，营造出办公学习氛围。

图表主体为一个环形结构，由六个色彩鲜明、形状不规则的扇形板块组成，围绕中心一枚金色奖牌（带五角星）排列，象征职称评审的核心价值。每个扇形板块通过细线连接至对应的说明框，形成清晰的信息关联。

各板块及其对应内容如下：

1. **30% - 薪资**
   - 主题：薪资待遇提升
   - 说明文字：“直接挂钩工资档次，享受相应的绩效津贴与岗位补助。”
   - 副标题：“30% 收入跃迁”
   - 视觉元素：黄色扇形，内含钱袋与向上箭头图标，象征收入增长。

2. **20% - 晋升**
   - 主题：关键晋升跳板
   - 说明文字：“选拔中高层管理岗位或技术带头人的硬性门槛条件。”
   - 副标题：“职场硬通货”
   - 视觉元素：蓝色扇形，内含人物攀登阶梯并指向上的箭头图标，象征职业晋升。

3. **15% - 落户**
   - 主题：城市安居优待
   - 说明文字：“在一线城市积分落户中占据高分值，享受人才安居房补贴。”
   - 副标题：“落户加速器”
   - 视觉元素：青绿色扇形，内含房屋与对勾图标，代表落户与住房保障。

4. **15% - 能力**
   - 主题：专业能力背书
   - 说明文字：“通过论文、专利与业绩考核系统化证明个人技术水平。”
   - 副标题：“权威认证”
   - 视觉元素：橙色扇形，内含文件与红色印章图标，象征资质认证。

5. **10% - 地位**
   - 主题：行业社会地位
   - 说明文字：“进入专家库，参与行业标准制定，获得社会广泛认可。”
   - 副标题：“专家身份”
   - 视觉元素：紫色扇形，内含皇冠与点赞手势图标，象征社会地位与权威。

6. **10% - 退休**
   - 主题：退休养老保障
   - 说明文字：“退休金核算的重要依据，保障晚年生活质量稳步提升。”
   - 副标题：“长期红利”
   - 视觉元素：粉色扇形，内含摇椅与盾牌图标，象征晚年安稳与保障。

所有数据加总为100%（30% + 20% + 15% + 15% + 10% + 10%），构成完整的职称评审价值体系。信息框采用半透明圆角矩形设计，配色与对应扇形一致，增强视觉统一性。整个信息图通过直观的饼状环形布局、明确的百分比标注、生动的图标以及详实的文字说明，全面解析了职称评审在职业发展中的多维价值，强调其不仅是能力认证，更是影响薪资、晋升、落户、退休等多方面的综合“硬通货”。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 50：Infographic 50</b></div><img src="../all_small/026.webp" alt="信息图案例 026" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 50 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 51：EARTHQUAKE: 活下去</b></div><img src="../all_small/031.webp" alt="信息图案例 031" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这是一张以“EARTHQUAKE: 活下去”为主题、采用漫画风格和拼贴设计的地震应急生存指南信息图。整体背景为米色带有网点纹理，搭配黑黄警示条纹、手绘线条、涂鸦式文字框和鲜艳的粉、黄、蓝高亮色块，营造出紧张又具视觉冲击力的氛围。标题“EARTHQUAKE: 活下去”位于顶部中央，其中“EARTHQUAKE”使用粉色与黄色方块交替填充，“活下去”三字则置于黄色背景上，黑色粗体字，两侧配有波浪形震动符号，强调地震主题。下方散布多个“STAY ALIVE”印章式标志，增强警示感。

信息图内容分为五个编号部分，每个部分均配有插图和说明文字，按逻辑顺序呈现地震应对步骤：

01. 你的生存盲盒
- 插图：一个打开的背包，内含水瓶、急救包（带十字标志）、食物盒、手电筒、口哨和手摇收音机等物品。
- 文字说明：“不要等地动山摇才开始恐慌。水、高热量食物、急救包、口哨和手摇收音机。把它们塞进包里，放在门边。这是你的底牌。”
- 视觉元素：文字框边缘呈撕纸状，底部有黑黄警示条纹，关键词“底牌”用粉色高亮。

02. 黄金十秒
- 插图：一部老式翻盖手机，屏幕亮起蓝色，周围环绕着表示信号或警报的弧形波纹线。
- 文字说明：“当预警警报撕裂寂静，哪怕只有几秒，也足以改变结局。切断电源，远离燃气。不要跳楼，绝对不要进电梯！”
- 视觉元素：文字框边缘撕纸状，关键词“绝对不要进电梯！”用蓝色高亮，并有蓝色箭头从手机指向该部分。

03. 伏地·遮挡·抓牢
- 插图：一个简笔画小人蜷缩在桌子下，双手紧抱桌腿，上方有粉红色虚线箭头表示掉落物。
- 文字说明：“Drop, Cover, Hold on. 物理学不会说谎。迅速压低重心躲进桌下，死死抓牢桌腿。护住头部，护住你的头部和颈部，这是最高指令。”
- 视觉元素：文字框撕纸状，关键词“护住头部”用黄色高亮。

04. 逃离钢铁森林
- 插图：多栋高楼大厦倾斜倒塌，部分建筑被X标记，暗示危险。
- 文字说明：“如果在室外，立刻远离建筑、高压线和立交桥。玻璃幕墙和巨大广告牌是高空掉掉落的致命杀手。向空旷处奔跑。”
- 视觉元素：文字框撕纸状，关键词“致命杀手”用粉色高亮。

05. 黑暗中的回声
- 插图：两只手在废墟中紧握，手指敲击岩石，周围有波浪线表示声音传播。
- 文字说明：“主震后必有余震。如果被困，不要盲目大喊耗尽体力。用石头敲击水管，传递莫尔斯码。保存体力，等待微光。”
- 视觉元素：文字框撕纸状，关键词“保存体力”用蓝色高亮。

整体布局呈非对称网格结构，五个要点分布在画面不同位置，通过箭头和视觉引导线连接，形成从预防（01）→ 预警响应（02）→ 室内避险（03）→ 室外逃生（04）→ 被困求生（05）的完整逻辑链。所有文本均为中文，仅标题中保留英文“EARTHQUAKE”和“STAY ALIVE”作为强调。字体多样，包括粗体、手写体、印章体，增强信息层级和视觉趣味性。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 52：Shattered Silks: Multi-dimensional Deconstruction of the New Costume Drama</b></div><img src="../all_small/032.webp" alt="信息图案例 032" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 52 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 53：SPRING FESTIVAL SERVICE GUIDE</b></div><img src="../all_small/035.webp" alt="信息图案例 035" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 53 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 54：Infographic 54</b></div><img src="../all_small/037.webp" alt="信息图案例 037" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这是一张以软木板为背景、采用拼贴艺术风格设计的中国航天主题信息图（Infographic），整体布局自由散落，模拟真实剪报或研究笔记的视觉效果。标题“CHINA AEROSPACE”以多种颜色和字体的拼贴字母呈现，下方配有中文“星辰大海”及英文“Infographic”，形成中英双语主标题。整个画面由多张照片、手写注释、票据、蓝图、便签纸等元素通过图钉、回形针固定在软木板上，营造出一种充满探索精神与手工质感的视觉氛围。

信息图内容分为五个核心板块，每个板块均包含一张代表性的图片和一段描述性文字，分别介绍中国航天工程的重要成就：

1. **天宫筑梦：宇宙中的中国家园**
   - 图片：中国空间站（天宫）在地球轨道上的俯瞰图，可见其大型太阳能帆板和模块化结构。
   - 文字：“天宫筑梦：宇宙中的中国家园。全面建成属于中国人的太空母港。常态化乘组轮换与太空授课，在失重空间里延续华夏文明的生活烟火。”
   - 旁边附有一张撕边的浅色票据，印有“SCONE”、“160.0”、“217 20.100”等数字，可能象征某种数据记录或纪念凭证。

2. **嫦娥奔月：跨越千年的双向奔赴**
   - 图片：月球表面的玉兔号月球车，带有太阳能板和天线，留下清晰车辙。
   - 文字：“嫦娥奔月：跨越千年的双向奔赴。从绕月、落月到采样返回，乃至实现人类首次月背软着陆。古老神话照进现实，带回月壤的独特浪漫。”
   - 该板块右侧用回形针固定，背景为黑色撕边纸张。

3. **长征破夜：剑指苍穹的力量**
   - 图片：三张不同角度的长征系列火箭发射场景，火焰喷射，塔架林立；另有一张大型卫星地面接收天线的照片。
   - 文字：“长征破夜：剑指苍穹的力量。不断刷新发射记录的运载火箭家族。高密度、高成功率发射，托举起华夏民族所有的飞天梦想。”
   - 此区域中央有一个白色网格状的“X”形装饰物，可能象征发射轨迹或技术符号。

4. **天问探火：红色星球的中国印记**
   - 图片：火星表面的红色沙丘地貌，橙黄色天空，展现荒凉而壮美的外星景观。
   - 文字：“天问探火：红色星球的中国印记。一次性实现绕、着、巡三大跨越。祝融号在火星大地留下中国足迹，丈量星际探索的全新边界。”
   - 上方附有一张“BOARDING PASS”登机牌式样的票根，红白配色，印有“地球至火星”字样和条形码，左侧标注坐标“20°39&#x27;8&quot;，22°5&#x27;36&quot;”。

5. **北斗指路：寰宇尽在掌握**
   - 图片：蓝色底的卫星轨道设计蓝图，标有“Satellite orbit design”，绘有卫星运行轨迹和角度刻度（如90°, 125°, 150°, 180°等），右下角为星图网格。
   - 文字：“北斗指路：寰宇尽在掌握。独立自主的全球卫星导航系统。无论白天黑夜、身处何方，夜空中最亮的中国星为你精准引航。”
   - 文字部分使用类似笔记本纸张的格子纸样式，左侧有打孔圆圈。

此外，画面中还散布着其他细节：
- 左上角有一张蓝白相间的网格图纸，被撕开一角。
- 右上角有一张撕边纸条，显示坐标“120°32&#x27;30&quot;，120°5&#x27;40&quot;”。
- 底部有一张小纸条，列有数字串：“800&#x27;50&#x27; 62&#x27;63&#x27; 456” 和 “200&#x27;6:04&#x27; 310636°”，可能是某种编码或测量数据。
- 多处使用黑色手绘线条勾勒圆形、箭头、波浪线，增强手作感和动态引导。
- 整体色调以棕色软木、蓝色科技、白色文字为主，搭配火箭火焰的橙黄、火星的赤红，色彩丰富且具有层次感。

该信息图通过非传统的、富有创意的拼贴形式，将中国航天四大标志性工程——空间站、探月、探火、导航系统——有机整合，兼具科普性与艺术感染力，生动诠释了“星辰大海”的探索主题。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 55：EARTHEN SOULS: THE RURAL VERTICALITY OF BEING</b></div><img src="../all_small/038.webp" alt="信息图案例 038" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 55 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 56：Judicial Landscape: Guizhou Civil Judgment Summary,</b></div><img src="../all_small/039.webp" alt="信息图案例 039" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 56 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 57：THE RENAISSANCE OF PAPER</b></div><img src="../all_small/042.webp" alt="信息图案例 042" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 57 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 58：PSYCHOLOGICAL GUIDE TO EMOTION MANAGEMENT: UNDERSTAND &amp; REGULATE YOUR FEELINGS EFFECTIVELY,</b></div><img src="../all_small/055.webp" alt="信息图案例 055" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 58 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 59：Infographic 59</b></div><img src="../all_small/064.webp" alt="信息图案例 064" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“体育界人物法律问题与争议技术剖面图”（Technical Diagram of Legal Issues &amp; Controversies in Sports Figures），采用工程蓝图（blueprint）风格设计，整体呈现为多层叠加的透明图纸，背景为浅灰色网格纸，营造出技术剖析的视觉效果。主色调为蓝色、白色和红色，关键警告信息以红色高亮显示，增强视觉冲击力。

图表结构分为三个主要层次：

1. **底层：体育生态系统（Sports Ecosystem）**
   - 位于最下方，以深蓝色网格背景呈现，描绘了体育产业的核心参与方及其相互关系。
   - 中心节点为“体育生态系统”，通过双向箭头连接五个实体：
     - 运动员
     - 俱乐部/联盟
     - 赞助商
     - 管理机构
     - 公众/媒体
   - 各实体间存在复杂的交互网络，体现系统性关联。

2. **中层：三大“故障模块”（Failure Modules）**
   - 以半透明图纸形式叠于底层之上，每个模块独立成块，包含具体违规类型、技术分析或法律框架。
   - 模块一：**兴奋剂违规（Stimulant Violation）**
     - 标题：“故障模块 兴奋剂违规”
     - 图标：化学分子式（HO-C6H4-CO-NH-CH(COOH)-COOH）、试管、烧杯
     - 关键事件链：
       - “检测失败” → “禁赛” → “声誉受损”
     - 技术分析文字：“物质代谢路径异常，样本污染风险，法律抗辩复杂性”
   - 模块二：**税务欺诈与财务纠纷（Tax Fraud &amp; Financial Disputes）**
     - 标题：“故障模块 税务欺诈与财务纠纷”
     - 图标：电子表格、金币堆、警戒三角
     - 关键事件链：
       - “审计追踪：离岸账户结构，虚假申报，资产转移路径” → “逃税漏税”
     - 无直接后续结果，但隐含法律后果。
   - 模块三：**合同违约与转会争议（Contract Breach &amp; Transfer Disputes）**
     - 标题：“故障模块 合同违约与转会争议”
     - 图标：断裂链条、合同文件、法槌
     - 法律框架说明：“解约金条款，肖像权归属，第三方所有权”

3. **顶层：争议与后果（Controversies &amp; Consequences）**
   - 位于最上层，以机械齿轮、问号、感叹号、破碎屏幕、法槌等符号构成动态流程图。
   - 中心主题：“争议与后果”，由多个故障模块触发。
   - 主要后果包括：
     - “职业生涯终结”（红色警示三角）
     - “公众信任崩塌”（破碎屏幕图标，出现两次）
     - “法律诉讼”（法槌击打产生火花）
     - “行为不端与暴力事件”（盾牌破裂、拳头、闪电，子项包括“性骚扰”、“家庭暴力”、“赛场冲突”）
   - 行为准则说明：“道德条款触发，刑事调查程序，公众舆论压力”
   - 系统性风险警告框（红色发光条）：
     - 文字：“系统性风险警告：法律边界模糊，监管滞后，利益冲突加剧”

**其他文本信息：**
- 图号：SPORT-LAW-001（在多个位置重复标注）
- 版本：冲突叠加版（右下角标注）
- 整体布局采用“剖面图”概念，将抽象的法律争议具象化为可拆解的技术系统，强调各模块之间的因果传导和系统性风险。

**视觉元素与数据编码：**
- 使用齿轮、电路板线条、箭头表示流程与关联。
- 警示符号（黄色三角、红色感叹号）用于标记高风险点。
- 破碎玻璃、断裂链条等图像象征系统崩溃。
- 颜色编码：蓝色代表系统正常运行，红色代表危机或警告，灰色代表中立或背景。

该信息图通过高度结构化的视觉语言，完整还原了体育界人物面临的主要法律争议类型、其内在机制、传导路径及最终导致的职业与社会后果，兼具教育性和警示意义。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 60：个人成长与命运：浮世绘卷中的启示</b></div><img src="../all_small/067.webp" alt="信息图案例 067" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图采用传统日本浮世绘艺术风格，整体布局分为上下两部分：上半部分为一幅寓意深刻的插画，下半部分为三栏式文字说明区域。整体色调以蓝、白、棕、粉为主，背景带有仿古纸张质感，边框装饰有云纹和波浪纹样，营造出古典东方美学氛围。

**上半部分插画描述：**
画面主体是一位身着深蓝色和服、腰系绿色裙裤的古代日本男子，正站在一艘木制小舟中，手持长桨划行于汹涌海浪之上。他右手高举一盏灯笼，灯笼上书“智慧”二字，象征内在指引。小舟位于画面中央偏右，正迎向右侧巨大的海浪，浪花翻腾，形成一个类似《神奈川冲浪里》的标志性浪形结构。浪峰处标有“命运之浪”，浪底则标注“自我意志”，暗示个体意志在命运洪流中的作用。

背景中，左侧山崖上有一座红色鸟居，其旁立有“起点”字样，山崖上樱花树繁花盛开，花瓣随风飘落，增添诗意与无常感。远处是标志性的富士山，山顶覆盖白雪，山后一轮金色圆日（或满月）高悬，上方标注“彼岸/目标”，象征终极理想或人生目标。天空右上角还有一弯新月，与太阳形成昼夜交替的意象，隐喻时间流转与人生阶段。

**下半部分文字说明区域：**
标题为“个人成长与命运：浮世绘卷中的启示”，字体较大，居中排列，下方分三栏展开论述，每栏包含中文标题、英文副标题及正文内容。

- **第一栏：命运的洪流 (The Torrent of Fate)**
  - 内容：“命运如巨浪，有时不可掌控。它包含环境、机遇与不可抗力。启示：接纳无常，顺势而为，不被巨浪吞噬。”

- **第二栏：成长的舟楫 (The Boat of Growth)**
  - 内容：“成长是手中的桨与舵，是内在的修行与智慧积累。它赋予我们驾驭的能力。启示：持续学习，磨炼意志，提升自我导航的能力。”

- **第三栏：交织与彼岸 (Interweaving and The Other Shore)**
  - 内容：“命运与成长交织前行。在波折中调整航向，最终抵达心中的彼岸。启示：在努力与接纳中寻找平衡，活出属于自己的精彩旅程。”

三栏之间以细线分隔，每栏顶部有装饰性云纹图案，底部亦有波浪纹饰呼应主题。整体设计将视觉隐喻与哲学思考紧密结合，通过浮世绘的经典元素（如富士山、巨浪、樱花、鸟居）构建了一个关于人生旅程的深刻隐喻系统：从“起点”出发，在“命运之浪”中凭借“智慧”与“修行”作为“舟楫”，依靠“自我意志”掌舵，最终驶向“彼岸/目标”。文字部分则系统化地提炼了这一旅程的三个核心维度——对命运的接纳、对成长的主动塑造、以及二者的动态平衡，构成完整的成长哲学框架。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 61：时尚潮流产品推广</b></div><img src="../all_small/073.webp" alt="信息图案例 073" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“时尚潮流产品推广”为主题，采用复古波普艺术风格设计，整体布局为方形，背景为仿旧纸张质感，边缘有撕裂效果。主标题“时尚潮流产品推广”位于顶部中央，字体粗大、黑色填充，带有蓝、粉、黄三色立体轮廓，视觉冲击力强。

信息图主体分为三个主要区域：左侧为“趋势洞察”，右侧为“推广策略”，底部为“核心价值”。中间区域为插画式视觉中心，描绘三位穿着潮流服饰的青年（一男两女），搭配滑板、运动鞋、背包、黑胶唱片等元素，色彩鲜艳，使用网点和几何图形构成，体现街头文化氛围。

**趋势洞察（左侧）**
标题：“趋势洞察”
包含四个子项，每个配有图标和说明文字：
1. 图标：磁带（彩色条纹）
   - 文字：“复古回潮 重塑经典，怀旧风尚。”
2. 图标：喷漆罐（粉色背景）
   - 文字：“街头文化 个性表达，城市脉搏。”
3. 图标：绿叶（黄色背景）
   - 文字：“可持续性 环保材料，循环时尚。”
4. 图标：像素化机器人头像（蓝色背景）
   - 文字：“数字融合 虚拟体验，科技赋能。”

**推广策略（右侧）**
标题：“推广策略”
包含四个子项，每个配有图标和说明文字：
1. 图标：手机屏幕内含Instagram、Facebook、Twitter图标
   - 文字：“社交媒体 内容种草，互动传播。”
2. 图标：两个交叠的圆环（蓝粉双色）
   - 文字：“跨界联名 强强联合，破圈效应。”
3. 图标：店铺门面（蓝顶红墙）
   - 文字：“限时快闪 稀缺体验，制造话题。”
4. 图标：麦克风与星星（粉色背景）
   - 文字：“KOL合作 意见领袖，精准触达。”

**核心价值（底部）**
标题：“核心价值”
包含三个子项，每个配有图标和说明文字：
1. 图标：人脑轮廓内含齿轮（粉蓝配色）
   - 文字：“个性表达 独特自我，不被定义。”
2. 图标：地球仪上分布四个人脸（蓝底）
   - 文字：“文化认同 连接社群，共鸣归属。”
3. 图标：灯泡内含闪电符号（黄底）
   - 文字：“创新体验 前沿设计，引领未来。”

**视觉元素与风格**
- 配色方案：以黑、白为主色调，辅以高饱和度的亮蓝、粉红、明黄，形成强烈对比。
- 字体：标题使用粗体无衬线字体，正文为简洁清晰的现代字体。
- 图形风格：采用复古印刷效果，如网点、墨迹晕染、边缘磨损，营造80-90年代街机海报感。
- 中央插画：人物姿态动感，服装细节丰富（如连帽衫、工装裤、棒球帽），运动鞋为Nike Air Jordan 1风格，突出潮流单品。

该信息图通过结构化分区、图文结合、色彩对比和复古美学，系统呈现了时尚潮流产品的市场趋势、推广方法及品牌核心理念，适用于品牌宣传、营销策划或行业分析场景。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 62：Understanding Investment Fees &amp; Net Returns</b></div><img src="../all_small/075.webp" alt="信息图案例 075" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 62 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 63：疫情期间活动举办防控指南与文明倡议</b></div><img src="../all_small/088.webp" alt="信息图案例 088" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以像素风格游戏界面为设计主题，整体背景为深蓝色，顶部显示“SCORE 5”和“STATS”进度条，营造出一种互动游戏的视觉氛围。标题为“疫情期间活动举办防控指南与文明倡议”，采用白色粗体字置于蓝色边框内，居中突出。

整体布局采用对称结构，以一座横跨峡谷的木制吊桥为中心，象征“跨间隙桥梁主体”，连接左右两侧的“活动前防控准备环节”（左侧城堡图标）和“活动中现场防控管控”（右侧体育场图标）。中间区域标注“间隙区域”，解释为“筹备期到活备期到活动开展期的疫情传播风险与防控落地难点”，并配有多个病毒图案和黄色警告标志，强调潜在风险。

### 左侧：“活动前：筑牢防控前置防线”
- 标题下方注明“三类准备工作缺一不可”。
- 包含三个具体措施，每个均配有像素风格插图：
  1. **人员风险排查**：一名戴口罩男子手持清单和手机，手机显示绿色健康码。
  2. **场地全面消杀**：一名穿防护服、戴口罩的工作人员喷洒消毒液。
  3. **防疫物资储备**：展示口罩、消毒液、体温计等物品图标。

### 右侧：“活动中：严抓现场动态管控”
- 标题下方注明“全流程落实防控要求”。
- 包含四个具体措施，每个均配有像素风格插图：
  1. **进场双核验**：一名安保人员使用设备扫描二维码。
  2. **常态化巡检**：一名清洁人员推着清洁车进行巡查。
  3. **应急快处置**：两名医护人员在门口处理突发情况。
  4. （注：此处原图中第四个措施未明确标示名称，但根据上下文和图标可推断为应急响应或医疗保障）

### 中央吊桥区域：“跨间隙桥梁主体”
- 标题为“全员文明参与倡议”，下方有“桥梁分步标识”说明。
- 包含三项公民行为倡议，配图分别为：
  1. 主动报备不隐瞒：一人举麦克风，旁有绿色对勾。
  2. 遵守秩序不聚集：三人保持间距排队。
  3. 健康监测不松懈：一人测量体温。
- 底部总结性标语：“共防共建：文明参与倡议”，并强调“每个人都是防控第一责任人”。

### 视觉与数据编码
- 图表类型：结构化流程图 + 分类清单。
- 数据编码：通过图标、颜色（如红色警示、绿色健康）、文字标签和空间位置（左右分区、中心桥梁）传达信息。
- 风格：复古像素艺术，模仿经典电子游戏界面，增强视觉吸引力与记忆点。
- 文本语言：全部为中文，无英文或其他语言内容。

该信息图通过游戏化设计将复杂的疫情防控流程可视化，清晰划分了“事前准备—事中管控—全民参与”的三大模块，并强调了各阶段的关键行动项与责任主体，兼具教育性与传播性。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 64：成人礼vs高考誓师大会核心差异对比</b></div><img src="../all_small/091.webp" alt="信息图案例 091" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以漫画风格设计，采用红、黄、蓝、黑为主色调，背景包含放射状线条和网点图案，营造出动态、醒目的视觉效果。整体布局为2x2网格结构，共四个主要模块，每个模块均有独立标题和内容区块，通过边框、颜色和图标区分不同主题。

标题位于顶部红色横幅中，使用白色粗体字加黑色描边，内容为“成人礼&amp;高考誓师大会活动策划信息总览”。

---

**第一模块：成人礼vs高考誓师大会核心差异对比**

位于左上角，标题为“成人礼vs高考誓师大会核心差异对比”，采用黄色字体加黑色描边。模块内以表格形式呈现，分为“对比维度”、“18岁成人礼”、“高考誓师大会”三列。

- **对比维度**（蓝色背景）：
  - 核心定位
  - 参与主体
  - 举办时间
  - 核心环节

- **18岁成人礼**（红色背景）：
  - 核心定位：公民身份宣告，开展责任、感恩教育
  - 参与主体：年满18岁的高二/高三学生、全体家长、校领导、校外德育嘉宾
  - 举办时间：五四青年节前后、固定的集体18岁纪念日
  - 核心环节：加冠/授成人纪念章、宪法宣誓、亲子交换家书

- **高考誓师大会**（蓝色背景）：
  - 核心定位：高考冲刺动员，强化奋斗目标与备考信念
  - 参与主体：高三全体师生、家长代表、校领导
  - 举办时间：高考前100天、高考前30天
  - 核心环节：授冲刺战旗、集体宣誓、目标墙签名、班级喊出征口号

该模块在“参与主体”行左侧配有小人图标，在“18岁成人礼”和“高考誓师大会”标题旁分别有奖牌和喇叭图标。

---

**第二模块：标准化活动流程参考**

位于右上角，标题为“标准化活动流程参考”，采用黄色字体加黑色描边。模块内分为两个并列流程图：“成人礼流程”（黄色标题框）和“高考誓师流程”（红色标题框），均采用编号步骤展示。

- **成人礼流程**：
  ① 开场：奏国歌、介绍到场嘉宾、校长致辞  
  ② 核心环节：长者加冠/颁发成人纪念章、集体宣读成年公民宪法誓词、学生与父母互换手写家书  
  ③ 收尾：齐唱《歌唱祖国》、全体合影、发放成年纪念礼  

- **高考誓师流程**：
  ① 开场：奏国歌、往届优秀学子分享备考经验  
  ② 核心环节：校领导为各班授冲刺战旗、各班喊出征口号、全体学生宣誓、在高考目标墙签名  
  ③ 收尾：齐唱励志歌曲、为学生发放冲刺祝福礼包、全体合影  

两流程图均使用圆圈编号，背景为灰白相间网点，右侧边缘有彩色条纹装饰。

---

**第三模块：活动报道核心方向**

位于左下角，标题为“活动报道核心方向”，采用黄色字体加黑色描边。模块内分上下两个部分：

- **成人礼报道要点**（黄色标题框）：
  1. 核心突出“责任、感恩、成长”主题，重点挖掘亲子家书交换、长辈致辞、学生发言等情感瞬间  
  2. 结合普法宣传，强调成年公民的权利与义务，可联动后续社区志愿服务等实践活动报道  
  3. 避免过度形式化报道，聚焦学生的真实感受与成长感悟  

- **高考誓师报道要点**（黄色标题框）：
  1. 核心突出“奋斗、理想、坚持”主题，挖掘学生备考、老师陪伴、家长支持的感人细节  
  2. 兼顾励志引导与心理减压，避免过度渲染竞争氛围、制造焦虑  
  3. 可搭配备考技巧、心理疏导指南等实用内容，提升内容实用性  

此模块使用对话气泡形状的文本框，底部有蓝色和红色渐变色块延伸，增强视觉引导。

---

**第四模块：活动组织通用注意事项**

位于右下角，标题为“活动组织通用注意事项”，采用黄色字体加黑色描边。模块内以四个带箭头标签的形式列出建议，标签颜色交替为蓝色和红色。

- **鼓励学生参与**（蓝色标签）：
  鼓励学生参与环节设计，提升自主参与感，避免完全由校方包办

- **关注心理状态**（红色标签）：
  高考誓师环节控制节奏，避免安排过多鸡血演讲，增加心理疏导相关环节

- **做好影像记录**（蓝色标签）：
  全程做好摄影摄像记录，为学生留存青春纪念素材

- **邀请家长参与**（红色标签）：
  优先邀请家长参与，强化仪式的情感联结价值

---

整体而言，该信息图通过清晰的结构划分、鲜明的颜色编码和漫画式视觉元素，系统性地呈现了成人礼与高考誓师大会在定位、流程、报道和组织方面的全面对比与指导建议，适用于学校活动策划、媒体宣传或教育工作者参考。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 65：基因传递：孟德尔定律解析</b></div><img src="../all_small/094.webp" alt="信息图案例 094" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“基因传递：孟德尔定律解析”为主题，采用三维立体、科技感十足的视觉风格，整体布局呈放射状流程结构，从左至右、由上至下引导观者理解孟德尔遗传定律的发现、机制与应用。背景为柔和米色，搭配透明亚克力质感的模块和发光线条，营造出清晰、现代的科学氛围。

标题“基因传递：孟德尔定律解析”位于顶部中央，使用半透明冰蓝色立体字，字体边缘有光晕效果，下方标注关键比例“3:1, 9:3:3:1”，点明核心数据。

信息图主要分为五个逻辑模块，通过发光路径连接，形成完整知识流：

1. **观察与实验基础**（左侧模块）
   - 标题：“观察与实验基础 数学规律”
   - 视觉元素：一个打开的豌豆荚，内含三颗黄色豆子和一颗绿色豆子，周围散落几颗黄绿相间的豆子。
   - 数据编码：通过豆子颜色比例隐喻性地表示孟德尔实验中性状分离现象，暗示3:1的表型比。
   - 该模块代表孟德尔豌豆杂交实验的原始观察数据。

2. **分离定律 3:1**（中部偏左模块）
   - 标题：“分离定律 3:1”
   - 视觉元素：一个透明球体置于发光底座上，内部有一对同源染色体（一蓝一橙），正在分离，箭头指示分离方向，旁边有DNA双螺旋小图标。
   - 数据编码：明确标示“3:1”的比例，对应一对等位基因在子代中的表型分离比。
   - 该模块解释了单因子杂交实验的遗传规律。

3. **自由组合定律 9:3:3:1**（中部偏右模块）
   - 标题：“自由组合定律 9:3:3:1”
   - 视觉元素：网格状棋盘，上面排列着多组彩色珠串（模拟染色体或基因型），每组珠串由不同颜色珠子组成（如蓝-橙、绿-棕等），右侧有一个由多个珠子组成的复杂分子结构。
   - 数据编码：明确标示“9:3:3:1”的比例，对应两对独立基因在子代中的表型组合比。
   - 该模块展示双因子杂交实验的遗传规律。

4. **细胞生物学基础 减数分裂**（底部中央模块）
   - 标题：“细胞生物学基础 减数分裂”
   - 视觉元素：一个透明球体，内部包含三对不同颜色的染色体（红-红、蓝-蓝、橙-橙），呈现减数分裂过程中染色体配对与分离的状态。
   - 数据编码：无显式数值，但通过染色体图像直观展示分离定律和自由组合定律的细胞学基础。
   - 该模块将遗传规律与细胞过程关联，说明其物理实现机制。

5. **数学规律分析 预测验证**（右上模块）
   - 标题：“数学规律分析 预测验证”
   - 视觉元素：两个半透明圆角矩形框，分别标注“3:1”和“9:3:3:1”，周围漂浮着DNA双螺旋、细胞核等微小图标。
   - 数据编码：重复强调关键比例，体现孟德尔利用数学方法进行预测和验证的过程。
   - 该模块突出遗传学的定量分析特征。

6. **广泛验证与应用 现代遗传学**（右下模块）
   - 标题：“广泛验证与应用 现代遗传学”
   - 视觉元素：一个绿色豌豆荚、一个眼状细胞结构、一个包含多种细胞器的细胞切面图、一个装有多种珠子的透明容器。
   - 数据编码：无显式数值，但通过多样化的生物结构象征遗传学在医学、农业、生物技术等领域的广泛应用。
   - 该模块总结孟德尔定律的当代意义。

所有模块之间由金色发光线条连接，形成从“观察→定律→机制→验证→应用”的完整逻辑链条。整体设计融合了生物学符号、数学比例和现代科技美学，使复杂的遗传学概念变得直观易懂。文字全部为中文，符合中国教育和科普语境。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 66：East to West Illusion</b></div><img src="../all_small/097.webp" alt="信息图案例 097" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“EARTH’S ROTATION”（地球自转）为标题，采用手绘风格的插画与清晰排版，系统性地阐释地球自转及其引发的天文现象。整体背景为浅米色带点阵网格，文字与图形以棕色、黑色为主色调，辅以少量蓝色和黄色，营造出温暖、易读的教育氛围。布局分为中心主图与左右两侧说明区，结构清晰、逻辑分明。

**标题与整体设计：**
- 标题“EARTH’S ROTATION”位于顶部中央，字体为粗体手写风格，带有棕色描边和阴影效果，右侧配有波浪装饰线。
- 左上角有太阳升起于地平线的小图标，作为主题引导。

**左侧区域 — Key / Symbols（符号说明）：**
此部分垂直排列四个符号及其对应名称，用于解读图中元素：
1. **Sun (光源)**：一个黄色太阳图标，周围放射短线条，表示光源。
2. **Rotation (自转方向)**：两个反向弯曲箭头组成的循环符号，表示旋转运动。
3. **Terminator (晨昏线)**：一个半圆内含斜线填充的图标，代表昼夜分界线。
4. **Solar Rays (太阳辐射)**：三个平行箭头指向右方，表示太阳光照射方向。
所有符号下方均标注英文与中文双语名称，便于理解。

**中心区域 — 地球自转主图：**
- 展示一个倾斜的地球球体，大陆轮廓简略勾勒，可见欧亚非大陆等。
- 地轴倾斜角度明确标注为“23.5°”，由虚线连接北极点与垂直方向，并用弧形箭头标示。
- 地球自转方向由环绕球体的粗黑箭头指示，箭头旁标注“W→E”，即自西向东。
- 太阳光从左侧水平射入，由多条平行箭头表示“Solar Rays”，照亮地球左半部。
- 昼夜分界线（Terminator）为一条斜线，将地球分为明亮的昼半球与阴影的夜半球。
- 图上方有一个指南针图标，标明N（北）、S（南）、E（东）、W（西）方位。

**右侧区域 — 科学解释与观察现象：**

**1. East to West Illusion（东升西落错觉）：**
- 标题为“East to West Illusion”，下方配有中文解释：“地球自西向东(W→E)自转, 如同坐在前进的车上窗外景物后退, 使我们产生太阳东升西落(E→W)的视觉错觉。”
- 关键词“相对运动”置于蓝色圆角矩形框内，强调原理。
- 配图展示一个小人抬头望向太阳与云朵，形象化表达观察者视角。

**2. Day &amp; Night Cycle（昼夜交替）：**
- 标题为“Day &amp; Night Cycle”，下方配有小图：一个被垂直分割的圆形，左半为白色并标注“昼 (Day)”，右半为斜线填充并标注“夜 (Night)”，太阳光从左侧照射。
- 中文解释：“地球是不透明的球体。对着太阳的一面为‘昼’(Day), 背着太阳的一面为‘夜’(Night)。随着自转, 两半球不断交替, 周期约为24小时。”
- “24小时”以橙色圆角矩形突出显示。

**底部区域 — Daily Observation（每日观测）：**
- 标题为“Daily Observation”，配有两个复古风格的仪器图标：一个指南针和一个怀表。
- Prompt：“正午时分, 太阳高度角达到最大”，描述日常可观察到的太阳位置变化规律。

**总结：**
该信息图通过符号定义、核心图形与多段文字结合的方式，全面解析地球自转机制及其对人类日常观测的影响。内容涵盖物理原理（如相对运动）、地理概念（晨昏线、昼夜周期）和生活现象（太阳东升西落），语言双语对照，图文并茂，适合科普教育使用。所有数据、标签与数值均精确呈现，无遗漏或近似处理。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 67：The Golden Bloom: Osmanthus Care,</b></div><img src="../all_small/019.webp" alt="信息图案例 019" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 67 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 68：Infographic 68</b></div><img src="../all_small/043.webp" alt="信息图案例 043" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“探索未知：中科院的科研与育人”为标题，采用简洁现代的扁平化设计风格，整体布局为2x2网格结构，分为四个象限，每个象限对应一个核心主题。顶部标题置于一个带有蓝色渐变边框的对话气泡中，字体粗大醒目，传达出探索与交流的主题氛围。

整体视觉元素以黑白线条勾勒为主，辅以少量高饱和度色彩（如蓝色、红色、黄色、绿色）作为重点标识，搭配几何符号、波纹、加号和省略号等装饰性图形，增强视觉层次感和科技感。

四个象限分别从不同维度展示中国科学院（中科院）的核心工作：

1. **左上角：前沿科研探索 (Frontier Research)**
   - 图标：一个由黑色线条构成的原子模型，中心为蓝色圆点，代表原子核，外围有三条椭圆形轨道，象征电子云。
   - 文字内容：
     - 标题：“前沿科研探索 (Frontier Research)”
     - 正文：“中科院致力于前沿基础研究。通过简单的几何原子模型符号，展现科学家们在物理、化学、生物等领域跨越未知、探寻物质本质的科学精神。”
   - 视觉辅助：图标周围点缀有加号、波浪线和省略号，营造动态探索的氛围。

2. **右上角：大科学装置 (Mega-Facilities)**
   - 图标：一个抛物面天线（类似射电望远镜），白色抛物面，红色底座，顶部有一个红色圆点发出三道弧形信号波纹。
   - 文字内容：
     - 标题：“大科学装置 (Mega-Facilities)”
     - 正文：“大科学装置是国家科技实力的体现。利用扁平化的抛物面与信号波纹，将宏大的‘中国天眼’等顶尖科研平台转化为清晰易懂的视觉坐标。”
   - 视觉辅助：同样配有加号、波浪线和省略号，强调信号传播与技术先进性。

3. **左下角：科教融合 (Integration)**
   - 图标：一本打开的黄色书本，书页上方放置一顶黄色学士帽，帽子中央嵌入一个齿轮，象征学术与实践的结合。
   - 文字内容：
     - 标题：“科教融合 (Integration)”
     - 正文：“依托高水平科研平台培养拔尖创新人才。‘书本与齿轮’的几何组合，直观呈现了学术理论与科研实践深度融合的中科院特色教育模式。”
   - 视觉辅助：周围装饰有加号、波浪线和省略号，突出教育与创新的互动。

4. **右下角：成果转化 (Application)**
   - 图标：一个绿色地球仪，表面有经纬网格，右侧有一支向上箭头的绿色色块，象征发展与进步。
   - 文字内容：
     - 标题：“成果转化 (Application)”
     - 正文：“将科研成果转化为国家发展的动力。摒弃复杂的细节，用纯粹的地球与上升箭头色块，聚焦科技创新在航天、生态等服务国家战略领域的实际贡献。”
   - 视觉辅助：同样使用加号、波浪线和省略号，强化成果落地与国家发展的关联。

所有文字均采用无衬线字体，中文与英文并列标注，便于国际理解。颜色编码与主题呼应：蓝色代表基础研究，红色代表大型设施，黄色代表教育，绿色代表应用转化。整体设计逻辑清晰，信息层级分明，旨在以可视化方式系统呈现中科院在科研与育人方面的战略布局与核心价值。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 69：Infographic 69</b></div><img src="../all_small/048.webp" alt="信息图案例 048" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 69 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 70：International Student Higher Education Pathway,</b></div><img src="../all_small/050.webp" alt="信息图案例 050" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 70 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 71：Infographic 71</b></div><img src="../all_small/052.webp" alt="信息图案例 052" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 71 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 72：Degree Upgrade Planning Resource Hub</b></div><img src="../all_small/059.webp" alt="信息图案例 059" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 72 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 73：网络小说的类型与特点讨论</b></div><img src="../all_small/070.webp" alt="信息图案例 070" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“网络小说的类型与特点讨论”为主题，采用金字塔式层级结构，将网络小说的类型划分为四个层次：基础层、中间层、上层和顶层。整体设计风格融合了奇幻、科技与自然元素，背景为朦胧的森林光斑，主体为一座由岩石、电路板、布料和水晶构成的立体金字塔，每一层均通过材质与视觉符号体现其对应类型的特点。

标题位于图像顶部，使用金色浮雕字体：“网络小说的类型与特点讨论”，具有强烈的视觉冲击力。

金字塔从下至上依次为：

1. **基础层：玄幻/仙侠**
   - 视觉表现：底层由覆盖青苔的岩石构成，表面刻有神秘符文，点缀着发光的水晶簇，象征宏大世界观与奇幻元素。
   - 文字内容：
     - 标题：“基础层：玄幻/仙侠”
     - 特点：“世界观宏大，修炼升级，奇幻元素”

2. **中间层：都市/言情**
   - 视觉表现：中层为一块带有水滴和墨迹的米色布料，象征现实生活的细腻情感与职场纠葛，布料边缘缠绕着微弱的金色光丝。
   - 文字内容：
     - 标题：“中间层：都市/言情”
     - 特点：“现实背景，情感纠葛，职场生活”

3. **上层：科幻/游戏/系统**
   - 视觉表现：上层为一块复杂的绿色电路板，嵌有微型显示屏和数据流，周围环绕蓝色与金色的光效线条，体现未来科技与虚拟现实感。
   - 文字内容：
     - 标题：“上层：科幻/游戏/系统”
     - 特点：“未来科技，虚拟现实，数据面板/任务”

4. **顶层：爽感与共鸣**
   - 视觉表现：顶端是一个发光的多面体晶体，连接着类似神经元或星系的金色网络结构，象征快速反馈与情感满足。
   - 文字内容：
     - 标题：“顶层：爽感与共鸣”
     - 核心：“快速反馈，代入感强，情感满足”

整个金字塔被金色光丝贯穿，象征各类型之间的内在联系与能量流动。底部地面覆盖青苔与碎石，散落着发光晶体，营造出一种神秘而生机勃勃的氛围。所有文字均采用统一的金色字体，与背景形成鲜明对比，确保可读性。

该信息图通过视觉隐喻与清晰的文字标注，系统地呈现了网络小说类型的层级结构及其核心特征，兼具艺术美感与信息传达功能。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 74：Infographic 74</b></div><img src="../all_small/084.webp" alt="信息图案例 084" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为《儿童营养补充全指南：科学建议+产品选购要点》，采用漫画风格设计，色彩鲜明，以红、黄、蓝为主色调，布局清晰分为左右两大板块，每个板块又细分为多个模块，图文并茂地呈现了儿童营养补充的科学指导与实用建议。

整体结构分为“科学参考指引”和“实操应用指南”两大核心部分，通过卡通插图、图标、爆炸式对话框、标签等视觉元素增强可读性与吸引力。

---

**第一部分：科学参考指引**

1. **分龄营养补充重点清单**
   - 标题：“分龄营养补充重点清单”，副标题：“分龄补营养，精准更高效；对应年龄段按需补充，避免过度摄入”
   - 内容按年龄分三个阶段：
     - **0-6月龄**：每日常规补充维生素D 400IU，纯母乳喂养宝宝需额外补充维生素K。配图：婴儿头像、Vit D注射器、Vit K胶囊。
     - **7月龄-3岁**：重点补充铁（Fe）、锌（Zn）、DHA，每日维生素D补充量维持在400-600IU。配图：幼儿头像、放大镜观察胶囊、Fe和Zn符号。
     - **4-12岁**：重点补充钙（Ca）、维生素A、B族维生素（B_B），保证每日蛋白质摄入量达标。配图：男孩头像、Ca气泡、B_B气泡、鸡蛋、牛奶瓶、眼睛图标。

2. **营养补充原则&amp;常见避坑指南**
   - 标题：“营养补充原则&amp;常见避坑指南”，副标题：“科学补营养，这些坑要避开”
   - 包含两个核心原则：
     - **优先膳食摄入**（绿色对勾）：核心原则1：日常均衡饮食是营养摄入的首要来源，不可用补充剂代替正常三餐。配图：孩子用餐场景，盘中有蔬菜、水果、肉类。
     - **按需适量补充**（红色STOP标志）：核心原则2：营养素补充并非越多越好，过量摄入维生素A、钙等可能引发中毒或代谢负担。配图：多瓶补剂被红色叉号覆盖。
   - **避坑指南**（黄色标签）：
     - ① 不做体检评估盲目跟风补 ❌
     - ② 把网红补剂当零食给孩子吃 ❌
     - ③ 用成人补充剂减量给儿童服用 ❌
     - 配图：红色“避坑”爆炸框，带有闪电效果。

---

**第二部分：实操应用指南**

1. **儿童营养补充产品3步选购法**
   - 标题：“儿童营养补充产品3步选购法”，副标题：“儿童补剂选购3步判断法”
   - 三步法分别由放大镜图标引导：
     - **看合规标识**：优先选择带蓝帽标识的保健食品，或有婴幼儿/儿童专用备案标识的正规产品，拒绝三无产品。配图：放大镜聚焦“蓝帽”标志。
     - **看配料成分**：优先选择无额外添加蔗糖、香精、人工色素、防腐剂的产品，致敏原标注清晰明确。配图：文件上贴有“无添加”印章，绿色对勾。
     - **看适配年龄**：选择标注对应适用年龄段的儿童专用产品，不要自行将成人补充剂减量给孩子服用。配图：药瓶标签上“年龄”被红圈突出。

2. **常见儿童补剂适用场景对照表**
   - 标题：“常见儿童补剂适用场景对照表”
   - 表格形式，两列：左侧“补剂类型”，右侧“适用场景”，背景色交替为红、蓝。
   - 具体内容：
     - **维生素D滴剂** → 全年龄段儿童日常常规补充，预防佝偻病、促进钙吸收。配图：滴管瓶、骨头图标。
     - **铁剂** → 体检确诊缺铁性贫血，或日常红肉、动物肝脏摄入不足的儿童。配图：滴管瓶、儿童头像。
     - **DHA藻油** → 日常深海鱼摄入不足的儿童，辅助促进视网膜和大脑发育。配图：鱼形胶囊、大脑与眼睛图标。
     - **钙剂** → 日常奶量不足、身高增长偏缓，经体检确认缺钙的儿童。配图：白色药片、儿童测量身高图。

---

**视觉与排版特征：**
- 整体采用网格化布局，四个主要模块分布在2x2的象限中。
- 使用大量漫画元素：如爆炸框、对话气泡、箭头、感叹号、禁止符号等。
- 图标系统丰富：Vit D、Fe、Zn、Ca、B_B、蓝帽、无添加、年龄、STOP等均有专属图形标识。
- 字体加粗、阴影、边框强调关键信息，如标题、数字、警示语。
- 色彩编码明确：黄色用于提示重点，蓝色用于说明步骤，红色用于警示或禁止。

该信息图内容全面，逻辑清晰，兼具科学性和实用性，适合家长快速掌握儿童营养补充的核心知识与选购技巧。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 75：从潮玩出圈的随机消费新玩法</b></div><img src="../all_small/085.webp" alt="信息图案例 085" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“盲盒文化全解析：核心逻辑、跨领域应用与社会影响”为标题，副标题为“从潮玩出圈的随机消费新玩法”，采用中文作为主要语言。整体布局为网格结构，分为三列和三行，共九个内容模块，左侧一列为纵向标题栏，右侧两列为横向内容栏，每个模块均用灰色边框框定，并配有尺寸标注（如300px x 200px等），呈现清晰的信息层级与视觉组织。

信息图顶部标有“盲盒文化核心逻辑”的小标题，主标题字体加粗且字号较大，位于页面最上方居中位置，下方是副标题，字体稍小。

左侧纵向标题栏包含三个垂直排列的灰色矩形模块，分别标注：
- 盲盒文化的基础认知
- 盲盒模式的跨领域应用
- 盲盒文化的双面社会影响

这三个模块高度均为200px，宽度为120px，底部标注尺寸“120px x 200px”。

中间和右侧内容区域被划分为三行三列的网格，每行对应左侧的一个主题，每列对应一个子维度：

**第一行（对应“盲盒文化的基础认知”）**

- **第一列（核心要素与驱动力）**：
  - 文本描述：“核心运行逻辑：以‘随机抽选+隐藏款/限定款稀缺激励’为核心，精准击中消费者的收藏欲、猎奇心理与社交分享需求”
  - 图形元素：一个立方体盒子图标，表面有叉号，连接两个圆角矩形标签：“随机抽选”、“隐藏款激励”

- **第二列（发展演变与落地场景）**：
  - 文本描述：“发展脉络：最早可追溯到日本福袋、扭蛋机制，2010国内潮玩品牌将其标准化推广，2019年前后进入大众视野成为消费热点”
  - 图形元素：一条水平线段，线上有三个圆点，两端各有一个带叉的方块，表示发展阶段或路径

- **第三列（受众特征与社会影响）**：
  - 文本描述：“核心受众特征：以18-35岁年轻群体为绝对主力，女性占比超6成，Z世代是消费核心人群”
  - 图形元素：四个简笔人物剪影（两男两女），上方有一个对话气泡，内写“Z世代主力受众”

**第二行（对应“盲盒模式的跨领域应用”）**

- **第一列**：
  - 列表形式，四项内容，每项前有带叉的方块图标：
    - 零售领域：降低商家库存损耗，提升消费趣味性
    - 文旅领域：拉动文旅消费，为传统文化传播提供年轻化载体
    - 美妆领域：降低消费者试错成本，提升品牌复购率
    - 互联网运营领域：有效提升用户活跃度与平台留存率

- **第二列**：
  - 列表形式，四项内容，每项前无图标：
    - 零售领域：推出文具盲盒、生鲜盲盒、临期商品盲盒
    - 文旅领域：推出考古盲盒、景区门票盲盒、非遗文创盲盒
    - 美妆领域：推出小样盲盒、节日限定盲盒
    - 互联网运营领域：推出盲盒式抽奖、盲盒式内容推送
  - 右侧有四个椭圆形标签，分别标注“零售”、“文旅”、“美妆”、“互联网运营”，通过线条指向对应内容，形成分类关联。

- **第三列**：
  - 文本：“零售、文旅、美妆、互联网运营领域的消费者、商家、平台”
  - 图形：一个带有交叉对角线的平行四边形，象征覆盖范围或网络连接。

**第三行（对应“盲盒文化的双面社会影响”）**

- **第一列**：
  - 正面影响：
    - 带动潮玩、文创等相关产业增长；
    - 考古盲盒、非遗盲盒等产品降低了传统文化的接触门槛；
    - 盲盒的交换、收藏属性为年轻群体提供了新的社交话题与圈层连接方式
    - 图形：一个上升趋势的柱状图
  - 负面影响：
    - 部分商家利用盲盒销售劣质、过期商品，侵害消费者合法权益；
      - 过度营销稀缺、收藏概念，容易诱导未成年人产生非理性消费、过度消费行为；
      - 部分热门盲盒出现恶意炒价乱象，扰乱正常市场秩序
    - 图形：一个装有钱币的袋子图标

- **第二列**：
  - 文本：“2022年国内潮玩盲盒市场规模突破150亿元”
  - 下方重复正面影响中的三点内容：
    - 考古盲盒、非遗盲盒等产品降低了传统文化的接触门槛
    - 盲盒的交换、收藏属性为年轻群体提供了新的社交话题与圈层连接方式
  - 下方三个带叉方块标签：
    - 劣质、过期商品侵害权益
    - 诱导非理性消费
    - 恶意炒价乱象

- **第三列**：
  - 正面影响（三项）：
    - 产业拉动
    - 文化传播
    - 社交连接
    - （三项均以椭圆形标签呈现，由文字“正面：产业增长 文化传播 社交连接”引出）
  - 负面影响（三项）：
    - 消费侵权风险
    - 非理性消费
    - 炒价乱象
    - （三项均以椭圆形标签呈现，由文字“负面：消费侵权风险 非理性消费 炒价乱象”引出）

整体设计风格简洁、现代，使用灰白配色，辅以红色尺寸标注线，增强结构感。所有文本均使用中文，字体清晰易读，图表与文字结合紧密，旨在系统性地解析盲盒文化的核心机制、广泛应用及其复杂的社会效应。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 76：Infographic 76</b></div><img src="../all_small/098.webp" alt="信息图案例 098" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 76 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 77：Infographic 77</b></div><img src="../all_small/099.webp" alt="信息图案例 099" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 77 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 78：加强党的建设·严明纪律要求</b></div><img src="../all_small/029.webp" alt="信息图案例 029" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图是一份正式的党内纪律指导文件，整体采用庄重典雅的设计风格，以米黄色为背景，配以深红色与金色边框及装饰元素，营造出严肃、权威的视觉氛围。顶部中央是一个红色圆形徽章，内含金色镰刀锤子党徽图案，两侧延伸出带有传统回纹与齿轮、藤蔓、麦穗等元素的对称装饰带，象征工业与农业结合、传统与现代并存。边框四周布满精致的金色卷草纹、回纹、齿轮和麦穗图案，底部则为连续的云纹与花卉装饰，体现中国传统文化审美。

标题部分位于顶部居中位置，使用醒目的红色粗体字，内容为“党建引领与纪律监察委员会 暨自我净化自我完善自我革新指导小组”，字体较大，突出发文机构的权威性。标题下方是一个发光的五角星图案，周围有放射状光芒，两侧点缀着金色莲花纹饰，象征纯洁与光辉。

主标题为“加强党的建设·严明纪律要求”，采用金色立体艺术字，字号最大，居中显示，极具视觉冲击力。其下附有副标题：“关于新形势下全面从严治党与自我革命的纲领性要求”，字体较小，颜色为深红，进一步说明文件的核心主题。

正文部分以“致：[全体党员干部]”开头，表明文件的受众对象，文字为黑色，加粗，并用下划线标示，清晰明确。

核心内容为“六大纪律准则”，以大号黑色黑体字呈现，下方列出了六条具体纪律要求，每条前均有一个红色方块项目符号，内容如下：

- 必须严明政治纪律，把准方向；
- 严明组织纪律，凝聚力量；
- 严明廉洁纪律，守住底线；
- 严明群众纪律，巩固根基；
- 严明工作纪律，激发担当；
- 严明生活纪律，磨炼品格。

其中，“严明”二字在每条中均以红色加粗字体突出，强调其重要性；其余关键词如“把准方向”、“凝聚力量”、“守住底线”、“巩固根基”、“激发担当”、“磨炼品格”则分别以金色或黑色字体呈现，形成视觉层次。

文件右下角为发布日期“发布日期 二〇二五年一月十五日”，字体为黑色，清晰可读。紧邻日期的是一个红色圆形印章，印文为“铁纪执行专用章”，印章中心图案为天平与剑的组合，象征公正与纪律的权威性，整体设计具有强烈的官方文书特征。

整张信息图无数据图表，纯属文本公告类设计，通过层级分明的排版、色彩对比和装饰元素，强化了文件的严肃性、规范性和号召力，旨在向全体党员干部传达新时代全面从严治党的核心要求。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 79：Infographic 79</b></div><img src="../all_small/056.webp" alt="信息图案例 056" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 79 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 80：EXHIBITION POSTPONEMENT &amp; ADJUSTMENTS: A GUOCHAO PERSPECTIVE UNDER THE EPIDEMIC</b></div><img src="../all_small/068.webp" alt="信息图案例 068" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 80 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 81：AGE DISCRIMINATION IS FLOURISHING</b></div><img src="../all_small/074.webp" alt="信息图案例 074" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 81 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 82：Infographic 82</b></div><img src="../all_small/020.webp" alt="信息图案例 020" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“房屋买卖合同：从交易流程到法律风控”为标题，采用蓝白主色调的现代科技风格设计，整体布局为垂直分层结构，结合等轴测立体图形与图标元素，清晰呈现房屋交易的全流程与法律风险控制体系。标题位于顶部中央，字体加粗醒目，右侧配有盾牌箭头与金色对勾图标，象征安全与确认。

信息图主体由一个类似工业流水线或数据处理链的立体结构组成，包含多个模块化方块，通过链条和虚线连接，体现交易环节的连续性与关联性。每个模块上方放置文件、房屋、天平、盾牌等图标，直观对应其功能主题。

整个信息图分为四个主要部分，分别从不同维度解析房屋买卖的法律与操作框架：

一、房屋买卖全周期风控模型
位于左上角，说明房屋交易是受合同严格约束的“连续履约链条”，每个环节伴随法律风险，需通过严密条款保障最终权益。配图为天平图标，象征法律平衡与公正。

二、法律条款防范（护航架构）
位于左侧中部，强调法律条款作为交易的契约系统，提供法律保障。内容包括：
- 标的物尽职调查（查封/抵押）
- 违约责任与解除权设定
- 不可抗力与情势变更
- 管辖与争议解决机制
并指出这些条款“渗透并约束每个交易环节”。视觉上以文件堆叠与盾牌图标表示。

三、基础交易流程（履约底座）
位于左下角，列出了房屋买卖的实质性推进步骤，共五个阶段：
1. 意向与定金签约
2. 购房资质与网签
3. 资金监管与按揭贷款
4. 完税与产权过户
5. 物业交割与尾款
此部分被描述为“履约底座”，即交易执行的基础路径，视觉上以深蓝色基座承载多个文件模块，末端连接一座小房子，代表交易完成。

四、条款间的串联与制动
位于右侧中部，解释合同精髓在于“相互制约”，例如“按揭审批失败的合同解除权（防范）直接关联资金退还（流程）”，优化触发点是规避纠纷的关键。视觉上用彩色链条连接不同模块，突出联动机制。

五、交易安全与权益实现
位于右上角，阐述合同法律架构设计的终极目标是“最大化保障该区域——防范钱房两空风险，确资产合法、安全置换”，并强调“零纠纷的产权转移与资金落袋”。配图为绿色盒子内含房屋与证书，外有盾牌保护，下方是带对勾的盾牌图标，强化安全与成功交付的概念。

整体设计通过模块化、链条式结构，将复杂的法律与交易流程可视化，运用图标、颜色编码（如蓝色代表流程、绿色代表安全）、虚实线条（虚线表示逻辑关联，实线表示物理连接）等多种数据编码方式，使抽象概念具象化。所有文本均使用简体中文，语言专业严谨，适合房地产从业者、购房者或法律专业人士阅读理解。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 83：Infographic 83</b></div><img src="../all_small/021.webp" alt="信息图案例 021" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“绿茵定格：巨星轶事”为标题，采用深色科技感界面设计，模拟录像监控或战术分析系统，整体风格酷似军事或体育战术分析软件。背景为黑白色调的足球场模糊影像，叠加了绿色、橙色、黄色、青色等高对比度线条和图形，形成视觉引导与数据层叠效果。右上角有红色“REC”录制标识及“[3x3]”网格标记，右侧边缘垂直显示比赛时间轴（MATCH 1:45 03:39:08、00:8.0 Domt 19:38:00），左侧边缘显示“MATCH Z5 FLB Doit 12:23:00”，底部有刻度尺与“100”标记，增强科技感与数据化氛围。

信息图主体为一个虚拟的3x3网格系统（GRID_SYSTEM），覆盖在足球场之上，将球场划分为九个区域，用于分析球员的空间感知与战术布局。该网格由绿色垂直线与橙色水平线构成，交点处用不同颜色圆圈标记关键位置。右上角还嵌入一个黄金螺旋曲线（斐波那契螺旋），象征美学与自然比例在运动中的体现。

信息图包含六个核心模块，每个模块均配有英文标签、中文标题、详细解释文字及关键词总结：

1. **GRID_SYSTEM: 球场上帝视角**  
   - 中文标题：绿茵分割 (Xavi&#x27;s Vision)  
   - 内容：“哈维的头脑中仿佛自带井字网格。他曾坦言自己全场比赛都在‘寻找空间’。通过虚拟的网格切割，战术大师们将球场划分为9个区块，精准找到对手的防守盲区。”  
   - 关键词：“防守盲区”

2. **POWER_POINTS: 致命十分角**  
   - 中文标题：能量交叉点 (The Top Bins)  
   - 内容：“网格的四个交叉点如同球门的死角。贝克汉姆的圆月弯刀、齐达内的天外飞仙，巨星们最不可思议的进球往往精准命这些视觉与物理的双重能量爆发点。”  
   - 关键词：“双重能量”

3. **HORIZON_LINE: 地空博弈法则**  
   - 中文标题：水平线置换 (Ground vs Air)  
   - 内容：“拒绝平庸的五五开。当C罗腾空2.56米头球砸门时，天空是他的领地（上1/3）；而马拉多纳连过五人时，大地则是他的舞台（下1/3）。主次分明，成就绝杀。”  
   - 关键词：“主次分明”

4. **NEGATIVE_SPACE: 空间阅读者**  
   - 中文标题：留白与跑位 (Raumdeuter)  
   - 内容：“托马斯·穆勒自称‘空间阅读者’。在网格留白的区域，无球跑动远比持球更具杀伤力。这种看似不平衡的站位，为致命一传创造了巨大的战术张力。”  
   - 关键词：“战术张力”

5. **EYE_FLOW: 传切视线引导**  
   - 中文标题：传切轨迹 (Tiki-Taka Flow)  
   - 内容：“克鲁伊夫的‘全攻全守’利用无死角的对角线跑位引导观众与防守者的视线。如同三分法构图，让足球在网格交叉点间流畅运转，绝不陷入死板的中心停滞。”  
   - 关键词：“流畅运转”

6. **附加视觉元素**  
   - 图中足球场中央偏下位置有一个青色雷达扫描环，指向球门方向，象征探测与定位。  
   - 右下方有一条从球员脚部延伸出的虚线箭头，指向足球，表示传球或射门路径。  
   - 多个球员剪影分布在不同区域，强化场景真实感。

整体结构清晰，以网格为核心框架，六大模块围绕网格交点分布，形成辐射状信息架构。文本内容结合具体球员案例（哈维、C罗、马拉多纳、贝克汉姆、齐达内、托马斯·穆勒、克鲁伊夫）与战术术语，深入浅出地阐释足球运动中的空间策略、能量爆发点、地空博弈、无球跑动与传切配合等高级战术理念。设计语言融合科技、几何、运动美学，使抽象战术具象化、可视化。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 84：旬之味：主厨定制三部曲</b></div><img src="../all_small/034.webp" alt="信息图案例 034" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“旬之味：主厨定制三部曲”为主题，呈现了一套由主厨精心设计的三道式高级料理套餐。整体设计风格典雅深邃，采用暗色调木质背景搭配石质台面，营造出日式侘寂美学氛围。画面中点缀有红枫叶与竹枝剪影，增强季节感与自然意境。

标题“旬之味：主厨定制三部曲”位于顶部中央，字体为金色古朴书法体，下方配有一个方形印章图案，内含“煎堂”字样（可能为餐厅或品牌标识）。标题两侧延伸出两条细金线，形成视觉平衡。

三道菜品分别以“COURSE 01｜先付”、“COURSE 02｜强肴”、“COURSE 03｜甘味”命名，并配有对应描述文字及图像：

- **COURSE 01｜先付**  
  文字说明：“北海道马粪海胆，鲜甜爆浆”  
  图像展示：一个透明高脚玻璃杯中盛放着橙黄色的海胆，顶部点缀黑色鱼子酱，底部可见绿色装饰物（如黄瓜片），置于左侧台面上。杯旁散落几片红色枫叶，增添秋意。

- **COURSE 02｜强肴**  
  文字说明：“备长炭烤A5和牛，入口即化”  
  图像展示：两块切面呈粉红色、带有丰富大理石纹路的熟成和牛，表面焦香微脆，摆放在深色圆形盘中，位于画面中央偏下位置，是视觉焦点。

- **COURSE 03｜甘味**  
  文字说明：“手打宇治抹茶，清苦回甘”  
  图像展示：一个深色浅口盘中放置一颗绿色抹茶团子（或抹茶大福）与两颗白色小丸子（可能是白玉团子或糯米球），位于右侧台面上，背景有灰色岩石状装饰。

三道菜通过细线从文字标签连接至对应图像，构成清晰的信息关联。整体布局呈三角形构图，中央为和牛主菜，两侧分别为前菜与甜点，层次分明。光线柔和聚焦于食物，突出质感与色泽。整个画面无多余元素，强调食材本真与匠人精神，传递高端日式料理的精致与仪式感。

所有文本均使用中文，包括标题、课程编号、菜名及描述，语言风格典雅且富有诗意，符合高端餐饮宣传语境。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 85：Young Pioneers of China Admission Ceremony &amp; Educational Activities Guide</b></div><img src="../all_small/058.webp" alt="信息图案例 058" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 85 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 86：超实用居家生活小贴士</b></div><img src="../all_small/086.webp" alt="信息图案例 086" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这是一张以黑板报风格设计的中文信息图，标题为“超实用居家生活小贴士”，整体采用手绘粉笔字和简笔画风格，背景为深黑色，边框为浅木色，营造出温馨、亲切的居家氛围。信息图通过白色线条分割成左右两大区域，左侧聚焦于两道简单快捷的家常菜食谱，右侧则提供多种日常生活的实用技巧。

**整体布局与视觉元素：**
- 标题“超实用居家生活小贴士”位于顶部中央，字体较大，带有白色描边，两侧点缀黄色和蓝色星星图案。
- 从标题向下延伸出两条白色曲线箭头，分别指向左右两个主模块。
- 左侧模块包含两个步骤（Step 1 和 Step 2），每个步骤用彩色圆角矩形框标注，配以手绘食物插图（如面条碗、鸡胸肉）和勾选标记。
- 右侧模块分为“美食制作避坑&amp;食材保鲜技巧”和“3个秒上手生活小技巧”两个子部分，每个子部分下有三个带绿色叶子图标的小技巧，配有简笔画（如锅、米饭、生菜、餐具、衣服、冰箱）。
- 图中使用了多种颜色的粉笔效果文字（白、黄、橙、蓝、绿）和装饰性元素（星星、箭头、波浪线、勾选框），增强可读性和趣味性。

---

**左侧模块：食谱教程**

**Step 1 - 10分钟番茄鸡蛋面（早餐首选）**
- **准备食材**：1个番茄、2个鸡蛋、1把挂面、2勺生抽、半勺盐，可按需准备少量番茄酱。
- **制作步骤**：
  1. 番茄切小块下锅炒出沙；
  2. 加足量清水煮开后下挂面煮3分钟；
  3. 淋入搅好的蛋液，加调料搅匀即可出锅。
- **优点提示**（带勾选框）：
  - ✅ 加番茄酱口感升级
  - ✅ 备齐基础食材
  - ✅ 按步骤操作零失败
- 配图：一碗面条、筷子、两个鸡蛋。

**Step 2 - 蒜香黄油鸡胸肉（懒人便当）**
- **准备食材**：2块鸡胸肉、3勺生抽、1勺蚝油、10g黄油、5瓣蒜切片。
- **制作步骤**：
  1. 鸡胸肉切1.5cm厚片，加调料抓匀腌制20分钟；
  2. 平底锅放黄油融化，下鸡胸肉小火两面各煎3分钟即可。
- **优点提示**（带勾选框）：
  - ✅ 提前腌制更入味
  - ✅ 小火慢煎不发柴
  - ✅ 冷藏可存放3天
- 配图：一只鸡、一块鸡胸肉、一个盘装鸡胸肉。

---

**右侧模块：生活小技巧**

**美食制作避坑&amp;食材保鲜技巧**
- **美食制作额外小贴士**（副标题）
  - **厨具使用避坑**：
    - 制作番茄、柠檬等酸性食物时不要用铁锅，会析出铁锈影响口感，长期食用不利于身体健康。
    - 配图：红色小锅。
  - **剩米饭利用**：
    - 吃不完的剩米米饭可以做成饭团、蛋炒饭，比二次蒸饭口感更好，还能变换口味。
    - 配图：一碗米饭。
  - **绿叶菜延长保鲜**：
    - 生菜、菠菜等绿叶菜吃不完时用干燥厨房纸完全包裹再放冷藏，保鲜时长可延长3-5天。
    - 配图：一颗生菜。

**3个秒上手生活小技巧**
- **餐具防碎去毒**：
  - 新买的陶瓷餐具放入加了食盐的沸水煮10分钟，不仅能降低碎裂概率，还可减少餐具残留的重金属析出。
  - 配图：叉子和勺子。
- **衣物去油免洗**：
  - 衣物不慎沾到油渍不用整件水洗，在油渍处挤少量洗洁精干搓2分钟，再用湿纸巾擦净即可完全去除油渍。
  - 配图：一件T恤。
- **冰箱平价除臭**：
  - 冰箱内放一卷拆开的卷装卫生纸，可快速吸附异味，每半个更换一次，比商用除臭剂性价比高5倍以上。
  - 配图：一个冰箱门。

---

该信息图结构清晰，内容实用，适合家庭主妇、上班族或学生等需要快速解决日常饮食和生活问题的人群。所有文本均为中文，无英文或其他语言内容。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 87：好习惯养成指南：重塑人生的底层逻辑</b></div><img src="../all_small/090.webp" alt="信息图案例 090" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“好习惯养成指南：重塑人生的底层逻辑”为主题，采用柔和水彩风格背景，整体布局为四象限结构，内容分为四个主要板块，辅以插画和图标增强可读性。主标题位于顶部中央，字体较大，颜色为深蓝色，具有引导性。

第一板块：“培养好习惯的核心价值”，位于左上角。配有一幅插画，描绘一个人走在由齿轮和植物组成的树干上，象征习惯的自动化与成长。下方三个核心价值点分别用图标和文字说明：
- 减少意志力消耗：图标为大脑与齿轮，文字解释“将需要刻意坚持的行为转化为自动化动作，大幅减少意志力消耗。”
- 复利增长效应：图标为金币与向上箭头，文字说明“长期坚持好习惯的复利效应，是普通人实现人生突破的最低门槛路径。”
- 提升人生掌控感：图标为指南针，文字说明“稳定的好习惯体系能提升人生掌控感，降低焦虑情绪的发生概率。”
底部总结句：“好习惯是人生性价比最高的投资，无需巨额成本，长期坚持即可收获超额回报。”

第二板块：“好习惯的全维度人生影响”，位于右上角。包含四个子维度，每个子维度配有插画和文字说明：
- 健康维度（绿色标签）：插画为女性跑步、蔬果，文字说明“规律作息、均衡饮食、定期运动。降低30%以上的慢性疾病发病风险，提升平均寿命5-8年。”
- 认知维度（绿色标签）：插画为女性树下阅读，文字说明“每日阅读、定期复盘、持续拓宽认知边界，10年累积知识储备量远超同龄人。”
- 事业维度（蓝色标签）：插画为男性办公，文字说明“要事优先、及时反馈、日清日结，提升40%以上的工作效率，获得更多晋升机会。”
- 社交维度（橙色标签）：插画为两人握手交谈，文字说明“诚实守信、情绪稳定、换位思考。构建更健康的人脉关系，获得更多信任与支持。”
底部总结句：“好习惯渗透生活的每一个角落。”

第三板块：“好习惯vs坏习惯的10年人生差”，位于左下角。主题句为“每天差30分钟，10年差整个人生”。分为左右两列对比：
- 左列“长期坚持好习惯”：
  - 每日行为差异：花30分钟阅读/运动（插画为女性读书）
  - 1年状态：养成稳定习惯，知识储备、体能状态优于同龄人30%（插画为女性看书+书堆）
  - 5年状态：在所在领域有一定知识积累，身体素质远高于平均水平（插画为男性西装+奖杯）
  - 10年状态：成为领域内专业人士，几乎无慢性基础病，人生选择权更多（插画为男性西装+天平与灯泡）
- 右列“长期保持坏习惯”：
  - 每日行为差异：花30分钟刷无营养短视频/吃垃圾食品（插画为男性看手机+零食）
  - 1年状态：无明显变化，偶尔出现颈椎、肠胃不适（插画为男性头痛）
  - 5年状态：知识储备无明显提升，已出现1-2种慢性基础病（插画为男性头痛+脑部图标）
  - 10年状态：核心竞争力无增长，健康问题频发，人生选择空间被大幅压缩（插画为男性痛苦+龙卷风）

第四板块：位于右下角，包含三个子部分：
- 统计数据：柱状图显示“降低30%以上的慢性疾病发病风险”，X轴无刻度，Y轴为趋势上升箭头，数据来源标注“相关健康研究”。
- 引用语录：卷轴样式，内容为“好习惯是人生性价比高的投资。”——佚名
- 关键术语：三个图标与文字组合：
  - 自动化动作（齿轮循环图标）
  - 复利效应（金币生长图标）
  - 人生掌控感（指南针图标）

整张信息图使用大量手绘风格插画，色彩清新自然，主要色调为蓝、绿、米白，视觉层次分明，信息结构清晰，旨在通过多维度对比和数据支撑，强调好习惯对个人发展的深远影响。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 88：根基与重构：乡村振兴的动态图谱</b></div><img src="../all_small/016.webp" alt="信息图案例 016" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“根基与重构：乡村振兴的动态图谱”为主题，采用拼贴艺术风格，融合粗麻布纹理、颜料泼洒、线条放射等视觉元素，营造出充满活力与有机质感的视觉效果。整体布局呈垂直结构，中心为一个由黄色、橙色和红色颜料从中心向外辐射的爆炸式图案，象征着乡村振兴的动能与扩散效应；背景则由绿色、棕色、米色等大地色调构成，辅以白色、银色、金色的线条和滴落颜料，形成多层次、立体化的视觉层次。

标题“根基与重构：乡村振兴的动态图谱”以深棕色粗体字置于中央绿色横幅之上，字体醒目，强调主题。

信息图围绕四个核心维度展开，每个维度以独立文本框呈现，内容如下：

1. **农业现代化 (Tech Acceleration)**
   - 位置：左下方
   - 文本内容：“智慧农机与生物技术的介入。这是对传统耕作方式的机械化重构，虽然精准且理性，但在整体的生命网络中，它是推动效率跃迁的核心动能。”
   - 视觉特征：文本框为深棕色，标签“农业现代化”置于白色撕纸状标签上，下方有胶带状装饰，体现手工拼贴感。

2. **产业兴旺 (Industrial Synergy)**
   - 位置：右下方
   - 文本内容：“一二三产的深度融合打破了单一农业的界限。就像颜料在空中交织，数字农业、乡村旅游与现代物流构建起复杂的价值交换网络，激发经济活力。”
   - 视觉特征：同样采用深棕色文本框与白色撕纸标签，标签上方有一条橙色胶带装饰。

3. **生态宜居**
   - 位置：底部中央偏右
   - 文本内容：“这是振兴的基石。绿水青山不仅是色彩的铺陈，更是生命的流动。自然资源的保护与利用如同底色，决定了整幅画作的调性与生命长度。”
   - 标签：“生态宜居”置于白色撕纸标签上，其右侧有一个标有“03”的浅棕色麻布袋标签，下方还有一个绿色标签注明“‘基石’”，突出其基础地位。

4. **共同富裕 (Shared Prosperity)**
   - 位置：顶部中央
   - 文本内容：“人才与要素的自由流动。城乡边界在这些细密线条的渗透下变得模糊。这一种更高维度的平衡，让现代文明的成果如同雨滴般均匀滋润每一片土地。”
   - 视觉特征：位于白色放射状线条中心，背景为麻布纹理，文字置于深棕色框内，英文标注“(Shared Prosperity)”位于中文下方。

此外，整个画面通过大量白色、银色、绿色、金色的线条连接各个模块，模拟网络或神经元结构，象征各要素之间的联动与渗透。颜料的飞溅与滴落效果增强了动态感，而散落的谷物颗粒、麻布纹理等细节强化了乡土与自然的主题。

该信息图并非传统数据图表，而是概念性视觉叙事，通过艺术化手法将乡村振兴的四个关键支柱——农业现代化、产业兴旺、生态宜居、共同富裕——有机整合，表达其相互关联、协同推进的动态过程。所有文本均以中文为主，辅以英文术语作为补充说明，整体语言风格兼具诗意与理性，旨在传达乡村振兴不仅是经济指标的提升，更是系统性、生态性的社会重构。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 89：IP is the Ultimate Asset</b></div><img src="../all_small/024.webp" alt="信息图案例 024" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 89 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 90：共生共赢：企业社会价值跃迁之路</b></div><img src="../all_small/027.webp" alt="信息图案例 027" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“向上向善：2024年度企业发展与社会责任里程碑”为副标题，主标题为“共生共赢：企业社会价值跃迁之路”，整体采用清新明亮的蓝绿色调，营造出可持续发展与成长的视觉氛围。布局呈阶梯式结构，象征企业从第一季度到第四季度（年度）的逐步进阶与价值跃迁。中心是一组由透明蓝色台阶连接的三个立体平台，分别代表Q1、Q2、Q3三个季度，顶部平台则代表年度总结（ANNUAL），形成从底部到顶端的清晰路径。

每个平台均配有主题图标和说明文字框，通过对话气泡形式呈现关键信息，并标注了对应季度和阶段名称：

- **STEP_Q1: 第一季度（战略启航）**  
  平台位于最底层，配有一个齿轮和一个心形图标，象征启动与初心。右侧文字框标题为“Q1: 锚定ESG蓝图 战略蓝图”，内容为：“发布年度战略。启动‘绿色办公’倡议，业务研发投入同比增长15%，为全年增长打下坚实基石。”

- **STEP_Q2: 第二季度（创新突破）**  
  平台位于中层，配有太阳能板、金币和植物幼苗等元素，突出科技创新与生态环保。右侧文字框标题为“Q2: 科技赋能生态”，内容为：“首个低碳智慧园区投产。通过技术迭代减少碳排放3000吨，商业版图向可持续发展领域深度延伸。”

- **STEP_Q3: 第三季度（责任深耕）**  
  平台位于上层，配有橙色爱心（内含家庭剪影）、植物幼苗和云朵，强调社会责任与人文关怀。右侧文字框标题为“Q3: 点亮公益微光”，内容为：“启动‘乡村振兴’专项计划。企业志愿者服务时长超1万小时，将商业红利转化广泛的社会温度。”

- **ANNUAL: 高质量发展典范 双重突破 ✨**  
  位于顶层平台，装饰有金色地球仪奖杯、多个奖杯、绿植和齿轮，象征卓越成就。左侧文字框内容为：“获评‘年度社会责任标杆’。实现业绩增长与ESG评分的双重突破，向着百年基业稳健迈进。”

整体设计风格为3D卡通化，具有现代感和亲和力。视觉元素包括发光的台阶、漂浮的云朵、小齿轮、太阳能板、金币、植物幼苗、爱心等，强化了环保、科技、人文、成长等核心主题。所有文本均为中文，语言正式且富有激励性，传达企业积极履行社会责任、追求可持续发展的愿景。信息流从Q1到Q3再到ANNUAL，逻辑清晰，层层递进，完整展现企业在2024年各阶段的战略部署与成果。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 91：Rhymes of Nature: 诗与思</b></div><img src="../all_small/030.webp" alt="信息图案例 030" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>这张信息图的标题为“Rhymes of Nature: 诗与思”，采用中英双语呈现，主标题使用黑色粗体无衬线字体，居于画面右上角。背景为浅蓝色，整体风格为现代扁平化插画风，色彩明快，线条简洁流畅，充满童趣与诗意。

画面布局呈非对称分散式结构，围绕四个核心主题展开，每个主题由一个英文关键词（Listen, Flow, Look Up, Grow）及其对应的中文翻译和一段富有哲理的文字说明组成，文字内容置于白色对话框内，边角圆润，带有轻微阴影，增强视觉层次感。

各主题及其对应视觉元素如下：

1. **Listen (倾听)**
   - 位置：左上区域。
   - 视觉元素：一位粉红色上衣、紫红色裤子的人物侧身蹲坐，双手张开如接住落叶，周围飘散着几片彩色叶子和抽象的白色风形线条。
   - 文字内容：
     &gt; Listen (倾听)
     &gt; 听风起，看叶落。万物皆有定时，在自然的律动中感受时光的诗意，无须急躁。

2. **Flow (顺应)**
   - 位置：中左区域。
   - 视觉元素：一位蓝绿色调的人物蜷缩成S形，身体融入一条由多种蓝色色块拼接而成的河流或波浪形态，手中捧着水滴，周围点缀着绿叶、小圆球（象征水珠）和一只飞翔的小鸟。
   - 文字内容：
     &gt; Flow (顺应)
     &gt; 上善若水。生命如同河流，遇到阻碍时便绕行，在柔软中蕴含着穿透坚石的力量。

3. **Look Up (仰望)**
   - 位置：右侧中上区域。
   - 视觉元素：一位紫色人物坐在一朵巨大的蘑菇上，抬头仰望星空，背景中有弯月、星星和云朵，旁边还有两只小鸟飞过。
   - 文字内容：
     &gt; Look Up (仰望)
     &gt; 身处沟壑，也要仰望星空。宇宙的浩瀚让人释怀眼前的烦恼，寻找内心的辽阔。

4. **Grow (生长)**
   - 位置：底部中央区域。
   - 视觉元素：一棵绿色大树占据中心，树干棕色，树冠由多个绿色圆形拼接而成。四位不同颜色（粉、紫、蓝、绿）的人物手拉手环绕树干跳跃，姿态活泼，象征团结与成长。树旁有一株含苞待放的粉色花朵。
   - 文字内容：
     &gt; Grow (生长)
     &gt; 向下扎根，向上生长。每一粒种子都蕴含着冲破泥土的渴望，在阳光下尽情绽放。

此外，画面顶部散布着半圆形、星形、月亮等装饰性几何图形，增强了梦幻与自然的主题氛围。整体构图均衡而不呆板，人物动作充满动态美感，色彩搭配和谐，营造出一种宁静、积极、富有生命力的视觉体验。该信息图旨在通过自然意象传达人生哲理，鼓励人们倾听内心、顺应变化、仰望星空、努力生长。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 92：College Entrance Pathway Reforce Comparison</b></div><img src="../all_small/045.webp" alt="信息图案例 045" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 92 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 93：Web Accessibility Guideline Compilation</b></div><img src="../all_small/047.webp" alt="信息图案例 047" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 93 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 94：Embroidery 101: Core Techniques &amp; Real-World Applications</b></div><img src="../all_small/053.webp" alt="信息图案例 053" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 94 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 95：A Complete Guide to Project Feasibility Study Report Compilation &amp; Professional Consulting Services,</b></div><img src="../all_small/060.webp" alt="信息图案例 060" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该案例展示了 U1-8B-MoT-Infographic 在第 95 个信息图场景中的生成效果，重点体现模型在版式规划、图表结构、文字渲染、背景协调性和整体视觉美观度上的能力。上方图片为对应的生成结果，可用于对比基础 8B-MoT 模型和 Infographic 专用模型在复杂信息图生成任务中的表现。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 96：高效传递信息，加快救援响应</b></div><img src="../all_small/083.webp" alt="信息图案例 083" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为《救援服务全解析：实用指南+真实案例分享》，旨在系统性地解析救援服务的类型、适用场景、操作流程，并通过真实案例帮助公众掌握正确的求助方法，提升风险响应能力。整体布局清晰，采用分块结构，分为“概览”、“一、常见救援服务类型及适用场景”、“二、救援申请标准化操作流程”、“三、真实救援案例复盘”四个主要部分，辅以图标、插画和文字说明，风格简洁明了，视觉元素丰富，色彩搭配以灰白背景为主，配以红色、橙色、蓝色等强调色突出重点内容。

**概览部分**位于顶部，简要说明本图目的：“析图精援的商形文化，系统式明确救援服务类型、场景及和流程，结合实例案例，并功缆援应急急结合涵和真实案例，帮助掌握握正确的帮助方法，提升级风险响应。”（注：原文存在明显错别字，如“商形文化”应为“品牌形象”或类似，“功缆援应急急结合涵”可能为“功能救援与应急结合内涵”，但此处保留原始文本）。

**第一部分：常见救援服务类型及适用场景**
中心是一个圆形标题框，内文为“常见救援服务分类速查速查 分清类型，求助不走弯路”。从中心向外辐射出三大类别：
- **公共救援**：包含消防车（标有“119”）、救护车（标有“120”）、交通锥、反光背心、路障、救生圈、绳索、无人机等图标，对应“医疗物具”、“交通援具”、“水水/山救援工具”，并注明“免费充共工具，公绳成分物、天天”（原文表述不清，可能意指“免费公共工具，公共绳索等物资，全天候可用”）。
- **商业救援**：包含拖车、钥匙、会员卡、登山靴、背包、指南针等图标，说明“有费用户可付救服务通过代理热线”。
- **公益救援**：包含红十字背心、搜救犬、寻人启事海报（标有“搜名人协请版”、“报名”及二维码），说明“免费救援队商模型、大人窜件售帮公方或官众号部或紧急部门”（原文表述混乱，可能意指“免费救援队商业模式、成人事件求助可通过官方公众号或紧急部门”）。

**第二部分：救援申请标准化操作流程**
标题为“高效传递信息，加快救援响应”，分为三个步骤，横向排列：
- **第一步：避险自保**：图标为两个绿色奔跑小人，文字说明“进入手法子或敕安全，避险危障”（原文错误，应为“进入安全区域或采取自救措施，避免危险障碍”）。
- **第二步：准确报信**：图标为手机通话，列出“4关键信息点”：1. 位置；2. 人亡/状态；3. 风险类型；4. 联系联系（原文重复，应为“联系方式”）。
- **第三步：原地等候**：图标为坐着看手机的人和信号波纹，文字说明“如紧紧急帮时，请求救求近附近的帮助”（原文错误，应为“如遇紧急情况，请请求附近人员的帮助”）。

**第三部分：真实救援案例复盘**
标题为“从真实案例学救援应对技巧”，下设“处置过程”和“案例总结”。
- **处置过程**：左侧展示地图、指南针、干粮等生存物资，文字描述：“2023年10月，3人迷巽者了迷3人的迷路，信号量湖弼弱，旦踉重量的小食物量册有限少。”（原文多处错别字，应为“2023年10月，3名迷路者迷失在山区，信号微弱，仅携带少量食物”）。中间展示登山者攀岩并使用手机的插画，文字说明：“他们他们叫110 溃明确的，洞性清晰清晰信息，风睑类型，荥定位置置，定动成功照救救援情况。”（原文错误，应为“他们拨打110，提供了明确、清晰的信息，包括风险类型、定位位置，成功启动救援”）。右侧展示警车、消防车、搜救犬及救援队伍，文字说明：“他们救援后找到:12小时，并会会救救援队取成功接购，2小时联合救援队及成功找到，无有死亡”（原文错误，应为“他们被救援队在12小时内找到，其中2小时由联合救援队成功营救，无人死亡”）。
- **案例总结**：左侧为扩音器图标，文字：“准确位置和人员信息不忺是信息，以敢韵救心的成功”（原文错误，应为“准确的位置和人员信息是成功救援的关键”）。右侧为剪贴板图标，列有三个带勾选框的要点：
    - 准确的位置和人员信息是核心为成功
    - 在前行去不褪娄理前，前行注衮规划注册向前登擢登区（原文错误，应为“在出行前，务必规划行程并注册登记登山区域”）
    - 购买相关相的户外救援服务（原文错误，应为“购买相关的户外救援保险或服务”）

整体而言，该信息图通过图文结合的方式，尽管存在大量错别字和语法错误，但仍试图传达救援服务的分类、标准化求助流程及真实案例经验，具有较强的教育意义，但在语言表达上需要大幅修正以确保信息准确传递。</code></pre>
</div></details></div></td>
</tr>
<tr>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 97：Infographic 97</b></div><img src="../all_small/087.webp" alt="信息图案例 087" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“磷化工行业核心维度信息对比表”，采用未来科技感设计风格，以深色背景搭配霓虹蓝、紫、粉等荧光线条与文字，整体呈现数字化仪表盘或虚拟界面视觉效果。图表结构为横向多列、纵向三行的表格布局，左侧为三大主题分类，顶部为五个分析维度：统计数据、核心术语、风格偏好、布局偏好、其他要求。

图表内容分为三个主要部分：

**第一部分：磷化工行业的发展价值与现状**
- **统计数据**：
  - 全球最大生产国/消费国
  - 磷矿石储量/磷肥产量 &gt;40% 全球总量
  - 新能源相关磷化工产品需求增速 &gt;30%/年
- **核心术语**：
  - 磷肥生产
  - 新能源电池
  - 食品加工
  - 阻燃材料
  - 精细化升级
  - 高附加值
  - 磷酸铁锂
- **风格偏好**（配图）：
  - 农业领域（农田+化肥桶）
  - 新能源领域（绿色手机电池）
  - 食品/材料领域（食品与化学品）
  - 绿色转型（地球循环图标）
- **布局偏好**（图文结合）：
  - 磷化工：支撑多领域的基础性产业（上升趋势柱状图）
  - 我国磷化工产业规模全球领先，正朝高端化升级（环形增长图表）
- **其他要求**：
  - 国民经济基础性产业
  - 不可或缺

**第二部分：磷化工行业现存环境风险**
- **统计数据**：
  - 磷石膏年排放量 &gt;8000万吨
  - 综合利用率 &lt;50%
  - 传统采收率 &lt;60%
- **核心术语**：
  - 含氟废气
  - 含磷氨氮废水
  - 大气污染
  - 水体富营养化
  - 饮用水安全
  - 磷石膏堆存
  - 土壤/地下水污染
  - 乱采滥挖
  - 植被破坏
  - 水土流失
- **风格偏好**（配图）：
  - 磷化工发展伴生三类环境风险（工厂冒黑烟）
  - 三废排放、资源浪费问题亟待解决（禁止符号覆盖矿山）
- **布局偏好**（红色警示框）：
  - 废气/废水污染
  - 磷石膏堆存风险
  - 矿区生态破坏
- **其他要求**：
  - 威胁生态安全
  - 亟待解决

**第三部分：磷化工环境治理的主流路径与未来方向**
- **统计数据**：
  - 磷资源采收率提升至 &gt;70%
  - 规上企业主要污染物达标排放率 100%
  - 2025年磷石膏综合利用率提升至 &gt;60%
- **核心术语**：
  - 源头管控
  - 准入门槛
  - 低品位磷矿利用
  - 过程治理
  - 废气脱硫脱硝
  - 废水深度处理
  - 末端资源化
  - 磷石膏制建材
  - 制酸技术
- **风格偏好**（配图）：
  - 全链条治理推动磷化工绿色转型（智能化工厂）
  - 减污降碳与资源循环利用并行（地球循环箭头）
- **布局偏好**（蓝色流程图）：
  - 源头资源高效利用 → 过程污染达标排放 → 末端副产物资源化
- **其他要求**：
  - 绿色转型
  - 可持续发展

整个信息图通过清晰的分类和数据对比，系统展示了磷化工行业的现状、挑战与发展方向，强调其在国民经济中的基础地位，同时指出环境污染问题及绿色转型的必要性。视觉元素如箭头、图表、禁令符号和工业场景插图增强了信息传达的直观性和警示性。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 98：核心逻辑：双向作用的共生关系</b></div><img src="../all_small/089.webp" alt="信息图案例 089" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“核心逻辑：双向作用的共生关系”为主题，旨在梳理职场个人成长与企业文化的相互作用机制，提供适配度判断方法及相应成长策略，帮助职场人士理清两者之间的共生关系，并找到最大化成长收益的落地路径。整体设计采用暖色调背景（浅米色），内容布局清晰，分为左右两大板块，通过中央垂直分割线和“企业文化”、“双向作用”两个竖向标签进行结构化分隔，视觉上强调“双向奔赴”的核心概念。

左侧部分标题为「适配度判断：3个核心评估维度」，包含三个评分项（1-5分自行打分），每个维度均配有插图、说明文字和评分提示：

1. **价值导向匹配度**  
   - 插图：放大镜观察箭头方向，与指南针形成对比，象征方向一致性。  
   - 说明文字：“个人职业发展目标是否和企业倡导的价值方向一致，例如希望深耕技术的员工是否处于「技术优先」而非「关系优先」的文化环境”。

2. **成长资源匹配度**  
   - 插图：浇水壶浇灌植物，旁边有书籍、梯子、灯泡等象征成长资源的元素。  
   - 说明文字：“企业是否提供个人成长所需的资源支持，包括培训机会、试错空间、跨部门项目权限、晋升通道透明度等”。

3. **氛围适配度**  
   - 插图：左侧为平静工作的女性（电脑前、咖啡杯、时钟），右侧为焦虑抓头的男性（文件堆积、混乱思绪），中间以“VS”分隔，体现工作氛围差异。  
   - 说明文字：“企业的工作节奏、沟通文化、奖惩规则是否符合个人工作习惯，例如偏好扁平高效沟通的员工是否需要应对复杂层级审批和无意义内耗”。

右侧部分标题为「成长落地：不同适配度的应对策略」，根据适配度分数划分为三种策略，每种策略配有插图和详细说明：

- **适配度80分以上：深度绑定**  
  - 插图：一人拥抱大树，象征与企业深度融合。  
  - 策略说明：“深度绑定策略，主动参与企业文化建设，争取核心项目资源，成为企业文化的受益者和传播者，获得最快成长速度”。

- **50-80分：求同存异**  
  - 插图：一人用锤子修桥过河，象征克服障碍、搭建连接。  
  - 策略说明：“求同存异策略，抓住适配部分获取成长资源，主动避开不适配的规则内耗，专注个人核心能力提升，待能力成熟后再选择更适配的平台”。

- **50分以下：及时止损**  
  - 插图：一人背着背包奔跑在阳光小径上，象征果断离开、开启新旅程。  
  - 策略说明：“及时止损策略，评估企业文化调整的可能性，若现有文化完全阻碍个人成长，及时更换赛道避免消耗职业黄金期”。

在左右两部分之间，有一个橙色对话框标注“核心逻辑：不是单向选择，是双向奔赴”，强化主题。整张信息图使用简洁的卡通风格插画、清晰的标题层级、统一的配色方案（主色为橙黄、棕色系）和直观的图标，使内容易于理解且富有亲和力。所有文本均为中文，语言准确、专业，适合职场人群阅读与自我评估。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 99：Infographic 99</b></div><img src="../all_small/093.webp" alt="信息图案例 093" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图以“地球四季与生命律动”为标题，采用垂直分层结构，通过三维立体视觉效果和透明玻璃板分层设计，系统阐述了地球四季形成机制及其对生命活动的驱动作用。整体风格科技感十足，背景为深蓝色宇宙空间，点缀发光羽毛和光效线条，营造出神秘而富有动感的氛围。

图像从上至下分为四个主要层级，每个层级均配有文字说明框（半透明圆角矩形），内容围绕地球运动与生命响应展开：

**第一层：地球公转与地轴倾斜**
- 中心为一个三维地球模型，其自转轴倾斜23.5°，并标有“23.5°”字样。
- 地球沿椭圆形轨道绕太阳公转，轨道呈现彩虹色光泽，轨道上有多个地球位置点，表示不同季节的位置。
- 左侧文字框标题：“地轴倾斜成因”，内容：“地球自转轴倾斜 23.5 度，公转中角度不变，导致阳光照射区域变化。”
- 右侧文字框标题：“公转轨道变化”，内容：“地球绕太阳公转，不同位置接收阳光角度与强度不同，形成四季循环。”

**第二层：温度与能量变化及植被生长周期**
- 层级上方是一个渐变色平面（橙黄至蓝白），象征太阳辐射强度分布。
- 平面下方是四个透明圆盘，分别展示植物在不同季节的状态：
  - 春季：嫩芽萌发；
  - 夏季：绿叶繁茂；
  - 秋季：叶片泛黄；
  - 冬季：枝干枯萎。
- 左侧文字框标题：“温度与能量”，内容：“直射区温度升高，斜射区温度降低，影响生物生存与繁殖策略。”
- 右侧文字框标题：“植被生长周期”，内容：“春生夏长，秋收冬藏，周期性生长帮助植物适应环境变化。”

**第三层：动物迁徙与繁殖行为**
- 层级展示三只候鸟飞行轨迹，带有发光尾迹，象征迁徙路径。
- 下方拼图式地形包含多种生态元素：绿色植物、落叶树、昆虫、鸟类等。
- 左侧文字框标题：“动物迁徙繁殖”，内容：“候鸟随季节迁徙，寻找温暖温嗳地区繁殖，适应不同季节环境。”（注：“温嗳”应为“温暖”的笔误）
- 右侧文字框标题：“生态系统影响”，内容：“季节变化驱动食物链，支持微生物、昆虫及消费者生存。”

**整体布局与视觉设计**
- 图像采用分层堆叠的立体结构，每一层代表一个主题模块，由上至下逻辑递进：从物理成因 → 能量传递 → 植物响应 → 动物行为 → 生态系统效应。
- 文字框使用统一的深灰色半透明背景，白色或浅色字体，配有点状装饰图标，增强可读性。
- 所有图形元素具有光滑质感和光影反射，突出科技感与现代感。
- 标题“地球四季与生命律动”位于顶部，字体为银白色金属质感，下方环绕着叶子、羽毛和星球装饰元素，强化主题。

本信息图完整呈现了地球四季形成的天文基础及其对生命系统的多层次影响，图文结合严谨，逻辑清晰，适合用于科普教育场景。</code></pre>
</div></details></div></td>
<td valign="top" width="25%"><div align="center" style="margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><b>案例 100：Infographic 100</b></div><img src="../all_small/095.webp" alt="信息图案例 095" width="100%"><div style="margin-top: 8px;"><details><summary><b>Prompt</b></summary><div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
<pre><code>该信息图题为“根系微观：矿物质吸收机制”，以深色土壤背景和发光的植物根系为中心视觉元素，采用科技感十足的蓝白色调与半透明结构设计，整体风格现代、富有未来感。图像中央是一棵半透明的植物根系，从土壤中延伸而出，主根与侧根清晰可见，并伴有蓝色光点与能量流线条，象征养分传输路径。围绕中心根系分布着八个信息模块，每个模块均包含一个插图和一段说明文字，通过细线连接至根系相应位置，形成逻辑关联。

所有文本均为中文，字体为无衬线体，标题使用较大加粗字体，内容框内文字清晰可读。每个信息模块均采用带发光边框的矩形框，内部包含高分辨率科学插图和简明解释文字。

以下是各模块详细内容：

1. **根毛与根尖生长**
   - 插图：放大显示根尖区域，可见大量细长根毛从表皮细胞伸出，根部浸于土壤中。
   - 文字：“根毛极大增加接触面积，根尖深入土壤寻找养分。”

2. **离子交换机制**
   - 插图：细胞膜结构示意图，载体蛋白在膜上选择性转运离子，同时释放细胞内离子。
   - 文字：“载体选择性吸收离子，同时释放细胞内离子。”

3. **质子泵与 ATP**
   - 插图：细胞膜上的质子泵（H⁺-ATPase）利用ATP水解能量将H⁺泵出细胞，形成跨膜电位差。
   - 文字：“利用 ATP 能量泵出 H⁺，形成电位差驱动离子进入。”

4. **养分感应机制**
   - 插图：根细胞感知外部环境变化，激素分子（如紫色小颗粒）被激活并传递信号。
   - 文字：“感应浓度变化，通过激素调节根生长与分化。”

5. **信号转导途径**
   - 插图：细胞内信号通路示意图，包括受体、第二信使、核内基因表达调控等过程。
   - 文字：“激活信号途径，改变基因表达以合成转运蛋白。”

6. **营养素运输**
   - 插图：木质部和韧皮部纵向管道结构，箭头指示向上和向下的物质流动方向。
   - 文字：“通过木质部和韧皮部将养分运输至植物全身。”

7. （注：实际图中左侧第三个模块为“养分感应机制”，其下方应为“营养素运输”；右侧第三个模块为“信号转导途径”，其下方应为“质子泵与ATP”的补充或相关联模块。但根据布局，左侧从上到下依次为：根毛与根尖生长 → 离子交换机制 → 养分感应机制 → 营养素运输；右侧从上到下依次为：质子泵与ATP → 信号转导途径 → （无明确标签的模块，可能为补充说明或重复）→ （无明确标签模块）。经仔细核对，右侧第二个模块为“质子泵与ATP”，第三个为“信号转导途径”，最下方为“营养素运输”的对应插图。）

8. （补充：左侧第一个模块上方为“根毛与根尖生长”，其下方为“离子交换机制”，再下方是“养分感应机制”，最下方是“营养素运输”。右侧从上到下依次为“质子泵与ATP”、“信号转导途径”，以及底部与“营养素运输”对应的插图。）

整体布局呈左右对称结构，左侧侧重物理结构与初步吸收，右侧侧重分子机制与信号调控，底部则统一归结到运输环节。信息流由外而内、由结构到功能、由局部到整体，层层递进，完整呈现了植物根系吸收矿物质的微观生理过程。

数据编码方式主要依赖视觉隐喻：
- 发光线条代表能量流或信号传递；
- 箭头指示物质或信号方向；
- 不同颜色区分不同分子或结构（如ATP为橙黄色，H⁺为蓝色，激素为紫色）；
- 放大插图用于揭示微观细节。

此信息图兼具教育性与艺术性，适合用于生物学教学或科普传播，系统地展示了植物根系如何通过复杂的生物化学机制高效吸收和分配矿物质养分。</code></pre>
</div></details></div></td>
</tr>
</table>
