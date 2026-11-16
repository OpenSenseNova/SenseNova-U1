# SenseNova-U1 Infographic Showcases

<p align="center"><strong>English</strong> | <a href="./u1_infographic_showcases_CN.md">简体中文</a></p>

The [infographic model guide](u1_infographic_model.md) covers V1–V3 weights, run commands, and evaluations. This page shows V3 examples; earlier examples and comparisons are in the [archive](u1_infographic_archive.md).

## SenseNova-U1-8B-MoT-Infographic-V3 Showcase

The V3 edits below are grouped by local text, local content, overall style, and overall layout changes.

### Local Text Editing

#### Region-Marked Precise Editing

<table width="100%" style="table-layout: fixed;">
<tr>
<th width="28%" align="left">Instruction</th>
<th width="36%" align="left">Before Edit</th>
<th width="36%" align="left">After Edit</th>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">将蓝色框中的屋顶变为红色，将红框中的标题改为比较简短的格式，将绿框中的屋子改为红色屋顶的咖啡屋，将紫色框中改为共享自行车站，将粉色框中的内容改为书柜，将右下角青色框中的内容改为，室内植物玻璃房</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/bbc1e512-b947-4a7f-a2bf-219716b57e47" alt="V3 edit showcase 26 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/ec2479e2-3aa4-4b8a-850b-64c6ccd5e73d" alt="V3 edit showcase 26 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把红色框的“经典·从容·每一刻”换成经典时尚手表，蓝色框换成“100米防水”，橙色框换成电池寿命约10年</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/05802591-b36b-49fc-93aa-e625ca29af8e" alt="V3 edit showcase 03 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/7eb76c56-af18-4025-aec4-1feb6417d164" alt="V3 edit showcase 03 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">修复红框内模糊标题为“量子互联网的第一条链路”，去掉红框，其余不动。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/d7bc06a3-db7e-4fd2-802c-71a7489bde7a" alt="V3 edit showcase 04 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/08631c76-390c-448c-81fd-5d087b0fa7fd" alt="V3 edit showcase 04 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">请在图像中红色边界框标出的空白区域内插入指定内容，并使新增内容与真实环境的材质、透视、光照和空间关系自然融合。<br>目标区域：<br>“画面左下角主面板下方的空白深色区域，位于左侧四个数据卡片下方、底部图表左侧的红色边界框内部”<br>需要插入的文字或内容为：<br>“年度结余目标：120,000元”<br>请严格将新增内容限制在红框内部。完成插入后，需要彻底去除红色边界框，使最终图像中不再显示任何红线、控制点、箭头或编辑标记。<br>在插入内容之前，请先分析目标区域的实际环境，包括：<br>深色金融仪表盘背景的平面方向；<br>目标区域的矩形仪表盘面板边界；<br>玻璃拟态卡片、深色渐变背景和细线边框材质；<br>光源方向和亮度；<br>环境阴影与微弱高光；<br>字体应该像金融仪表盘中的数字标签和状态卡片，而不是普通贴纸文字。<br>新增文字必须按照目标表面的透视自然放置。由于目标区域是正面仪表盘平面，文字应保持水平排版、居中或左对齐，并与原有卡片网格对齐。<br>新增内容的颜色、亮度和清晰度必须符合环境光照。建议使用浅色文字配合绿色或金色强调数字，使其与原图“储蓄率 41%”“目标完成 76%”等数据卡片风格一致。不能使用与场景完全无关的纯白贴字，也不能让文字看起来悬浮在表面前方。<br>如果目标区域是仪表盘卡片：<br>文字应在红框内部合理居中；<br>保留原有深色面板边缘、微弱发光和阴影；<br>不得覆盖相邻卡片、图表或边框；<br>可以根据原有设计加入适当内边距和小型图标。<br>插入的文字必须完整、清晰、无乱码、无错别字。字体风格应与金融仪表盘和数据卡片用途相符，不得使用与场景冲突的字体。<br>除红框内部区域和红框本身之外，图像中的其他内容必须保持完全不变。不得移动标题“个人年度财务仪表盘”、左侧数据卡、现金流图、储蓄仪表、支出分类图、目标追踪列表或任何其他物体，不得改变整体光照、颜色和构图。<br>不得扩大目标区域，不得把新增内容放到红框外，也不得保留红框作为最终设计元素。<br>最重要的是，新增内容必须像原本就存在于该金融仪表盘中，而不是后来粘贴上去。最终画面应具有正确的材质融合、环境光照、卡片层级和自然对齐关系。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/fd54835f-44fe-46a3-948e-27f974553a90" alt="V3 edit showcase 05 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/f5b9dfc2-0aa2-40eb-98fe-136efd49c850" alt="V3 edit showcase 05 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">将原西班牙语标题替换为英文“STOP USING SHOWER SPONGES!”和副标题“YOUR SKIN DESERVES BETTER”，移除红框，框外所有文字、人物、颜色和布局完全不变。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/95f80cb9-0784-4b6b-bacf-9861fa9bb274" alt="V3 edit showcase 06 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/a1569284-f926-488c-a6c9-c9fe2c72b79f" alt="V3 edit showcase 06 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">翻译红框中的内容</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/fd532f0a-12ac-4eb8-9cb9-99c755f54596" alt="V3 edit showcase 07 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/5c347441-83e7-4bca-9812-3c2febddd670" alt="V3 edit showcase 07 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把红色框的“我不见，黄河之水地下来”替换成“君不见，黄河之水天上来”，把蓝色框的“我不见，高堂明镜悲白发”替换成“君不见，高堂明镜悲白发”。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/da548cba-72bc-474f-8dbb-1b14d84e8c68" alt="V3 edit showcase 08 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/38d4121e-44b2-4c6d-bb1b-d0307720937e" alt="V3 edit showcase 08 output" width="100%"></td>
</tr>
</table>

#### Natural-Language Prompt Editing

<table width="100%" style="table-layout: fixed;">
<tr>
<th width="28%" align="left">Instruction</th>
<th width="36%" align="left">Before Edit</th>
<th width="36%" align="left">After Edit</th>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">主题风格换成漫画风格，需要进行以下修改：<br>第一项修改：将原图中的文字：“STYLE”替换为：“时尚杂志”<br>第二项修改：将原图中的文字：“FORWARD”替换为：“美丽的补充力”</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/d828ad2e-dc25-41dc-b4fa-09c6430cf602" alt="V3 edit showcase 01 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/886a00d5-8b33-405d-84a1-0d6b1ec466fc" alt="V3 edit showcase 01 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;"><br>第一项修改：将原图中的文字：“不够高级“替换为：“DESIGN！OR  DIE！”第二项修改：将原图中的文字：“没有惊喜感”，换成“缺乏设计感会导致”</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/6668849d-6200-4bdd-bd0f-9c942a5858a9" alt="V3 edit showcase 09 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/a6bb0c48-7f53-4978-8cfe-06c0a96e879a" alt="V3 edit showcase 09 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;"><br>第一项修改：将原图中的文字：“城市低碳生活地图”替换为：“城市绿色出行地图”第二项修改：将原图中的文字：“绿色出行 42%”替换为：“绿色出行 56%”第三项修改：将位于画面右上角卡片中的“节能建筑 36栋”，调整为“低碳建筑 48栋”。第四项修改：删除文字“社区回收 18站”中的“社区”，保留为“回收 18站”，并确保删除后语句自然、排版完整。第五项修改：修正原图中的数据：“公共绿地 128公顷”修改为“公共绿地 156公顷”。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/12d33583-2451-43b0-bf81-96059e2eaf3e" alt="V3 edit showcase 10 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/09d654dd-f43c-4a44-91bb-8aff15663b0c" alt="V3 edit showcase 10 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把“主要航海壮举”换成“主要事件”，把“地理大发现时代”换成“！！地理大发现时期！！”，把“克里斯托弗·哥伦布”，换成“Cristóbal Colón”</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/611832d7-a8c7-44e1-9370-15b8f2d888ce" alt="V3 edit showcase 18 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/f5465b9a-6ef6-42b9-aaad-ab5ae50e50c7" alt="V3 edit showcase 18 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把“人类进化史”换成“人类——我们的故事”，把“概述”换成“总结”，把“主要进化阶段”换成“演化路径”。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/12dd48dc-1a75-41ca-a6b5-e33bf5cd75ed" alt="V3 edit showcase 19 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/eadbd53f-65ae-43a6-80f8-6556ed0da8d7" alt="V3 edit showcase 19 output" width="100%"></td>
</tr>
</table>

### Local Content Editing

<table width="100%" style="table-layout: fixed;">
<tr>
<th width="28%" align="left">Instruction</th>
<th width="36%" align="left">Before Edit</th>
<th width="36%" align="left">After Edit</th>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">删除海面下的塑料袋、塑料瓶和吸管，将中央口号改为白色手写体“SAVE OUR OCEAN”，把右侧海龟移到文字下方并在底部新增回收标志，保持海洋分层结构清晰。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/3c6b175d-5319-4086-b557-9494b9957f41" alt="V3 edit showcase 02 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/7bc71122-4daa-4d26-bd9a-c6ef7d3cf6fa" alt="V3 edit showcase 02 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">透明杯壁上加入“CURLY &amp; PROUD”，文字需随杯面弧度自然变形，并呈现透过玻璃与茶水后的透明度和色偏。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/6415a2b5-be35-4a26-8ab1-d2ad1dad2b8f" alt="V3 edit showcase 11 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/e6c5c364-9e43-4f3b-918a-404f68aa713c" alt="V3 edit showcase 11 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">复古电脑屏幕内加入绿色单色像素字“DESIGN MODE: ON”，匹配屏幕透视、颗粒噪点与玻璃反光；</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/1051f104-7155-4744-aede-a4d8d91f4a29" alt="V3 edit showcase 12 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/dc63618b-375c-458d-958c-ba553443483e" alt="V3 edit showcase 12 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">将四代员工使用个人AI工具的比例依次改为：Gen Z 92%、Millennials 81%、Gen X 68%、Boomers 55%，并同步调整四根彩色柱子的高度。保留人物插画、英文标题、配色纹理、来源和品牌信息不变。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/627f4998-cadd-4890-b265-46ece2c5b52b" alt="V3 edit showcase 13 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/00320589-4760-44b1-b52f-e77b6dd4b867" alt="V3 edit showcase 13 output" width="100%"></td>
</tr>
<tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">在顶部蓝色标题区右上角自然加入一枚金黄色奖杯徽章，奖杯内写“TOP GROWTH”，采用与原图一致的扁平矢量风格、粗线条和蓝黄配色，并添加少量庆祝星光装饰。调整徽章大小避免遮挡“2023 HIGHLIGHTS”，其他文字、数据、插画和布局保持不变。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/83ef4291-4b31-4aff-bd64-a211cbe7d7cd" alt="V3 edit showcase 16 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/1d6ecedc-838b-4daf-b16d-5cbe51277c92" alt="V3 edit showcase 16 output" width="100%"></td>
</tr>
</table>

### Global Style Editing

<table width="100%" style="table-layout: fixed;">
<tr>
<th width="28%" align="left">Instruction</th>
<th width="36%" align="left">Before Edit</th>
<th width="36%" align="left">After Edit</th>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">主题风格换成乐高风格。将原图中的文字：“2022“替换为：“2025”第二项修改：将原图中的文字：“FINESTBUSINESSTRENDS”换成“HELLO！ WORLD！”</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/9a4889be-ba56-4952-bc84-682f31bd86c7" alt="V3 edit showcase 17 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/6a4c1a30-cb2b-40d7-a26e-ff46a4628999" alt="V3 edit showcase 17 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把图片风格换成乐高风格和中国春节风格，所有文字换成像素字体。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/bee3637e-a452-40bf-b610-bfa7561e6975" alt="V3 edit showcase 20 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/8370e414-1be9-482b-8fce-24f3ddcf438e" alt="V3 edit showcase 20 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">将顶部主标题“城市应急避险指挥图”替换为“城市防汛应急联动图”。风格换成赛博朋克风格。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/d5b2191f-44e6-494e-aa02-9451c4a406c5" alt="V3 edit showcase 21 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/2d2437ff-ffd3-4e94-bd13-8c75b50db2aa" alt="V3 edit showcase 21 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">把图片风格换成中国传统风格。字体换成宋体</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/8e9254a7-041b-4591-b624-23121fd9a5e5" alt="V3 edit showcase 22 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/652069f7-65f5-4479-9279-c51f2bb250ff" alt="V3 edit showcase 22 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">保持内容不变，迁移为亮色风格</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/6d0595cc-7c75-43ed-8f7a-f59f9dba70f2" alt="V3 edit showcase 23 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/f7b2310a-fc49-411d-be5f-242b586ad6f0" alt="V3 edit showcase 23 output" width="100%"></td>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">风格换成复古羊皮纸地图风；保留全部文字、数据与版式。</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/7d53cde2-f182-487d-acf1-7eba7b0aa117" alt="V3 edit showcase 24 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/39ed9c3f-1756-4389-a472-6ffd47f0e2ac" alt="V3 edit showcase 24 output" width="100%"></td>
</tr>
</table>

### Global Layout Editing

<table width="100%" style="table-layout: fixed;">
<tr>
<th width="28%" align="left">Instruction</th>
<th width="36%" align="left">Before Edit</th>
<th width="36%" align="left">After Edit</th>
</tr>
<tr>
<td valign="top" width="28%"><details><summary><b>Instruction</b></summary><div style="margin-top: 8px; font-size: 0.9em; line-height: 1.5;">信息大致不变，对布局进行美化</div></details></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/28ac52b2-1518-4234-ab76-26884dc9f8cd" alt="V3 edit showcase 25 input" width="100%"></td>
<td valign="top" width="36%"><img src="https://github.com/user-attachments/assets/0338f4ff-5f76-4f8c-91b2-71087bb503f9" alt="V3 edit showcase 25 output" width="100%"></td>
</tr>
</table>

## Earlier examples

The [archive](u1_infographic_archive.md) keeps V2 and first-release examples, plus comparisons between infographic releases and the original U1 checkpoint. It also has a [larger first-release gallery](u1_infographic_archive.md#more-examples-from-the-first-infographic-model).
