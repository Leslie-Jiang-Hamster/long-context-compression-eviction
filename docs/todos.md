# TODOs (Aligned with 毕业设计模板.docx)

## A. 模板硬性结构
- [x] 封面信息定稿：题目、院系、班级、学号、导师、日期
- [ ] 原创性声明与授权书页完整（含作者签名、导师签名与日期）
- [ ] 中文摘要、英文摘要、关键词完整且规范
- [ ] 目录、参考文献、附录、致谢完整

## B. 章节结构与写作规范
- [x] 章节结构对齐：绪论、方案论证、系统设计、系统实现、性能测试与分析、总结与展望
- [x] 除总结章外，每章首段有“引言”
- [x] 第2章到倒数第2章每章有“本章小结”
- [x] 第2章到倒数第2章每章除小结外至少3个小节
- [ ] 正文避免“我/我们/本文”口语化表达，统一学术叙述

## C. 实验与结果（论文主线）
- [x] 主实验设置锁定：`hotpotqa` + `length >= 8000` + `full_kv vs AHEC` + 固定30条样本
- [x] 主实验跑完：固定30条样本（不再执行`max_samples=143`或top-100）
- [x] 输出主结果表：Faithfulness / Answer Relevancy / Context Precision + 吞吐 / 时延 / 峰值显存 / KV估计（基于30条）
- [x] 消融实验完成：AHEC w/o adaptive gate
- [x] 消融实验完成：AHEC w/o sink preserve
- [x] 消融实验完成：AHEC w/o compression
- [x] 消融实验完成：AHEC w/o eviction
- [x] 参数敏感性实验（看趋势即可）：调整 `pressure` 映射参数（如 `1500/6000`）观察语义与资源变化
- [x] 参数敏感性实验（看趋势即可）：调整 `eviction_rate` / `compression_rate` 取值区间（低/中/高三档）观察趋势
- [x] 参数敏感性实验（看趋势即可）：调整 `sink_keep`（0/1/2）观察质量-资源权衡
- [ ] 参数实验结果整理为图和表：至少包含 1 张趋势图 + 1 张汇总表（可直接入论文）
- [x] 失败样本误差分析完成

### 本次消融结果记录（2026-05-06）
- 结果文件：`results/semantic_eval_20260506T152534Z.json`
- 生成文件：`results/generation_eval_20260506T151537Z.json`
- 评测设置：`hotpotqa`，`context length >= 8000`，`max_samples=30`，judge `repeats=1`
- `AHEC`：Faithfulness `0.856667`，Answer Relevancy `0.845000`，Context Precision `0.838333`，Throughput `15.473872`，Latency `2.781081`，Peak VRAM `7.313614`，KV `380.326042`
- `ahec_wo_adaptive_gate`：Faithfulness `0.853333`，Answer Relevancy `0.815000`，Context Precision `0.846667`，Throughput `12.862967`，Latency `3.740421`，Peak VRAM `7.972074`，KV `611.703386`
- `ahec_wo_sink_preserve`：Faithfulness `0.816667`，Answer Relevancy `0.810000`，Context Precision `0.813333`，Throughput `16.170559`，Latency `2.768003`，Peak VRAM `7.288457`，KV `371.665365`
- `ahec_wo_compression`：Faithfulness `0.830000`，Answer Relevancy `0.795000`，Context Precision `0.825000`，Throughput `12.726843`，Latency `3.557100`，Peak VRAM `7.892798`，KV `583.787240`
- `ahec_wo_eviction`：Faithfulness `0.843333`，Answer Relevancy `0.836667`，Context Precision `0.830000`，Throughput `12.916009`，Latency `3.651410`，Peak VRAM `7.935748`，KV `598.932031`

### 本次参数实验结果记录（2026-05-07）
- 低档生成文件：`results/generation_eval_20260507T031153Z.json`
- 中档生成文件：`results/generation_eval_20260507T034000Z.json`
- 高档生成文件：`results/generation_eval_20260507T034514Z.json`
- 低档语义文件：`results/semantic_eval_20260507T040735Z.json`
- 中档语义文件：`results/semantic_eval_20260507T041010Z.json`
- 高档语义文件：`results/semantic_eval_20260507T041218Z.json`
- 统一设置：`hotpotqa`，`context length >= 8000`，`max_samples=30`，judge `repeats=1`
- `low`：Faithfulness `0.800000`，Answer Relevancy `0.786667`，Context Precision `0.803333`，Throughput `11.182593`，Latency `4.044168`，KV `633.492708`
- `mid`：Faithfulness `0.856667`，Answer Relevancy `0.845000`，Context Precision `0.835000`，Throughput `16.417557`，Latency `2.611728`，KV `380.326042`
- `high`：Faithfulness `0.865000`，Answer Relevancy `0.866667`，Context Precision `0.845000`，Throughput `20.732302`，Latency `1.741736`，KV `169.505729`

### 失败样本误差分析记录（2026-05-07）
- 分析依据：`results/semantic_eval_20260507T040735Z.json`（low）、`results/semantic_eval_20260507T041010Z.json`（mid）、`results/semantic_eval_20260507T041218Z.json`（high）
- 失败判定：`min(Faithfulness, Answer Relevancy, Context Precision) < 0.75`
- 失败数量：`low=6`，`mid=5`，`high=4`（30条样本）
- 主要错误类型：
- `证据缺失/未命中`：模型回复“上下文未提供相关信息”，但参考答案在原始长上下文存在（典型样本：`sample_id=88b095...`、`228572...`、`72a769...`、`139d2f...`）
- `实体混淆`：答案结构正确但实体替换错误（典型样本：`sample_id=933655...`，将 `Michael Swango` 误答为 `Wayne Williams`）
- `方向正确但细节错`：回答到相关人物/领域，但关键信息（地点、对象、出生年份）偏差（典型样本：`sample_id=105e7f...`、`bb5fb5...`）
- 结论：随着参数从 `low -> mid -> high`，失败样本数下降，说明在本批样本上更激进的压缩淘汰策略未导致失败增多，反而提高了有效证据命中率与答案相关性。

## D. 图表公式与格式
- [ ] 图题在图下，编号连续，图中文字小于正文字号
- [ ] 表题在表上，编号连续，排版规范
- [ ] 公式编号连续，且在正文中被引用
- [ ] 关键算法至少给出流程图或伪代码
- [ ] 无大段空白、无错页、无章节格式错位

## E. 参考文献与附录
- [ ] 参考文献不少于25篇
- [ ] 英文文献不少于10篇
- [ ] 近三年文献不少于5篇
- [ ] 文献按正文首次引用顺序编号
- [ ] 附录补齐：复现实验命令、环境版本、关键结果文件索引

## F. 收尾核对
- [x] 术语统一：全部使用 `AHEC`（不再出现 `ours_hybrid`）
- [ ] 论文中的所有数字与结果文件逐项核对一致
- [ ] 全文完成一次格式自检和一次内容自检
