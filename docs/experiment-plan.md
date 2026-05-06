# 实验计划

## 阶段 1：基线复现
- Base: Full KV（不压缩、不淘汰）
- Eviction-only: H2O, StreamingLLM
- Compression-only: CompLLM 风格

## 阶段 2：融合方法（Ours）
- Step A: 压缩候选集（压缩低重要性 token）
- Step B: 淘汰策略（保留 sink + 高频重要 token）
- Step C: 自适应门控
  - 输入特征：注意力分数、最近激活次数、层间稳定性
  - 输出：压缩率 `r_c`、淘汰率 `r_e`

## 阶段 3：评测设置
- 数据：LongBench（优先多文档问答）
- 长度分桶：8k / 16k / 32k / 64k（按模型能力裁剪）
- 主指标：Faithfulness / Answer Relevancy / Context Precision
- 次指标：F1 / Rouge-L / Accuracy
- 资源指标：KV 内存、峰值显存、吞吐

## 阶段 4：消融
- 只压缩 vs 只淘汰 vs 融合
- 去掉 sink 保留机制
- 固定阈值 vs 自适应阈值

## 产出图表
- 语义质量-内存 Pareto 曲线
- 字符串质量-内存对照曲线
- 不同上下文长度下的稳定性曲线
- 各层保留率热力图
