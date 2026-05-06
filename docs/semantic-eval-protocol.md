# Semantic Evaluation Protocol

## 主评估指标
- Faithfulness: 生成答案是否被给定上下文支持
- Answer Relevancy: 回答是否切题
- Context Precision: 保留上下文是否真正有助于回答

## 推荐工具
- RAGAS（优先）或 ARES

## 评估设置
- Judge 模型固定
- temperature = 0
- 同一样本重复评测 3 次取平均
- 至少 10% 样本做人工复核

## 报告格式
每个方法汇报：
- Faithfulness
- Answer Relevancy
- Context Precision
- F1 / Rouge-L / Accuracy（次指标）
- KV Memory Reduction(%)
- Peak VRAM(GB)
- Throughput(tokens/s)

## 判定原则
- 主结论以语义指标和资源指标为准
- 字符串指标仅用于横向可比和稳定性补充
