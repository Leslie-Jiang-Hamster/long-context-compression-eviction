# Long-Context Compression + Eviction for Reasoning

## 目标
围绕毕设题目「面向长上下文推理的压缩与淘汰机制研究」，实现并评测一个融合方案：
- 压缩：参考 CompLLM
- 淘汰：参考 H2O / StreamingLLM
- 融合：自适应门控（按 token 重要性动态分配压缩率与淘汰率）

## 评测策略（语义为主）
- 主评估：语义指标（Faithfulness / Answer Relevancy / Context Precision）
- 次评估：LongBench 官方字符串指标（F1 / Rouge-L / Accuracy）
- 资源评估：KV cache 占用、峰值显存、吞吐（tokens/s）

这样可以同时满足：
1) 导师要求的语义导向；
2) 与已有工作可对比的可复现指标。

## 基准任务（首选）
- LongBench（先聚焦 Multi-Document QA 相关子任务）

## 目录
- `docs/experiment-plan.md`：实验路线与里程碑
- `docs/semantic-eval-protocol.md`：语义评测协议
- `docs/baselines.md`：基线与消融设计
- `configs/`：实验配置
- `scripts/`：本地/远程运行脚本
- `src/`：方法实现

## 里程碑
1. 跑通基线（Full KV / H2O / StreamingLLM / CompLLM）
2. 实现融合策略 V1（静态阈值）
3. 升级融合策略 V2（自适应门控）
4. 完成语义主结果 + 字符串次结果 + 资源曲线
