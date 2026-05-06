# Long-Context Compression + Eviction for Reasoning

## 目标
围绕毕设题目「面向长上下文推理的压缩与淘汰机制研究」，实现并评测一个融合方案：
- 压缩：参考 CompLLM
- 淘汰：参考 H2O / StreamingLLM
- 融合：自适应门控（按 token 重要性动态分配压缩率与淘汰率）

## 基准任务（首选）
- LongBench（先聚焦 Multi-Document QA 相关子任务）

## 核心指标
- 任务效果：EM / F1 / Rouge（按子任务）
- 资源指标：KV cache 占用、峰值显存、吞吐（tokens/s）
- 折中指标：质量下降 vs 内存节省率

## 目录
- `docs/experiment-plan.md`：实验路线与里程碑
- `docs/baselines.md`：基线与消融设计
- `configs/`：实验配置
- `scripts/`：运行脚本
- `src/`：方法实现

## 里程碑
1. 复现实验基线（Full KV / H2O / StreamingLLM / CompLLM）
2. 实现融合策略 V1（静态阈值）
3. 升级融合策略 V2（自适应门控）
4. 完成主结果 + 消融 + 可视化
