# Stage11 31-60 专家与分段融合说明

[English](./stage11_31_60_only_and_segmented_fusion_20260408.md) | [中文](./stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md)

## 1. 目标

这份文档记录当前 Stage11 重设计的核心内容：

- 独立的 `31-60 only` expert
- 分段 `11-3` 融合
- second-stage shortlist 重排
- 有边界的 policy gate

## 2. 设计约束

这轮重设计保持以下内容不变：

- Stage09 候选契约
- Stage11_1 pool build 契约
- `11-30` 冻结 expert 路径
- label、split、candidate boundary、metric 含义

## 3. 为什么 31-60 需要独立专家

问题不在于 `31-60` 学不会，而在于：

- `11-2` 的局部救援判断可以学到
- 但 `11-3` 中释放到最终 top10 的能力仍然偏弱
- `11-30 + 31-60` 混训会改变全榜校准，反过来伤到边界段

因此当前方案拆开了：

- `11-30` 边界救援
- `31-40 / 41-60` 中段救援

## 4. 31-60 训练设计

独立 `31-60` 路径采用：

- 只训练 `rescue_31_60`
- 每个正样本生成多组 typed slate
- `PAIR_LOCAL_LISTWISE_MAX_RIVALS = 4`
- 优先增加每用户 slate 覆盖，而不是加大单个 slate 宽度

必需的 slate 类型：

1. `31-60 vs same-band`
2. `31-60 vs 11-30`
3. `31-60 vs head-anchor`

对子段的处理：

- `31-40`：更靠近边界，更多 boundary blocker
- `41-60`：保留更多 same-band blocker，目标先释放到 `top20 / top30`

## 5. 当前冻结线中的 61-100 角色

当前冻结线也接入了独立 `61-100` expert`，但策略保持保守。

当前口径是：

- deep expert 要训练出来
- 先让它改善深段名次
- 当前冻结线不把它做成激进的 top30 晋升策略

这样做的目的是在不打乱前沿排序的前提下，先把 deep 路径稳定下来。

## 6. 11-3 分段融合

当前分段路由：

- `11-30 -> v101`
- `31-40 -> 31-60 expert`
- `41-60 -> 31-60 expert`
- `61-100 -> 61-100 expert`

当前 second-stage 设计：

- 在 first-stage blend 之后做 shortlist rerank
- 做 route-local normalization
- 用有边界的 gate 和 cap-rank policy

当前策略口径：

- `31-40`：top10 救援通道
- `41-60`：top20 救援通道
- `61-100`：保守名次提升通道

## 7. 当前参考结果

训练侧：

- `31-60 only` 最优快照：
  - `31-40 true win = 0.7385`
  - `41-60 true win = 0.7605`
- `61-100 only` 最优快照：
  - `61-100 true win = 0.8626`

评估侧：

- `v120 @ alpha=0.80`
  - 当前双模型最优离线结果
- `v124 @ alpha=0.36`
  - 当前三模型冻结基线

## 8. 这份文档的用途

这份文档用于说明：

- 为什么 Stage11 不再以通用 SFT/DPO 作为主训练路径
- 为什么要引入分段专家
- 为什么当前 tri-band 路线对 deep rescue 保持保守

这份文档不用于宣称当前 tri-band policy 已经是最终生产冠军。
