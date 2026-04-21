# 评估说明

这份文档是给面试官和招聘方看的离线评估页，重点不是把所有 CSV 摊开，而是把实验设计、分桶逻辑和当前边界讲清楚。

## 当前评估回答的问题

| 问题 | 当前证据 |
| --- | --- |
| 召回有没有把真值保住？ | Stage09 route-aware recall summary |
| 结构化精排能不能跨不同用户密度分桶带来提升？ | Stage10 `bucket2 / bucket5 / bucket10` 指标 |
| bounded RM rescue 有没有在不变成 full-list rerank 的前提下带来价值？ | Stage11 参考线和分段专家训练信号 |
| 这套方法有没有可解释的取舍？ | 分桶定义、coverage 指标、案例分析、gated vs freeze 对照 |

## Baseline vs Current

| 模块 | baseline | 当前线 | takeaway |
| --- | --- | --- | --- |
| `Stage09` 候选漏斗 | early gate: `truth_in_pretrim150 = 0.7248`，`hard_miss = 0.1616` | source-parity structural v5: `0.7451`，`0.1190` | 真值保留更高，硬缺失更低 |
| `Stage10` 结构化精排 | `PreScore@10` | `LearnedBlendXGBCls@10` | 在 `bucket2 / bucket5 / bucket10` 上都有正向提升 |
| `Stage11` 救援重排 | 分段单带 / 双带历史线 | `v124` tri-band freeze，`v120` 保留为 best-known reference | 救援增益成立，但 freeze 选择同时考虑覆盖和稳定性 |

## 各 Stage 的增益贡献

### Stage09

- `truth_in_pretrim150`: `0.7248 -> 0.7451`（`+0.0203`）
- `hard_miss`: `0.1616 -> 0.1190`（`-0.0426`）

解释：

- 当前 route-aware 召回给下游交付了更强的候选池
- 但当前线仍然存在非小的 hard miss，所以 recall 依然是主边界之一

### Stage10

| bucket | PreScore recall / ndcg | Learned recall / ndcg | delta |
| --- | --- | --- | --- |
| `bucket2` | `0.1098 / 0.0513` | `0.1127 / 0.0522` | `+0.0028 / +0.0009` |
| `bucket5` | `0.0935 / 0.0440` | `0.1261 / 0.0581` | `+0.0326 / +0.0141` |
| `bucket10` | `0.0569 / 0.0265` | `0.0772 / 0.0341` | `+0.0203 / +0.0076` |

解释：

- 最大的离线提升出现在当前主展示口径 `bucket5`
- `bucket2` 也有正向提升，但幅度更小，这和含冷启动用户集的稀疏、噪声和混合意图特征一致

### Stage11

当前公开与 showcase 结果可以组织成一条受控对照线：

| 结果线 | Recall@10 | NDCG@10 | cohort 说明 |
| --- | ---: | ---: | --- |
| `v117 segmented` | `0.1663` | `0.0739` | 早期 segmented rescue 参考线 |
| `v120 two-band best-known` | `0.1973` | `0.0898` | 当前 517 用户 rescue cohort 上的 best-known line |
| `v121 joint12 gate` | `0.2259` | `0.1049` | 分数更高，但只保留了 `394` 个 surviving users，丢掉 `123` 个 |
| `v124 tri-band freeze` | `0.1857` | `0.0838` | 当前冻结对外口径 |

解释：

- `v121` 说明更强的 gate 可以把 surviving subset 推得更高
- 但 `v124` 更适合当前 freeze，因为它更容易解释，也更稳

## 分桶评估与 Cold-Start Portability

当前仓库使用三条用户密度口径：

- `bucket2`：含冷启动、在 leave-two-out 下仍可训练的用户集
- `bucket5`：中高交互用户集，也是当前主展示口径
- `bucket10`：高交互用户集，早期主要用于结构验证

这些 bucket 说明：

- 这套方法不是只在一个高密度切片上成立
- Stage10 主线可以迁移到更冷的 `bucket2`，但收益更弱
- `bucket5` 是当前质量、覆盖和可解释性的平衡点

当前 checked-in release 表只汇总公开 `bucket2` 总口径。更新后的
Stage09 / Stage10 脚本已经支持通过显式 cohort CSV 重放更细的冷启动子集，
例如 `0-3` 或 `4-6` 交互用户；对应入口是
`CANDIDATE_FUSION_USER_COHORT_PATH` 和 `RANK_EVAL_USER_COHORT_PATH`。
这些 finer slice 目前只是脚本支持，还没有冻结进 `current_release`
摘要表。

## Coverage / Tail / Novelty 信号

当前 Stage10 指标文件里已经有比 Recall/NDCG 更工业化的字段。

| bucket | 模型 | user coverage | item coverage | tail coverage | novelty |
| --- | --- | ---: | ---: | ---: | ---: |
| `bucket2` | `PreScore@10` | `1.0000` | `0.6229` | `0.0526` | `8.1657` |
| `bucket2` | `LearnedBlendXGBCls@10` | `1.0000` | `0.6073` | `0.0484` | `8.1058` |
| `bucket5` | `PreScore@10` | `1.0000` | `0.3782` | `0.0481` | `8.3756` |
| `bucket5` | `LearnedBlendXGBCls@10` | `1.0000` | `0.3971` | `0.0506` | `8.4324` |
| `bucket10` | `PreScore@10` | `1.0000` | `0.3138` | `0.0604` | `8.7737` |
| `bucket10` | `LearnedBlendXGBCls@10` | `1.0000` | `0.3266` | `0.0709` | `8.8256` |

解释：

- 当前 checked-in Stage10 线在 user coverage 上保持完整
- `bucket5` 和 `bucket10` 不只是排序质量提高，item / tail coverage 和 novelty 也在改善
- `bucket2` 则提醒我们：冷启动可迁移性存在，但不是无成本的

## Ablation 与受控对照

### Stage10

当前 Stage10 文件已经支持一个紧凑的 ablation 故事：

- `ALS@10_from_candidates`：传统 baseline
- `PreScore@10`：候选顺序 baseline
- `LearnedBlendXGBCls@10`：当前结构化精排主线

这能证明收益不只是来自 recall 提升，learned reranker 本身也贡献了增益。

### Stage11

当前公开和 showcase 文件支持这样一条受控 rescue 叙事：

- `v117`：segmented rescue 可以工作
- `v120`：two-band joint rescue 达到当前 best-known offline 结果
- `v121`：更强 gate 可以推高 surviving subset
- `v124`：freeze 选择牺牲部分峰值，换取更干净的对外口径

## 已知失败案例与当前边界

- 当前 `bucket5` 线里，`Stage09 hard_miss = 0.1190`，说明仍有一部分用户根本没有把真值交给下游。
- `bucket2` 的增益更小，说明在更稀疏、更噪声的用户集上，可迁移性成立但幅度有限。
- `v121` 的 headline 指标更高，但它是通过过滤到 `394` 个 surviving users、丢掉 `123` 个用户得到的，因此必须结合 coverage 一起解释。
- `61-100` 专家训练信号是好的，但当前 frozen tri-band line 里仍是保守使用，没有形成强 top-rank rescue。

## 主要指标文件入口

- [../data/metrics/current_release/stage09/stage09_bucket5_route_aware_recall_snapshot.csv](../data/metrics/current_release/stage09/stage09_bucket5_route_aware_recall_snapshot.csv)
- [../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv](../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv)
- [../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv](../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v117_segmented_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v117_segmented_metrics.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v120_joint12_default_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v120_joint12_default_metrics.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v121_joint12_gate_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v121_joint12_gate_metrics.csv)
