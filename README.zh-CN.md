# Yelp 离线排序系统

[English](./README.md) | [中文](./README.zh-CN.md)

这是一个面向 Yelp 餐饮推荐场景的离线推荐与重排项目。当前版本围绕三层主线展开：

- 召回路由（`Stage09`）
- 结构化精排（`Stage10`）
- 基于奖励模型的救援式重排（`Stage11`）

## 当前基线总览

| 主线 | 当前冻结展示口径 | 当前结果 |
| --- | --- | --- |
| 召回路由（`Stage09`） | `bucket5` 候选漏斗 | `truth_in_pretrim150 = 0.7451`，`hard_miss = 0.1190` |
| 结构化精排（`Stage10`） | 三条用户集评估线 | `bucket2 = 0.1127 / 0.0522`，`bucket5 = 0.1261 / 0.0581`，`bucket10 = 0.0772 / 0.0341` |
| 救援式重排（`Stage11`） | 双模型研究峰值 | `v120 @ alpha=0.80`，`Recall@10 = 0.1973`，`NDCG@10 = 0.0898` |
| 救援式重排（`Stage11`） | 三模型冻结基线 | `v124 @ alpha=0.36`，`Recall@10 = 0.1857`，`NDCG@10 = 0.0838` |

下面再按三层分别展开。

## 当前基线结果

### 召回路由（`Stage09`）

中高交互用户集（`bucket5`，最小交互门槛 7）上的候选漏斗结果：

| 指标 | 早期版本 | 当前主线 |
| --- | ---: | ---: |
| `truth_in_pretrim150` | `0.7248` | `0.7451` |
| `hard_miss` | `0.1616` | `0.1190` |

### 结构化精排（`Stage10`）

按最小交互门槛定义的三组用户集评估线上的 `PreScore@10 -> LearnedBlendXGBCls@10`：

| 用户集 | 基线 recall / ndcg | 精排后 recall / ndcg |
| --- | --- | --- |
| 含冷启动、在留二法切分下仍可训练的用户集（`bucket2`） | `0.1098 / 0.0513` | `0.1127 / 0.0522` |
| 中高交互用户集（`bucket5`） | `0.0935 / 0.0440` | `0.1261 / 0.0581` |
| 高交互用户集（`bucket10`） | `0.0569 / 0.0265` | `0.0772 / 0.0341` |

### 救援式重排（`Stage11`）

训练侧结果：

- `11-30` 专家模型
  - 当前冻结的边界救援专家，负责前排边界救援
- `31-60` 专家模型
  - `31-40 true win = 0.7385`
  - `41-60 true win = 0.7605`
- `61-100` 专家模型
  - `61-100 true win = 0.8626`

评估侧结果（`bucket5`）：

| 结果线 | Recall@10 | NDCG@10 | 说明 |
| --- | ---: | ---: | --- |
| 双模型研究峰值 `v120 @ alpha=0.80` | `0.1973` | `0.0898` | `11-30 + 31-60` 当前最优离线档 |
| 三模型冻结基线 `v124 @ alpha=0.36` | `0.1857` | `0.0838` | `11-30 + 31-60 + 61-100` 当前冻结口径 |

当前三模型口径里，深段模型（`61-100`）先按保守方式使用，主要体现为更深位置候选的名次提升，而不是激进地抢占前排位置。

## 这版最值得看的点

1. 这次升级不是只换了大模型，而是把召回、精排和救援式重排三层一起重做了一遍，属于整条排序链路的升级。
2. 大模型不做全量候选重排，只在结构化精排给出的候选窗口上工作，成本、时延和回退风险都更可控。
3. `Stage11` 的增益不是单纯靠“把 alpha 调大”硬推出来的，`11-30`、`31-60`、`61-100` 三个专家在训练侧都已经学到了稳定信号。
4. 三模型冻结版本里，前排收益主要由 `11-30` 和 `31-60` 承担，`61-100` 先按保守策略做深段名次提升，避免打乱主链稳定性。

## 为什么先做 `bucket10`，再扩到 `bucket5`

这条实验路线是刻意这样安排的，不是事后挑结果。

- `bucket10` 是高交互用户集，用户语义最丰富，历史行为更完整，更适合先验证模型结构本身是否成立。
- 在这一组上先做实验，能更快判断：
  - 召回路由有没有把候选池搭对
  - 结构化精排能不能稳定建立全局排序骨架
  - 奖励模型能不能在局部竞争里识别被低估的真值
- `bucket10` 证明方向成立之后，再把同一套方法扩到 `bucket5`，目的是做更大覆盖面的验证，确认这套方法不是只对高交互用户有效。

所以当前仓库里：

- `bucket10` 更像早期结构验证集
- `bucket5` 更像当前主展示口径，因为覆盖面更大，也更接近实际主线使用场景
- `bucket2` 用来验证 `Stage09 -> Stage10` 在含冷启动用户集上是否仍有可迁移性

## 案例分析入口

首页这里只放三个缩略入口，详细过程放到单独文档：

- Prompt 构造样本：展示 Stage11 训练时，用户侧证据、当前候选、局部竞争者和排名窗口是怎么拼成一个可学习的局部判断题的。
- `11-30` 用户案例：展示一个边界真值为什么能在不泄露真值位置的前提下被救回前十。
- `31-60` 用户案例：展示一个中段真值为什么适合用奖励模型做局部救援，而不是直接重排全榜。

详细案例入口：

- [docs/stage11/stage11_case_notes_20260409.zh-CN.md](./docs/stage11/stage11_case_notes_20260409.zh-CN.md)

## 这次版本升级了什么

相对上一版仓库冻结版本，这次升级不是单独调整 `Stage11`，而是三层一起升级。

1. 召回路由（`Stage09`）从通用候选融合升级成按路由组织的候选层，候选保真率更高，硬缺失更低。
2. 结构化精排（`Stage10`）接入了更多来自召回层的特征，形成了可迁移到不同交互门槛用户组的精排主线。
3. 救援式重排（`Stage11`）从通用 `SFT / DPO` 辅助重排，切换成更贴近排序任务的奖励模型方案。

## 三层系统结构

### 召回路由（`Stage09`）

这一层负责把不同来源的候选组织成统一候选池，并控制召回预算、候选通道和挑战者通道。

### 结构化精排（`Stage10`）

这一层是全局排序骨架。它消费匹配特征、文本特征、相对交叉特征和分组差距特征，输出主排序结果。

### 救援式重排（`Stage11`）

这一层不接管全排序，而是在 `Stage10` 输出的受控候选窗口上做局部救援。当前方案包括：

- 分段专家模型
- 候选短名单重排
- 有边界的门槛与保护规则

## 为什么不用全量大模型重排

这个项目没有让大模型或奖励模型对全量候选做重排。

当前方案只对 `Stage10` 输出窗口做重排，原因是：

- 成本更低
- 行为更可控
- 出问题时更容易回退
- 对前排结果的扰动更小

因此，`Stage10` 仍然是全局排序骨架，`Stage11` 是受控的增益层，而不是替代层。

## 数据泄露控制

当前 `Stage11` 主线按“不在推理时使用真值信息”设计：

- 专家路由依据的是候选当前所在的排名窗口，而不是隐藏的真值分段
- 候选短名单重排只使用当前分数和当前名次
- 真值只参与训练监督和离线评估

## 用户集定义

系统当前使用三条按最小交互门槛定义的用户集评估线：

- 含冷启动、在留二法切分下仍可训练的用户集（`bucket2`，最小交互门槛 4）
- 中高交互用户集（`bucket5`，最小交互门槛 7）
- 高交互用户集（`bucket10`，最小交互门槛 12）

这三条用户集不是互斥分层，而是三种不同门槛下的评估口径。这样做的目的，是验证整条排序栈在不同数据密度下是否都成立，而不是只在单一切片上有效。

## 对外公开表面

- [scripts/stage01_to_stage08](./scripts/stage01_to_stage08)：项目早期流水线，保留作主线历史补充
- [scripts/launchers](./scripts/launchers)：当前统一启动入口
- [docs/contracts/launcher_env_conventions.zh-CN.md](./docs/contracts/launcher_env_conventions.zh-CN.md)：启动脚本变量口径
- [docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md](./docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md)：Stage11 分段专家设计说明
- [docs/stage11/stage11_case_notes_20260409.zh-CN.md](./docs/stage11/stage11_case_notes_20260409.zh-CN.md)：Prompt 样本与真实救援案例
- [data/output/current_release](./data/output/current_release)：当前冻结结果面
- [data/output/showcase_history](./data/output/showcase_history)：展示用历史结果面
- [data/metrics/current_release](./data/metrics/current_release)：当前冻结指标快照
- [data/metrics/showcase_history](./data/metrics/showcase_history)：展示用历史指标快照

## 推荐入口

当前推荐直接使用 `scripts/launchers` 下的统一入口，而不是根目录下的兼容脚本：

- Stage09: [scripts/launchers/stage09_bucket5_mainline.sh](./scripts/launchers/stage09_bucket5_mainline.sh)
- Stage10: [scripts/launchers/stage10_bucket5_mainline.sh](./scripts/launchers/stage10_bucket5_mainline.sh)
- Stage11:
  - [scripts/launchers/stage11_bucket5_11_1.sh](./scripts/launchers/stage11_bucket5_11_1.sh)
  - [scripts/launchers/stage11_bucket5_export_only.sh](./scripts/launchers/stage11_bucket5_export_only.sh)
  - [scripts/launchers/stage11_bucket5_train.sh](./scripts/launchers/stage11_bucket5_train.sh)
  - [scripts/launchers/stage11_bucket5_eval.sh](./scripts/launchers/stage11_bucket5_eval.sh)

变量定义统一写在：

- [docs/contracts/launcher_env_conventions.zh-CN.md](./docs/contracts/launcher_env_conventions.zh-CN.md)

## 对外技术说明

- 启动脚本变量口径：
  [docs/contracts/launcher_env_conventions.zh-CN.md](./docs/contracts/launcher_env_conventions.zh-CN.md)
- Stage11 分段专家设计说明：
  [docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md](./docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md)
- Stage11 Prompt 与案例说明：
  [docs/stage11/stage11_case_notes_20260409.zh-CN.md](./docs/stage11/stage11_case_notes_20260409.zh-CN.md)

## 仓库边界

当前仓库版本化的内容包括：

- 代码
- 小型指标文件
- manifest
- 对外技术说明文档

以下内容不做版本化：

- 原始 Yelp 数据
- 大模型权重
- 大体量云端日志
- 全量预测结果

当前对外展示用的小结果文件统一放在：

- [data/output/current_release](./data/output/current_release)
- [data/output/showcase_history](./data/output/showcase_history)
- [data/metrics/current_release](./data/metrics/current_release)
- [data/metrics/showcase_history](./data/metrics/showcase_history)

原始冻结 pack 和内部收口文档继续保留在本地，用于审计与追溯，不纳入当前对外公开表面。
