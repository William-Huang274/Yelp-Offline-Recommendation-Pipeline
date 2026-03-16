# Yelp 离线推荐流水线

[English](./README.md) | [简体中文](./README.zh-CN.md)

这是一个基于 Yelp 数据集、面向路易斯安那州餐饮发现场景的离线推荐与重排项目。仓库覆盖 `01 -> 11` 全链路：数据准备、重标注、business profiling、stage09 候选融合、stage10 结构化重排，以及 stage11 QLoRA / DPO sidecar 实验。

这个仓库的定位不是单独展示某一个模型，而是展示一个完整的推荐系统作品集项目：既有分桶离线评估、对齐后的模型对比，也有面向 LLM 重排的数据集构造、release 验证工具和文档化的模型 lineage。

## 项目亮点

- 不是单点模型 demo，而是完整的离线推荐流水线
- 不是把所有用户混在一起评估，而是按 `bucket_2`、`bucket_5`、`bucket_10` 做 user slicing
- 在同一 cohort 上对比 heuristic ranking、XGBoost rerank 和 QLoRA / DPO sidecar
- stage11 不是直接拿 prompt 做实验，而是有明确的数据工程链路：pointwise -> rich SFT -> DPO pairs
- 仓库包含 release manifest、validator、monitoring 检查和 rollback / readiness 文档

## 结果概览

当前公开冻结版本：`internal_pilot_v1_champion_20260313`

| Model | Bucket | Recall@10 | NDCG@10 | Role |
| --- | --- | ---: | ---: | --- |
| `PreScore@10` | `bucket10` | `0.056911` | `0.026467` | 应急基线 |
| `LearnedBlendXGBCls@10` | `bucket10` | `0.065041` | `0.029217` | 结构化 fallback |
| `QLoRASidecar@10` | `bucket10` | `0.067751` | `0.029935` | 当前 champion |

规模速览：

- Stage10 结构化重排训练：`47,271` 行，`2,251` 个 train users，`2,251` 个正样本
- Stage11 pointwise 数据集：`22,327` 行，`7,855` 个正样本，`3,618` 个总用户
- Stage11 rich SFT 数据集：`27,743` 行，`7,855` 个正样本
- Stage11 DPO 数据集：约 `5,836` 条 train pairs，`1,452` 条 eval pairs
- 最终对齐的 release cohort：`738` 个 eval users，`candidate_topn = 250`，`top_k = 10`

release cohort 审计证据：

- `truth_in_all = 0.947208`
- `truth_in_pretrim = 0.882255`
- `hard_miss = 0.052792`

## 本地运行与验证

这个仓库在本地有三种不同的运行 / 验证方式，含义并不一样。

### 1. 仓库级 Smoke Test

如果你的目标是确认一个 fresh clone 能正常工作，先走这条路径。

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python tools/check_release_readiness.py
python tools/check_release_monitoring.py
```

这条路径验证的是：

- 路径工具是否正常
- latest / prod run pointer 是否正常
- validator 行为是否正常
- release readiness 和 monitoring 工具链是否正常

这条路径不依赖原始 Yelp 数据。

### 2. 冻结产物验证

如果你本地已经有冻结的 stage 输出，可以走这条路径检查 artifact lineage。

```bash
python tools/validate_stage_artifact.py --kind stage09_candidate --run-dir data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion
python tools/validate_stage_artifact.py --kind stage11_dataset --run-dir data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset
```

这条路径验证的是：

- stage09 冻结候选产物
- stage11 冻结数据集产物
- 与当前公开 freeze 一致的 release lineage

### 3. 基于原始数据全量重跑

只有在你持有原始 Yelp 数据、并且希望重跑各阶段流水线时，才需要走这条路径。

需要先知道：

- 这个公开仓库不包含原始 Yelp source data
- stage09 到 stage11 的大体量运行产物没有完整地版本化进仓库
- 当前公开 release 状态主要由 `data/output/_prod_runs` 下的控制文件、已跟踪指标和文档表示
- 当前稳定的下游 business-profile contract 起点是 [docs/contracts/data_contract.md](./docs/contracts/data_contract.md) 里记录的 stage08 merged profile 输出

可选的 stage11 训练环境：

```bash
python -m pip install -r requirements-stage11-qlora.txt
```

internal pilot 辅助命令：

```bash
python scripts/pipeline/internal_pilot_runner.py --mode validate
python scripts/pipeline/internal_pilot_runner.py --mode monitor
```

Windows 包装脚本：

```bat
scripts/run_internal_pilot.bat --mode validate
```

## Bucket 策略

stage09 在 [scripts/09_candidate_fusion.py](./scripts/09_candidate_fusion.py) 中定义了 `MIN_TRAIN_REVIEWS_BUCKETS = [2, 5, 10]`，对应三个最小历史深度 bucket：

- `bucket_2`
- `bucket_5`
- `bucket_10`

对每个 bucket，stage09 还会额外应用 `min_user = min_train + 2` 作为 user-side cohort floor。

bucket 内部还会再划分 user segment：

- `light`：train interactions `<= 7`
- `mid`：train interactions `8-19`
- `heavy`：train interactions `>= 20`

这里的 segment 和 bucket 不是一回事。bucket 用于 cohort 切片，segment 用于分段 candidate budget 和 fusion 行为。

为什么当前公开的 stage11 主线先从 `bucket10` 开始：

- 这是第一个能提供更丰富、更杂偏好的 user slice
- 最难修复的 head-ordering 问题，尤其 heavy user 问题，会在这个 bucket 上暴露得更明显
- 这个 bucket 最适合验证 text-aware SFT / DPO sidecar 是否能在结构化 fallback 之上继续提升重排效果

## 训练数据构造

### Stage10 结构化重排

冻结 fallback 模型运行：`20260307_210530_stage10_1_rank_train`

bucket10 的训练量来自冻结模型元数据：

- `train_rows = 47271`
- `train_pos = 2251`
- `train_users = 2251`
- `valid_users = 369`
- `test_users = 701`

这是 sidecar champion 之前对齐使用的结构化 fallback。

### Stage11 Pointwise 数据集

冻结数据集运行：`20260311_011112_stage11_1_qlora_build_dataset`

`run_meta.json` 中的关键构造参数：

- `buckets_processed = [10]`
- `candidate_file = candidates_pretrim150.parquet`
- `topn_per_user = 120`
- `eval_user_frac = 0.2`
- `prompt_mode = full_lite`
- `include_valid_pos = true`
- `valid_pos_weight = 0.6`

bucket10 的基础 pointwise 数据量：

| Slice | Rows | Positives | Negatives |
| --- | ---: | ---: | ---: |
| train | `17796` | `6276` | `11520` |
| eval | `4531` | `1579` | `2952` |
| total | `22327` | `7855` | `14472` |

补充 cohort 信息：

- `users_total = 3618`
- `users_with_positive = 3386`
- `users_no_positive = 232`

### Rich SFT 数据集

当前 SFT 主线并不是直接使用基础 pointwise export，而是使用同一次 bucket10 数据集构造生成的 `rich_sft` export。

冻结 `rich_sft` 数据量：

| Slice | Rows | Positives | Negatives |
| --- | ---: | ---: | ---: |
| train | `22111` | `6276` | `15835` |
| eval | `5632` | `1579` | `4053` |
| total | `27743` | `7855` | `19888` |

`rich_sft` 的构造方式：

- 起点仍然是 bucket10 的冻结 `pretrim150` 候选池
- 与基础 stage11 dataset build 共享同一套 train / eval user 切分
- 使用 `full_lite` prompt，包含 user evidence、item evidence 和 history anchors
- 保留 true positives，并可选纳入权重为 `0.6` 的 valid positives
- 每个用户采样负例的形状是 `1 explicit + 2 hard + 2 near + 1 mid + 0 tail`
- hard negative 限制在 `pre_rank <= 20`
- mid negative 限制在 `pre_rank <= 60`
- negative rating 要求 `<= 2.5`
- 当前冻结 run 中 `rich_sft_allow_neg_fill = false`

### DPO Pair 数据集

当前 DPO 线不是从 raw base weights 直接做 pairwise，而是先从 SFT adapter warm-start，再继续做 DPO。

当前 DPO 的关键形状与构造逻辑：

- pairwise source mode: `rich_sft`
- pair policy: `v2a`
- top-k cutoff: `10`
- high-rank cutoff: `20`
- loser 优先来自同一 bucket10 候选池中的 `hard`、`near` 和 outranking confusers
- 审计后的规模约为 `5836` 条 train pairs 和 `1452` 条 eval pairs

## 仓库结构

- [scripts](./scripts)：各阶段脚本与 pipeline 工具
- [tests](./tests)：可移植测试与本地 artifact smoke checks
- [tools](./tools)：release 检查、monitoring 与验证工具
- [config](./config)：后续阶段使用的配置资产
- [docs](./docs)：分类后的文档索引
- [docs/contracts](./docs/contracts)：contract 与配置参考
- [docs/release](./docs/release)：freeze、readiness、monitoring、rollback 文档
- [docs/repo](./docs/repo)：repo hygiene、path、pointer、smoke-test、runner 文档
- [docs/stage09](./docs/stage09)：candidate fusion 与 bucket10 audit 文档
- [docs/stage10](./docs/stage10)：结构化重排训练文档
- [docs/stage11](./docs/stage11)：QLoRA、sidecar 与 cloud runbook 文档
- [docs/dpo](./docs/dpo)：DPO 指南与建议
- [docs/labeling](./docs/labeling)：标注手册

## 关键文档

- 文档总索引：[docs/README.md](./docs/README.md)
- 数据契约：[docs/contracts/data_contract.md](./docs/contracts/data_contract.md)
- release readiness：[docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md](./docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md)
- champion closeout：[docs/release/first_champion_closeout_20260313.md](./docs/release/first_champion_closeout_20260313.md)
- rollback 与 monitoring：[docs/release/rollback_and_monitoring.md](./docs/release/rollback_and_monitoring.md)
- stage11 cloud run profile：[docs/stage11/stage11_cloud_run_profile_20260309.md](./docs/stage11/stage11_cloud_run_profile_20260309.md)
- smoke-test 说明：[docs/repo/gl10_smoke_tests_20260313.md](./docs/repo/gl10_smoke_tests_20260313.md)

## 当前限制

- 这仍然是离线 batch pipeline，不是在线 serving 系统
- 当前 readiness report 仍然是 `WARN`，不是 `PASS`
- 冻结 stage11 champion 仍记录 `enforce_stage09_gate=false`
- release manifest 仍标记 `production_ready=false`
- `GL-01` 凭证轮换当时因为引用的云机器是临时环境而被推迟

## License

本仓库采用 [MIT License](./LICENSE)。
