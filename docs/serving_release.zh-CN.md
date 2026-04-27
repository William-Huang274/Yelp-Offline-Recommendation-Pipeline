# Serving 与 Release 说明

这个仓库没有对外暴露真实在线服务，但已经有一套成型的 batch-style ranking / release 管理表面。

## 什么算 Release Artifact

| 表面 | 作用 |
| --- | --- |
| `data/output/current_release` | 当前 README 和 demo 使用的 compact outward-facing 结果面 |
| `data/metrics/current_release` | 当前对外主线的 compact 指标快照 |
| `config/serving.yaml` | mock serving 使用的策略、版本、fallback 顺序和 latency budget |
| `data/output/showcase_history` | 用于对照和讲故事的历史 run metadata |
| `data/metrics/showcase_history` | 用于受控比较的历史指标 |
| `data/output/_prod_runs` | release manifest、active pointer 和 rollback snapshot |

当前仓库刻意选择 compact release files，而不是直接把 full prediction dump 或完整在线服务资产都塞进仓库。

## `_prod_runs` 里每类文件的作用

当前仓库已经 check in 了这些典型文件：

| 文件模式 | 作用 |
| --- | --- |
| `release_manifest_*.json` | 被提升为发布线的 manifest，定义了各 stage run 的引用关系 |
| `release_policy.json` | 当前 policy block，描述 champion、aligned fallback、emergency baseline |
| `stage09_release.json` | emergency baseline 的 release pointer |
| `stage10_release.json` | 结构化精排的 aligned fallback pointer |
| `stage11_release.json` | 当前 champion rescue line 的 pointer |
| `rollback_snapshot_*.json` | publish 前对 active pointer 的快照 |
| `rollback_applied_*.json` | 已执行 rollback 的审计记录 |

## Version / Checkpoint / Config 怎么组织

当前仓库使用的是分层版本管理方式：

- 每个 stage run 都会写 `run_meta.json`
- 对外 summary 文件会回指到 stage 级 run metadata
- Stage11 的 train / eval run 保留各自的 adapter / checkpoint 目录和 run metadata
- launcher wrapper 和 runtime shell 把环境配置和主 Python 逻辑分开

所以这不是一个“大 notebook + 手工改 manifest”的结构，而是一套能指向具体 stage run、配置和指标快照的 release surface。

## Batch Inference 怎么走

当前主线有两种 batch-style 使用方式。

### Review-First 路径

直接消费 checked-in 的 compact artifact 和 demo helper：

```bash
python tools/release/run_release_checks.py --skip-pytest
python tools/serving/batch_infer_demo.py
python tools/serving/batch_infer_demo.py --strategy baseline
python tools/serving/batch_infer_demo.py --strategy xgboost
python tools/serving/batch_infer_demo.py --strategy reward_rerank
python tools/serving/mock_serving_api.py --self-test
python tools/serving/load_test_mock_serving.py --requests 20 --concurrency 4 --simulate-fallback-every 5
python tools/demo/demo_recommend.py
python tools/demo/demo_recommend.py show-case --case boundary_11_30
```

这是 README 和 demo 验证命令使用的轻量路径。

### Stage-Level Batch 路径

使用 launcher 或本地 wrapper：

- Stage09 本地 wrapper：
  [../tools/stage/run_stage09_local.ps1](../tools/stage/run_stage09_local.ps1)
- Stage10 本地 wrapper：
  [../tools/stage/run_stage10_bucket5_local.ps1](../tools/stage/run_stage10_bucket5_local.ps1)
- Stage11 云端 inventory / pull helper：
  [../tools/stage/cloud_stage11.py](../tools/stage/cloud_stage11.py)

### Mock Serving 路径

为了把这套离线系统翻译成更像上线服务的形态，仓库现在额外提供两个轻量入口：

- batch inference demo：
  [../tools/serving/batch_infer_demo.py](../tools/serving/batch_infer_demo.py)
- HTTP mock serving：
  [../tools/serving/mock_serving_api.py](../tools/serving/mock_serving_api.py)

示例：

```bash
python tools/serving/batch_infer_demo.py --format json
python tools/serving/mock_serving_api.py --host 127.0.0.1 --port 8000
```

当前 API surface 只保留最小必要 contract：

- `GET /health`：返回当前 release id 和服务状态
- `POST /rank`：输入一条 mock 用户画像、候选商户列表和可选策略，输出 Stage09 -> Stage10 -> Stage11 的排序结果

当前支持三种策略：

- `baseline`：使用候选基础分作为轻量 baseline
- `xgboost`：使用 Stage10 结构化精排逻辑
- `reward_rerank`：使用 Stage11 bounded reward-model rescue 逻辑

服务响应会显式返回：

- `strategy_requested`
- `strategy_used`
- `fallback_used`
- `fallback_reason`
- `serving_metrics.latency_ms`
- `serving_metrics.fallback_count`

如果要看内部 release runner 入口，见：

- [../scripts/pipeline/internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py)

## Fallback 逻辑

当前 release model 是分层的：

1. `stage11_release` 是 champion path
2. `stage10_release` 是 aligned fallback
3. `stage09_release` 是 emergency baseline

这不只是概念层，而是已经被 release policy 显式记录下来的角色关系。

实际含义是：

- rescue 层不可读或不稳定时，回退到 Stage10
- structured rerank 工件不可信时，再回退到 Stage09
- rollback 优先通过改 pointer 完成，而不是直接改代码

## Rollback 与 Monitoring

当前仓库已经有 internal-pilot 级别的 rollback / monitoring 表面。

主要命令：

```bash
python scripts/pipeline/internal_pilot_runner.py --mode monitor
python scripts/pipeline/internal_pilot_runner.py --mode publish --manifest-path data/output/_first_champion_freeze_20260313/first_champion_manifest.json
python scripts/pipeline/internal_pilot_runner.py --mode rollback
```

当前 monitor 会检查：

- Stage09 recall 质量
- candidate row 数和 evaluated users
- Stage11 / Stage10 / PreScore 的排序关系
- training artifact 可读性
- gate 状态一致性

详细规则在：

- [archive/release/rollback_and_monitoring.md](./archive/release/rollback_and_monitoring.md)

## 这说明了什么

从工程角度看，这意味着仓库已经不只是“离线打分脚本”。

它已经具备：

- 版本化 release pointer
- champion / fallback / baseline 三层角色
- rollback snapshot
- monitor signal
- 不依赖完整云端训练也能审阅的 compact release artifact

这就是它从研究型 ranking stack 向可审阅、可回退、可解释 batch release surface 迈出的工程化一步。
