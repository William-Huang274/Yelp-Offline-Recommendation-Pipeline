# 仓库地图

[English](./repository_map.md) | [中文](./repository_map.zh-CN.md)

这份文档是当前公开仓库表面的深链接地图，方便第一次进入项目的人快速找到主入口。

## 主要代码路径

| 路径 | 作用 |
| --- | --- |
| `scripts/launchers` | 面向当前主线的 stage 级 launcher 入口 |
| `tools` | 校验、demo、云端 inventory 和 mock serving 工具 |
| `tests` | 公开链接、release 文件、launcher 和指标的 smoke tests |
| `docs/contracts` | launcher 变量和环境约定 |
| `docs/stage11` | Stage11 对外设计说明和案例分析 |
| `docs/project` | 复现说明、冻结线说明、设计取舍和工程审阅文档包 |

## Checked-In 结果面

| 路径 | 作用 |
| --- | --- |
| `data/output/current_release` | 当前对外展示的 release 输出 |
| `data/metrics/current_release` | 当前主线的 compact 指标快照 |
| `data/output/showcase_history` | 用于受控对照的历史输出 |
| `data/metrics/showcase_history` | 用于评估故事线的历史指标 |
| `data/output/_prod_runs` | release manifest、active pointer 和 rollback snapshot |

## 推荐入口

- Stage09 launcher：
  [../../scripts/launchers/stage09_bucket5_mainline.sh](../../scripts/launchers/stage09_bucket5_mainline.sh)
- Stage10 launcher：
  [../../scripts/launchers/stage10_bucket5_mainline.sh](../../scripts/launchers/stage10_bucket5_mainline.sh)
- Stage11 数据集 / 导出 / 训练 / 评估：
  [../../scripts/launchers/stage11_bucket5_11_1.sh](../../scripts/launchers/stage11_bucket5_11_1.sh)，
  [../../scripts/launchers/stage11_bucket5_export_only.sh](../../scripts/launchers/stage11_bucket5_export_only.sh)，
  [../../scripts/launchers/stage11_bucket5_train.sh](../../scripts/launchers/stage11_bucket5_train.sh)，
  [../../scripts/launchers/stage11_bucket5_eval.sh](../../scripts/launchers/stage11_bucket5_eval.sh)
- 公开校验入口：
  [../../tools/release/run_release_checks.py](../../tools/release/run_release_checks.py)
- Demo 工具：
  [../../tools/demo/demo_recommend.py](../../tools/demo/demo_recommend.py)，
  [../../tools/serving/batch_infer_demo.py](../../tools/serving/batch_infer_demo.py)，
  [../../tools/serving/mock_serving_api.py](../../tools/serving/mock_serving_api.py)

## 历史材料说明

- `scripts/stage01_to_stage08` 保留作项目早期可复现实验历史。
- `docs/` 和 `tools/` 下的 archive 目录适合审计和追溯，但不是当前主入口。
- 大体量原始数据、云端日志和模型权重不纳入当前公开仓库表面。
