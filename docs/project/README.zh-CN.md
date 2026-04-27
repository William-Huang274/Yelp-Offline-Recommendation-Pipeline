# 项目工程文档

[English](./README.md) | [中文](./README.zh-CN.md)

这里汇总当前推荐系统工程的设计说明、数据链路、复现路径、验证报告和运行入口。根目录 README 保留项目总览；本目录提供更细的技术文档。

## 阅读顺序

第一次阅读时可先看 [guide_index.zh-CN.md](./guide_index.zh-CN.md)，它按架构、数据、评估、复现和 demo 入口组织文档。

查看当前冻结版本：

1. [current_frozen_line.zh-CN.md](./current_frozen_line.zh-CN.md)
2. [data_lineage_and_storage.md](./data_lineage_and_storage.md)
3. [evaluation_and_casebook.md](./evaluation_and_casebook.md)
4. `python tools/run_release_checks.py`

复现主线时参考 [reproduce_mainline.md](./reproduce_mainline.md)。

## 文档入口

- [guide_index.zh-CN.md](./guide_index.zh-CN.md)：中文总索引和使用指引
- [environment_setup.md](./environment_setup.md)：环境安装与运行前置
- [data_lineage_and_storage.md](./data_lineage_and_storage.md)：数据来源、
  Parquet 落地、各阶段产物与冻结结果面
- [reproduce_mainline.md](./reproduce_mainline.md)：当前主线复现路径
- [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md)：挑战、取舍、
  风险控制与当前边界
- [evaluation_and_casebook.md](./evaluation_and_casebook.md)：冻结指标和案例
- [current_frozen_line.zh-CN.md](./current_frozen_line.zh-CN.md)：详细冻结线、
  数据规模和参考结果线
- [design_choices.zh-CN.md](./design_choices.zh-CN.md)：bounded rerank 方案、
  分桶路线和泄露控制
- [repository_map.zh-CN.md](./repository_map.zh-CN.md)：主入口、结果面和仓库地图
- [demo_runbook.md](./demo_runbook.md)：demo 运行脚本
- [demo_serving_entry.zh-CN.md](./demo_serving_entry.zh-CN.md)：
  demo 范围、mock serving 命令和工程压测报告入口
- [demo_reproducibility_matrix.zh-CN.md](./demo_reproducibility_matrix.zh-CN.md)：
  demo 最小用例、接口矩阵和本地验证记录
- [cloud_and_local_demo_runbook.zh-CN.md](./cloud_and_local_demo_runbook.zh-CN.md)：
  云端 Stage11 与本地 Stage01-10 调试说明
- [acceptance_checklist.md](./acceptance_checklist.md)：验证清单
- [repo_navigation.md](./repo_navigation.md)：当前主线与 archive 的边界

## 常用命令

```bash
python tools/run_release_checks.py
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
```

PowerShell 也可以直接运行：

```powershell
.\tools\run_release_checks.ps1
```
