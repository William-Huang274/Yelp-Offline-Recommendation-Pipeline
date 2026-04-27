# 工程材料索引

这份索引按使用场景整理当前项目的工程文档、复现入口、demo 命令和验证报告。

## 先看什么

查看当前冻结版本和关键技术证据：

1. [current_frozen_line.zh-CN.md](./current_frozen_line.zh-CN.md)：当前冻结线、数据规模和参考结果。
2. [data_lineage_and_storage.md](./data_lineage_and_storage.md)：数据摄取、Parquet 存储、Stage09/10/11 输入输出。
3. [evaluation_and_casebook.md](./evaluation_and_casebook.md)：当前冻结指标、Stage11 案例和结果解释。
4. [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md)：工程挑战、资源约束、取舍和回退策略。
5. [acceptance_checklist.md](./acceptance_checklist.md)：验证清单。

运行 demo：

1. [demo_serving_entry.zh-CN.md](./demo_serving_entry.zh-CN.md)：demo 范围、mock serving 命令和工程压测报告入口。
2. [demo_reproducibility_matrix.zh-CN.md](./demo_reproducibility_matrix.zh-CN.md)：最小用例、接口矩阵和本地验证记录。
3. [demo_runbook.md](./demo_runbook.md)：demo 运行顺序和命令。
4. [cloud_and_local_demo_runbook.zh-CN.md](./cloud_and_local_demo_runbook.zh-CN.md)：云端 Stage11 和本地 Stage01-10 调试方式。
5. [repo_navigation.md](./repo_navigation.md)：哪些目录是当前主线，哪些只是 archive。
6. [reproduce_mainline.md](./reproduce_mainline.md)：冻结版本复现路径和完整 pipeline 路径。

环境与复现：

1. [environment_setup.md](./environment_setup.md)：环境安装、Spark/Java/Hadoop/GPU 前置要求。
2. [reproduce_mainline.md](./reproduce_mainline.md)：主线复现路径。
3. [repository_map.zh-CN.md](./repository_map.zh-CN.md)：主入口、结果面和仓库地图。

## 主线架构

当前仓库主线：

- `Stage09`: route-aware recall funnel，负责把真实目标尽量保留在候选池。
- `Stage10`: structured rerank，负责用结构化特征做主排序。
- `Stage11`: Qwen3.5-9B segmented reward-model rescue rerank，负责在 bounded shortlist 内做分段救援式重排。

旧的 `ALS-only baseline`、`XGBoost regressor`、`PostgreSQL/FastAPI serving` 不属于当前冻结主线，只作为历史探索或后续方向保留。

## 本地验证命令

在仓库根目录运行：

```powershell
python tools/release/run_release_checks.py
```

如果只想快速检查文档、release surface 和 demo CLI，可以运行：

```powershell
python tools/release/run_release_checks.py --skip-pytest
```

Windows PowerShell 也可以使用包装脚本：

```powershell
.\tools\release\run_release_checks.ps1
```

## Demo 命令

在仓库根目录运行：

```powershell
python tools/demo/demo_recommend.py
python tools/demo/demo_recommend.py summary
python tools/demo/demo_recommend.py list-cases
python tools/demo/demo_recommend.py show-case --case boundary_11_30
python tools/demo/demo_recommend.py show-case --case mid_31_40
python tools/demo/demo_recommend.py walkthrough
python tools/stage/cloud_stage11.py local-check
```

`python tools/demo/demo_recommend.py` 不带参数时默认等同于 `summary`。这些命令不重新训练模型，只读取当前冻结结果和案例说明。

## 压缩包使用说明

阅读打包后的 zip 内容时：

- 先打开压缩包根目录的 `00_工程材料总索引.md`。
- 文档主体在 `docs/project/` 下。
- 轻量 demo 和验收脚本在 `tools/` 下。
- 深层代码、数据和完整训练产物以完整 GitHub 仓库为准。
