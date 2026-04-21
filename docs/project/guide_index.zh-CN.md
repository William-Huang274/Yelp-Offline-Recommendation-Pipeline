# 工程材料总索引

这份索引是面向老师审阅、组内分工和最终汇报准备的中文入口。它把当前项目已经补齐的工程文档、demo 入口和验收命令按使用场景整理在一起。

## 先看什么

如果只想快速判断仓库是否满足老师 briefing 要求，建议按这个顺序阅读：

1. [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)：老师要求与仓库证据的逐项映射。
2. [data_lineage_and_storage.md](./data_lineage_and_storage.md)：数据摄取、Parquet 存储、Stage09/10/11 输入输出。
3. [evaluation_and_casebook.md](./evaluation_and_casebook.md)：当前冻结指标、Stage11 案例和结果解释。
4. [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md)：工程挑战、资源约束、取舍和回退策略。
5. [acceptance_checklist.md](./acceptance_checklist.md)：交付前验收清单。

如果要准备 20 分钟 demo，优先看：

1. [demo_runbook.md](./demo_runbook.md)：现场演示顺序、命令和讲解口径。
2. [cloud_and_local_demo_runbook.zh-CN.md](./cloud_and_local_demo_runbook.zh-CN.md)：云端 Stage11 和本地 Stage01-10 调试方式。
3. [repo_navigation.md](./repo_navigation.md)：哪些目录是当前主线，哪些只是 archive。
4. [reproduce_mainline.md](./reproduce_mainline.md)：冻结版本复现路径和完整 pipeline 路径。

如果要写 proposal 或 final report，优先看：

1. [proposal_template_content.md](./proposal_template_content.md)：按老师要求整理的 proposal 内容模板。
2. [final_report_outline.md](./final_report_outline.md)：final report 推荐结构。
3. [environment_setup.md](./environment_setup.md)：环境安装、Spark/Java/Hadoop/GPU 前置要求。

## 核心故事线

当前仓库主线应该讲成：

- `Stage09`: route-aware recall funnel，负责把真实目标尽量保留在候选池。
- `Stage10`: structured rerank，负责用结构化特征做主排序。
- `Stage11`: Qwen3.5-9B segmented reward-model rescue rerank，负责在 bounded shortlist 内做分段救援式重排。

汇报时不要再把当前冻结版本讲成旧的 `ALS-only baseline`、`XGBoost regressor` 或 `PostgreSQL/FastAPI serving` 主线。这些可以作为历史探索或未来方向，但不是当前冻结结果的核心交付。

## 本地验收命令

在仓库根目录运行：

```powershell
python tools/run_release_checks.py
```

如果只想快速检查文档、release surface 和 demo CLI，可以运行：

```powershell
python tools/run_release_checks.py --skip-pytest
```

Windows PowerShell 也可以使用包装脚本：

```powershell
.\tools\run_release_checks.ps1
```

## Demo 命令

在仓库根目录运行：

```powershell
python tools/demo_recommend.py
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
python tools/demo_recommend.py show-case --case mid_31_40
python tools/demo_recommend.py walkthrough
python tools/cloud_stage11.py local-check
```

`python tools/demo_recommend.py` 不带参数时默认等同于 `summary`。这些命令不重新训练模型，只读取当前冻结结果和案例说明，适合老师现场演示和组内排练。

## 压缩包使用说明

如果你正在阅读打包后的 zip 内容：

- 先打开压缩包根目录的 `00_工程材料总索引.md`。
- 文档主体在 `docs/project/` 下。
- 轻量 demo 和验收脚本在 `tools/` 下。
- 压缩包主要用于审阅和汇报准备；深层代码、数据和完整训练产物仍以完整 GitHub 仓库为准。

## 仍需人工补齐

仓库已经补齐工程证据和可审阅表面，但以下课程交付仍需要组内人工完成：

- 最终 team names 和成员信息。
- 每个成员的 effort estimate / man-day 分工。
- 最终 PPT。
- individual reflection report。
- peer review。
