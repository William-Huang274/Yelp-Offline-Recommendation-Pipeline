# 项目工程文档包

[English](./README.md) | [中文](./README.zh-CN.md)

这里收纳的是面向老师审阅、项目演示和工程复现的文档包。它的目标不是替代根目录
README，而是把当前仓库已有的技术证据整理成一套更适合 practice module
proposal、demo 和 final report 使用的工程表面。

## 建议使用顺序

如果是第一次打开这套材料，建议先看
[guide_index.zh-CN.md](./guide_index.zh-CN.md)，它是中文总索引和使用指引。

只需要审阅当前冻结版本时：

1. 先看 [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)
2. 再看 [data_lineage_and_storage.md](./data_lineage_and_storage.md)
3. 然后运行 `python tools/run_release_checks.py`
4. 最后用 `python tools/demo_recommend.py summary` 和案例命令做演示

需要复现主线时，直接看 [reproduce_mainline.md](./reproduce_mainline.md)。

## 文档入口

- [guide_index.zh-CN.md](./guide_index.zh-CN.md)：中文总索引和使用指引
- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)：
  老师要求和仓库证据的映射
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
- [demo_runbook.md](./demo_runbook.md)：20 分钟 demo 运行脚本
- [cloud_and_local_demo_runbook.zh-CN.md](./cloud_and_local_demo_runbook.zh-CN.md)：
  云端 Stage11 与本地 Stage01-10 调试说明
- [acceptance_checklist.md](./acceptance_checklist.md)：审阅验收清单
- [repo_navigation.md](./repo_navigation.md)：当前主线与 archive 的边界
- [proposal_template_content.md](./proposal_template_content.md)：proposal 内容模板
- [final_report_outline.md](./final_report_outline.md)：final report 结构模板

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
