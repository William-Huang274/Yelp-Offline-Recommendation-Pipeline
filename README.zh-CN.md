# Yelp 离线推荐流水线

[English](./README.md) | [简体中文](./README.zh-CN.md)

这是一个基于 Yelp 数据集的离线推荐与重排项目，聚焦路易斯安那州餐饮发现场景。仓库覆盖从数据准备、商家重标注、候选召回，到结构化重排以及 LLM sidecar 重排实验的完整链路。

这个仓库更适合作为作品集和研究交付项目展示，而不是可直接上线的在线服务。

## 项目亮点

- 覆盖 `01 -> 11` 阶段的端到端离线推荐流程
- 具有审计和冻结发布能力的稳定召回底座
- 以 XGBoost 为 fallback 的结构化重排
- 以 QLoRA / DPO sidecar rerank 为当前 champion 路径的大模型实验
- 包含发布校验、回退、监控和数据契约文档

## 当前冻结结果

- 发布标签：`internal_pilot_v1_champion_20260313`
- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- `eval_users = 738`
- `candidate_topn = 250`
- `top_k = 10`

同一批评测下的主要结果：

| Model | Recall@10 | NDCG@10 | Role |
| --- | ---: | ---: | --- |
| `PreScore@10` | `0.056911` | `0.026467` | emergency baseline |
| `LearnedBlendXGBCls@10` | `0.065041` | `0.029217` | aligned fallback |
| `QLoRASidecar@10` | `0.067751` | `0.029935` | current champion |

## 仓库结构

- [scripts](./scripts)：各阶段脚本与流水线工具
- [docs](./docs)：冻结说明、审计记录、运行手册与修复笔记
- [tests](./tests)：路径、指针与校验器的 smoke tests
- [tools](./tools)：发布校验和监控辅助工具
- [config](./config)：后续阶段使用的配置资产

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python tools/check_release_readiness.py
python tools/check_release_monitoring.py
```

Stage11 GPU 环境可以额外安装：

```bash
python -m pip install -r requirements-stage11-qlora.txt
```

## 公开仓库说明

- 仓库不包含原始 Yelp 数据
- stage09-11 的大体量运行产物不会完整纳入版本控制
- 当前公开版本的 release 状态主要由 `docs/`、`data/metrics/` 和 `data/output/_prod_runs/` 表示

## 当前局限

- 这是离线批处理流水线，不是在线 serving 系统
- 当前 readiness report 仍然是 `WARN`，不是 `PASS`
- 冻结的 stage11 champion 仍记录 `enforce_stage09_gate=false`
- 发布 manifest 仍标记 `production_ready=false`

## 定位

这个项目更准确的定位是：

`offline recommendation and rerank research pipeline with release discipline`

而不是：

`fully productionized recommendation platform`
