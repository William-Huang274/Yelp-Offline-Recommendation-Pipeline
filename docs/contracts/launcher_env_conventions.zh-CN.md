# Launcher 变量口径

[English](./launcher_env_conventions.md) | [中文](./launcher_env_conventions.zh-CN.md)

## 目的

这份文档统一当前 launcher 表面里 `TOP_K`、`RERANK_TOPN`、`PAIRWISE_POOL_TOPN` 的含义。

这几个名字都和候选窗口有关，但指向的不是同一个环节。

## 统一定义

| 概念 | 阶段 / 环境变量 | 含义 | 典型用途 |
| --- | --- | --- | --- |
| `TOP_K` | Stage10/Stage11，`RANK_EVAL_TOP_K` | 最终对外汇报的指标截断位 | Recall@10、NDCG@10 |
| `RERANK_TOPN` | Stage10，`RANK_RERANK_TOPN` | Stage10 学习模型真正重打分的头部候选窗口 | 结构化精排窗口 |
| `RERANK_TOPN` | Stage11，`QLORA_RERANK_TOPN` | Stage11 sidecar / RM 真正重打分的候选窗口 | 救援重排窗口 |
| `PAIRWISE_POOL_TOPN` | Stage11_1，`QLORA_PAIRWISE_POOL_TOPN` | 只用于导出 pairwise pool 时的 learned-rank 截断位 | Stage11 训练挖样 |
| `TOPN_PER_USER` | Stage11_1，`QLORA_TOPN_PER_USER` | Stage11 数据构造前，每个用户先导出的候选规模 | Stage11 数据面 |

## 区分规则

1. `TOP_K` 是汇报口径，不是模型打分窗口。
2. `RERANK_TOPN` 是模型打分窗口，不是训练样本池大小。
3. `PAIRWISE_POOL_TOPN` 是训练数据挖样窗口，不是线上或离线最终重排目标位。
4. `TOPN_PER_USER` 是 Stage11 数据构造的导出范围，通常会大于 `PAIRWISE_POOL_TOPN`。

## 共享路径锚点

当前 active 兼容 launcher 通过 `scripts/launchers/_path_contract.sh`
统一解析远端路径默认值。

主要锚点包括：

- `BDA_REMOTE_PROJECT_ROOT`
- `BDA_REMOTE_PROJECT_OUTPUT_ROOT`
- `BDA_REMOTE_PROJECT_SPARK_TMP_ROOT`
- `BDA_REMOTE_FAST_ROOT`
- `BDA_REMOTE_HF_ROOT`
- `BDA_REMOTE_PYTHON_BIN`

这样做的目的，是在保持默认展开路径不变的前提下，避免每个
`run_stage*.sh` 都重复写死某台机器的绝对路径。

## 例子

### Stage10

- `RANK_EVAL_TOP_K=10`
- `RANK_RERANK_TOPN=150`

含义：

- 最终汇报 top10 指标
- 用 Stage10 学习模型重打分前 150 个候选

### Stage11_1

- `QLORA_TOPN_PER_USER=250`
- `QLORA_PAIRWISE_POOL_TOPN=100`

含义：

- 每个用户先导出最多 250 个候选进入 Stage11 数据面
- 构造 pairwise pool 时，只在 learned-rank 前 100 的区域里挖正负样本

### Stage11_3

- `RANK_EVAL_TOP_K=10`
- `QLORA_RERANK_TOPN=100`

含义：

- 最终汇报 top10 指标
- 允许 Stage11 对当前 top100 候选窗口做重打分

## 实现位置

当前主链对应实现：

- Stage10 打分：
  - [10_2_rank_infer_eval.py](../../scripts/10_2_rank_infer_eval.py)
- Stage11 数据构造：
  - [11_1_qlora_build_dataset.py](../../scripts/11_1_qlora_build_dataset.py)
- Stage11 sidecar 评估：
  - [11_3_qlora_sidecar_eval.py](../../scripts/11_3_qlora_sidecar_eval.py)

launcher 暴露面见：

- [scripts/launchers/README.zh-CN.md](../../scripts/launchers/README.zh-CN.md)

