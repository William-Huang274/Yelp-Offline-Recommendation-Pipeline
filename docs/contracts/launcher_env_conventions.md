# Launcher Variable Conventions

[English](./launcher_env_conventions.md) | [中文](./launcher_env_conventions.zh-CN.md)

## Purpose

This note standardizes the meanings of `TOP_K`, `RERANK_TOPN`, and
`PAIRWISE_POOL_TOPN` across the current launcher surface.

These names appear similar but describe different parts of the pipeline.

## Canonical Definitions

| concept | stage / env | meaning | typical use |
| --- | --- | --- | --- |
| `TOP_K` | Stage10/Stage11, `RANK_EVAL_TOP_K` | final metric cutoff reported to users | Recall@10, NDCG@10 |
| `RERANK_TOPN` | Stage10, `RANK_RERANK_TOPN` | head candidate window rescored by the learned Stage10 model | structured rerank before final ranking |
| `RERANK_TOPN` | Stage11, `QLORA_RERANK_TOPN` | candidate window rescored by the Stage11 sidecar / RM path | rescue rerank window |
| `PAIRWISE_POOL_TOPN` | Stage11_1, `QLORA_PAIRWISE_POOL_TOPN` | learned-rank cutoff used only when exporting pairwise-pool candidates | pair mining for Stage11 training |
| `TOPN_PER_USER` | Stage11_1, `QLORA_TOPN_PER_USER` | candidate rows exported per user before Stage11 dataset construction | Stage11 dataset build surface |

## Separation Rules

1. `TOP_K` is a reporting cutoff, not a scoring window.
2. `RERANK_TOPN` is a scoring window, not a training-pool limit.
3. `PAIRWISE_POOL_TOPN` is a training-data mining limit, not an online or offline rerank target.
4. `TOPN_PER_USER` is the exported candidate surface for Stage11 dataset build; it is usually larger than `PAIRWISE_POOL_TOPN`.

## Shared Path Anchors

Current active compatibility launchers resolve remote paths through
`scripts/launchers/_path_contract.sh`.

The main anchors are:

- `BDA_REMOTE_PROJECT_ROOT`
- `BDA_REMOTE_PROJECT_OUTPUT_ROOT`
- `BDA_REMOTE_PROJECT_SPARK_TMP_ROOT`
- `BDA_REMOTE_FAST_ROOT`
- `BDA_REMOTE_HF_ROOT`
- `BDA_REMOTE_PYTHON_BIN`

These anchors keep launcher defaults consistent while avoiding repeated
machine-specific absolute paths inside every `run_stage*.sh`.

## Practical Examples

### Stage10

- `RANK_EVAL_TOP_K=10`
- `RANK_RERANK_TOPN=150`

Meaning:

- report Recall@10 / NDCG@10
- rescore the top 150 candidates with the learned Stage10 model

### Stage11_1

- `QLORA_TOPN_PER_USER=250`
- `QLORA_PAIRWISE_POOL_TOPN=100`

Meaning:

- export up to 250 candidates per user into the Stage11 dataset surface
- when building pairwise-pool data, only mine positives / negatives from the top 100 learned-rank region

### Stage11_3

- `RANK_EVAL_TOP_K=10`
- `QLORA_RERANK_TOPN=100`

Meaning:

- report final top10 metrics
- allow Stage11 to rescore the current top100 candidate window

## Implementation Notes

Current active scripts use these mappings:

- Stage10 scoring:
  - [10_2_rank_infer_eval.py](../../scripts/10_2_rank_infer_eval.py)
- Stage11 dataset build:
  - [11_1_qlora_build_dataset.py](../../scripts/11_1_qlora_build_dataset.py)
- Stage11 sidecar eval:
  - [11_3_qlora_sidecar_eval.py](../../scripts/11_3_qlora_sidecar_eval.py)

The launcher surface that exposes these variables is:

- [scripts/launchers/README.md](../../scripts/launchers/README.md)

