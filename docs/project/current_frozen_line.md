# Current Frozen Line

[English](./current_frozen_line.md) | [中文](./current_frozen_line.zh-CN.md)

This page keeps the detailed frozen-line notes moved out of the root README.

## Outward-Facing Scope: `bucket5`

The main outward-facing `Stage11` line is currently centered on the
mid-to-high interaction set (`bucket5`).

| scope item | current `bucket5` line |
| --- | ---: |
| businesses | `1,798` |
| users | `9,765` |
| train interactions | `133,048` |
| validation users | `9,765` |
| test users | `9,765` |
| fixed `Stage10` eval users | `1,935` |
| `Stage11` rescue eval users | `517` |

## Data Scale

The public result surface is built on one shared source-parity scope after
filtering:

- `196,939` users
- `1,798` businesses

The three evaluation lines are defined by minimum interaction thresholds:

| user set | users | train interactions | validation users | test users |
| --- | ---: | ---: | ---: | ---: |
| cold-start-inclusive trainable set under leave-two-out (`bucket2`) | `26,686` | `178,636` | `26,686` | `26,686` |
| mid-to-high interaction set (`bucket5`) | `9,765` | `133,048` | `9,765` | `9,765` |
| high-interaction set (`bucket10`) | `3,618` | `93,789` | `3,618` | `3,618` |

Current mainline evaluation sizes:

- `Stage10` fixed eval users: `bucket2 = 5,344`, `bucket5 = 1,935`, `bucket10 = 738`
- `Stage11` rescue eval users on `bucket5`: `517`

## Current Baseline Results

### Recall Routing (`Stage09`)

`bucket5` candidate funnel improvement on the mid-to-high interaction set:

| metric | early gate | source-parity structural v5 |
| --- | ---: | ---: |
| `truth_in_pretrim150` | `0.7248` | `0.7451` |
| `hard_miss` | `0.1616` | `0.1190` |

### Structured Reranking (`Stage10`)

| user set | PreScore@10 recall / ndcg | LearnedBlendXGBCls@10 recall / ndcg |
| --- | --- | --- |
| cold-start-inclusive trainable set under leave-two-out (`bucket2`) | `0.1098 / 0.0513` | `0.1127 / 0.0522` |
| mid-to-high interaction set (`bucket5`) | `0.0935 / 0.0440` | `0.1261 / 0.0581` |
| high-interaction set (`bucket10`) | `0.0569 / 0.0265` | `0.0772 / 0.0341` |

### Rescue Reranking (`Stage11`)

`Stage11` uses one shared `Qwen3.5-9B` reward-model backbone with three expert
checkpoints specialized for the rerank windows `11-30`, `31-60`, and `61-100`.

Training-side evidence:

- `11-30` expert
  - `11-30 true win = 0.9560`
- `31-60` expert
  - `31-40 true win = 0.7385`
  - `41-60 true win = 0.7605`
- `61-100` expert
  - `61-100 true win = 0.8626`

Evaluation-side reference points on `bucket5`:

| line | recall@10 | ndcg@10 | note |
| --- | ---: | ---: | --- |
| `v120 two-band @ alpha=0.80` | `0.1973` | `0.0898` | current best-known offline result for `11-30 + 31-60` |
| `v124 tri-band @ alpha=0.36` | `0.1857` | `0.0838` | current frozen line for `11-30 + 31-60 + 61-100` |

Under the current tri-band policy, the deep expert stays conservative:

- `11-30`: primary top10 rescue path
- `31-60`: top10 / top20 rescue path
- `61-100`: rank-uplift path rather than an aggressive top-rank push

## What Matters Most In This Version

1. The upgrade touches recall routing, structured reranking, and rescue
   reranking together.
2. The LLM does not rerank the full candidate list. It only operates on the
   bounded candidate window produced by `Stage10`.
3. The Stage11 gains do not come only from score scaling. The `11-30`,
   `31-60`, and `61-100` experts all show stable training-side signals.
4. In the frozen tri-band line, most front-rank gains come from `11-30` and
   `31-60`, while `61-100` is intentionally conservative.
