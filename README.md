# Yelp Offline Ranking Stack

[English](./README.md) | [中文](./README.zh-CN.md)

This repository contains an offline recommendation and reranking project for
Yelp restaurant discovery. The current version is organized around three main
layers:

- recall routing (`Stage09`)
- structured reranking (`Stage10`)
- reward-model rescue reranking (`Stage11`)

`Stage11` is built on one shared `Qwen3.5-9B` reward-model backbone and split
into three expert checkpoints by rerank window:

- `11-30` boundary-rescue expert
- `31-60` mid-rank rescue expert
- `61-100` deep-rank uplift expert

## Current Baseline Overview

| line | current frozen display scope | current result |
| --- | --- | --- |
| recall routing (`Stage09`) | `bucket5` candidate funnel | `truth_in_pretrim150 = 0.7451`, `hard_miss = 0.1190` |
| structured reranking (`Stage10`) | three evaluation lines | `bucket2 = 0.1127 / 0.0522`, `bucket5 = 0.1261 / 0.0581`, `bucket10 = 0.0772 / 0.0341` |
| rescue reranking (`Stage11`) | two-band research peak | `v120 @ alpha=0.80`, `Recall@10 = 0.1973`, `NDCG@10 = 0.0898` |
| rescue reranking (`Stage11`) | tri-band freeze baseline | `v124 @ alpha=0.36`, `Recall@10 = 0.1857`, `NDCG@10 = 0.0838` |

The sections below expand those four lines in more detail.

## Current Outward-Facing Scope: `bucket5`

The main outward-facing `Stage11` line is currently centered on the
mid-to-high interaction set (`bucket5`). This is the public scope behind the
root README metrics.

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

`Stage11` is not three unrelated models. It is one shared `Qwen3.5-9B`
reward-model backbone with three expert checkpoints specialized for the rerank
windows `11-30`, `31-60`, and `61-100`.

Training-side evidence:

- `11-30` expert
  - `11-30 true win = 0.9560`
  - current frozen boundary-rescue expert for front-rank rescue
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

1. This is not only a Stage11 refresh. The upgrade touches recall routing,
   structured reranking, and rescue reranking together.
2. The LLM does not rerank the full candidate list. It only operates on the
   bounded candidate window produced by `Stage10`, which keeps cost, latency,
   and rollback risk under control.
3. The Stage11 gains do not come only from score scaling. The `11-30`,
   `31-60`, and `61-100` experts all show stable training-side signals.
4. In the frozen tri-band line, most front-rank gains come from `11-30` and
   `31-60`, while `61-100` is intentionally kept conservative and used mainly
   for rank uplift.

## Why Start With `bucket10` And Then Expand To `bucket5`

This sequencing was deliberate rather than chosen after the fact.

- `bucket10` is the high-interaction set. It contains the richest user
  semantics and the cleanest behavioral evidence, so it is the best place to
  validate whether the model design itself works.
- We used `bucket10` first to test:
  - whether recall routing built the right candidate pool
  - whether the structured reranker could establish a stable global backbone
  - whether the reward model could rescue underweighted truth items under local
    competition
- After that direction was validated, we expanded to `bucket5` to test the
  same design on a broader and more practical coverage range.

In the current repository line:

- `bucket10` is best understood as the early architecture-validation set
- `bucket5` is the main outward-facing line because it has wider coverage and
  sits closer to the intended mainline use case
- `bucket2` is used to validate whether the `Stage09 -> Stage10` path remains
  portable on a cold-start-inclusive trainable set

## Case Notes

The root README keeps only a short summary. Detailed Stage11 case notes cover:

- a real prompt-construction sample
- a concrete `11-30` rescue example
- a concrete `31-60` rescue example
- why the current `61-100` policy stays conservative

- [docs/stage11/stage11_case_notes_20260409.md](./docs/stage11/stage11_case_notes_20260409.md)

## What Changed In This Release

Compared with the previous frozen repository version, this release upgrades all
three layers together.

1. `Stage09` moves from generic candidate fusion to route-aware candidate
   routing, with better candidate retention and lower hard misses.
2. `Stage10` adds more route-derived and competition-aware features and becomes
   a stronger rerank mainline across multiple interaction thresholds.
3. `Stage11` moves from generic `SFT / DPO` sidecar reranking to a
   ranking-oriented reward-model design.

## Three-Layer System Structure

### Recall Routing (`Stage09`)

This layer organizes candidates from multiple sources into one candidate pool
and controls recall budgets, candidate lanes, and challenger lanes.

### Structured Reranking (`Stage10`)

This layer is the global ranking backbone. It consumes match, text,
relative-cross, and group-gap features and produces the main ranked list.

### Rescue Reranking (`Stage11`)

This layer does not replace the full ranking stack. It operates only on the
bounded candidate window produced by `Stage10`. The current design combines:

- segmented expert checkpoints
- shortlist reranking
- bounded gate and protection rules

## Why Not Use Full-List LLM Reranking

The project does not let the LLM or reward model rerank the full candidate
list.

It only reranks the `Stage10` output window, for four reasons:

- lower cost
- more controllable behavior
- easier rollback
- smaller front-rank disruption

## Leakage Control

The Stage11 line is built to avoid inference-time label leakage.

- expert routing uses the candidate's current rank window, not the hidden truth band
- shortlist reranking uses current scores and current ranks only
- supervision uses true labels in training and evaluation, but those labels are not exposed at inference time

## User-Set Definitions

The repository uses three evaluation lines defined by minimum interaction
thresholds:

- cold-start-inclusive trainable set under leave-two-out (`bucket2`, minimum interaction floor 4)
- mid-to-high interaction set (`bucket5`, minimum interaction floor 7)
- high-interaction set (`bucket10`, minimum interaction floor 12)

These are not mutually exclusive tiers. They are three evaluation lines with
different data-density thresholds, used to check whether the ranking stack is
portable beyond one curated slice.

## Public Repository Surface

- [scripts/stage01_to_stage08](./scripts/stage01_to_stage08): earlier local pipeline stages kept as reproducible project history
- [scripts/launchers](./scripts/launchers): outward-facing launcher surface
- [docs/contracts/launcher_env_conventions.md](./docs/contracts/launcher_env_conventions.md): launcher variable meanings
- [docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.md](./docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.md): Stage11 segmented-expert design note
- [docs/stage11/stage11_case_notes_20260409.md](./docs/stage11/stage11_case_notes_20260409.md): prompt sample and concrete rescue cases
- [data/output/current_release](./data/output/current_release): current release result surface
- [data/output/showcase_history](./data/output/showcase_history): selected historical reference results
- [data/metrics/current_release](./data/metrics/current_release): current release metric snapshot
- [data/metrics/showcase_history](./data/metrics/showcase_history): selected historical metric references

## Recommended Entry Points

Use the launcher wrappers rather than the long legacy root launchers:

- Stage09: [scripts/launchers/stage09_bucket5_mainline.sh](./scripts/launchers/stage09_bucket5_mainline.sh)
- Stage10: [scripts/launchers/stage10_bucket5_mainline.sh](./scripts/launchers/stage10_bucket5_mainline.sh)
- Stage11 dataset/export/train/eval:
  - [scripts/launchers/stage11_bucket5_11_1.sh](./scripts/launchers/stage11_bucket5_11_1.sh)
  - [scripts/launchers/stage11_bucket5_export_only.sh](./scripts/launchers/stage11_bucket5_export_only.sh)
  - [scripts/launchers/stage11_bucket5_train.sh](./scripts/launchers/stage11_bucket5_train.sh)
  - [scripts/launchers/stage11_bucket5_eval.sh](./scripts/launchers/stage11_bucket5_eval.sh)

Launcher variable definitions are documented here:

- [docs/contracts/launcher_env_conventions.md](./docs/contracts/launcher_env_conventions.md)

## Public Technical Notes

- launcher variable conventions:
  [docs/contracts/launcher_env_conventions.md](./docs/contracts/launcher_env_conventions.md)
- Stage11 segmented-expert design note:
  [docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.md](./docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.md)
- Stage11 case notes:
  [docs/stage11/stage11_case_notes_20260409.md](./docs/stage11/stage11_case_notes_20260409.md)

## Internal Version Comparison Note

The repository also keeps one internal comparison note for release review. It
explains:

- what exactly was frozen in the previous version
- why the current version improves more across the upgraded stack
- why `Stage11` moved from generic `SFT / DPO` to reward-model rescue rerank

Local path:

- `docs/release/version_comparison_previous_vs_current_20260410.md`

This note is kept as a local release-comparison artifact and is not part of the
current public technical surface.

## Repository Boundary

This repository tracks code, small metrics, manifests, and public technical notes.

It does not version:

- raw Yelp source data
- large cloud logs
- large model weights
- full prediction dumps

The outward-facing small result files used by the current closeout are kept under:

- [data/output/current_release](./data/output/current_release)
- [data/output/showcase_history](./data/output/showcase_history)
- [data/metrics/current_release](./data/metrics/current_release)
- [data/metrics/showcase_history](./data/metrics/showcase_history)

The original frozen provenance pack and internal closeout notes are kept locally
for auditing, but are intentionally excluded from the public repository surface.
