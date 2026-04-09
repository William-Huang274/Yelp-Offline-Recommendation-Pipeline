# Yelp Offline Ranking Stack

[English](./README.md) | [中文](./README.zh-CN.md)

Route-aware offline ranking stack for Yelp restaurant recommendation.

The current repository version is built around a unified `stage09 -> stage10 -> stage11` line:

- `stage01 -> stage08`: earlier local pipeline foundation kept for repository completeness
- `stage09`: route-aware recall design and candidate funnel control
- `stage10`: structured rerank on top of route-derived features
- `stage11`: reward-model-based rescue rerank with segmented experts and bounded shortlist reranking

## Current Release Position

The current line is no longer the earlier `bucket10`-only generic sidecar proof of concept.

It is now positioned as:

- a shared `stage09` recall contract
- a stronger `stage10` rerank baseline across `bucket2`, `bucket5`, and `bucket10`
- a `bucket5 stage11` rescue stack built on top of the upgraded `stage09 -> stage10` path

Two Stage11 reference points are kept intentionally:

- `two-band research peak`: best offline `11-30 + 31-60` result after alpha grid
- `tri-band freeze baseline`: conservative `11-30 + 31-60 + 61-100` freeze candidate

## Current Metrics Snapshot

### Stage09

`bucket5` candidate funnel improvement on the mid-to-high interaction set:

| metric | early gate | source-parity structural v5 |
| --- | ---: | ---: |
| `truth_in_pretrim150` | `0.7248` | `0.7451` |
| `hard_miss` | `0.1616` | `0.1190` |

### Stage10

| user set | PreScore@10 recall / ndcg | LearnedBlendXGBCls@10 recall / ndcg |
| --- | --- | --- |
| cold-start-inclusive trainable set under leave-two-out (`bucket2`) | `0.1098 / 0.0513` | `0.1127 / 0.0522` |
| mid-to-high interaction set (`bucket5`) | `0.0935 / 0.0440` | `0.1261 / 0.0581` |
| high-interaction set (`bucket10`) | `0.0569 / 0.0265` | `0.0772 / 0.0341` |

### Stage11

Training-side evidence:

- `11-30 only expert`
  - frozen boundary-rescue expert used for front-rank rescue
- `31-60 only expert`
  - `31-40 true win = 0.7385`
  - `41-60 true win = 0.7605`
- `61-100 only expert`
  - `61-100 true win = 0.8626`

Evaluation-side reference points on `bucket5`:

| line | recall@10 | ndcg@10 | note |
| --- | ---: | ---: | --- |
| `v120 two-band @ alpha=0.80` | `0.1973` | `0.0898` | current two-band best-known offline result |
| `v124 tri-band @ alpha=0.36` | `0.1857` | `0.0838` | current tri-band freeze baseline |

Current tri-band policy keeps the deep expert conservative:

- `11-30`: primary top10 rescue path
- `31-60`: top10 / top20 rescue path
- `61-100`: rank uplift path, not a forced top10 push

## Why Start With `bucket10` And Then Expand To `bucket5`

This sequencing was deliberate.

- `bucket10` is the high-interaction set. It contains the richest user semantics
  and the cleanest behavioral evidence, so it is the best place to validate
  whether the model design itself works.
- We used `bucket10` first to test:
  - whether the recall routing produced the right candidate pool
  - whether the structured reranker could establish a stable global backbone
  - whether the reward model could rescue underweighted truth items in local
    competition
- After that direction was validated, we expanded to `bucket5` to test the same
  design on a broader and more practical coverage range.

In the current repository line:

- `bucket10` is best understood as the early architecture-validation set
- `bucket5` is the main outward-facing line because it has wider coverage and is
  closer to the intended mainline use case
- `bucket2` is used to validate whether the `stage09 -> stage10` path remains
  portable on a cold-start-inclusive trainable set

## Case Notes

The root README keeps only a short summary. Detailed Stage11 case notes cover:

- a real prompt-construction sample
- a concrete `11-30` rescue example
- a concrete `31-60` rescue example
- why the current `61-100` policy stays conservative

- [docs/stage11/stage11_case_notes_20260409.md](./docs/stage11/stage11_case_notes_20260409.md)

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

## Entry Points

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
