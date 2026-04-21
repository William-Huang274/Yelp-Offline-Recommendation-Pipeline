# Proposal Template Content

## Purpose

This note converts the briefing's proposal requirements into a repository-aware
template that the team can fill quickly.

## 1. Project Title

Suggested title:

`Spark-Based Yelp Recommendation Ranking Stack With Route-Aware Recall,
Structured Re-Ranking, and Reward-Model Rescue`

## 2. Project Members

Fill manually:

- member name
- student ID if needed
- primary contribution area

## 3. Overview

Suggested baseline wording:

> This project addresses Yelp restaurant discovery as a big-data recommendation
> problem. The system processes large multi-source Yelp data, builds a Spark
> parquet-based pipeline, generates routed candidate sets, reranks them with a
> structured ranker, and applies bounded reward-model rescue reranking for
> difficult local ranking decisions.

## 4. General Architecture

Reference:

- [data_lineage_and_storage.md](./data_lineage_and_storage.md)

Suggested architecture summary:

1. ingest raw Yelp JSON and optional photo assets
2. store curated parquet layers
3. run Stage09 route-aware recall
4. run Stage10 structured rerank
5. run Stage11 bounded rescue rerank
6. publish compact result and metrics surfaces

## 5. Scope Of Work

Fill or adapt the following:

### Analytics Problem

- improve restaurant recommendation quality under sparse, noisy, and long-tail
  interaction data

### Ingestion

- use Spark-based JSON ingest into parquet
- use optional photo-summary aggregation for additional business-side evidence

### Storage

- use local parquet as the curated analytical layer
- use structured stage outputs and compact release surfaces for review

### Designed / Developed Components

- Stage09 route-aware candidate routing
- Stage10 structured rerank with route-derived features
- Stage11 reward-model rescue rerank with segmented experts
- validators, launchers, and compact release surface

### Why This Demonstrates Course Mastery

- `BEAD`: multi-source ingest, parquet storage, Spark processing
- `RCS`: recommender-system design and evaluation
- `PBDA`: project structuring, evaluation, bounded deployment thinking, and
  monitoring/rollback awareness

## 6. Effort Estimates

Fill manually with the actual team plan.

Suggested table:

| task | owner | estimated effort (man-days) | notes |
| --- | --- | ---: | --- |
| raw data ingest and parquet setup | `TBD` | `TBD` | |
| Stage09 mainline and recall audit | `TBD` | `TBD` | |
| Stage10 training and evaluation | `TBD` | `TBD` | |
| Stage11 dataset/train/eval | `TBD` | `TBD` | |
| validation, demo prep, and report writing | `TBD` | `TBD` | |

## 7. Attachments To Reuse

Use these repository assets when preparing the proposal:

- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)
- [data_lineage_and_storage.md](./data_lineage_and_storage.md)
- [evaluation_and_casebook.md](./evaluation_and_casebook.md)
