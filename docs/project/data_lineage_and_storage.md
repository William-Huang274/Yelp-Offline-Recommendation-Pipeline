# Data Lineage And Storage

## Purpose

This note explains how the project moves from raw Yelp source data to the
current frozen release surface.

## 1. Raw Sources

The project uses multiple Yelp-derived sources rather than a single flat table.

| source | original format | typical business value |
| --- | --- | --- |
| business metadata | JSON | merchant attributes, categories, location |
| review logs | JSON | user-item interaction evidence and text signals |
| tip logs | JSON | lightweight preference and intent evidence |
| check-in logs | JSON | temporal and popularity context |
| user metadata | JSON | user-level profiling support |
| photo dump | zipped tar containing `photos.json` | merchant visual/scene evidence |

## 2. Ingestion Into Curated Parquet

The repository keeps the raw-to-curated conversion explicit.

### Core Yelp JSON Ingest

Canonical early ingest script:

- [../../scripts/stage01_to_stage08/01_data prep.py](../../scripts/stage01_to_stage08/01_data%20prep.py)

This script:

- reads the Yelp JSON files with Spark
- writes Snappy parquet outputs under `data/parquet`
- keeps partition counts moderate for local-machine review

The curated outputs follow the pattern:

- `data/parquet/yelp_academic_dataset_business`
- `data/parquet/yelp_academic_dataset_checkin`
- `data/parquet/yelp_academic_dataset_tip`
- `data/parquet/yelp_academic_dataset_user`
- `data/parquet/yelp_academic_dataset_review`

### Optional Photo Ingest

The photo asset path is handled separately:

- [../../scripts/archive/release_closeout_20260409/stage09_nonmainline/09_yelp_photo_summary_build.py](../../scripts/archive/release_closeout_20260409/stage09_nonmainline/09_yelp_photo_summary_build.py)

This script:

- streams `photos.json` from the zip/tar asset
- aggregates business-level photo summaries
- writes a parquet summary layer

## 3. Derived Processing Layers

After raw data is converted into curated parquet, the pipeline builds several
stage-level outputs.

| layer | typical output root | purpose |
| --- | --- | --- |
| early foundation and history | `scripts/stage01_to_stage08` outputs | historical preparation and early modeling foundation |
| Stage09 recall routing | `data/output/09_candidate_fusion/...` | candidate generation, funnel control, truth retention |
| Stage10 structured rerank | `data/output/...stage10...` | learned reranking on top of routed candidates |
| Stage11 dataset / train / eval | `data/output/...stage11...` | bounded RM rescue reranking workflow |

Current Stage09 directly consumes multiple curated or derived business-side
assets, including:

- review evidence
- tip signals
- check-in context
- item/business context surfaces

See [../../scripts/09_candidate_fusion.py](../../scripts/09_candidate_fusion.py)
for the current path anchors.

## 4. Frozen Repository Review Surface

The repository does not keep all large run directories under version control.
Instead, it keeps a compact review surface.

### Current Visible Output Surface

- [../../data/output/current_release](../../data/output/current_release)

This is the current outward-facing result surface for:

- Stage09 route-aware recall summary files
- Stage10 mainline rerank summary files
- Stage11 expert summaries and freeze-baseline evaluation files

### Current Visible Metrics Surface

- [../../data/metrics/current_release](../../data/metrics/current_release)

This holds compact metric snapshots that can be cited in docs, demos, and
reports without shipping full experiment dumps.

### Historical Comparison Surface

- [../../data/output/showcase_history](../../data/output/showcase_history)
- [../../data/metrics/showcase_history](../../data/metrics/showcase_history)

These provide selected historical reference points for comparison and demo
storytelling.

### Provenance Surface

- `data/output/_prod_runs`

This is the larger provenance area referenced by the curated release files. It
is intentionally not the main reviewer entry point.

## 5. What Is Versioned vs Not Versioned

Versioned in the repository:

- code
- curated small metrics
- compact run summaries
- technical notes
- launchers and validators

Not versioned in the repository:

- raw Yelp source dumps
- large cloud logs
- large model weights
- full prediction dumps

## 6. Reviewer-Friendly Summary

The simplest way to explain the storage design to a reviewer is:

1. raw Yelp data lives outside the repository
2. Spark converts the raw data into local parquet layers
3. the three-stage ranking stack consumes those layers and writes large stage
   outputs under `data/output`
4. the repository only checks in the compact, teacher-friendly release surface
