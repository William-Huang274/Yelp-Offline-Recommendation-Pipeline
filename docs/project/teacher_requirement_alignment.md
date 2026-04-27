# Teacher Requirement Alignment

## Purpose

This note maps the practice-module briefing requirements to the current
repository evidence so that proposal, demo, and final-report preparation can
start from the canonical engineering surface rather than from scattered notes.

## 1. Course-Module Competency Mapping

| briefing requirement | repository evidence | primary references |
| --- | --- | --- |
| `BEAD`: ingest from multiple data sources, choose storage, build scalable data processing pipelines with Spark | Yelp JSON sources are ingested into parquet, optional photo assets are aggregated into business-level parquet, and the active pipeline is Spark-centric | [../../scripts/stage01_to_stage08/01_data prep.py](../../scripts/stage01_to_stage08/01_data%20prep.py), [../../scripts/09_candidate_fusion.py](../../scripts/09_candidate_fusion.py), [data_lineage_and_storage.md](./data_lineage_and_storage.md) |
| `RCS`: apply recommender-system concepts to build a recommender | The current repository line is an end-to-end ranking stack with recall routing, structured reranking, and reward-model rescue reranking | [../../README.md](../../README.md), [evaluation_and_casebook.md](./evaluation_and_casebook.md) |
| `PBDA`: structure the big data project, cover technical architecture, model development, aggregation, and monitoring | The repository documents stage boundaries, bounded reranking, evaluation contracts, rollout monitoring, and rollback rules | [../../docs/release/stage09_stage10_stage11_upgrade_audit_20260409.md](../../docs/release/stage09_stage10_stage11_upgrade_audit_20260409.md), [../../docs/archive/release/rollback_and_monitoring.md](../../docs/archive/release/rollback_and_monitoring.md), [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md) |

## 2. Project-Stage Requirement Mapping

The briefing asks the project to cover an analytics problem through ingestion,
storage, processing, analytics solution, interpretation, and engineering
challenges.

| briefing stage | current repository evidence |
| --- | --- |
| selection of analytics project involving big data | Yelp restaurant recommendation with large review, business, tip, check-in, and photo data |
| analytics method | three-stage ranking stack: `Stage09` recall routing, `Stage10` structured rerank, `Stage11` reward-model rescue rerank |
| big data ingestion | Yelp JSON to parquet conversion and optional photo-zip aggregation |
| big data storage | curated parquet layer under `data/parquet`, stage outputs under `data/output`, frozen review surface under `data/output/current_release` |
| big data processing | Spark-based feature building, candidate routing, reranking, dataset export, and release validation |
| analytics solution | frozen `bucket5` Stage09/10/11 results with current release metrics |
| interpretation | Stage11 case notes and the engineering casebook in this folder |
| articulation of challenges encountered | trade-offs, bounded LLM use, leakage control, rollback/monitoring, and resource constraints |

## 3. Deliverable Mapping

| deliverable from briefing | repository support added in this pack | still manual / outside repository |
| --- | --- | --- |
| project proposal | [proposal_template_content.md](./proposal_template_content.md), [teacher_requirement_alignment.md](./teacher_requirement_alignment.md), [data_lineage_and_storage.md](./data_lineage_and_storage.md) | team names, final effort estimates, final project scope wording |
| demo presentation | [demo_runbook.md](./demo_runbook.md), `python tools/demo/demo_recommend.py ...`, [evaluation_and_casebook.md](./evaluation_and_casebook.md) | final slide design and speaker allocation |
| final project report | [final_report_outline.md](./final_report_outline.md), [environment_setup.md](./environment_setup.md), [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md) | final narrative writing, screenshots, and team-authored conclusion |
| source code submission | repository code, launchers, frozen result surface, validators | none |
| individual reflection report | not a shared repository artifact | each student must write and submit separately |
| peer review | not a shared repository artifact | must be submitted through the course workflow separately |

## 4. Current Strengths For Review

The repository is strongest when presented as:

1. a big-data recommendation project rather than as a single-model notebook
2. one coherent `stage09 -> stage10 -> stage11` upgrade line
3. a bounded and reviewable engineering stack with explicit release surfaces,
   validators, and rollback thinking

## 5. Current Boundaries

These claims are supported now:

- the repository demonstrates the engineering and modeling stack required for
  the group project
- the frozen release line is reproducible at the documentation and result-surface
  level
- the demo can be run from checked-in frozen assets without requiring a GPU

These still require manual group work outside the repository:

- named team-member allocation
- exact man-day estimates per member
- individual reflection report
- peer-review submission
