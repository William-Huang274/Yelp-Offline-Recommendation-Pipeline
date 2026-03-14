# Yelp LA Offline Recommendation Pipeline

An offline recommendation project built on the Yelp dataset, focused on Los
Angeles food and restaurant discovery. The repository covers the full path from
data preparation and business relabeling to candidate generation, structured
rerank, and LLM-based sidecar rerank experiments.

This repository is suitable as a portfolio and research-delivery project. It is
not a production-ready online service.

## What This Project Demonstrates

- end-to-end offline recommender workflow across `01 -> 11` stages
- stable recall pipeline with audit and frozen release lineage
- structured rerank with XGBoost fallback
- QLoRA / DPO sidecar rerank experiments as the current champion path
- release validation, rollback, monitoring, and data-contract documentation

## Current Frozen Result

Release label:

- `internal_pilot_v1_champion_20260313`

Aligned release contract:

- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- `bucket = 10`
- `eval_users = 738`
- `candidate_topn = 250`
- `top_k = 10`

Model ranking on the same release cohort:

| Model | Recall@10 | NDCG@10 | Role |
| --- | ---: | ---: | --- |
| `PreScore@10` | `0.056911` | `0.026467` | emergency baseline |
| `LearnedBlendXGBCls@10` | `0.065041` | `0.029217` | aligned fallback |
| `QLoRASidecar@10` | `0.067751` | `0.029935` | current champion |

Current audited recall evidence for the release source:

- `truth_in_all = 0.947208`
- `truth_in_pretrim = 0.882255`
- `hard_miss = 0.052792`

Supporting documents:

- release readiness: [docs/release_readiness_report_internal_pilot_v1_champion_20260313.md](./docs/release_readiness_report_internal_pilot_v1_champion_20260313.md)
- champion closeout: [docs/first_champion_closeout_20260313.md](./docs/first_champion_closeout_20260313.md)
- go-live checklist: [docs/go_live_readiness_20260312.md](./docs/go_live_readiness_20260312.md)

## Repository Layout

- [scripts](./scripts): stage scripts and pipeline utilities
- [docs](./docs): freeze notes, audit notes, runbooks, and repair records
- [tests](./tests): smoke tests for path handling, pointers, and validators
- [tools](./tools): validation and release-check helpers
- [config](./config): config assets used by later stages
- [data/metrics](./data/metrics): small tracked metrics summaries
- [data/output/_prod_runs](./data/output/_prod_runs): release pointers, manifests, and rollback snapshots

Main active scripts:

- stage07: [scripts/07_relabel_then_cluster.py](./scripts/07_relabel_then_cluster.py)
- stage09: [scripts/09_user_profile_build.py](./scripts/09_user_profile_build.py), [scripts/09_item_semantic_build.py](./scripts/09_item_semantic_build.py), [scripts/09_candidate_fusion.py](./scripts/09_candidate_fusion.py), [scripts/09_1_recall_audit.py](./scripts/09_1_recall_audit.py)
- stage10: [scripts/10_1_rank_train.py](./scripts/10_1_rank_train.py), [scripts/10_2_rank_infer_eval.py](./scripts/10_2_rank_infer_eval.py)
- stage11: [scripts/11_1_qlora_build_dataset.py](./scripts/11_1_qlora_build_dataset.py), [scripts/11_2_qlora_train.py](./scripts/11_2_qlora_train.py), [scripts/11_2_dpo_train.py](./scripts/11_2_dpo_train.py), [scripts/11_3_qlora_sidecar_eval.py](./scripts/11_3_qlora_sidecar_eval.py)

## Pipeline Summary

1. `01-08`: prepare Yelp data, filter LA food businesses, relabel and cluster businesses, and build business-profile assets.
2. `09`: build user profile and item semantics, fuse multi-route candidates, and audit recall quality.
3. `10`: train structured rerank models and evaluate aligned fallback ranking.
4. `11`: build QLoRA / DPO datasets, train adapters, and run sidecar rerank evaluation.

## Quick Start

CPU-friendly smoke and validation environment:

```bash
python -m pip install -r requirements.txt
```

Optional GPU-heavy stage11 environment:

```bash
python -m pip install -r requirements-stage11-qlora.txt
```

Run smoke tests:

```bash
python -m pytest tests -q
```

Run release validation:

```bash
python tools/check_release_readiness.py
python tools/check_release_monitoring.py
```

Run the internal-pilot helper:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode validate
python scripts/pipeline/internal_pilot_runner.py --mode monitor
```

Windows wrapper:

```bat
scripts\run_internal_pilot.bat --mode validate
```

## Key Documentation

- data contract and asset inventory: [docs/data_contract.md](./docs/data_contract.md)
- path unification note: [docs/gl09_path_unification_20260313.md](./docs/gl09_path_unification_20260313.md)
- smoke test note: [docs/gl10_smoke_tests_20260313.md](./docs/gl10_smoke_tests_20260313.md)
- batch runner note: [docs/gl12_batch_runner_20260313.md](./docs/gl12_batch_runner_20260313.md)
- rollback and monitoring: [docs/rollback_and_monitoring.md](./docs/rollback_and_monitoring.md)
- cloud execution profile: [docs/stage11_cloud_run_profile_20260309.md](./docs/stage11_cloud_run_profile_20260309.md)

## Public Repo Notes

- raw Yelp source data is not included in this repository
- large stage09-11 run artifacts are not fully versioned here
- the canonical public release state is represented by:
  - tracked docs under [docs](./docs)
  - tracked metrics under [data/metrics](./data/metrics)
  - tracked release-control files under [data/output/_prod_runs](./data/output/_prod_runs)
- the current stable business-profile contract for downstream use is the
  `stage08` merged profile output documented in [docs/data_contract.md](./docs/data_contract.md)

## Current Limitations

- this is still an offline batch pipeline, not an online serving system
- the current readiness report is `WARN`, not `PASS`
- the frozen stage11 champion still records `enforce_stage09_gate=false`
- the release manifest still marks `production_ready=false`
- `GL-01` credential rotation was deferred because the referenced cloud machine was temporary

## Why This Is Portfolio-Worthy

This project is strong as a job-search portfolio item because it combines:

- data engineering and schema management
- recommender-system retrieval and rerank design
- experiment alignment and offline evaluation discipline
- LLM fine-tuning workflow integration
- release-readiness, rollback, and monitoring thinking beyond pure modeling

The right positioning is:

`offline recommendation and rerank research pipeline with release discipline`

not:

`fully productionized recommendation platform`

## License

This repository is released under the [MIT License](./LICENSE).
