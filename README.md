# Yelp Offline Recommendation Pipeline

[English](./README.md) | [Chinese](./README.zh-CN.md)

Offline recommender for Louisiana restaurant discovery on the Yelp dataset. The repo covers the full `01 -> 11` offline path: data prep, relabeling, business profiling, stage09 candidate fusion, stage10 structured rerank, and stage11 QLoRA / DPO sidecar experiments.

This repository is meant to show an end-to-end recommender-system build, not just a standalone model notebook. It combines bucketed offline evaluation, aligned model comparison, dataset engineering for LLM reranking, and release validation utilities.

## Why This Project Stands Out

- end-to-end offline recommendation pipeline instead of a single ranking-model demo
- bucket-aware user slicing (`bucket_2`, `bucket_5`, `bucket_10`) rather than treating all users as one cohort
- same-cohort comparison across heuristic ranking, XGBoost rerank, and QLoRA / DPO sidecar models
- stage11 training is backed by explicit dataset engineering: pointwise -> rich SFT -> DPO pairs
- includes release manifests, validators, monitoring checks, and rollback / readiness documentation

## Results Snapshot

Current public frozen release: `internal_pilot_v1_champion_20260313`

| Model | Bucket | Recall@10 | NDCG@10 | Role |
| --- | --- | ---: | ---: | --- |
| `PreScore@10` | `bucket10` | `0.056911` | `0.026467` | emergency baseline |
| `LearnedBlendXGBCls@10` | `bucket10` | `0.065041` | `0.029217` | structured fallback |
| `QLoRASidecar@10` | `bucket10` | `0.067751` | `0.029935` | current champion |

Scale at a glance:

- Stage10 structured rerank training: `47,271` rows, `2,251` train users, `2,251` positives
- Stage11 pointwise dataset: `22,327` rows, `7,855` positives, `3,618` total users
- Stage11 rich SFT dataset: `27,743` rows and `7,855` positives
- Stage11 DPO dataset: about `5,836` train pairs and `1,452` eval pairs
- Final aligned release cohort: `738` eval users, `candidate_topn = 250`, `top_k = 10`

Release cohort evidence:

- `truth_in_all = 0.947208`
- `truth_in_pretrim = 0.882255`
- `hard_miss = 0.052792`

## Run And Verify Locally

There are three useful ways to run this repository locally, and they do not mean the same thing.

### 1. Repo Smoke Test

Use this path when you want to confirm a fresh clone is wired correctly.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python tools/check_release_readiness.py
python tools/check_release_monitoring.py
```

What this verifies:

- path helpers
- latest / prod run pointers
- validator behavior
- release readiness and monitoring tooling

This path does not require raw Yelp source data.

### 2. Frozen Artifact Validation

Use this path when you already have the frozen local stage outputs and want to verify artifact lineage.

```bash
python tools/validate_stage_artifact.py --kind stage09_candidate --run-dir data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion
python tools/validate_stage_artifact.py --kind stage10_rank_model --run-dir data/output/10_rank_models/20260307_210530_stage10_1_rank_train
python tools/validate_stage_artifact.py --kind stage10_infer_eval --run-dir data/output/10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval
python tools/validate_stage_artifact.py --kind stage11_dataset --run-dir data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset
```

What this verifies:

- frozen stage09 candidate artifacts
- frozen stage10 rank-train artifacts
- frozen stage10 infer/eval artifacts
- frozen stage11 dataset artifacts
- release-lineage consistency with the tracked public freeze

Bucket2/5 stage10 gate helper:

```bash
python scripts/pipeline/bucket_stage10_gate_runner.py --bucket 5 --mode full --dry-run
```

This helper keeps future bucket2/5 gate runs isolated under
`data/output/stage10_gate` and `data/metrics/stage10_gate`.
The tracked submission surface is the small metrics and manifests under
`data/metrics/stage10_gate`; the large isolated run directories remain local-only.
It only rebuilds `stage09 -> stage10` for `bucket_2` or `bucket_5`; any stage11
admission decision happens later.

### 3. Full Rebuild From Raw Data

Use this path only if you have the raw Yelp source data and want to rerun the pipeline stages.

Important notes:

- raw Yelp source data is not included in this public repository
- large stage09-stage11 run artifacts are not fully versioned here
- the canonical public release state is represented by tracked docs, tracked metrics, and release-control files under `data/output/_prod_runs`
- the stable downstream business-profile contract begins at the tracked stage08 merged profile output documented in [docs/contracts/data_contract.md](./docs/contracts/data_contract.md)

Optional stage11 training environment:

```bash
python -m pip install -r requirements-stage11-qlora.txt
```

Internal-pilot helpers:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode validate
python scripts/pipeline/internal_pilot_runner.py --mode monitor
```

Windows wrapper:

```bat
scripts/run_internal_pilot.bat --mode validate
```

## Bucketing Strategy

Stage09 defines three minimum-history buckets from `MIN_TRAIN_REVIEWS_BUCKETS = [2, 5, 10]` in [scripts/09_candidate_fusion.py](./scripts/09_candidate_fusion.py):

- `bucket_2`
- `bucket_5`
- `bucket_10`

For each bucket, stage09 also applies `min_user = min_train + 2` as the user-side cohort floor.

Inside each bucket, users are separately tagged as:

- `light` when train interactions `<= 7`
- `mid` when train interactions are `8-19`
- `heavy` when train interactions `>= 20`

This segment split is different from the bucket definition and is used for per-segment candidate budgets and fusion behavior.

Why the current public stage11 line starts with `bucket10`:

- it is the first slice where users have richer and more diverse preference signals
- it exposes the hardest head-ordering errors, especially for heavy users
- it is the most meaningful bucket for testing whether text-aware SFT / DPO sidecars can improve reranking beyond the structured fallback model

`bucket_2` and `bucket_5` now also have completed `stage09 -> stage10` gate
artifacts, but the repo still treats `bucket10` as the frozen public release
slice and the only bucket currently admitted to stage11.

## Training Data Construction

### Stage10 Structured Rerank

Frozen fallback model run: `20260307_210530_stage10_1_rank_train`

Bucket10 training volume from the frozen model metadata:

- `train_rows = 47271`
- `train_pos = 2251`
- `train_users = 2251`
- `valid_users = 369`
- `test_users = 701`

This is the aligned structured fallback used before the sidecar champion.

### Stage11 Pointwise Dataset

Frozen dataset run: `20260311_011112_stage11_1_qlora_build_dataset`

Key build settings from `run_meta.json`:

- `buckets_processed = [10]`
- `candidate_file = candidates_pretrim150.parquet`
- `topn_per_user = 120`
- `eval_user_frac = 0.2`
- `prompt_mode = full_lite`
- `include_valid_pos = true`
- `valid_pos_weight = 0.6`

Base pointwise dataset on bucket10:

| Slice | Rows | Positives | Negatives |
| --- | ---: | ---: | ---: |
| train | `17796` | `6276` | `11520` |
| eval | `4531` | `1579` | `2952` |
| total | `22327` | `7855` | `14472` |

Additional cohort facts:

- `users_total = 3618`
- `users_with_positive = 3386`
- `users_no_positive = 232`

### Rich SFT Dataset

The current SFT mainline does not train from the base pointwise export. It trains from the richer `rich_sft` export of the same bucket10 run.

Frozen `rich_sft` volume:

| Slice | Rows | Positives | Negatives |
| --- | ---: | ---: | ---: |
| train | `22111` | `6276` | `15835` |
| eval | `5632` | `1579` | `4053` |
| total | `27743` | `7855` | `19888` |

How `rich_sft` is constructed:

- start from the frozen bucket10 `pretrim150` candidate pool
- keep the same train / eval user split as the base stage11 dataset build
- use `full_lite` prompts with user evidence, item evidence, and history anchors
- keep true positives and optionally include valid positives with weight `0.6`
- sample negatives per user as `1 explicit + 2 hard + 2 near + 1 mid + 0 tail`
- cap hard negatives at `pre_rank <= 20`
- cap mid negatives at `pre_rank <= 60`
- require negative rating `<= 2.5`
- keep `rich_sft_allow_neg_fill = false` in the frozen run

### DPO Pair Dataset

The current DPO line warm-starts from the SFT adapter instead of training pairwise from raw base weights.

Current DPO shape and construction:

- pairwise source mode: `rich_sft`
- pair policy: `v2a`
- top-k cutoff: `10`
- high-rank cutoff: `20`
- loser selection favors `hard`, `near`, and outranking confusers from the same bucket10 candidate pool
- audited pair volume is roughly `5836` train pairs and `1452` eval pairs on bucket10

## Repository Layout

- [scripts](./scripts): stage scripts and pipeline utilities
- [tests](./tests): portable tests and local artifact smoke checks
- [tools](./tools): release-check, monitoring, and validation helpers
- [config](./config): config assets used by later stages
- [docs](./docs): categorized documentation index
- [docs/contracts](./docs/contracts): contracts and config references
- [docs/release](./docs/release): freeze, readiness, monitoring, and rollback notes
- [docs/repo](./docs/repo): repo hygiene, path, pointer, smoke-test, and runner notes
- [docs/stage09](./docs/stage09): candidate-fusion and bucket10 audit notes
- [docs/stage10](./docs/stage10): structured rerank training notes
- [docs/stage11](./docs/stage11): QLoRA, sidecar, and cloud runbooks
- [docs/dpo](./docs/dpo): DPO guides and recommendations
- [docs/labeling](./docs/labeling): labeling manuals

## Key Documentation

- docs index: [docs/README.md](./docs/README.md)
- data contract: [docs/contracts/data_contract.md](./docs/contracts/data_contract.md)
- release readiness: [docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md](./docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md)
- champion closeout: [docs/release/first_champion_closeout_20260313.md](./docs/release/first_champion_closeout_20260313.md)
- rollback and monitoring: [docs/release/rollback_and_monitoring.md](./docs/release/rollback_and_monitoring.md)
- stage11 cloud run profile: [docs/stage11/stage11_cloud_run_profile_20260309.md](./docs/stage11/stage11_cloud_run_profile_20260309.md)
- smoke-test note: [docs/repo/gl10_smoke_tests_20260313.md](./docs/repo/gl10_smoke_tests_20260313.md)

## Current Limitations

- this is still an offline batch pipeline, not an online serving system
- the current readiness report is `WARN`, not `PASS`
- the frozen stage11 champion still records `enforce_stage09_gate=false`
- the release manifest still marks `production_ready=false`
- `GL-01` credential rotation was deferred because the referenced cloud machine was temporary

## License

This repository is released under the [MIT License](./LICENSE).
