# GL-13 Bucket Stage10 Gate Readiness (2026-03-18)

## 1. Current Position

The repo now treats `bucket_2`, `bucket_5`, and `bucket_10` as three slices that
should all be comparable at `stage10` before any broader `stage11` admission
decision.

`bucket_10` remains the current frozen public release slice because:

- it already has frozen stage09, stage10, and stage11 pointers under `data/output/_prod_runs`
- it is the only bucket currently admitted to stage11
- its public metrics are already tracked in `README.md` and release-control files

`bucket_5` and `bucket_2` now have completed isolated `stage09 -> stage10`
chains. Those runs were first generated on 2026-03-17 under the legacy local-only
root `data/output/backfill`. The repo-facing naming is now `stage10_gate`; future
isolated runs should default to `data/output/stage10_gate`.

## 2. Readiness Matrix

| Bucket | Role | Stage09 run | truth_in_all | truth_in_pretrim | pretrim_cut_loss | Stage10 run | Selected model | Recall@10 | NDCG@10 | Stage11 status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| `bucket_10` | frozen public release | `20260311_005450_full_stage09_candidate_fusion` | `0.947208` | `0.882255` | `0.064953` | `20260313_193213_stage10_2_rank_infer_eval` | `LearnedBlendXGBCls@10` | `0.065041` | `0.029217` | admitted |
| `bucket_5` | stage10 gate complete | `20260317_220710_full_stage09_candidate_fusion` | `0.838402` | `0.724834` | `0.113569` | `20260317_224300_stage10_2_rank_infer_eval` | `LearnedBlendXGBCls@10` | `0.088115` | `0.040522` | pending admission |
| `bucket_2` | stage10 gate complete | `20260317_225205_full_stage09_candidate_fusion` | `0.823128` | `0.696920` | `0.126208` | `20260317_235955_stage10_2_rank_infer_eval` | `LearnedBlendXGBCls@10` | `0.108533` | `0.050402` | pending admission |

## 3. Shared Contracts And Local Safety

All bucket2/5 stage10 gate runs reuse the current shared upstream contracts:

- stage08 business profile:
  `data/output/08_cluster_labels/full/20260304_204941_full_profile_merged`
- stage09 user profile:
  `data/output/09_user_profiles/20260304_234037_full_stage09_user_profile_build`
- stage09 item semantic:
  `data/output/09_item_semantics/20260305_000408_full_stage09_item_semantic_build`

Local Windows fallback policy for these runs:

- `stage09` uses `LOCAL_PARQUET_WRITE_MODE=driver_parquet`
- `stage10_2` uses `XGB_BATCH_MODE=hash_partition_memory`
- release pointers remain unchanged and bucket10-only

## 4. Submission Surface

Commit the small, auditable bucket gate artifacts:

- `data/metrics/stage10_gate/bucket_2/*.csv`
- `data/metrics/stage10_gate/bucket_5/*.csv`
- `data/metrics/stage10_gate/manifests/*.json`
- `data/metrics/stage10_gate/bucket_stage10_gate_summary_20260318.csv`

Do not commit the large run directories under `data/output/stage10_gate/` or the
legacy `data/output/backfill/` root. Those stay local-only and are already
covered by the tracked summary manifests.

## 5. Runner And Validation Entry Points

Preferred runner:

- `python scripts/pipeline/bucket_stage10_gate_runner.py --bucket 5 --mode full`
- `python scripts/pipeline/bucket_stage10_gate_runner.py --bucket 2 --mode full`
- `python scripts/pipeline/bucket_stage10_gate_runner.py --bucket 5 --mode validate`

Compatibility:

- `python scripts/pipeline/bucket_backfill_runner.py ...` still works as a thin wrapper
- legacy CLI flags `--backfill-output-root` and `--backfill-metrics-root` are still accepted

Required validators:

- `python tools/validate_stage_artifact.py --kind stage09_candidate --run-dir <stage09_run_dir>`
- `python tools/validate_stage_artifact.py --kind stage10_rank_model --run-dir <stage10_train_run_dir>`
- `python tools/validate_stage_artifact.py --kind stage10_infer_eval --run-dir <stage10_eval_run_dir>`

## 6. Admission Rule

This note does not change the current release policy.

- `bucket_10` stays the public frozen release slice
- `bucket_2` and `bucket_5` are now ready for a bucket-level `stage10` comparison
- stage11 admission for non-bucket10 slices should only happen after that explicit gate
