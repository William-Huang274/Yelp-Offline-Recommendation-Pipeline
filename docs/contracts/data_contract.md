# Data Contract and Asset Inventory

## 1. Scope

This document defines the current artifact contract for the frozen internal-pilot
release:

- `release_label = internal_pilot_v1_champion_20260313`
- champion pointer: `stage11_release`
- fallback pointer: `stage10_release`
- emergency baseline pointer: `stage09_release`

Release comparison contract:

- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- `bucket = 10`
- `eval_users = 738`
- `candidate_topn = 250`
- `top_k = 10`

This contract is for offline batch recommendation only. It is not an online API
service contract.

## 2. Storage Layout

Current storage split:

- local repo root: `D:\5006_BDA_project`
- local release pointers: `data/output/_prod_runs`
- local frozen release artifacts: `data/output/_v1_freeze_20260313` and `data/output/_first_champion_freeze_20260313`
- cloud repo root: `/root/5006_BDA_project`
- cloud active stage09 root: `/root/autodl-tmp/stage09_fs`
- cloud active stage11 root: `/root/autodl-tmp/stage11_fs`

Current policy:

1. Treat the local frozen copies and local `_prod_runs` pointers as the canonical
   release contract for `internal_pilot_v1_champion_20260313`.
2. Treat `_latest_runs` as experiment pointers only.
3. Do not let smoke runs or partial runs overwrite `_prod_runs`.
4. For future release comparison, align all models on the same release cohort
   even if training splits differ.

## 3. Contract Rules

General rules for all critical artifacts:

1. Every release-critical run directory must be timestamped and immutable after
   freeze.
2. Every release-critical run must have machine-readable metadata:
   - `run_meta.json`
   - or a stage-specific metadata file such as `bucket_meta.json`
3. Downstream consumers must read frozen pointers from `_prod_runs`, not from
   ad hoc path edits.
4. Field meaning is part of the contract. Do not change label definition,
   candidate boundary, split logic, or metric definition without explicit version
   bump and parity validation.
5. Training contract and release-evaluation contract are separate:
   - `stage10` and `stage11` may use different training splits
   - release comparison must still share the same `source_run_09`, eval cohort,
     `candidate_topn`, and `top_k`

## 4. Canonical Release Lineage

Current release-critical lineage:

1. business profile contract: `08_cluster_labels/full/20260304_204941_full_profile_merged`
2. user profile contract: `09_user_profiles/20260304_234037_full_stage09_user_profile_build`
3. item semantic contract: `09_item_semantics/20260305_000408_full_stage09_item_semantic_build`
4. release source recall contract: `09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion`
5. aligned fallback eval: `10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval`
6. shared eval cohort: `11_eval_cohorts/20260311_011112_bucket10_eval_users.csv`
7. stage11 dataset contract: `11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset`
8. champion model contract: `11_qlora_models/20260312_221630_stage11_2_dpo_train_9b_v3full_ckpt485_eval_run`
9. champion eval contract: `11_qlora_sidecar_eval/20260313_151256_stage11_3_qlora_sidecar_eval`
10. release control contract: `data/output/_prod_runs/*.json`

## 5. Artifact Contracts

### 5.1 Raw Source Layer

Artifact type:

- raw Yelp JSON and early parquet conversion inputs

Path pattern:

- external/manual inputs used by `01stage`

Producer:

- external dataset intake

Consumers:

- `01_data prep.py`
- `02_data_analysis.py`

Required contract:

- business, user, and review entities must remain logically consistent
- downstream release does not currently read raw JSON directly

Release note:

- raw source files are outside the current repo freeze
- the first stable downstream business-profile contract for this release begins
  at `stage08`

### 5.2 Stage07 Relabel Working Assets

Artifact type:

- relabel work products and review queues

Path pattern:

- `data/output/07_embedding_cluster/<run_id>_full_relabel_*`

Producer:

- `07_relabel_only.py`
- `07_relabel_then_cluster.py`

Consumers:

- `08_merge_cluster_profile.py`
- manual label review

Required files:

- `biz_labels_audit_full.csv`
- `biz_labels_review_queue.csv`
- `run_meta.csv`

Stability note:

- the current local historical copy of `biz_relabels.csv` under
  `20260210_234723_full_relabel_minilm_relabel_only_ab_b` is not reliable as a
  downstream release contract because the file content is corrupted locally
- for the current release, downstream consumers should treat `stage08`
  `biz_profile_recsys.csv` as the stable business-profile contract

### 5.3 Stage08 Business Profile Contract

Canonical run:

- `data/output/08_cluster_labels/full/20260304_204941_full_profile_merged`

Producer:

- `08_merge_cluster_profile.py`

Consumers:

- `09_item_semantic_build.py`
- `09_candidate_fusion.py`
- stage09 profile joins and explainability outputs

Required files:

- `biz_profile_merged.csv`
- `biz_profile_recsys.csv`
- `run_meta.csv`

Required columns in `biz_profile_recsys.csv`:

- `business_id`
- `name`
- `city`
- `categories`
- `final_l1_label`
- `final_l2_label_top1`
- `final_l2_label_top2`
- `final_label_confidence`
- `base_cluster`
- `cluster_parent`
- `cluster_level`
- `cluster_for_recsys`
- `cluster_label_for_recsys`
- `in_cluster_strict_input`

### 5.4 Stage09 User Profile Contract

Canonical run:

- `data/output/09_user_profiles/20260304_234037_full_stage09_user_profile_build`

Producer:

- `09_user_profile_build.py`

Consumers:

- `09_candidate_fusion.py`
- `11_1_qlora_build_dataset.py`

Required files:

- `user_profiles.csv`
- `user_profile_evidence.csv`
- `user_profile_sentences.csv`
- `user_profile_tag_profile_long.csv`
- `user_profiles_summary.csv`
- `run_meta.json`

Required columns in `user_profiles.csv`:

- `user_id`
- `n_train`
- `tier`
- `last_train_ts`
- `days_since_last_train`
- `used_window_months`
- `profile_confidence`
- `profile_confidence_v1`
- `profile_confidence_v2`
- `profile_conf_interaction`
- `profile_conf_text`
- `profile_conf_freshness`
- `profile_conf_consistency`
- `n_reviews_selected`
- `n_sentences_selected`
- `profile_keywords`
- `profile_tag_support`
- `profile_top_pos_tags`
- `profile_top_neg_tags`
- `profile_text_short`
- `profile_text_long`
- `profile_text`

Required columns in `user_profile_tag_profile_long.csv`:

- `user_id`
- `tag`
- `tag_type`
- `pos_w`
- `neg_w`
- `net_w`
- `abs_net_w`
- `support`
- `tag_confidence`

Required metadata keys in `run_meta.json`:

- `run_id`
- `run_profile`
- `run_tag`
- `summary`

### 5.5 Stage09 Item Semantic Contract

Canonical run:

- `data/output/09_item_semantics/20260305_000408_full_stage09_item_semantic_build`

Producer:

- `09_item_semantic_build.py`

Consumers:

- `09_candidate_fusion.py`
- `11_1_qlora_build_dataset.py`

Required files:

- `item_semantic_features.csv`
- `item_tag_evidence.csv`
- `item_tag_profile_long.csv`
- `run_meta.json`

Required columns in `item_semantic_features.csv`:

- `business_id`
- `semantic_score`
- `semantic_confidence`
- `semantic_support`
- `semantic_tag_richness`
- `semantic_pos_weight_sum`
- `semantic_neg_weight_sum`
- `top_pos_tags`
- `top_neg_tags`

Required metadata keys in `run_meta.json`:

- `run_id`
- `run_tag`
- `summary`

### 5.6 Stage09 Candidate Fusion Contract

Canonical release source:

- `data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion`

Producer:

- `09_candidate_fusion.py`

Consumers:

- `09_1_recall_audit.py`
- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`
- `11_1_qlora_build_dataset.py`

Required bucket path pattern:

- `data/output/09_candidate_fusion/<run_id>_full_stage09_candidate_fusion/bucket_<bucket>/`

Release-critical bucket:

- `bucket_10`

Required files in each consumed bucket:

- `truth.parquet`
- `train_history.parquet`
- `candidates_all.parquet`
- `candidates_pretrim150.parquet`
- `bucket_meta.json`

Required columns in `truth.parquet`:

- `user_idx`
- `true_item_idx`
- `valid_item_idx`
- `user_id`

Required columns in `candidates_all.parquet`:

- `user_idx`
- `item_idx`
- `source_rank`
- `source_score`
- `source_confidence`
- `source`

Required columns in `candidates_pretrim150.parquet`:

- `user_idx`
- `item_idx`
- `business_id`
- `signal_score`
- `quality_score`
- `item_train_pop_count`
- `name`
- `city`
- `categories`
- `primary_category`
- `user_train_count`
- `user_segment`
- `source_set`
- `als_rank`
- `cluster_rank`
- `profile_rank`
- `popular_rank`
- `semantic_score`
- `semantic_confidence`
- `semantic_support`
- `semantic_tag_richness`
- `als_backbone_topn`
- `semantic_effective_score`
- `als_backbone_score`
- `pre_score`
- `user_pretrim_top_k`
- `pre_rank`
- `tower_score`
- `seq_score`
- `tower_inv`
- `seq_inv`

Compatibility note:

- current release compares all downstream ranking models on the same
  `candidates_pretrim150.parquet` contract

### 5.7 Stage09 Recall Audit Contract

Canonical audit run:

- `data/output/09_recall_audit/20260313_191007_stage09_recall_audit`

Producer:

- `09_1_recall_audit.py`

Consumers:

- release readiness validation
- freeze notes and go-live decisions

Required files:

- `data/metrics/stage09_recall_audit_summary_latest.csv`
- stage-specific audit run directory under `data/output/09_recall_audit`

Required columns in `stage09_recall_audit_summary_latest.csv`:

- `source_run_09`
- `bucket`
- `truth_in_all`
- `truth_in_pretrim`
- `pretrim_cut_loss`
- `hard_miss`

### 5.8 Shared Release Eval Cohort Contract

Canonical file:

- `data/output/11_eval_cohorts/20260311_011112_bucket10_eval_users.csv`

Producer:

- extracted from the `stage11_1` dataset split used for aligned comparison

Consumers:

- `10_2_rank_infer_eval.py`
- `11_3_qlora_sidecar_eval.py`

Required columns:

- `user_idx`

Compatibility note:

- this file is the release-evaluation cohort contract
- it is not a requirement that `stage10` and `stage11` share the same training split

### 5.9 Stage10 Rank Model Contract

Canonical fallback model run:

- `data/output/10_rank_models/20260307_210530_stage10_1_rank_train`

Producer:

- `10_1_rank_train.py`

Consumers:

- `10_2_rank_infer_eval.py`

Required files:

- `rank_model.json`
- `models/`
- `splits/`

Required keys in `rank_model.json`:

- `run_id`
- `run_tag`
- `source_stage09_run`
- `feature_columns`
- `params`

Compatibility note:

- historical training source in this frozen model is `20260305_002746`
- this does not invalidate aligned release comparison because the release-eval
  cohort and release source contract were re-aligned in `GL-07`

### 5.10 Stage10 Aligned Infer/Eval Contract

Canonical fallback eval run:

- `data/output/10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval`

Producer:

- `10_2_rank_infer_eval.py`

Consumers:

- release readiness
- champion/fallback selection
- release notes

Required files:

- `run_meta.json`
- metrics file referenced by the release pointer:
  `data/metrics/recsys_stage10_results_gl07.csv`

Required keys in `run_meta.json`:

- `source_stage09_run`
- `eval_candidate_topn`
- `top_k`

Required columns in `data/metrics/recsys_stage10_results_gl07.csv`:

- `run_id_10`
- `source_run_09`
- `bucket_min_train_reviews`
- `model`
- `recall_at_k`
- `ndcg_at_k`
- `user_coverage_at_k`
- `item_coverage_at_k`
- `tail_coverage_at_k`
- `novelty_at_k`
- `n_users`
- `n_items`
- `n_candidates`

### 5.11 Stage11 Dataset Contract

Canonical dataset run:

- `data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset`

Producer:

- `11_1_qlora_build_dataset.py`

Consumers:

- `11_2_qlora_train.py`
- `11_2_dpo_train.py`
- `11_3_qlora_sidecar_eval.py`

Required bucket paths:

- `bucket_10/all_parquet`
- `bucket_10/train_json`
- `bucket_10/eval_json`
- `bucket_10/rich_sft_train_json`
- `bucket_10/rich_sft_eval_json`
- `bucket_10/user_evidence_table`
- `bucket_10/item_evidence_table`
- `bucket_10/pair_evidence_audit`

Required files:

- `run_meta.json`

Required columns in the `all_parquet` contract:

- `bucket`
- `user_idx`
- `item_idx`
- `business_id`
- `label`
- `sample_weight`
- `label_source`
- `neg_tier`
- `neg_pick_rank`
- `neg_is_near`
- `neg_is_hard`
- `pre_rank`
- `pre_score`
- `prompt`
- `target_text`
- `user_evidence_text`
- `history_anchor_text`
- `item_evidence_text`
- `pair_evidence_summary`
- `split`

Required metadata keys in `run_meta.json`:

- `run_id`
- `run_tag`
- `source_run_09`
- `summary`

### 5.12 Stage11 Model Adapter Contract

Canonical champion model run:

- `data/output/11_qlora_models/20260312_221630_stage11_2_dpo_train_9b_v3full_ckpt485_eval_run`

Producer:

- `11_2_dpo_train.py`

Consumers:

- `11_3_qlora_sidecar_eval.py`
- release pointer `stage11_release`

Required files:

- `run_meta.json`
- `adapter/`

Required metadata keys in `run_meta.json`:

- `training_method`
- `source_stage11_dataset_run`
- `train_pairs`
- `eval_pairs`
- `train_runtime_sec`
- `train_loss`
- `eval_loss`
- `enforce_stage09_gate`

Operational note:

- current frozen champion still records `enforce_stage09_gate=false`
- this is why release readiness remains `WARN`, not `PASS`

### 5.13 Stage11 Sidecar Eval Contract

Canonical champion eval run:

- `data/output/11_qlora_sidecar_eval/20260313_151256_stage11_3_qlora_sidecar_eval`

Producer:

- `11_3_qlora_sidecar_eval.py`

Consumers:

- champion freeze
- release readiness
- release monitor checks

Required files:

- `run_meta.json`
- `alpha_sweep_best.json`
- `alpha_sweep_grid.csv`
- `bucket_10_scores.csv`
- `qlora_sidecar_metrics.csv`

Required keys in `run_meta.json`:

- `source_run_09`
- `source_run_11_1_data`
- `source_run_11_2`
- `enforce_stage09_gate`
- `top_k`
- `rerank_topn`
- `metrics_file`

Required columns in `qlora_sidecar_metrics.csv`:

- `run_id_11`
- `source_run_09`
- `source_run_11_2`
- `bucket_min_train_reviews`
- `model`
- `recall_at_k`
- `ndcg_at_k`
- `n_users`
- `n_items`
- `n_candidates`

Required columns in `bucket_10_scores.csv`:

- `user_idx`
- `item_idx`
- `pre_rank`
- `label_true`
- `pre_score`
- `qlora_prob`
- `blend_score`

### 5.14 Release Control Contract

Canonical path:

- `data/output/_prod_runs`

Producer:

- `scripts/pipeline/internal_pilot_runner.py`
- `scripts/pipeline/project_paths.py`

Consumers:

- `tools/check_release_readiness.py`
- `tools/check_release_monitoring.py`
- future batch publish / rollback flows

Required files:

- `release_policy.json`
- `stage09_release.json`
- `stage10_release.json`
- `stage11_release.json`
- latest `release_manifest_*.json`
- latest `rollback_snapshot_*.json`

Required keys in `release_policy.json`:

- `release_label`
- `champion_pointer`
- `aligned_fallback_pointer`
- `emergency_baseline_pointer`
- `release_contract`

Required keys in stage-specific release pointers:

- `pointer_name`
- `run_dir`
- `release_label`
- `release_role`
- release-specific source and metric references

## 6. Downstream Consumption Rules

1. New release comparison must start from `_prod_runs/release_policy.json`.
2. Any new champion proposal must update:
   - stage release pointer JSON
   - release manifest
   - readiness report
   - rollback snapshot
3. Historical experiment outputs can exist under `data/output`, but they are not
   allowed to silently replace the canonical release lineage.
4. If a required file exists but schema changes, bump the contract version in
   docs and rerun parity validation before promoting it to `_prod_runs`.

## 7. Current Open Risks

The data contract is now documented, but the current release is still not
production-ready because:

- `GL-01` credential rotation remains open by user choice on a temporary cloud machine
- the frozen stage11 champion still records `enforce_stage09_gate=false`
- the current release manifest still marks `production_ready=false`

## 8. Completion Statement

`GL-14` is complete for `internal_pilot_v1_champion_20260313` when this document
is used as the canonical artifact inventory and contract reference for future
repair cycles.
