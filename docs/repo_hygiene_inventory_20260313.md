# Repo Hygiene Inventory (2026-03-13)

## Purpose

This note records the current repository hygiene state after adding the root
`README.md` and `.gitignore`.

It exists to make the remainder of `GL-03` explicit: the main remaining issue is
not cache noise, but real project source files that are still outside the tracked
git baseline.

## Snapshot

Git worktree snapshot taken on `2026-03-13`:

- modified tracked files: `10`
- ignored local-only paths: `26`
- untracked paths still visible to git: `36`

## 1. Modified Tracked Source Files

These are real source files already in the tracked baseline and still modified:

- `scripts/09_1_recall_audit.py`
- `scripts/09_candidate_fusion.py`
- `scripts/09_item_semantic_build.py`
- `scripts/09_profile_calibration_train.py`
- `scripts/09_user_profile_build.py`
- `scripts/11_0_qlora_data_gate_audit.py`
- `scripts/11_1_qlora_build_dataset.py`
- `scripts/11_2_dpo_train.py`
- `scripts/11_2_qlora_train.py`
- `scripts/11_3_qlora_sidecar_eval.py`

Interpretation:

- these are intentional active pipeline files
- they should stay visible in git status
- they are not repo-noise and should not be hidden by `.gitignore`

## 2. Ignored Local-Only Paths

These are now correctly hidden as local machine noise or generated artifacts:

- `.venv-wsl/`
- `=1.6.0,`
- `_recovered_winfr/`
- `tmp/`
- `data/parquet/`
- `data/spark-tmp/`
- `data/tmp/`
- `data/output/07_embedding_cluster/`
- `data/output/08_cluster_labels/`
- `data/output/09_candidate_fusion/`
- `data/output/09_item_semantics/`
- `data/output/09_recall_audit/`
- `data/output/09_user_profiles/`
- `data/output/10_2_rank_infer_eval/`
- `data/output/10_5_transformer_rerank_eval/`
- `data/output/10_rank_models/`
- `data/output/11_eval_cohorts/`
- `data/output/11_qlora_data/`
- `data/output/11_qlora_models/`
- `data/output/11_qlora_sidecar_eval/`
- `data/output/11_qlora_train_remote/`
- `data/output/_v1_freeze_20260313/remote_snapshot/`
- `data/output/manual_label_seed200_stratified.csv`
- `scripts/__pycache__/`
- `scripts/pipeline/__pycache__/`
- `tools/__pycache__/`

Interpretation:

- these are the paths that should remain local-only
- this is the part of `GL-03` that is already working

## 3. Untracked Paths Still Requiring Classification

These paths are still visible to git and need an explicit decision.

### Root-level project files

- `.gitignore`
- `README.md`
- `AGENTS.md`
- `requirements-stage11-qlora.txt`

These should stay visible and should be part of the repo baseline.

### Real project directories

- `config/`
- `docs/`
- `tools/`
- `scripts/pipeline/`
- `scripts/legacy_stage07/`

These also look like real project content and should be reviewed for inclusion
rather than ignored.

### Untracked stage scripts

- `scripts/09_0_recover_user_profile_vectors.py`
- `scripts/09_2_bucket10_prerank_probe.py`
- `scripts/09_user_profile_stats.py`
- `scripts/10_10_pitfall_memory_report.py`
- `scripts/10_1_rank_train.py`
- `scripts/10_2_rank_infer_eval.py`
- `scripts/10_3_xgb_diagnose.py`
- `scripts/10_4_autotune_seq.py`
- `scripts/10_5_transformer_rerank_eval.py`
- `scripts/10_6_two_tower_seq_probe.py`
- `scripts/10_7_tower_seq_09_probe.py`
- `scripts/10_8_xgb_tower_seq_eval.py`
- `scripts/10_9_autopilot_xgb.py`
- `scripts/10_rerank_and_eval.py`
- `scripts/11_1_reweight_pairwise_pool.py`
- `scripts/11_2_dpo_checkpoint_audit.py`
- `scripts/11_2_dpo_export_pairs.py`
- `scripts/11_2_qwen35_path_benchmark.py`
- `scripts/11_2_qwen35_preflight_smoke.py`
- `scripts/11_2_score_pointwise_confusers.py`
- `scripts/activate_stage11_recovery_env.bat`
- `scripts/check_dpo_env.bat`
- `scripts/run_dpo_low_memory.bat`
- `scripts/run_dpo_optimized.bat`
- `scripts/run_stage09_bucket10_enriched_audit_waiter.sh`
- `scripts/stage07_core.py`

Interpretation:

- these are not temporary outputs
- they look like real source files that belong to the project
- leaving them untracked is the main reason `GL-03` is not fully done yet
- default keep decision for the current repair cycle:
  - keep these files in the intended repo baseline unless a later review explicitly
    moves a file into `legacy/` or deletes it as accidental debris

### Data and doc paths still visible

- `data/`
- `docs/`

Interpretation:

- `data/` remains visible because curated files such as metrics and the v1 freeze
  manifest are intentionally not globally ignored
- `docs/` remains visible because documentation should stay reviewable and versioned

## 4. What GL-03 Still Needs

To fully close `GL-03`, do not add more broad ignore rules.

Instead:

1. review the current untracked source files and decide which ones are part of the
   intended repo baseline
2. keep generated outputs ignored
3. keep active source changes visible
4. only delete clearly accidental sync debris after confirmation

## 5. Recommended Next Action

The next low-risk repo-hygiene action is:

- classify the visible untracked source files into:
  - keep in repo
  - move to archive / legacy
  - delete if accidental

After that, `GL-03` can be considered truly complete.

## 6. Current Working Decision

As of `2026-03-13`, the working decision is:

- local generated outputs stay ignored
- active modified tracked scripts stay visible
- the visible untracked source files are treated as real project files, not noise

That means `GL-03` is operationally complete for go-live planning purposes.

The remaining step is release housekeeping:

- explicitly adopt the intended source files into the long-term git baseline in a
  later repo-cleanup pass
