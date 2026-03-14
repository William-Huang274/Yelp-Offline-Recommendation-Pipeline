# Stage09 Bucket10 Recovery Plan (2026-03-13)

## Why This Plan Exists

The current bucket10 bottleneck is no longer a small scoring tweak problem.

Existing audits already show:

- `stage11` consumes `pre_rank` directly at `scripts/11_1_qlora_build_dataset.py:1031`
- current bucket10 `truth_in_top150` is only `301 / 738`
- low-risk layered replay only moves that to `305 / 738`
- linear and nonlinear audit probes on the frozen fused table both regress versus baseline

Relevant audit references:

- `docs/stage09_bucket10_prerank_audit_20260313.md`
- `docs/stage09_bucket10_prerank_followup_20260313.md`

This means the next loop must solve two things in order:

1. make stage09 reruns reproducible again on the remote machine
2. expose richer pre-rank evidence before changing mainline bucket10 logic

## Execution Stages

### Stage 1. Unblock Stage09 Rerun Preconditions

Goal:

- make remote stage09 reruns executable without relying on a broken historical profile-vector path

Current blocker:

- prior rerun attempt failed because `user_profile_vectors.npz` was missing under the resolved `09_user_profiles` run

Actions:

1. audit remote `09_user_profiles` run directories and metadata
2. confirm whether the failure came from:
   - a missing asset in the selected run
   - an incorrect explicit `USER_PROFILE_RUN_DIR`
   - or an unexpected root-path mismatch between repo and data disk
3. choose the safest recovery path:
   - point stage09 to an existing valid user-profile run if one already exists
   - otherwise rebuild `09_user_profile_build` once and pin the new run explicitly
4. document the pinned run path so later stage09 reruns do not depend on `pick_latest_run`
5. standardize recovery embedding on local `bge-m3` so rebuilt `user_profile_vectors.npz` does not depend on transient remote model availability

Exit criteria:

- one explicit remote `USER_PROFILE_RUN_DIR` is verified to contain:
  - `user_profile_vectors.npz`
  - `user_profiles.csv`
  - optional `user_profile_tag_profile_long.csv`
- a dry-run resolution check in `09_candidate_fusion.py` succeeds

### Stage 1A. `bge-m3` Recovery Runbook

Purpose:

- rebuild missing `user_profile_vectors.npz` for historical `09_user_profiles` runs without rerunning the full user-profile stage

Successful remote run on 2026-03-13:

- input run:
  - `/root/autodl-tmp/project_data/data/output/09_user_profiles/20260304_234037_full_stage09_user_profile_build`
- output run:
  - `/root/autodl-tmp/project_data/data/output/09_user_profiles/20260313_191253_full_full_stage09_user_profile_vector_recovery`
- local model path:
  - `/root/autodl-tmp/hf_models/BAAI__bge-m3`
- files written:
  - `user_profile_vectors.npz`
  - `vector_recovery_meta.json`
  - copied `user_profiles.csv`

Successful runtime parameters:

- `PROFILE_VECTOR_RECOVERY_MODEL_NAME=bge-m3`
- `BGE_LOCAL_MODEL_PATH=/root/autodl-tmp/hf_models/BAAI__bge-m3`
- `PROFILE_VECTOR_DEVICE=cuda`
- `PROFILE_VECTOR_RECOVERY_BATCH_SIZE=16`
- `PROFILE_VECTOR_RECOVERY_MAX_LENGTH=512`
- `PROFILE_VECTOR_RECOVERY_NORMALIZE=true`
- backend:
  - `transformers_mean_pool`

Observed output:

- `vectors_written=50653`
- `dim=1024`

Observed throughput probe on the same machine:

- CPU baseline:
  - `batch_size=4`
  - about `0.78 rows/s`
- CUDA baseline:
  - `batch_size=16`
  - about `34.15 rows/s`

Next-run recommendation:

- default to `PROFILE_VECTOR_DEVICE=cuda` whenever the GPU is idle
- start from `PROFILE_VECTOR_RECOVERY_BATCH_SIZE=64`
- if memory is still comfortably below about `12g`, probe `96` then `128`
- if CUDA OOM appears, fall back to `32`
- do not share the GPU with active stage11 evaluation or training runs

### Stage 2. Build Bucket10 Enriched Audit Export

Goal:

- create a bucket10-only audit export from stage09 that preserves raw route evidence before final pretrim/head ordering

Why:

- the frozen fused table currently exposes mostly route membership and rank-order summaries
- probe results show that changing the scorer over the current column set is not enough

Required new audit fields:

- raw ALS score or calibrated ALS percentile
- raw cluster similarity / route score
- raw profile route score
- route confidence after thresholding
- per-route rank percentile
- profile bridge / shared-tag route evidence when present
- source-gap features such as:
  - best nonpopular route vs ALS
  - profile-vs-ALS gap
  - cluster-vs-ALS gap
- pretrim-drop reason markers for candidates lost before head cut

Implementation rule:

- this export is audit-only first
- do not change label definition, candidate boundary, split logic, or metric definition
- do not add heavy Spark actions inside loops

Exit criteria:

- one bucket10 export can be produced from a fixed stage09 run
- export is sufficient to rerun cohort-level pre-rank ceiling probes without re-deriving features outside stage09

### Stage 3. Gate Before Mainline Fusion Changes

Only after Stage 2 is complete should we test a richer bucket10 scorer.

Go / no-go gate:

- cohort `truth_in_top150 >= 320`
- heavy-user `truth_in_top150 >= 110`
- no drop in `truth_in_pretrim`

If that gate fails, stop score work and move upstream to route generation quality.

### Stage 4. Structural Bucket10 Fusion v2

Only if the gate passes:

- add heavy-user-specific head lanes
- reduce the global ALS-heavy front guard
- reserve explicit head capacity for profile/cluster-backed candidates
- rank inside each lane using enriched raw route evidence, not only fused rank summaries

### Stage 5. Exact Replay And Downstream Validation

Validation order:

1. bucket10 exact replay parity
2. bucket10 truth-in-pool / truth-in-top150 / truth-in-top250 audit
3. stage11 `top150` evaluation
4. optional `top250` evaluation if bucket10 pool expansion still matters

## Working Rules

- CPU-only until a real bucket10 ceiling improvement is shown
- do not touch the running GPU stage11 task
- prefer explicit pinned run directories over `latest` auto-resolution for critical upstream assets
- no production score rewrite before enriched audit export proves there is real upside

## Current Status

Started:

- Stage 1 remote rerun-blocker investigation
- Stage 2 local implementation of bucket10 enriched audit export path
- profile-vector recovery standardized on `bge-m3`
- recovered profile-vector run completed at:
  - `/root/autodl-tmp/project_data/data/output/09_user_profiles/20260313_191253_full_full_stage09_user_profile_vector_recovery`
- bucket10 enriched audit export auto-started from:
  - `/root/autodl-tmp/stage09_fs/20260313_191907_full_stage09_candidate_fusion`

Not started:

- structural bucket10 fusion v2
- downstream stage11 validation on new stage09 output

## 2026-03-13 Route Restore Run

Goal:

- restore bucket10 profile recall routes `vector/shared/bridge_user/bridge_type`
- rerun bucket10 fusion with the restored upstream tag-long assets
- measure whether route restoration alone materially improves `truth_in_top150`

Code / runtime notes:

- `09_user_profile_build.py` and `09_item_semantic_build.py` now accept env-driven Spark and GPU settings
- both scripts now prefer `sentence-transformers` but safely fall back to `transformers_mean_pool`
- on this host we intentionally kept the base Python env unchanged because it already had `transformers=5.2.0`; runtime used the fallback backend with the same local `bge-m3` weights

Completed upstream runs:

- user profile build:
  - `/root/autodl-tmp/project_data/data/output/09_user_profiles/20260313_195212_full_stage09_user_profile_build`
  - files confirmed:
    - `user_profile_vectors.npz`
    - `user_profile_tag_profile_long.csv`
  - runtime backend:
    - `transformers_mean_pool`
    - `device=cuda`
    - `batch_size=64`

- item semantic build:
  - `/root/autodl-tmp/project_data/data/output/09_item_semantics/20260313_195956_full_stage09_item_semantic_build`
  - files confirmed:
    - `item_semantic_features.csv`
    - `item_tag_profile_long.csv`
  - runtime backend:
    - `transformers_mean_pool`
    - `device=cuda`
    - `batch_size=128`

Completed stage09 run:

- fusion run:
  - `/root/autodl-tmp/stage09_fs/20260313_200615_full_stage09_candidate_fusion`
- recall audit:
  - `/root/autodl-tmp/stage09_fs/audits/20260313_201040_stage09_recall_audit`
- truth topn audit:
  - `/root/autodl-tmp/stage09_fs/audits/20260313_201056_bucket10_route_restore_truth_topn`

Route restore verification:

- `profile_recall_enabled_routes = ['vector', 'shared', 'bridge_user', 'bridge_type']`
- `profile_recall_rows_total = 1183200`
- `profile_recall_rows_shared = 394877`
- `profile_recall_rows_bridge_user = 645868`
- `profile_recall_rows_bridge_type = 463633`

Bucket10 result on `2997` users:

- `truth_in_all = 2603 / 2997 = 0.8685`
- `truth_in_pretrim = 2300 / 2997 = 0.7674`
- `truth_in_top150 = 1242 / 2997 = 0.4144`
- `truth_in_top250 = 1564 / 2997 = 0.5219`

Direct comparison vs the route-missing run `/root/autodl-tmp/stage09_fs/20260313_191907_full_stage09_candidate_fusion`:

- `truth_in_all: 2598 -> 2603`
- `truth_in_pretrim: 2304 -> 2300`
- `truth_in_top150: 1245 -> 1242`
- `truth_in_top250: 1561 -> 1564`

Interpretation:

- restoring the four profile routes worked technically
- with the current `coverage_stage2` pool shape, route restoration alone does not fix the bucket10 head problem
- the top150 bottleneck remains mainly a pre-rank / head-structure issue, not simply “profile routes were missing”
