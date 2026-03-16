# Go-Live Readiness Plan (2026-03-12)

## 0. Status Update (2026-03-13)

To preserve the latest successful cloud lineage before further repairs, a `v1` local freeze
was completed on `2026-03-13`.

After `GL-06` and `GL-07` were completed on the same day, the first internal-pilot
champion freeze was also completed.

Reference files:

- `docs/release/v1_freeze_20260313.md`
- `data/output/_v1_freeze_20260313/v1_freeze_manifest.json`
- `docs/release/first_champion_closeout_20260313.md`
- `data/output/_first_champion_freeze_20260313/first_champion_manifest.json`

Important note:

- this freeze was executed before `GL-01` credential rotation because preserving the current
  cloud lineage had higher immediate priority
- treat `GL-02` as completed for the current frozen lineage
- `GL-01` remains the next required repair item
- `GL-14` was completed on `2026-03-13` with `docs/contracts/data_contract.md`
- for the current temporary cloud machine, `GL-01` remains an accepted but explicit risk
## 1. Scope

This document is the execution checklist for bringing the current `01stage -> 11stage`
project to an internal go-live state.

It is written for the current repository and the current cloud layout:

- Local repo root: `D:\5006_BDA_project`
- Cloud repo root: `/root/5006_BDA_project`
- Cloud active data roots:
  - `/root/autodl-tmp/stage09_fs`
  - `/root/autodl-tmp/stage11_fs`

This plan is for batch/offline recommendation go-live, not a real-time API service.

## 2. Current Judgment

### 2.1 What is already true

- `01-08` stages are complete enough to support project delivery and methodology explanation.
- `09` stage recall is the strongest and most stable part of the current mainline.
- `10` stage aligned XGB rerank now beats `PreScore` under the `GL-07` contract.
- `11` stage now holds the first frozen internal-pilot champion path.

### 2.2 What is not yet true

- The project is not yet ready for a stable internal go-live.
- The project is not yet ready for production-style handoff.
- The repository is not yet in a freezeable release state.

## 3. Current Evidence Snapshot

### 3.1 Stage09

Current strongest audited recall evidence in repo metrics:

- `data/metrics/stage09_recall_audit_summary_latest.csv`
- current recorded source run: `20260311_005450_full_stage09_candidate_fusion`
- bucket10:
  - `truth_in_all = 0.947208`
  - `truth_in_pretrim = 0.882255`
  - `pretrim_cut_loss = 0.064953`
  - `hard_miss = 0.052792`

### 3.2 Stage10

Current latest aligned rerank evidence:

- `data/metrics/recsys_stage10_results_gl07.csv`
- aligned run: `20260313_193213`
- current result:
  - `PreScore@10 recall = 0.056911`
  - `LearnedBlendXGBCls@10 recall = 0.065041`

Conclusion:

- aligned stage10 XGB is a valid fallback path
- stage10 is no longer behind `PreScore` under the aligned `GL-07` contract
- stage10 is still behind the current stage11 champion

### 3.3 Stage11

Current frozen champion evidence:

- pinned `11_1`: `/root/autodl-tmp/stage11_fs/input11data/20260311_011112_stage11_1_qlora_build_dataset`
- pinned `11_2` DPO eval run: `/root/autodl-tmp/stage11_fs/models/20260312_221630_stage11_2_dpo_train_9b_v3full_ckpt485_eval_run`
- pinned `11_3`: `/root/autodl-tmp/stage11_fs/sidecar_eval/20260313_151256_stage11_3_qlora_sidecar_eval`
- metrics:
  - `PreScore@10 recall = 0.056910569105691054`
  - `QLoRASidecar@10 recall = 0.06775067750677506`

Current conclusion:

- stage11 is the strongest current path under the aligned release contract
- stage11 is now frozen as the first internal-pilot champion
- stage10 aligned XGB remains the immediate fallback

## 4. Temporary Release Policy

Current frozen champion policy:

- champion path: `stage11 QLoRASidecar@10`
- aligned fallback path: `stage10 LearnedBlendXGBCls@10`
- emergency baseline path: `PreScore@10`

This means:

1. Use the frozen champion path for version closeout discussion and internal pilot planning.
2. Do not treat this as production-ready until `GL-08` and later operational items are completed.
3. Keep fallback and emergency-baseline paths explicit in every release note.

## 5. Release Target Definition

The project should pass three levels in order:

### Level A: Project Delivery Ready

- Can explain all stages.
- Has one top-level README.
- Has one clear run map from raw data to final metrics.

### Level B: Internal Go-Live Ready

- Has one frozen champion path.
- Has aligned source run IDs for recall, rank, and evaluation.
- Has reproducible config and no critical path ambiguity.
- Has smoke tests, release manifest, and rollback pointer.

### Level C: Production Ready

- Has monitoring, rollback, data contract, and scheduled pipeline runner.
- Has credential hygiene and least-privilege access.
- Has service-level operational ownership.

Current status:

- Level A: complete
- Level B: complete for internal pilot
- Level C: partially complete, not ready

## 6. Repair Order

Follow the items below in strict order.

### GL-01 Rotate Cloud Credentials

Priority: `P0`

Problem:

- Cloud access has been used in a shared/manual way.
- Current collaboration flow has exposed direct root credentials.

Why this blocks go-live:

- This is an immediate security and ownership risk.

How to repair:

1. Create a non-root cloud user for project operations.
2. Switch from password login to SSH key login.
3. Disable direct root login for normal project operation.
4. Move all future training and eval runs to a named service account or dedicated operator account.
5. Store the new access method outside repo files.

Output:

- one new SSH key-based operator account
- one short runbook with login instructions

Done when:

- root password is rotated
- normal project operations no longer require root login

### GL-02 Freeze a Safe Baseline Release

Priority: `P0`

Problem:

- There is no single frozen release baseline.
- repo state is dirty
- run lineage is split across local and cloud

Why this blocks go-live:

- No one can tell which code version and which artifact version are the actual release.

How to repair:

1. Create a release name, for example:
   - `internal_pilot_v1_20260312`
2. Freeze one temporary baseline:
   - source recall baseline: `20260305_002746_full_stage09_candidate_fusion`
   - ranking baseline: `PreScore@10`
   - stage11: disabled by default
3. Record the baseline in a release note file:
   - `docs/release_internal_pilot_v1_20260312.md`
4. Record exact source runs and metrics in a release manifest JSON:
   - suggested path: `data/output/_release_manifests/internal_pilot_v1_20260312.json`
5. Commit the repository state used for this release note.
6. Tag the commit with a non-ambiguous tag name.

Output:

- one release note
- one release manifest
- one git tag

Done when:

- a new teammate can identify exactly which code and which data artifacts define the release

### GL-03 Repair Repository Hygiene

Priority: `P0`

Status update on `2026-03-13`:

- root `README.md` has now been added
- root `.gitignore` has now been added
- remaining work under this item is to classify the current dirty worktree into:
  - real source files to keep
  - generated local artifacts to ignore
  - temporary sync debris to delete after confirmation
- current classification note:
  - `docs/repo/repo_hygiene_inventory_20260313.md`

Problem:

- root `README.md` is missing
- `.gitignore` is missing
- many core files are untracked or modified

Why this blocks go-live:

- The repository is not handoff-safe.

How to repair:

1. Add root `README.md`.
2. Add `.gitignore` covering at least:
   - `data/`
   - `.venv-wsl/`
   - `__pycache__/`
   - temporary logs
   - local spark temp
3. Review `git status`.
4. Separate:
   - files that are real source code
   - files that are local outputs
   - files that are cloud sync copies
5. Commit source/config/docs only.

Output:

- clean root README
- clean `.gitignore`
- reduced noisy git status

Done when:

- `git status` shows only intentional source changes

### GL-04 Repair Broken Documentation

Priority: `P0`

Status update on `2026-03-13`:

- a repo scan found no remaining zero-byte `.md` files in the current working tree
- the structural part of this repair item is effectively cleared
- remaining work is content hardening when specific docs are revised or newly referenced

Problem:

- several critical md files are zero-byte placeholders locally

Affected files confirmed in local repo:

- `docs/labeling/labeling_manual_v1.md`
- `docs/labeling/labeling_manual_v1_1.md`
- `docs/stage10/stage09_stage10_three_step_report_20260220.md`
- `docs/stage10/stage10_calibration_plan.md`
- `docs/stage10/stage10_pretrain_audit_20260226.md`
- `docs/stage10/stage10_pretrain_audit_20260226_1.md`
- `docs/stage11/stage11_execution_plan_20260227.md`
- `docs/stage11/stage11_qlora_prerequisites.md`

Cloud note:

- current cloud `docs/` only contains:
  - `stage11_cloud_run_profile_20260309.md`
  - `stage11_data_text_feature_guideline_20260308.md`
- so cloud cannot fully restore the zero-byte files

Why this blocks go-live:

- Key design decisions and operating assumptions are currently not recoverable from repo docs.

How to repair:

1. For each zero-byte file, try recovery in this order:
   - previous git history
   - other local backups
   - other synced machine copies
   - cloud copies if available
   - manual rewrite from surviving scripts, metrics, and run_meta
2. If original content is unavailable, replace with a reconstructed doc that clearly states:
   - `reconstructed on 2026-03-12`
   - evidence source
   - what was inferred vs what was directly observed
3. Do not leave any zero-byte md file in the final release branch.

Output:

- all critical docs readable
- no zero-byte md files

Done when:

- every md referenced in README opens with readable content

### GL-05 Write the Top-Level Project README

Priority: `P0`

Status update on `2026-03-13`:

- root `README.md` now exists
- the repo now has one top-level entry point for project scope, stage map, active scripts,
  v1 freeze status, and the current repair order
- future revisions should keep that README aligned with the next frozen champion lineage

Problem:

- the project has stage docs but no single top-level entry point

Why this blocks go-live:

- Handoff and onboarding cost is too high.

How to repair:

1. Create root `README.md` with these sections:
   - project goal
   - pipeline overview
   - stage map
   - local vs cloud roles
   - current champion path
   - current experimental path
   - how to reproduce metrics
   - known limitations
2. Link the active runbook docs, especially:
   - `docs/release/go_live_readiness_20260312.md`
   - `docs/stage11/stage11_cloud_run_profile_20260309.md`
   - `scripts/07_STAGE_GUIDE.md`
3. Explicitly state that `09-11` newest artifacts may live on cloud storage, not in the repo working tree.

Output:

- one root README that a new reader can start from

Done when:

- a new reader can identify the active pipeline in under 10 minutes

### GL-06 Re-Audit the Current Cloud Stage09 Mainline

Priority: `P0`

Status update on `2026-03-13`:

- local re-audit completed for `20260311_005450_full_stage09_candidate_fusion`
- output run: `data/output/09_recall_audit/20260313_191007_stage09_recall_audit`
- latest metrics file now points to source `20260311_005450_full_stage09_candidate_fusion`
- top-line bucket10 metrics match the old audited baseline exactly
- result note:
  - `docs/stage09/stage09_reaudit_20260313.md`

Problem:

- latest cloud stage09 mainline is `20260311_005450_full_stage09_candidate_fusion`
- current repo gate summary still points to the older `20260305_002746` source
- latest `11_1` and `11_3` used `enforce_stage09_gate=false`

Why this blocks go-live:

- the active cloud lineage is not aligned with the auditable gate summary

How to repair:

1. Run `09_1_recall_audit.py` against:
   - `/root/autodl-tmp/stage09_fs/20260311_005450_full_stage09_candidate_fusion`
2. Write updated outputs to:
   - `data/output/09_recall_audit/...`
   - `data/metrics/stage09_recall_audit_summary_latest.csv`
3. Compare the new summary against the older `20260305_002746` audited line.
4. Decide one of two outcomes:
   - keep `20260311_005450` as the new audited mainline
   - revert to `20260305_002746` as the release source
5. Do not keep a state where the active stage11 lineage depends on an unaudited stage09 source.

Output:

- one fresh stage09 audit for the active cloud source
- one explicit decision on which stage09 source is release-safe

Done when:

- stage09 release source and stage09 gate summary point to the same source run

### GL-07 Re-Align Stage10 and Stage11 Evaluation on the Same Source Run

Priority: `P0`

Status update on `2026-03-13`:

- first aligned comparison run completed using the `stage11` eval contract
- aligned stage10 artifacts:
  - `data/metrics/recsys_stage10_results_gl07.csv`
  - `data/output/10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval/run_meta.json`
- aligned contract:
  - source run `20260311_005450_full_stage09_candidate_fusion`
  - eval users `738`
  - candidate topn `250`
- aligned ordering is now:
  - `stage11 QLoRASidecar@10`
  - `stage10 LearnedBlendXGBCls@10`
  - `PreScore@10`
- result note:
  - `docs/repo/gl07_stage10_stage11_alignment_20260313.md`

Problem:

- stage09, stage10, and stage11 comparisons are not fully aligned on one source lineage
- stage10 `XGB/LR` and stage11 `SFT/DPO` are different model families and do not share the same training-set construction

Clarification:

- `GL-07` means align the **evaluation contract**, not force the same **training split**.
- For DPO, it is acceptable that `11_1` uses its own train/eval split for model construction and checkpoint selection.
- For release comparison, stage10 and stage11 only need to share:
  - the same `source_run_09`
  - the same candidate pool semantics
  - the same eval user cohort
  - the same `top_k` and rerank window
- They do **not** need to share the same training rows or label format.

Why this blocks go-live:

- champion selection is not trustworthy unless all candidates are compared on the same source run

How to repair:

1. Pick the release source run from `GL-06`.
2. Re-run or re-score:
   - stage10 `PreScore`
   - stage10 learned rerank
   - stage11 SFT sidecar
   - stage11 DPO sidecar
3. Use one evaluation profile for release comparison:
   - same bucket
   - same user cohort
   - same top-k
   - same rerank candidate depth
4. Keep two metric views separate:
   - `dev metric`: stage11 internal train/eval split used for SFT/DPO development
   - `release metric`: common cross-model comparison cohort used to compare `PreScore`, `XGB`, `SFT`, `DPO`
5. Preferred comparison rule:
   - do not compare stage10 against stage11 on `11_1` internal eval split unless stage10 is explicitly rescored on the same user cohort
6. Recommended implementation options:
   - Option A: use one shared explicit release cohort for both stage10 and stage11
   - Option B: export stage11 eval users from `11_1` and score stage10 on the same users
7. Write a comparison table:
   - suggested path: `docs/release_model_comparison_20260312.md`
8. Freeze one champion.

Recommended freeze rule:

- if learned ranker or sidecar does not beat `PreScore` with repeatable gain, keep `PreScore` as champion

Output:

- one aligned comparison table
- one explicit champion decision

Done when:

- there is a single answer to "what model/path goes live first"
- the answer is based on a shared evaluation cohort, not mixed train/eval conventions

### GL-08 Separate Production Pointers from Experiment Pointers

Priority: `P1`

Status update on `2026-03-13`:

- release pointer namespace created under `data/output/_prod_runs`
- first release pointer set written:
  - `stage09_release.json`
  - `stage10_release.json`
  - `stage11_release.json`
  - `release_policy.json`
- helper support added in `scripts/pipeline/project_paths.py`
- result note:
  - `docs/repo/gl08_prod_pointers_20260313.md`

Problem:

- current `_latest_runs` is used as an experiment convenience layer
- current `stage11_3` pointer can point to smoke audit output

Why this blocks go-live:

- production cannot depend on "latest by experiment"

How to repair:

1. Keep existing `_latest_runs` for experiment use only.
2. Add a second pointer namespace, for example:
   - `data/output/_prod_runs/stage09_release.json`
   - `data/output/_prod_runs/stage10_release.json`
   - `data/output/_prod_runs/stage11_release.json`
3. Only update `_prod_runs` after release approval.
4. Make runner scripts accept:
   - explicit input path
   - prod pointer
   - experimental pointer
5. Never let smoke or partial runs overwrite prod pointers.

Output:

- clear separation between experiment and release lineage

Done when:

- no production entry point reads a smoke pointer

### GL-09 Unify Config and Remove Hardcoded Paths

Priority: `P1`

Status update on `2026-03-13`:

- active release-path scripts now use `scripts/pipeline/project_paths.py`
- `GL-09` patch closed for:
  - `scripts/10_1_rank_train.py`
  - `scripts/10_5_transformer_rerank_eval.py`
  - `scripts/07_relabel_then_cluster.py`
- `env_or_project_path(...)` now auto-normalizes legacy repo-root strings
- config reference added:
  - `docs/contracts/config_reference.md`
- result note:
  - `docs/repo/gl09_path_unification_20260313.md`

Problem:

- `09` and `11` are partly path-abstracted
- `07` and most of `10` still contain hardcoded `D:/5006 BDA project/...` paths

Examples:

- `scripts/07_relabel_then_cluster.py`
- `scripts/10_1_rank_train.py`
- `scripts/10_2_rank_infer_eval.py`
- `scripts/10_5_transformer_rerank_eval.py`

Why this blocks go-live:

- the pipeline is not portable across local and cloud

How to repair:

1. Standardize all stage scripts on `scripts/pipeline/project_paths.py`.
2. Replace direct absolute roots with:
   - `env_or_project_path(...)`
   - `project_path(...)`
3. Add env var names for each important root path.
4. Keep defaults conservative for local execution.
5. Document all required env vars in one place:
   - suggested file: `docs/contracts/config_reference.md`

Output:

- no hardcoded project root in active critical path scripts

Done when:

- local and cloud runs use the same path resolution model

### GL-10 Add Minimal Smoke Tests

Priority: `P1`

Status update on `2026-03-13`:

- first pytest smoke suite added under `tests/`
- reusable stage artifact validators added:
  - `scripts/pipeline/run_validators.py`
  - `tools/validate_stage_artifact.py`
- current run status:
  - `python -m pytest tests -q` -> `9 passed`
  - local frozen `stage09` validator -> `PASS`
  - local frozen `stage11` dataset validator -> `PASS`
- result note:
  - `docs/repo/gl10_smoke_tests_20260313.md`

Problem:

- there is almost no formal test coverage

Why this blocks go-live:

- breakage will be discovered only at full run time

How to repair:

1. Add a `tests/` directory.
2. Add at least these tests:
   - `project_paths` resolution
   - latest/prod pointer read and write
   - `run_meta.json` required field validation
   - stage09 candidate directory shape validation
   - stage11 dataset directory shape validation
3. Add one tiny smoke fixture for path resolution and pointer logic.
4. Add one script that validates a candidate run without doing full training.

Suggested minimal command:

- `python -m pytest tests -q`

Output:

- one minimal test suite that runs quickly

Done when:

- critical non-training logic is covered by fast checks

### GL-11 Add Release Validation Scripts

Priority: `P1`

Status update on `2026-03-13`:

- release validation script added:
  - `tools/check_release_readiness.py`
- first report artifact generated:
  - `docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md`
- current report result:
  - overall `WARN`
  - `PASS=52 WARN=2 FAIL=0`
- current warning reasons:
  - `stage11` champion run still records `enforce_stage09_gate=false`
  - release manifest still records `production_ready=false`

Problem:

- release checks are currently manual and scattered

Why this blocks go-live:

- human error is too easy

How to repair:

1. Add a validation script that checks:
   - required files exist
   - pointer targets exist
   - metrics files match the source run
   - champion run has evaluation results
   - no zero-byte docs in required doc list
2. Suggested script:
   - `tools/check_release_readiness.py`
3. Make the script emit:
   - PASS
   - WARN
   - FAIL
4. Save its output as a release artifact:
   - `docs/release_readiness_report_<release_id>.md`

Output:

- one repeatable release-readiness report

Current output path:

- `docs/release/release_readiness_report_internal_pilot_v1_champion_20260313.md`

Done when:

- release readiness can be checked with one command

### GL-12 Add One-Click Batch Runner

Priority: `P2`

Status update on `2026-03-13`:

- operator runner added:
  - `scripts/pipeline/internal_pilot_runner.py`
- wrappers added:
  - `scripts/run_internal_pilot.bat`
  - `scripts/run_internal_pilot.sh`
- supported modes now include:
  - `validate`
  - `recall`
  - `rank`
  - `eval`
  - `publish`
- publish now writes:
  - `data/output/_prod_runs/release_manifest_internal_pilot_v1_champion_20260313.json`
- validation run completed for:
  - `--mode validate`
  - `--mode publish`
  - wrapper `--mode validate`
  - dry-run `recall/rank/eval`
- result note:
  - `docs/repo/gl12_batch_runner_20260313.md`

Problem:

- the pipeline is a set of scripts, not a single operable batch entry point

Why this blocks production-style operation:

- operators need one consistent entry path

How to repair:

1. Add a runner for internal pilot mode.
2. Suggested entry forms:
   - `scripts/run_internal_pilot.bat`
   - `scripts/run_internal_pilot.sh`
   - or one Python runner under `scripts/pipeline/`
3. Make the runner support:
   - `--mode validate`
   - `--mode recall`
   - `--mode rank`
   - `--mode eval`
   - `--mode publish`
4. For first version, `publish` can simply update prod pointers and write a release manifest.

Output:

- one standard operator path for batch use

Done when:

- a release can be validated and published without manual path editing

### GL-13 Add Rollback and Monitoring

Priority: `P2`

Status update on `2026-03-13`:

- rollback and monitor contract added:
  - `docs/release/rollback_and_monitoring.md`
- monitor script added:
  - `tools/check_release_monitoring.py`
- runner extended with:
  - `--mode monitor`
  - `--mode rollback`
- publish now auto-writes rollback snapshots under:
  - `data/output/_prod_runs/rollback_snapshot_*.json`
- rollback writes audit records under:
  - `data/output/_prod_runs/rollback_applied_*.json`
- current monitor report:
  - `docs/release/release_monitor_report_internal_pilot_v1_champion_20260313.md`
- result note:
  - `docs/release/rollback_and_monitoring.md`
- validation run completed for:
  - `--mode publish` with snapshot creation
  - `--mode monitor`
  - `--mode rollback` from a recorded snapshot

Problem:

- there is no operational rollback contract
- there is no monitoring contract

Why this blocks production-style go-live:

- failures cannot be contained safely

How to repair:

1. Define rollback to previous prod pointer.
2. Store previous release manifest permanently.
3. Monitor at least:
   - candidate row count
   - users evaluated
   - truth-in-pretrim
   - hard miss
   - champion metric drift
   - training failures
4. Write one short ops doc:
   - `docs/release/rollback_and_monitoring.md`

Output:

- one rollback rule
- one monitoring rule set

Done when:

- the team can answer "what do we do if tonight's release regresses"

### GL-14 Add Data Contract and Asset Inventory

Priority: `P2`

Status update on `2026-03-13`:

- `docs/contracts/data_contract.md` now exists as the canonical artifact inventory
- the current frozen release lineage from `stage08` through `_prod_runs` is documented
- the `stage07` local relabel corruption caveat is documented, and the stable downstream
  business-profile contract is anchored at `stage08`
- this item is complete for `internal_pilot_v1_champion_20260313`

Problem:

- artifacts live across local repo, cloud repo, and cloud storage roots
- data contracts are implicit in scripts

Why this blocks long-term maintainability:

- future modifications will create silent incompatibilities

How to repair:

1. Write a data contract doc with:
   - raw sources
   - parquet tables
   - stage07/08 label assets
   - stage09 candidate files
   - stage10 model outputs
   - stage11 adapters and eval outputs
2. For each artifact type, define:
   - path pattern
   - required files
   - required columns
   - producer stage
   - consumer stage
3. Suggested path:
   - `docs/contracts/data_contract.md`

Output:

- one artifact inventory and contract

Done when:

- every critical stage input/output is documented

## 7. Immediate Execution Sequence

Use this sequence exactly.

1. `GL-01` rotate cloud credentials.
2. `GL-02` freeze temporary safe baseline release.
3. `GL-03` repair repo hygiene.
4. `GL-04` repair broken docs.
5. `GL-05` write root README.
6. `GL-06` re-audit active cloud stage09 source.
7. `GL-07` re-align stage10 and stage11 comparisons on the same source run.
8. Freeze the first true internal-go-live champion.
9. `GL-08` add production pointers.
10. `GL-09` unify config and remove hardcoded paths.
11. `GL-10` add smoke tests.
12. `GL-11` add release validation script.
13. `GL-12` add one-click batch runner.
14. `GL-13` add rollback and monitoring.
15. `GL-14` add data contract and asset inventory.

Status on `2026-03-13`:

- `GL-02` through `GL-14` are complete for `internal_pilot_v1_champion_20260313`
- `GL-01` remains open by user choice because the current cloud machine is temporary

## 8. First Champion Freeze

After `GL-06` and `GL-07`, the first frozen internal-pilot champion is:

- use `stage09` audited source lineage
- use `stage11 QLoRASidecar@10` as champion ranking path
- keep `stage10 LearnedBlendXGBCls@10` as aligned fallback
- keep `PreScore@10` as emergency baseline

Reason:

- the active stage09 source is now re-audited
- stage10 has been rescored on the same release contract as stage11
- stage11 still ranks first under the aligned comparison

## 9. Acceptance Checklist for Internal Go-Live

Do not declare internal go-live ready until all items below are true:

- one release note exists
- one release manifest exists
- one champion path is frozen
- stage09 audit source equals release source
- stage10/stage11 comparisons are aligned to the same source run
- no required docs are zero-byte
- root README exists
- `.gitignore` exists
- prod pointers exist and are separated from experimental pointers
- smoke tests pass
- release-readiness script passes

## 10. Acceptance Checklist for Production Readiness

Do not declare production ready until all items below are true:

- internal go-live checklist passes
- rollback flow is documented and tested
- monitoring checks are in place
- access is least-privilege
- batch runner exists
- data contract exists

## 11. How to Use This Document

For every future repair cycle:

1. pick the next unchecked `GL-*` item in order
2. implement the repair
3. write the concrete output artifact named in the item
4. mark the item done in the next revision of this file
5. do not skip forward unless the current item is blocked

If an item is blocked:

- record the blocker directly under the item
- record the decision to bypass it
- record the risk created by bypassing it

## 12. Revision Note

This file was created on `2026-03-12` based on:

- local repository inspection
- local metrics and output artifacts
- cloud read-only inspection of current `stage09_fs` and `stage11_fs` runs

It should be updated after every major repair milestone.
