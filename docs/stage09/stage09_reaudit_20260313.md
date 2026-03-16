# Stage09 Re-Audit Note (2026-03-13)

## Purpose

This note records the `GL-06` re-audit of the active cloud stage09 lineage.

Target source:

- `20260311_005450_full_stage09_candidate_fusion`

Audit script:

- [09_1_recall_audit.py](../../scripts/09_1_recall_audit.py)

Local audit output:

- [stage09_recall_audit_summary.csv](../../data/output/09_recall_audit/20260313_191007_stage09_recall_audit/stage09_recall_audit_summary.csv)
- [stage09_recall_audit.json](../../data/output/09_recall_audit/20260313_191007_stage09_recall_audit/stage09_recall_audit.json)

Latest metrics pointer updated by this run:

- [stage09_recall_audit_summary_latest.csv](../../data/metrics/stage09_recall_audit_summary_latest.csv)

## Execution

Run date:

- `2026-03-13`

Local execution profile:

- `SPARK_MASTER=local[2]`
- `SPARK_DRIVER_MEMORY=6g`
- `SPARK_EXECUTOR_MEMORY=6g`
- `SPARK_SQL_SHUFFLE_PARTITIONS=8`
- `SPARK_DEFAULT_PARALLELISM=8`
- bucket override: `10`

Input run:

- [20260311_005450_full_stage09_candidate_fusion](../../data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion)

## Result

Audited source run:

- `20260311_005450_full_stage09_candidate_fusion`

Audit run id:

- `20260313_191007`

Bucket 10 result:

- `n_users = 3618`
- `truth_in_all = 0.947208`
- `truth_in_pretrim = 0.882255`
- `pretrim_cut_loss = 0.064953`
- `hard_miss = 0.052792`
- `pretrim_file = candidates_pretrim150.parquet`
- `pretrim_top_k_used = 900`

## Comparison Against Previous Latest Audit

Previous latest metrics file before this re-audit:

- source run: `20260305_002746_full_stage09_candidate_fusion`
- run id: `20260305_003539`

Current re-audit metrics:

- source run: `20260311_005450_full_stage09_candidate_fusion`
- run id: `20260313_191007`

Observed comparison:

- `truth_in_all`: unchanged at `0.947208`
- `truth_in_pretrim`: unchanged at `0.882255`
- `hard_miss`: unchanged at `0.052792`

Conclusion:

- the current active cloud stage09 source now has a fresh local audit
- its top-line bucket10 recall metrics match the previous audited baseline
- from a recall-audit perspective, `20260311_005450` is safe to treat as the
  current audited stage09 mainline

## Decision

For all downstream release-alignment work after `2026-03-13`, use:

- `20260311_005450_full_stage09_candidate_fusion`

This resolves the earlier mismatch where:

- `stage11` was using the newer stage09 source
- but local latest recall audit still pointed to the older `20260305_002746` source

## What This Does Not Solve

This re-audit does not solve:

- `GL-07` shared release-cohort comparison between `stage10` and `stage11`
- final champion selection
- production pointer separation

It only closes the stage09 audit/source alignment problem.
