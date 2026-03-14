# First Champion Closeout (2026-03-13)

## Version

Release label:

- `internal_pilot_v1_champion_20260313`

This is the first frozen champion for internal pilot closeout.

It is not a production-ready release.

## Champion Decision

Frozen champion path:

- `stage11 QLoRASidecar@10`

Aligned fallback path:

- `stage10 LearnedBlendXGBCls@10`

Emergency baseline:

- `PreScore@10`

This ranking is frozen using the aligned contract established in `GL-06` and `GL-07`.

## Frozen Contract

- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- bucket = `10`
- eval users = `738`
- candidate contract = `pre_rank <= 250`
- metric = `@10`

## Why This Champion Won

Under the aligned contract:

- `PreScore@10`
  - recall `0.056911`
  - ndcg `0.026467`
- `stage10 LearnedBlendXGBCls@10`
  - recall `0.065041`
  - ndcg `0.029217`
- `stage11 QLoRASidecar@10`
  - recall `0.06775067750677506`
  - ndcg `0.02993468209586097`

Ordering:

1. `stage11 QLoRASidecar@10`
2. `stage10 LearnedBlendXGBCls@10`
3. `PreScore@10`

Champion edge over aligned stage10 fallback:

- recall `+0.00270967750677506`
- ndcg `+0.00071768209586097`

## Exact Frozen Lineage

Stage09 audited source:

- [20260311_005450_full_stage09_candidate_fusion](../data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion)
- re-audit note: [stage09_reaudit_20260313.md](./stage09_reaudit_20260313.md)

Stage11 champion lineage:

- [20260311_011112_stage11_1_qlora_build_dataset](../data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset)
- [20260312_221630_stage11_2_dpo_train_9b_v3full_ckpt485_eval_run](../data/output/11_qlora_models/20260312_221630_stage11_2_dpo_train_9b_v3full_ckpt485_eval_run)
- [20260313_151256_stage11_3_qlora_sidecar_eval](../data/output/11_qlora_sidecar_eval/20260313_151256_stage11_3_qlora_sidecar_eval)

Stage10 aligned fallback lineage:

- model source: [20260307_210530_stage10_1_rank_train](../data/output/10_rank_models/20260307_210530_stage10_1_rank_train)
- aligned eval meta: [run_meta.json](../data/output/10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval/run_meta.json)

## Closeout Artifacts

Machine-readable champion manifest:

- [first_champion_manifest.json](../data/output/_first_champion_freeze_20260313/first_champion_manifest.json)

Frozen snapshots:

- [stage09_recall_audit_summary_latest.csv](../data/output/_first_champion_freeze_20260313/snapshots/metrics/stage09_recall_audit_summary_latest.csv)
- [recsys_stage10_results_gl07.csv](../data/output/_first_champion_freeze_20260313/snapshots/metrics/recsys_stage10_results_gl07.csv)
- [qlora_sidecar_metrics.csv](../data/output/_first_champion_freeze_20260313/snapshots/metrics/qlora_sidecar_metrics.csv)

Supporting notes:

- [v1_freeze_20260313.md](./v1_freeze_20260313.md)
- [gl07_stage10_stage11_alignment_20260313.md](./gl07_stage10_stage11_alignment_20260313.md)

## What This Freeze Means

This freeze means:

- the first champion path is now explicit
- the aligned fallback path is now explicit
- the version can be discussed, handed over, and reviewed without ambiguity

This freeze does not mean:

- production-ready rollout
- monitoring complete
- rollback complete
- pointer separation complete
- credential hygiene complete

## Next Action After Closeout

The next repair item after this closeout is:

- `GL-08` production pointer separation

After that:

- `GL-09` path cleanup
- `GL-10` smoke tests
- `GL-11` release validation
- `GL-12` batch runner
- `GL-13` rollback and monitoring
- `GL-14` data contract
