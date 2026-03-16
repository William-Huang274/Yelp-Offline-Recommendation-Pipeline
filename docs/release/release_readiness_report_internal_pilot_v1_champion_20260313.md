# Release Readiness Report: internal_pilot_v1_champion_20260313

- generated_at: `2026-03-16 18:26:09 +0800`
- overall_status: `WARN`
- report_path: [release_readiness_report_internal_pilot_v1_champion_20260313.md](./release_readiness_report_internal_pilot_v1_champion_20260313.md)

## Summary

- champion_pointer: `stage11_release`
- aligned_fallback_pointer: `stage10_release`
- emergency_baseline_pointer: `stage09_release`
- source_run_09: `20260311_005450_full_stage09_candidate_fusion`
- release_contract: `bucket=10, eval_users=738, candidate_topn=250, top_k=10`
- check_counts: `PASS=57 WARN=2 FAIL=0`

## Key Metrics

- stage09 audit: `truth_in_all=0.947208` `truth_in_pretrim=0.882255` `hard_miss=0.052792`
- stage10 fallback: `LearnedBlendXGBCls@10` `recall@10=0.065041` `ndcg@10=0.029217`
- stage10 baseline: `PreScore@10` `recall@10=0.056911` `ndcg@10=0.026467`
- stage11 champion: `QLoRASidecar@10` `recall@10=0.06775067750677506` `ndcg@10=0.02993468209586097`

## Checks

| Level | Area | Message |
| --- | --- | --- |
| `PASS` | `pointers` | loaded production pointer: release_policy.json |
| `PASS` | `pointers` | loaded production pointer: stage09_release.json |
| `PASS` | `pointers` | loaded production pointer: stage10_release.json |
| `PASS` | `pointers` | loaded production pointer: stage11_release.json |
| `PASS` | `pointers` | stage09_release release_label matches policy |
| `PASS` | `pointers` | stage10_release release_label matches policy |
| `PASS` | `pointers` | stage11_release release_label matches policy |
| `PASS` | `stage09` | found path: stage09_release run_dir |
| `PASS` | `stage10` | found path: stage10_release run_dir |
| `PASS` | `stage11` | found path: stage11_release run_dir |
| `PASS` | `stage09` | stage09 candidate run validator passed |
| `PASS` | `stage11` | found path: stage11 dataset run |
| `PASS` | `stage11` | stage11 dataset run validator passed |
| `PASS` | `stage09` | found path: stage09 audit metrics file |
| `PASS` | `stage10` | found path: stage10 metrics file |
| `PASS` | `stage10` | found path: stage10 run_meta.json |
| `PASS` | `stage10` | found path: stage10 rank_model.json |
| `PASS` | `stage10` | found path: stage10 eval user cohort |
| `PASS` | `stage11` | found path: stage11 metrics file |
| `PASS` | `stage11` | found path: stage11 run_meta.json |
| `PASS` | `stage11` | found path: stage11 alpha sweep best file |
| `PASS` | `stage11` | found path: stage11 model run dir |
| `PASS` | `stage11` | found path: stage11 adapter dir |
| `PASS` | `stage09` | stage09 audit metrics match release source and bucket |
| `PASS` | `stage10` | stage10 run_meta source matches release contract |
| `PASS` | `stage10` | stage10 run_meta candidate topn matches release contract |
| `PASS` | `stage10` | stage10 run_meta top_k matches release contract |
| `PASS` | `stage10` | stage10 metrics source_run_09 matches release contract |
| `PASS` | `stage10` | stage10 selected model row matches release user count |
| `PASS` | `stage10` | stage10 metrics include PreScore@10 row |
| `PASS` | `stage10` | stage10 pointer recall matches metrics row |
| `PASS` | `stage11` | stage11 run_meta source matches release contract |
| `PASS` | `stage11` | stage11 run_meta rerank_topn matches release contract |
| `PASS` | `stage11` | stage11 run_meta top_k matches release contract |
| `WARN` | `stage11` | stage11 champion run_meta still records enforce_stage09_gate=false |
| `PASS` | `stage11` | stage11 metrics source_run_09 matches release contract |
| `PASS` | `stage11` | stage11 selected model row matches release user count |
| `PASS` | `stage11` | stage11 metrics include PreScore@10 row |
| `PASS` | `stage11` | stage11 pointer recall matches metrics row |
| `PASS` | `stage11` | alpha_sweep_best.json is readable |
| `PASS` | `ranking` | champion > fallback > PreScore ordering holds on frozen release metrics |
| `PASS` | `manifest` | located release manifest: release_manifest_internal_pilot_v1_champion_20260313.json |
| `WARN` | `manifest` | release manifest still marks production_ready=false |
| `PASS` | `ops` | rollback snapshot exists: rollback_snapshot_20260313_203053_internal_pilot_v1_champion_20260313.json |
| `PASS` | `ops` | release monitor report present: release_monitor_report_internal_pilot_v1_champion_20260313.md |
| `PASS` | `docs` | required doc present and readable: README.md |
| `PASS` | `docs` | required doc present and readable: go_live_readiness_20260312.md |
| `PASS` | `docs` | required doc present and readable: v1_freeze_20260313.md |
| `PASS` | `docs` | required doc present and readable: first_champion_closeout_20260313.md |
| `PASS` | `docs` | required doc present and readable: stage09_reaudit_20260313.md |
| `PASS` | `docs` | required doc present and readable: gl07_stage10_stage11_alignment_20260313.md |
| `PASS` | `docs` | required doc present and readable: gl08_prod_pointers_20260313.md |
| `PASS` | `docs` | required doc present and readable: gl09_path_unification_20260313.md |
| `PASS` | `docs` | required doc present and readable: gl10_smoke_tests_20260313.md |
| `PASS` | `docs` | required doc present and readable: gl12_batch_runner_20260313.md |
| `PASS` | `docs` | required doc present and readable: rollback_and_monitoring.md |
| `PASS` | `docs` | required doc present and readable: config_reference.md |
| `PASS` | `docs` | required doc present and readable: data_contract.md |
| `PASS` | `docs` | required doc present and readable: stage11_cloud_run_profile_20260309.md |
