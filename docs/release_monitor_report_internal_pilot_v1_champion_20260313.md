# Release Monitor Report: internal_pilot_v1_champion_20260313

- generated_at: `2026-03-13 21:06:46 +0800`
- overall_status: `WARN`
- report_path: [release_monitor_report_internal_pilot_v1_champion_20260313.md](./docs/release_monitor_report_internal_pilot_v1_champion_20260313.md)

## Summary

- truth_in_pretrim: `0.882255`
- hard_miss: `0.052792`
- users_evaluated: `738`
- candidate_rows_expected: `184500`
- stage11_recall_at_10: `0.06775067750677506`
- stage10_recall_at_10: `0.065041`
- prescore_recall_at_10: `0.056911`
- check_counts: `PASS=7 WARN=1 FAIL=0`

## Checks

| Level | Area | Message |
| --- | --- | --- |
| `PASS` | `stage09` | truth_in_pretrim healthy: 0.882255 |
| `PASS` | `stage09` | hard_miss within bound: 0.052792 |
| `PASS` | `traffic` | users evaluated stable at 738 |
| `PASS` | `traffic` | candidate row count stable at 184500 |
| `PASS` | `ranking` | champion drift healthy: stage11=0.067751 > stage10=0.065041 > prescore=0.056911 |
| `PASS` | `training` | stage10 rank_model.json is readable |
| `PASS` | `training` | stage11 training artifact healthy: runtime=397.6s train_pairs=6462 eval_pairs=1089 |
| `WARN` | `training` | stage11 model run still records enforce_stage09_gate=false |
