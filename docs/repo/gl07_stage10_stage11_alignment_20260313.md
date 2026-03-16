# GL-07 Alignment Note (2026-03-13)

## Purpose

This note records the first direct `stage10` vs `stage11` alignment run using the
`stage11` evaluation contract.

The goal of this run was:

- keep the same `stage09` source as the latest frozen `stage11`
- use the same `stage11` eval user cohort
- use the same candidate truncation contract
- re-run `stage10 XGB` under that contract before comparing against `stage11`

## Alignment Contract

Shared source and evaluation contract:

- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- eval users = `738` users from `stage11_1` eval split
- candidate truncation = `pre_rank <= 250`
- final metric = `@10`

Stage11 reference used for comparison:

- [qlora_sidecar_metrics.csv](../../data/output/11_qlora_sidecar_eval/20260313_151256_stage11_3_qlora_sidecar_eval/qlora_sidecar_metrics.csv)

Aligned stage10 local artifacts:

- [recsys_stage10_results_gl07.csv](../../data/metrics/recsys_stage10_results_gl07.csv)
- [run_meta.json](../../data/output/10_2_rank_infer_eval/20260313_193213_stage10_2_rank_infer_eval/run_meta.json)

## What Was Run

Stage10 model used:

- `20260307_210530_stage10_1_rank_train`

Aligned evaluation settings:

- explicit eval cohort file generated from `stage11_1` eval split
- `RANK_EVAL_CANDIDATE_TOPN = 250`
- `RANK_RERANK_TOPN = 250`
- bucket scope = `10`

Cloud execution was used for the aligned run because local execution was manually
interrupted and the cloud machine already held the active `stage09` and `stage11`
artifacts.

## Result

Aligned `stage10` result:

- `PreScore@10`
  - `recall = 0.056911`
  - `ndcg = 0.026467`
- `LearnedBlendXGBCls@10`
  - `recall = 0.065041`
  - `ndcg = 0.029217`

Same-contract frozen `stage11` result:

- `PreScore@10`
  - `recall = 0.056910569105691054`
  - `ndcg = 0.02646705876884783`
- `QLoRASidecar@10`
  - `recall = 0.06775067750677506`
  - `ndcg = 0.02993468209586097`

## Comparison

Stage10 aligned XGB vs PreScore:

- recall gain: `+0.008130`
- recall relative gain: `+14.285463%`
- ndcg gain: `+0.002750`
- ndcg relative gain: `+10.390297%`

Stage11 QLoRA vs aligned stage10 XGB:

- recall gain: `+0.00270967750677506`
- recall relative gain: `+4.166107%`
- ndcg gain: `+0.00071768209586097`
- ndcg relative gain: `+2.456385%`

Ordering under the aligned contract:

1. `stage11 QLoRASidecar@10`
2. `stage10 LearnedBlendXGBCls@10`
3. `PreScore@10`

## Decision

This aligned run resolves the main release-comparison ambiguity that existed before:

- `stage10` is no longer being compared on a different eval user set
- `stage10` is no longer being compared on a different candidate-count contract

Current takeaway:

- `stage10 XGB` is valid and competitive under the `stage11` contract
- but `stage11` still remains the strongest current result

## Remaining Caveat

This run aligned evaluation, not training lineage.

That means:

- `stage10` was evaluated on the `20260311_005450` source and the `stage11` eval cohort
- but the `stage10` model itself still comes from the earlier `20260307_210530` training run

If you want a fully refreshed final release decision, the next optional step is:

- retrain `stage10` on the now-audited `20260311_005450` lineage

But for the current `GL-07` comparison purpose, the aligned evaluation result is
already sufficient to rank the current candidates.
