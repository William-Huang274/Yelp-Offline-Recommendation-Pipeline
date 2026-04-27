# Evaluation And Casebook

## 1. Current Review Scope

The current outward-facing line is centered on `bucket5`, with supporting
Stage10 evidence across `bucket2`, `bucket5`, and `bucket10`.

Current `bucket5` scope:

| item | value |
| --- | ---: |
| businesses | `1,798` |
| users | `9,765` |
| train interactions | `133,048` |
| Stage10 fixed eval users | `1,935` |
| Stage11 rescue eval users | `517` |

## 2. Frozen Stage09 Evidence

Current `bucket5` route-aware recall summary:

| metric | current value |
| --- | ---: |
| `truth_in_pretrim150` | `0.7451` |
| `hard_miss` | `0.1190` |

Source:

- [../../data/output/current_release/stage09/bucket5_route_aware_sourceparity/summary.json](../../data/output/current_release/stage09/bucket5_route_aware_sourceparity/summary.json)

## 3. Frozen Stage10 Evidence

Current structured-rerank summary:

| bucket | PreScore@10 recall / ndcg | LearnedBlendXGBCls@10 recall / ndcg |
| --- | --- | --- |
| `bucket2` | `0.1098 / 0.0513` | `0.1127 / 0.0522` |
| `bucket5` | `0.0935 / 0.0440` | `0.1261 / 0.0581` |
| `bucket10` | `0.0569 / 0.0265` | `0.0772 / 0.0341` |

Main references:

- [../../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv](../../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv)
- [../../data/output/current_release/stage10/stage10_current_mainline_summary.json](../../data/output/current_release/stage10/stage10_current_mainline_summary.json)

## 4. Frozen Stage11 Evidence

Training-side expert evidence:

| expert | key signal |
| --- | --- |
| `11-30` | `11-30 true win = 0.9560` |
| `31-60` | `31-40 true win = 0.7385`, `41-60 true win = 0.7605` |
| `61-100` | `61-100 true win = 0.8626` |

Evaluation-side references:

| line | recall@10 | ndcg@10 | note |
| --- | ---: | ---: | --- |
| `v120 two-band @ alpha=0.80` | `0.1973` | `0.0898` | best-known two-band offline result |
| `v124 tri-band @ alpha=0.36` | `0.1857` | `0.0838` | current frozen tri-band line |

Primary references:

- [../../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv](../../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv)
- [../../data/output/current_release/stage11/experts/expert_training_summary.json](../../data/output/current_release/stage11/experts/expert_training_summary.json)

## 5. Case 1: Boundary Rescue (`11-30`)

Canonical case:

- `user_idx = 1072`
- truth `item_idx = 58`
- `learned_rank = 17`
- `blend_rank = 4`
- `final_rank = 8`
- `route_band = boundary_11_30`
- `reward_score = 13.4375`
- `rescue_bonus = 0.7579`

Interpretation:

- Stage10 already placed the truth near the front boundary
- Stage11 resolved a difficult local comparison among nearby rivals
- this shows why the front boundary is a good target for local rescue instead of
  full-list reranking

## 6. Case 2: Mid Rescue (`31-40`)

Canonical case:

- `user_idx = 1940`
- truth `item_idx = 92`
- `learned_rank = 36`
- `blend_rank = 1`
- `final_rank = 2`
- `route_band = rescue_31_40`
- `reward_score = 11.3125`
- `rescue_bonus = 1.0350`

Interpretation:

- the truth candidate started in the middle band
- Stage11 evaluated it against local competitors rather than against the whole
  candidate list
- controlled promotion rules then allowed it into the final front rank

## 7. Case 3: Deep Expert Policy

The current freeze line trains the `61-100` expert successfully but uses it
conservatively.

Interpretation:

- the expert learns useful signals
- the current policy emphasizes rank uplift rather than aggressive front-rank
  takeover
- this protects stability while preserving future extension space

## 8. CLI Support For Demo

The checked-in demo CLI exposes these review helpers:

```bash
python tools/demo/demo_recommend.py summary
python tools/demo/demo_recommend.py list-cases
python tools/demo/demo_recommend.py show-case --case boundary_11_30
python tools/demo/demo_recommend.py show-case --case mid_31_40
```

## 9. Recommended Reporting Usage

For the final report:

- use the Stage09 table as the candidate-quality evidence
- use the Stage10 table as the structured-baseline evidence
- use the Stage11 table and the two cases as the interpretation and impact
  section

The detailed case explanations remain in:

- [../../docs/stage11/stage11_case_notes_20260409.md](../../docs/stage11/stage11_case_notes_20260409.md)
- [../../docs/stage11/stage11_case_notes_20260409.zh-CN.md](../../docs/stage11/stage11_case_notes_20260409.zh-CN.md)
