# Stage11 31-60 Expert And Segmented Fusion

[English](./stage11_31_60_only_and_segmented_fusion_20260408.md) | [中文](./stage11_31_60_only_and_segmented_fusion_20260408.zh-CN.md)

## 1. Objective

This note records the Stage11 redesign that introduced:

- a dedicated `31-60 only` expert
- segmented `11-3` fusion
- second-stage shortlist reranking
- bounded policy gates

The current freeze line is built on one shared `Qwen3.5-9B` reward-model
backbone, then specialized into three rerank experts:

- `11-30` boundary rescue
- `31-60` mid-rank rescue
- `61-100` deep-rank uplift

## 2. Design Constraints

The redesign keeps the following fixed:

- Stage09 candidate contract
- Stage11_1 pool build contract
- `11-30` frozen expert path
- label definition, split logic, candidate boundary, and metric meaning

## 3. Why 31-60 Needed A Dedicated Expert

The core issue was not that `31-60` could not learn.

The issue was that:

- local rescue quality could be good in `11-2`
- but release into final top10 was still weak in `11-3`
- mixed `11-30 + 31-60` training changed full-list calibration and weakened the boundary path

The redesign therefore separated:

- `11-30` boundary rescue
- `31-40 / 41-60` mid rescue

## 3.1 Outward-Facing `bucket5` Scope

The current public Stage11 line is centered on the `bucket5` mid-to-high
interaction set. This is the scope behind the root README metrics.

| scope item | current `bucket5` line |
| --- | ---: |
| businesses | `1,798` |
| users | `9,765` |
| train interactions | `133,048` |
| validation users | `9,765` |
| test users | `9,765` |
| fixed `Stage10` eval users | `1,935` |
| `Stage11` rescue eval users | `517` |

## 4. 31-60 Training Design

The dedicated `31-60` path uses:

- `rescue_31_60` only
- multiple typed slates per positive
- `PAIR_LOCAL_LISTWISE_MAX_RIVALS = 4`
- richer per-user slate coverage instead of larger single-slate width

Required slate types:

1. `31-60 vs same-band`
2. `31-60 vs 11-30`
3. `31-60 vs head-anchor`

Subband treatment:

- `31-40`: closer to boundary, more boundary blockers
- `41-60`: more same-band blockers, target first release into `top20 / top30`

The current freeze line uses the following training scale:

- `11-30` expert:
  - `11-30 true win = 0.9560`
- `31-60` expert:
  - training users `891`
  - evaluation users `215`
  - `train_pairs = 2363`
  - `eval_pairs = 552`
- `61-100` expert:
  - training users `679`
  - evaluation users `178`
  - `train_pairs = 1532`
  - `eval_pairs = 393`

Both expert lines are built from slices of the `bucket5` `11-100` rescue pool,
rather than from a separately defined user population.

## 5. 61-100 Position In The Current Freeze

The current freeze also includes a dedicated `61-100` expert, but the policy is
kept conservative.

Current interpretation:

- train the deep expert
- allow it to improve deep-rank ordering
- do not force a strong top30 promotion policy in the current freeze line

This keeps the deep path useful without destabilizing the front of the list.

## 6. 11-3 Segmented Fusion

Current segmented route:

- `11-30 -> v101`
- `31-40 -> 31-60 expert`
- `41-60 -> 31-60 expert`
- `61-100 -> 61-100 expert`

Current second-stage design:

- shortlist rerank on top of first-stage blend
- route-local normalization
- bounded gate and cap-rank policy

Current policy interpretation:

- `31-40`: top10 rescue lane
- `41-60`: top20 rescue lane
- `61-100`: conservative rank-uplift lane

## 7. Current Reference Points

Training-side:

- `Stage11` mainline rescue evaluation on `bucket5` uses `517` users
- `11-30 only` frozen boundary snapshot:
  - `11-30 true win = 0.9560`
- `31-60 only` best snapshot:
  - `31-40 true win = 0.7385`
  - `41-60 true win = 0.7605`
  - `listwise win rate = 0.7518`
- `61-100 only` best snapshot:
  - `61-100 true win = 0.8626`
  - `listwise win rate = 0.8626`

Evaluation-side:

- `v120 @ alpha=0.80`
  - current two-band best-known offline result
- `v124 @ alpha=0.36`
  - current tri-band freeze baseline

## 8. What This Note Is For

Use this note to explain:

- why Stage11 moved away from generic SFT/DPO mainline training
- why segmented experts were introduced
- why the current tri-band line keeps deep rescue conservative

Do not use this note to claim that the current tri-band policy is the final
production champion.
