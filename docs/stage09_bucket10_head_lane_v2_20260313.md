# Stage09 Bucket10 Head Lane V2 Audit (2026-03-13)

## Scope

This note records the first `bucket10 head lane simulator v2` replay on the
large-pool bucket10 head-v2 table.

Input table:

- `/root/autodl-tmp/stage09_fs/audits/20260313_220737_stage09_bucket10_head_v2_table`

Audit output:

- `/root/autodl-tmp/stage09_fs/audits/20260313_235437_stage09_bucket10_head_lane_simulator_v2`

The simulator is audit-only. It does not modify `stage09` production logic.

## Goal

Test whether a structured top150 policy can outperform the current single-list
heuristic by splitting bucket10 head allocation into:

- `als_lane`
- `personal_lane`
- `recovery_lane`
- `flex_lane`

Heavy users and mid/light users use different lane quotas.

## Baseline

Large-pool baseline on the same bucket10 shared-user set (`2997` users):

- `truth_in_top80 = 885`
- `truth_in_top150 = 1227`
- `truth_in_top250 = 1598`

## Grid

The first v2 grid searched `336` configs around these families:

- heavy:
  - `als = 55/60`
  - `personal = 55/65`
  - `recovery = 20/30`
- mid/light:
  - `als = 75/80`
  - `personal = 35/40`
  - `recovery = 15/20`
- thresholds:
  - `personal_max_pre_rank = 220/300`
  - `recovery_min_pre_rank = 150/220`
  - `consensus_min_detail = 1/2`

## Best Result

Best config:

- heavy:
  - `als = 55`
  - `personal = 55`
  - `recovery = 20`
- mid/light:
  - `als = 80`
  - `personal = 40`
  - `recovery = 15`
- thresholds:
  - `personal_max_pre_rank = 220`
  - `recovery_min_pre_rank = 150`
  - `consensus_min_detail = 2`

Best result:

- `truth_in_top80 = 874`
- `truth_in_top150 = 1158`
- `truth_in_top250 = 1551`

Delta vs baseline:

- `top80: -11`
- `top150: -69`
- `top250: -47`

## What Improved

The v2 lane policy did recover some deep truths:

- recovered from `151-250` into `top150`: `76`
- recovered from `251+` into `top150`: `13`
- total recovered into `top150`: `89`

Recovered truth lanes:

- heavy:
  - `personal_lane = 31`
  - `recovery_lane = 4`
- mid:
  - `personal_lane = 45`
  - `recovery_lane = 9`

So the `personal/recovery` lanes are not useless. They can pull some truths up.

## Why It Still Failed

The problem is the tradeoff.

This v2 policy lost more existing top150 truths than it recovered:

- lost from baseline `top150`: `158`
- all `158` losses came from baseline bucket `81-150`
- losses from baseline `top80`: `0`

Segment split:

- heavy:
  - recovered: `35`
  - lost: `76`
- mid:
  - recovered: `54`
  - lost: `82`

Interpretation:

- the ALS backbone preserved the very front (`top80`)
- but the new personal/recovery lanes pushed out too many truths that were
  already sitting in baseline `81-150`
- net effect was negative

This means the first structured lane design is still too coarse. It is not a
`top80` problem; it is a boundary problem around the existing `81-150` band.

## Practical Conclusion

The first `head lane v2` idea is not good enough to write back into
`09_candidate_fusion.py`.

What it proved:

1. hard personal/recovery lanes can recover some `151+` truths,
2. but current quotas and gating displace too many existing `81-150` truths,
3. so a full lane replacement is too blunt.

## Recommended Next Step

Do not push this v2 policy to production.

Instead, the next audit should target the boundary band only:

- keep baseline `top80` fixed
- keep most of baseline `81-120` or `81-130` fixed
- only allow structured replacement inside a narrow tail window such as
  `121-220` or `131-220`
- treat this as a `boundary repair` problem, not a full top150 reallocation

That is the highest-signal lesson from the first v2 replay.
