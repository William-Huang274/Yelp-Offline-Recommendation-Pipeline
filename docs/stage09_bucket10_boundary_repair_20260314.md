# Stage09 Bucket10 Boundary Repair Audit (2026-03-14)

## Scope

This note records the first `bucket10 boundary repair simulator` replay on the
large-pool bucket10 head-v2 table.

Input table:

- `/root/autodl-tmp/stage09_fs/audits/20260313_220737_stage09_bucket10_head_v2_table`

Audit output:

- `/root/autodl-tmp/stage09_fs/audits/20260314_000822_stage09_bucket10_boundary_repair_simulator`

The simulator is audit-only. It does not modify `stage09` production logic.

## Goal

The previous full-lane v2 replay proved that a coarse top150 reallocation was
too blunt:

- it recovered some `151+` truths,
- but it displaced even more existing `81-150` truths.

So this replay changed the objective:

- keep the very front fixed,
- repair only the `top150` boundary,
- use a limited number of challenger promotions from `151+`,
- do not touch the whole top150 list.

## Baseline

Shared-user large-pool baseline (`2997` users):

- `truth_in_top80 = 885`
- `truth_in_top150 = 1227`
- `truth_in_top250 = 1598`

## Grid

The boundary-repair grid searched `648` configs over:

- `protect_topk = 100 / 110 / 120 / 130`
- `candidate_cap = 180 / 200 / 220`
- `heavy_replace = 6 / 10 / 14`
- `mid_replace = 4 / 8 / 12`
- `challenger_min_detail = 1 / 2`
- `min_margin = 0.00 / 0.03 / 0.06`

Interpretation:

- positions up to `protect_topk` stay fixed in baseline order,
- only the tail window `protect_topk+1 .. 150` is allowed to swap,
- challengers are drawn from `151 .. candidate_cap`.

## Best Result

Best config:

- `protect_topk = 130`
- `candidate_cap = 200`
- `heavy_replace = 6`
- `mid_replace = 8`
- `challenger_min_detail = 1`
- `min_margin = 0.00`

Best result:

- `truth_in_top80 = 885`
- `truth_in_top150 = 1238`
- `truth_in_top250 = 1598`

Delta vs baseline:

- `top80: +0`
- `top150: +11`
- `top250: +0`

## What Improved

This direction finally produced a positive net gain:

- recovered from `151-250` into `top150`: `28`
- recovered from `251+`: `0`
- total recovered into `top150`: `28`
- total lost from baseline `top150`: `17`
- net gain: `+11`

Segment split:

- heavy:
  - recovered: `10`
  - lost: `4`
  - net: `+6`
- mid:
  - recovered: `18`
  - lost: `13`
  - net: `+5`

## What This Means

The boundary-repair idea is directionally correct.

Compared with the previous full-lane replay:

- it kept `top80` intact,
- it reduced collateral damage on baseline `81-150`,
- it still recovered some `151+` truths,
- and it produced a positive net gain instead of a negative one.

## Why The Gain Is Still Small

The gain is only `+11`, so this is not yet enough to matter for a full
`stage11` retrain.

Important signals:

1. All recovered truths came from `151-250`, not from `251+`.
2. All losses still came from baseline `81-150`, but the loss count fell to
   only `17`.
3. The best config repeated across multiple `min_margin` and
   `challenger_min_detail` combinations, which suggests those thresholds are not
   the main bottleneck.

Interpretation:

- the useful action is concentrated in a narrow band near the existing top150
  boundary,
- deeper challengers beyond about `200` are not adding useful gain,
- the next bottleneck is likely challenger ordering quality inside the
  `151-200` band, not the replacement quota itself.

## Practical Conclusion

Do not write this directly back into `09_candidate_fusion.py` yet.

But unlike the previous full-lane attempt, this path is worth continuing.

## Recommended Next Step

The next audit should get even narrower:

- freeze baseline `top130`,
- compare only the boundary tail `131-150`,
- challenger pool should focus on `151-200`,
- improve the challenger-vs-incumbent comparison instead of widening the pool
  or adding more replacements.

In short:

- full top150 reallocation was too coarse,
- boundary repair is directionally right,
- but the current challenger scoring is still too weak to produce a large gain.
