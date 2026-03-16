# Stage09 Bucket10 Pairwise Boundary Probe (2026-03-14)

## Scope

This note records the first high-parallel `pairwise boundary probe` on the
large-pool bucket10 head-v2 table.

Input table:

- `/root/autodl-tmp/stage09_fs/audits/20260313_220737_stage09_bucket10_head_v2_table`

Audit output:

- `/root/autodl-tmp/stage09_fs/audits/20260314_004647_stage09_bucket10_pairwise_boundary_probe`

The probe is audit-only. It does not modify `stage09` production logic.

## Goal

The previous boundary-repair replay found a small positive gain by:

- fixing baseline `top130`,
- swapping only inside the `131-150` boundary band,
- promoting challengers from roughly `151-200`.

This probe tested whether a stronger challenger-vs-incumbent comparator could
improve on that result by using pairwise feature deltas:

- detail support delta
- cluster support delta
- profile-route support delta
- consensus-count delta
- popular-penalty delta

## Cloud Setup

This run used the large cloud configuration instead of the earlier conservative
single-process mode:

- `96` worker processes
- `5184` configs
- high-memory Linux host (`208` CPU cores, `754G` RAM observed on the box)

## Baseline

Shared-user large-pool baseline (`2997` users):

- `truth_in_top80 = 885`
- `truth_in_top150 = 1227`
- `truth_in_top250 = 1598`

## Best Result

Best result:

- `truth_in_top80 = 885`
- `truth_in_top150 = 1238`
- `truth_in_top250 = 1598`

Delta vs baseline:

- `top80: +0`
- `top150: +11`
- `top250: +0`

This exactly matches the earlier non-pairwise boundary-repair best.

## Best Config

Best config:

- `protect_topk = 130`
- `candidate_cap = 200`
- `heavy_replace = 6`
- `mid_replace = 8`
- `min_detail = 1`

Important finding:

- all pairwise delta weights were effectively unnecessary at the optimum:
  - `detail_w = 0.0`
  - `cluster_w = 0.0`
  - `profile_w = 0.0`
  - `consensus_w = 0.0`
  - `popular_w = 0.0`
  - `min_margin = 0.0`

In other words:

- the best point collapsed back to the same plain boundary-repair behavior,
- the new pairwise delta features did not produce additional gain.

## Why This Matters

This is a stronger negative result than the earlier narrow grid:

1. the cloud run was no longer resource-limited,
2. the search space was much wider,
3. but the best metric still stayed at `1238`.

So the current bottleneck is no longer:

- not enough CPU,
- not enough configs,
- not enough narrow-threshold search.

The bottleneck is now more likely:

- the feature surface itself,
- or the route evidence quality around the `131-200` boundary band.

## Practical Conclusion

The current bucket10 boundary problem does not appear solvable by:

- wider grid search,
- stronger static pairwise weighting,
- or simply throwing more CPU at the same feature family.

At this point, the search evidence says the current boundary-repair feature
surface is near its ceiling.

## Recommended Next Step

Do not continue expanding static pairwise weight grids.

The next useful move should be structural:

- add new boundary-specific evidence, or
- build a true incumbent-vs-challenger training table with richer raw route
  context, or
- change the upstream route construction feeding the `131-200` band.
