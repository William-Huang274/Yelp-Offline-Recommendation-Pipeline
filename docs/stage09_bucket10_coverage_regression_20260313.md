# Stage09 Bucket10 Coverage Regression Note (2026-03-13)

## Scope

This note explains why recent bucket10 runs no longer match the earlier
`truth_in_all > 0.92` level and separates that problem from the already-known
`top150` head-ranking problem.

Compared runs:

- Early high-coverage run:
  - `/root/autodl-tmp/stage09_fs/20260311_005450_full_stage09_candidate_fusion`
- Current restored-route run:
  - `/root/autodl-tmp/stage09_fs/20260313_200615_full_stage09_candidate_fusion`

The current run restored all four profile routes, so this note isolates what
still regressed after route recovery.

## Key Findings

### 1. The old `0.92+` result was real, but it was not just a cohort artifact

On the old `738`-user eval cohort, the earlier audit already showed:

- `truth_in_all = 687 / 738 = 93.09%`
- `truth_in_pretrim = 635 / 738 = 86.04%`
- `truth_in_top150 = 301 / 738 = 40.79%`

The more important comparison is on the shared bucket10 users that exist in both
runs.

Shared-user comparison (`2997` users):

- old run:
  - `truth_in_all = 2828 / 2997 = 94.36%`
  - `truth_in_pretrim = 2640 / 2997 = 88.09%`
  - `truth_in_top150 = 1209 / 2997 = 40.34%`
  - `truth_in_top250 = 1599 / 2997 = 53.35%`
- current restored-route run:
  - `truth_in_all = 2603 / 2997 = 86.85%`
  - `truth_in_pretrim = 2300 / 2997 = 76.74%`
  - `truth_in_top150 = 1242 / 2997 = 41.44%`
  - `truth_in_top250 = 1564 / 2997 = 52.19%`

Conclusion:

- The current run is not only missing some easy users.
- Even on the exact same `2997` users, coverage fell materially:
  - `truth_in_all: -225 users`
  - `truth_in_pretrim: -340 users`

### 2. The current run also dropped `621` old bucket10 users entirely

User-set comparison:

- old bucket10 truth users: `3618`
- current bucket10 truth users: `2997`
- overlap: `2997`
- old-only users: `621`

Those `621` old-only users were easier than the shared set:

- `truth_in_all = 599 / 621 = 96.46%`
- `truth_in_pretrim = 552 / 621 = 88.89%`
- `truth_in_top150 = 325 / 621 = 52.33%`
- `truth_in_top250 = 402 / 621 = 64.73%`

So part of the regression is a changed bucket10 user set, but that is only part
of the story. The shared-user coverage drop is still real.

### 3. Route restoration did not solve the main coverage regression

Current restored-route run bucket meta:

- `profile_recall_enabled_routes = ['vector', 'shared', 'bridge_user', 'bridge_type']`

But compared with the early run, the pool is still much tighter:

- old:
  - `pretrim_top_k_used = 900`
  - `als_top_k_used = 900`
  - `cluster_top_k_used = 520`
  - `profile_top_k_used = 700`
  - `profile_recall_rows_total = 2506700`
- current:
  - `pretrim_top_k_used = 520`
  - `als_top_k_used = 620`
  - `cluster_top_k_used = 360`
  - `profile_top_k_used = 400`
  - `profile_recall_rows_total = 1183200`

Conclusion:

- The current route set is complete again.
- The candidate budget is still much smaller than the earlier high-coverage run.
- This is the clearest explanation for why `truth_in_all` did not recover.

### 4. A large part of the loss is consistent with deeper truths being cut off

Among shared users that were `in_all` on the old run but `absent_all` on the
current run:

- `264` users moved from old `in_all` to new `absent_all`

Among the subset that had already reached old pretrim before disappearing:

- users: `204`
- old `pre_rank` median: `539`
- old `pre_rank` p75: `762.5`
- old `als_rank` median: `476`
- old `profile_rank` median: `417.5`
- old `cluster_rank` median: `415`

This lines up with the tighter new limits:

- many lost truths were already living in the deep half of the old pretrim pool
- shrinking `900/700/520` style route budgets down to `620/400/360` will
  predictably remove that tail

### 5. But tighter top-k alone does not explain everything

For the same `204` users lost from old pretrim to new `absent_all`:

- `123` still had old `als_rank <= 620`
- `40` still had old `profile_rank <= 400`
- `12` still had old `cluster_rank <= 360`

So the regression is not explained by top-k compression alone.

There is likely additional drift in one or more of:

- route score quality
- route thresholding
- user/item semantic artifacts used to build profile recall
- bucket/user-set construction upstream of stage09

## Interpretation

There are now two separate problems:

1. Coverage regression:
   - current route/data construction does not recover the earlier
     `truth_in_all > 0.92` level
   - restoring the missing profile routes was necessary but not sufficient
2. Head-ranking regression:
   - even when truth survives into pretrim, bucket10 still underperforms inside
     top150

These are different problems and should not be debugged with one scalar score.

## What This Means For Next Steps

### Replay Result: old-size bucket10 budget on current assets

I ran bucket10-only replays on the current restored-route assets and only
restored the earlier large-pool budget family.

Current restored-route baseline (`20260313_200615`, shared-user set of `2997`):

- `truth_in_all = 2603`
- `truth_in_pretrim = 2300`
- `truth_in_top150 = 1242`
- `truth_in_top250 = 1564`

Large-pool replay (`20260313_212048`, current assets, large budget, 3 profile
routes):

- `truth_in_all = 2829`
- `truth_in_pretrim = 2618`
- `truth_in_top150 = 1237`
- `truth_in_top250 = 1597`

Large-pool replay (`20260313_212850`, current assets, large budget, full 4
profile routes):

- `truth_in_all = 2829`
- `truth_in_pretrim = 2618`
- `truth_in_top150 = 1227`
- `truth_in_top250 = 1598`

Comparison against the early high-coverage run on the same shared-user set:

- early run shared users:
  - `truth_in_all = 2828`
  - `truth_in_pretrim = 2640`
  - `truth_in_top150 = 1209`
  - `truth_in_top250 = 1599`

Interpretation:

- Restoring the old large pool almost fully recovers `truth_in_all`.
- `truth_in_pretrim` also returns close to the early run.
- `top250` returns almost exactly to the early run.
- `top150` does not improve with the larger pool; it slightly worsens vs the
  current tighter-pool baseline.

This is the cleanest evidence so far that:

1. coverage regression is mostly a pool-budget problem,
2. head ranking is a separate problem,
3. and solving only head ranking will never recover the missing `truth_in_all`
   users.

### A. Do not treat current bucket10 as a pure head-ranking problem

The current run is worse than the old run before head-ranking even starts.

So:

- a better head policy may improve `top150`
- but it cannot by itself recover the missing `truth_in_all` users

### B. Route/data construction is still not good enough

This is now supported by three facts:

1. four profile routes are restored, but coverage is still far below the old
   run
2. shared-user coverage still regressed heavily
3. route budgets and total profile recall rows are much lower than before

### C. Head simulator is still worth running, but only for the second problem

The new audit-only script:

- `scripts/09_3_bucket10_head_simulator.py`

tests only this question:

- given the current pretrim pool, how much more `truth_in_top150` can a
  structured head allocation recover?

It does not answer the missing-coverage problem.

## Bottom Line

Current bucket10 is losing on both axes:

- it recalls fewer truths than the earlier high-coverage run
- and it still ranks the surviving truths poorly inside top150

So yes:

- the current route and data construction are still not good enough
- and the current pre-rank/head structure is also still not good enough

Both need to be handled, in that order.
