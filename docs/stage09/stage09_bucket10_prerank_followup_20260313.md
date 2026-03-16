# Stage09 Bucket10 Pre-Rank Follow-up Audit (2026-03-13)

## Scope

This note extends the earlier bucket10 truth-rank audit and answers a narrower question:

- If we keep the current stage09 candidate pool fixed, can a better pre-rank alone materially lift `truth_in_top150`?
- If not, what part of stage09 should change next?

The critical code path is still:

- `scripts/11_1_qlora_build_dataset.py:1031`
  - stage11 cuts the rerank pool directly by `pre_rank <= TOPN_PER_USER`
- `scripts/09_candidate_fusion.py:2498-2516`
  - stage09 builds `signal_score` from route weight, route rank, and route confidence
- `scripts/09_candidate_fusion.py:2684-2703`
  - `pre_score` and `head_score` are formed on top of those aggregated route signals
- `scripts/09_candidate_fusion.py:2863-2898`
  - layered pretrim can rewrite final `pre_rank`, but it still only reorders the current fused feature surface

## Baseline Facts

The earlier deep audit on the 738-user eval cohort already established:

- `truth_in_all = 687 / 738 = 93.09%`
- `truth_in_pretrim = 635 / 738 = 86.04%`
- `truth_in_top150 = 301 / 738 = 40.79%`
- `truth_in_top250 = 386 / 738 = 52.30%`

So the dominant miss is not recall source coverage. It is the `pretrim -> top150` ordering.

The earlier low-risk refusion replay also established that safe policy tweaks are near ceiling:

- exact replay with bucket10 `front_guard_topk=140` and layered final-rank rewrite:
  - cohort `truth_in_top150: 301 -> 305`
  - cohort `truth_in_top250: 386 -> 388`

That result came from:

- `/root/autodl-tmp/stage09_fs/audits/20260313_171436_bucket10_refusion_truth_topn_audit/refusion_truth_topn_audit.json`
- `/root/autodl-tmp/stage09_fs/audits/20260313_174603_bucket10_prerank_deep_audit/prerank_deep_audit.json`

## New Probe Design

I added an audit-only script:

- `scripts/09_2_bucket10_prerank_probe.py`

It does not touch GPU and does not rerun the full stage09 pipeline. It:

1. Loads existing bucket10 `candidates_pretrim150.parquet` and truth from the frozen stage09 run.
2. Materializes one audit table once.
3. Runs user-level CV probes on the fixed candidate pool.
4. Measures whether alternative scorers can beat the current `pre_rank`.

Remote source run:

- `/root/autodl-tmp/stage09_fs/20260311_005450_full_stage09_candidate_fusion`

Cohort used:

- `/root/5006_BDA_project/data/output/11_eval_cohorts/historical_report_cohort_738_bucket10_users.csv`

Loaded audit table:

- `697,299` candidate rows
- `738` users
- `635` positives

## Probe Results

### 1. Linear probe (`numpy_lr`)

Output:

- `/root/autodl-tmp/stage09_fs/audits/20260313_180639_stage09_bucket10_prerank_probe/probe_summary.json`

Result against baseline:

- `truth_in_top80: 202 -> 195`
- `truth_in_top150: 301 -> 271`
- `truth_in_top250: 386 -> 376`

Interpretation:

- Simple linear reweighting is not only insufficient; it is clearly worse.
- This is strong evidence that "just learn better weights over current stage09 columns" is not the answer.

### 2. Nonlinear probe (`xgboost_ranker`)

Output:

- `/root/autodl-tmp/stage09_fs/audits/20260313_181150_stage09_bucket10_prerank_probe/probe_summary.json`

Result against baseline:

- `truth_in_top80: 202 -> 177`
- `truth_in_top150: 301 -> 280`
- `truth_in_top250: 386 -> 375`

Interpretation:

- Even a nonlinear ranker on the current fused feature surface fails to beat the existing handcrafted order.
- That means the remaining error is not "wrong scalar weights on already-available columns".
- The more likely issue is that the current fused table does not expose enough discriminative signal for the truth item before the head cut.

## What The Probes Say About The Real Bottleneck

The feature importance output is still useful even though the probe regressed.

Top features in the nonlinear probe included:

- `inv_pre_rank`
- `pre_rank_filled`
- `cluster_x_heavy`
- `pre_x_heavy`
- `is_als_only`
- `has_all4`
- `pre_x_cluster`
- `inv_als_rank`
- `nonpopular_source_count`
- `pre_x_profile`
- `profile_x_heavy`

This lines up with the earlier deep audit:

- bucket10 failure is concentrated in `heavy` users
- cluster/profile-backed items matter
- `als-only` pressure is still too strong in the head

But the important point is this:

- those signals are already present in the frozen fused frame
- and even with a nonlinear ranker, they still do not beat the baseline

So the limitation is now upstream of the score family.

## Practical Conclusion

At this point, there is no evidence that another score rewrite on top of the current stage09 fused columns will materially lift bucket10 `truth_in_top150`.

The current evidence stack is:

- cheap head push: no meaningful gain
- layered policy change: only `+4`
- linear learned re-rank on current columns: worse
- nonlinear learned re-rank on current columns: worse

That is enough to stop iterating on "same columns, new score".

## What Should Change Next

The next stage09 improvement should not be a new weight grid. It should be a feature-surface change before final `pre_rank`.

### A. Expose raw route strength before fusion/pretrim

Current stage09 largely exposes route membership and route rank, but not enough raw route evidence.

Next audit should carry forward, per candidate:

- raw ALS score or score percentile, not only `als_rank`
- raw cluster similarity and confidence
- raw profile route similarity / bridge score / shared-tag score
- route-specific confidence after thresholding
- route rank percentile within its own source list
- route agreement statistics, such as best nonpopular route gap vs ALS

Without these, the head is trying to separate near-tied items using mostly coarse route-order signals.

### B. Split bucket10 head policy by segment, especially `heavy`

Current bucket10 weakness is heavy-user dominant. A single global head is too ALS-heavy for that segment.

The next structural experiment should be:

1. keep a small global front guard
2. reserve explicit head lanes for profile/cluster-backed items for heavy users
3. score within those lanes using raw route strength, not only fused rank positions

This is different from the earlier `front_guard=140` tweak because it requires new signals plus segment-conditioned allocation.

### C. Treat two failure modes separately

The deep audit shows two different problems:

- `52` users: truth is in `candidates_all` but lost before pretrim
- `334` users: truth survives pretrim but loses the top150 head fight

These should not be solved with one scalar score.

Recommended split:

- coverage repair for `in_all_not_pretrim`
- head repair for `in_pretrim_but_not_top150`

### D. Do not spend more time on post-hoc rerank rescue alone

Stage11 rerank cannot rescue items that never enter the top150 pool.
The current probes reinforce that the missing lift must come from stage09 structural changes.

## Recommended Next Experiment

The next experiment I would implement is:

1. Create a bucket10-only audit export from stage09 that retains raw per-route scores and confidences before final pretrim.
2. Re-run the same cohort probe with those enriched route features.
3. Gate implementation on a real ceiling:
   - cohort `truth_in_top150 >= 320`
   - heavy-user `truth_in_top150 >= 110`
   - no drop in `truth_in_pretrim`

If that enriched-feature probe still cannot beat `301`, then the conclusion becomes stronger:

- the issue is not stage09 score calibration
- the issue is route generation quality itself, especially profile/cluster route quality for heavy users

## Bottom Line

Current bucket10 pre-rank optimization on the existing fused feature surface looks exhausted.

If we want a meaningful jump over `301 / 738` in top150 truth coverage, the next version must change one of:

- what raw route signals survive into fusion
- how heavy users are lane-allocated in the head
- or the upstream route generation quality itself

It should not be another small `pre_score/head_score` weight search.
