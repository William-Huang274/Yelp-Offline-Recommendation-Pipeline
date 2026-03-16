# Stage09 Bucket10 Pre-Rank Audit (2026-03-13)

## Scope

- Audited stage09 source run:
  - `/root/autodl-tmp/stage09_fs/20260311_005450_full_stage09_candidate_fusion`
- Audited eval cohort:
  - `/root/5006_BDA_project/data/output/11_eval_cohorts/historical_report_cohort_738_bucket10_users.csv`
- Deep audit output:
  - `/root/autodl-tmp/stage09_fs/audits/20260313_174603_bucket10_prerank_deep_audit/prerank_deep_audit.json`
- Refusion / layered replay output:
  - `/root/autodl-tmp/stage09_fs/audits/20260313_171436_bucket10_refusion_truth_topn_audit/refusion_truth_topn_audit.json`

This audit targets only the stage09 `pre_rank` bottleneck. It does not evaluate stage11 rerank quality directly.

## Executive Summary

The current bucket10 problem is not source recall.

- `truth_in_all = 687 / 738 = 93.09%`
- `truth_in_pretrim = 635 / 738 = 86.04%`
- `truth_in_top150 = 301 / 738 = 40.79%`
- `truth_in_top250 = 386 / 738 = 52.30%`

The dominant loss is `pretrim -> top150`.

- Users with truth already in pretrim but still outside top150: `334`
- Users missing from all candidates: `51`
- Users in all candidates but cut before pretrim: `52`

So the mainline failure is head ordering inside the already-recalled pool, not recall source coverage.

## Code Path Evidence

### 1. Stage11 slices directly by `pre_rank`

- `scripts/11_1_qlora_build_dataset.py:1028-1031`
- `pre_rank <= TOPN_PER_USER` is the gate that decides which stage09 candidates survive into stage11 data.

Implication:

- Any meaningful improvement must move truth into the first `150` rows on `pre_rank`.

### 2. Current stage09 pre-score is source-weighted rank fusion plus quality / semantic / ALS backbone

- `scripts/09_candidate_fusion.py:2504-2516`
- `scripts/09_candidate_fusion.py:2685-2690`

Current structure:

- `signal_score = source_weight * inverse_log_rank * source_confidence`
- `pre_score = signal_score + quality_weight * quality_score + semantic_weight * semantic_effective_score + als_backbone_score`

### 3. Source weights are strongly ALS-skewed for bucket10-relevant users

- `scripts/09_candidate_fusion.py:300-303`

Current segment weights:

- `mid`: `als=0.70`, `cluster=0.15`, `popular=0.08`, `profile=0.07`
- `heavy`: `als=0.85`, `cluster=0.08`, `popular=0.03`, `profile=0.04`

### 4. Layered pretrim can only help if final `pre_rank` is rewritten

- `scripts/09_candidate_fusion.py:2715-2732`
- `scripts/09_candidate_fusion.py:2802-2904`

Implication:

- Layered seed protection alone is not enough if downstream still sees the old head order.

## Audit Findings

### Finding 1: Heavy users are the worst bucket10 subgroup

From the `738`-user cohort:

- `heavy`
  - users: `280`
  - `truth_in_top150 = 95 / 280 = 33.93%`
  - `truth_in_top250 = 125 / 280 = 44.64%`
- `mid`
  - users: `458`
  - `truth_in_top150 = 206 / 458 = 44.98%`
  - `truth_in_top250 = 261 / 458 = 56.99%`

Interpretation:

- Bucket10 head ranking is materially worse for heavy users.
- Any bucket10 pre-rank fix that does not treat heavy users separately is likely to underperform.

### Finding 2: Missed truth is usually not weak single-source noise

Among the `334` users whose truth is in pretrim but still outside top150:

- `299 / 334 = 89.52%` still have ALS support
- `250 / 334 = 74.85%` have profile or cluster support
- `223 / 334 = 66.77%` have at least `2` non-popular sources (`als/cluster/profile`)
- `72 / 334 = 21.56%` already have `als_rank <= 220`
- `60 / 334 = 17.96%` already have `profile_rank <= 140`
- `32 / 334 = 9.58%` already have `cluster_rank <= 120`
- `69 / 334 = 20.66%` are sitting in old rank band `151-220`

Interpretation:

- The problem is not that most misses have no corroboration.
- A large share of misses already have multi-route evidence, but their aggregate score still loses to other head items.

### Finding 3: The head is not dominated by ALS-only garbage

Across all top150 slots for the `738` users (`110,700` rows total):

- rows with profile or cluster support: `102,491 = 92.58%`
- `als_only`: `3,639 = 3.29%`
- `als+popular only`: `4,570 = 4.13%`
- `popular_only`: `0`

Top head combos:

1. `als+cluster+popular+profile`: `33,244`
2. `als+popular+profile`: `28,601`
3. `als+cluster+profile`: `18,363`
4. `als+profile`: `13,339`

Interpretation:

- The head is already full of multi-source items.
- So the failure is not simply “too many ALS-only items in front”.
- The real issue is that the scoring function favors the wrong type of multi-source corroboration.

### Finding 4: What wins top80 is different from what gets stuck in 151+

Truth items in `top80` are dominated by:

- `als+popular+profile`: `78`
- `als+cluster+popular+profile`: `73`

Truth items in `151-250` are dominated by:

- `als+profile`: `25`
- `als+cluster+profile`: `22`

Truth items in `251+` are dominated by:

- `als+profile`: `78`
- `als`: `73`
- `als+cluster+profile`: `26`
- `als+cluster`: `21`

Interpretation:

- The current pre-rank strongly prefers combinations that already include `popular`.
- Items with `als + profile` or `als + cluster + profile` support, but weaker popularity support, are consistently pushed down.
- That is a scoring bias, not a recall gap.

### Finding 5: There is still a small pretrim-cut subgroup, and it is profile-heavy

Among the `52` users with truth in `candidates_all` but not in pretrim:

- `profile present`: `33 = 63.46%`
- `als present`: `17 = 32.69%`
- `cluster present`: `17 = 32.69%`
- `popular present`: `5 = 9.62%`

Interpretation:

- A small but real subgroup is being lost before pretrim, mostly on profile-driven truth.
- This matters, but it is not the main bucket10 bottleneck.

### Finding 6: Low-risk layered reorder has very limited ceiling

Replay result on the exact `738`-user cohort:

- baseline:
  - `truth_in_top150 = 301`
  - `truth_in_top250 = 386`
- best low-risk replay tested:
  - `front_guard_topk = 140`
  - no rescue tier
  - `truth_in_top150 = 305`
  - `truth_in_top250 = 388`

Interpretation:

- Layered final-rank rewrite is directionally correct but not sufficient.
- Even a better head-protection policy only recovers a handful of users.
- That means the core problem is the score itself, not only the seed/rewrite mechanism.

## Root Cause Statement

The bucket10 pre-rank problem is:

1. already-recalled truth is being mis-scored inside the large pretrim pool,
2. especially for heavy users,
3. because the current score over-rewards ALS-backed, popularity-correlated corroboration,
4. and under-rewards profile / cluster corroboration when that corroboration is not also popular.

In short:

- `stage09` is not failing to find enough candidate truth,
- it is failing to choose the right `150` rows from the `635` users where truth is already present.

## What This Means Operationally

Do not expect a large win from:

- only increasing `top150 -> top250`
- only adding another lightweight layered rescue
- only tweaking a small head bonus

Those are second-order improvements.

The first-order problem is bucket10 pre-rank scoring.

## Recommended Next Step

The next audit / implementation loop should target score structure directly:

1. Build a bucket10 pre-rank audit table from existing pretrim rows.
2. Train or fit a small audit-only model on numeric stage09 features only:
   - `als_rank`, `cluster_rank`, `profile_rank`, `popular_rank`
   - source-set indicators
   - `signal_score`, `quality_score`, `semantic_score`, `semantic_confidence`
   - user segment / train-count
3. Use that model only as an offline probe first, to answer:
   - how much top150 ceiling is available if score structure changes
   - which features most strongly separate top150 hits from 151+ misses
4. If the ceiling is material, then decide between:
   - bucket10-specific hand-built score rewrite, or
   - a lightweight pre-ranker for `stage09 pretrim -> top150`

The current audit says that this is the point where simple fusion tuning stops paying off.
