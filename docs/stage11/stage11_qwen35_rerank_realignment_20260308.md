# Stage11 Qwen3.5 Rerank Realignment Guide (2026-03-08)

## Goal
- Keep `Qwen3.5-4B` as the first-round backbone.
- Keep `stage09` candidate boundary and `stage11_3` final metric meaning unchanged.
- Fix the main mismatch between stage11 training data and real rerank evaluation before changing model family.

## Scope
- Dataset stage: [11_1_qlora_build_dataset.py](../../scripts/11_1_qlora_build_dataset.py)
- Train stage: [11_2_qlora_train.py](../../scripts/11_2_qlora_train.py)
- Eval stage: [11_3_qlora_sidecar_eval.py](../../scripts/11_3_qlora_sidecar_eval.py)
- Base model to keep in round 1: `Qwen/Qwen3.5-4B`

## What Is Actually Wrong
- [11_1_qlora_build_dataset.py#L48](../../scripts/11_1_qlora_build_dataset.py#L48) mixes `valid` into positives, but [11_3_qlora_sidecar_eval.py#L1199](../../scripts/11_3_qlora_sidecar_eval.py#L1199) only gives credit to `truth.parquet`.
- [11_1_qlora_build_dataset.py#L36](../../scripts/11_1_qlora_build_dataset.py#L36) and [11_1_qlora_build_dataset.py#L37](../../scripts/11_1_qlora_build_dataset.py#L37) build a small sampled classification set (`top120`, `8` negatives per user), while [11_3_qlora_sidecar_eval.py#L1226](../../scripts/11_3_qlora_sidecar_eval.py#L1226) reranks the full `pretrim150`.
- [11_2_qlora_train.py#L578](../../scripts/11_2_qlora_train.py#L578) further rebalance negatives for pointwise token CE, so the model is optimized for sampled YES/NO classification, not user-level reranking.
- [11_3_qlora_sidecar_eval.py#L879](../../scripts/11_3_qlora_sidecar_eval.py#L879) currently uses `softmax([YES, NO])` probability as the sidecar score. That is usable, but not the most stable ranking score once we move toward ranking-style training.

## Decision For Round 1
- Do not switch away from `Qwen3.5-4B` yet.
- Do not redesign the prompt family yet.
- Do not change `stage09` candidates, split logic, or final `Recall/NDCG` definition.
- First prove that `Qwen3.5-4B + FlashAttention + real-pool rerank-aligned supervision` can learn a useful signal.

## Why Keeping Qwen3.5 Is Reasonable
- `Qwen3.5-4B` already fits the current prompt assets and LoRA path.
- FlashAttention helps both training and eval throughput on a `4090 24G`.
- The current failure mode is mainly target mismatch, not clear evidence that the backbone itself is unusable.
- If round 1 still fails after data and scoring are aligned, then switching to a dedicated reranker is justified.

## First-Round Design

### 1. Rebuild Stage11_1 Around Real Rerank Supervision
- Main label mode: `truth_only`
- Candidate source: `pretrim150`
- Keep only users with `truth in pool` for the primary rerank training set.
- Do not use all negatives. Sample `16-24` negatives per positive from the real pool.
- Sample negatives by rank bands instead of the current `hard/near/easy/fill` proxy:
  - `1-10`
  - `11-30`
  - `31-80`
  - `81-150`
- Keep the existing prompt feature fields if they are still useful, but stop redefining label semantics.

### 2. Keep Training Simple in Round 1
- Round 1 objective: pointwise YES/NO supervision on rerank-aligned samples.
- Keep the current LoRA path and most trainer settings.
- Remove or disable the current post-build negative rebalance once the dataset already has controlled banded negatives.
- Do not start with pairwise/listwise in the first round. Add pairwise only after the new data path proves useful.

### 3. Change the Online Score Used by Stage11_3
- Keep the current prompt construction and candidate rerank loop.
- Add a new score mode:
  - `prob`: current `softmax([YES, NO])`
  - `logit_margin`: `yes_logit - no_logit`
- Make `logit_margin` the default ranking score for new experiments.
- Normalize the model score user-wise before blending with `pre_norm`.

### 4. Add a Small Rerank-Like Validation Set
- Build `mini_rerank_eval` inside stage11 data output.
- Use a fixed small eval-user subset.
- Keep all `150` candidates per selected user.
- Use it to select runs and detect regressions before running the full `stage11_3`.

## What Stage11_1 Should Output After Round 1
- `pointwise_train.jsonl`
  - One row per `(user, item)` pair
  - Labels: `truth_only`
  - Negatives: sampled from real `pretrim150`
- `mini_rerank_eval.parquet`
  - Small fixed user set
  - Full `150` candidates per user
  - Includes `true_item_idx`, `pre_rank`, `pre_score`, and prompt features needed by `11_3`
- Optional but recommended:
  - `data_audit.json`
  - `label_source_audit.csv`
  - `rank_band_audit.csv`

## New Env Knobs To Add

### Stage11_1
- `QLORA_LABEL_MODE=truth_only|truth_plus_valid`
- `QLORA_NEG_SOURCE_MODE=real_pool`
- `QLORA_NEG_BANDS=1-10:4,11-30:6,31-80:6,81-150:4`
- `QLORA_MINI_EVAL_USERS=128`
- `QLORA_BUILD_POINTWISE=true`
- `QLORA_BUILD_MINI_RERANK_EVAL=true`

### Stage11_2
- `QLORA_TRAIN_OBJECTIVE=pointwise_bce`
- `QLORA_USE_REBALANCE=false`
- `QLORA_SCORE_TARGET=yes_no`

### Stage11_3
- `QLORA_SCORE_MODE=logit_margin|prob`
- `QLORA_USE_MINI_RERANK_EVAL=false`

## Round-1 Execution Plan

### Step 1. Implement Data Realignment
- Modify [11_1_qlora_build_dataset.py](../../scripts/11_1_qlora_build_dataset.py) to support `truth_only` labels and real-pool banded negatives.
- Keep existing output files for backward compatibility if needed.
- Add new outputs rather than deleting old ones immediately.

### Step 2. Train a Pilot Pointwise Model
- Keep `Qwen3.5-4B`, QLoRA, and FlashAttention.
- Keep prompt mode fixed.
- Train only on the new `pointwise_train`.
- Recommended pilot settings on `4090 24G`:
  - `max_seq_len=512`
  - `epochs=1`
  - `batch_size=4`
  - `grad_acc=4`
  - `lr=1e-4`
  - `4bit=true`

### Step 3. Gate on Mini Rerank Eval
- Evaluate checkpoints on `mini_rerank_eval`.
- Primary gate:
  - `NDCG@10_truth_in_pool`
- Secondary gate:
  - `Recall@10_truth_in_pool`
- Report but do not select on:
  - `NDCG@10_all_users`
  - `Recall@10_all_users`

### Step 4. Run Full Stage11_3
- Run the existing full rerank eval on the same `pretrim150`.
- Use `QLORA_SCORE_MODE=logit_margin`.
- Compare against `PreScore` and the current `checkpoint700` baseline.

## First-Round Experiments

### Experiment A: Minimal-Risk Realignment
- Change data only:
  - `truth_only`
  - real-pool banded negatives
  - no `valid` as positive
- Keep pointwise YES/NO training.
- Keep blend enabled.
- Purpose:
  - test whether the main issue is data alignment

### Experiment B: Same Model, Better Score
- Use the same trained checkpoint as experiment A.
- Compare `prob` vs `logit_margin` in [11_3_qlora_sidecar_eval.py](../../scripts/11_3_qlora_sidecar_eval.py).
- Purpose:
  - test whether the current score definition is suppressing ranking signal

### Experiment C: Blend Sensitivity After Realignment
- Use the best score mode from experiment B.
- Sweep a small alpha grid such as:
  - `0.00`
  - `0.02`
  - `0.05`
  - `0.10`
- Purpose:
  - confirm whether the new sidecar signal is additive instead of harmful

## What Success Looks Like In Round 1
- `qlora-only` ranking quality is clearly better than the current near-random behavior on the true rerank pool.
- `blend_alpha > 0` no longer consistently hurts `PreScore`.
- `mini_rerank_eval` and full `stage11_3` move in the same direction.
- We can explain any gain without relying on `valid`-as-positive.

## What Not To Do In Round 1
- Do not mix `valid` back into the main positive label just to increase positive count.
- Do not use all `149` negatives per user for training. That wastes compute and usually adds low-value signal.
- Do not jump to pairwise/listwise before proving the new data path.
- Do not switch model family and data path at the same time.

## Answer To The Positive-Sample Concern
- Yes, `truth_only` will reduce the raw positive row count compared with `truth + valid`.
- That is acceptable because the extra `valid` positives are not aligned with the final metric and act like label noise for rerank.
- In rerank training, the useful unit is not only the number of positive rows. The useful unit is `one positive matched against several informative negatives from the same user`.
- A single truth item paired with `16-24` real hard negatives can produce much more useful learning signal than one truth plus one loosely related `valid` positive inside a proxy classification set.
- If positive coverage is still too small after the rebuild, the safe next move is:
  - keep `truth_only` as the hard label
  - oversample positive users during training
  - or add `valid` as an auxiliary feature or soft target, not as the main label

## Phase-2 Direction If Round 1 Works
- Add `pairwise` training on `(truth, neg)` pairs.
- Keep `logit_margin` as the score.
- Reuse the same `mini_rerank_eval` selection gate.
- Only after this should we decide whether a dedicated reranker base is worth the migration cost.
