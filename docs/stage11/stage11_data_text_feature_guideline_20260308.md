# Stage11 Data and Text Feature Guideline (2026-03-08)

## Goal
- Provide one execution guide for the stage11 data rebuild and text-feature rebuild.
- Make later implementation and experiment review comparable against a fixed checklist.
- Keep `stage09` candidate boundary and `stage11_3` final metric semantics unchanged.

## Scope
- Data construction: [11_1_qlora_build_dataset.py](../../scripts/11_1_qlora_build_dataset.py)
- Training: [11_2_qlora_train.py](../../scripts/11_2_qlora_train.py)
- Eval/scoring: [11_3_qlora_sidecar_eval.py](../../scripts/11_3_qlora_sidecar_eval.py)
- Prompt assembly: [qlora_prompting.py](../../scripts/pipeline/qlora_prompting.py)
- User profile source: [09_user_profile_build.py](../../scripts/09_user_profile_build.py)
- Item semantic source: [09_item_semantic_build.py](../../scripts/09_item_semantic_build.py)

## Current Audit

### 1. Label and pool mismatch
- [11_1_qlora_build_dataset.py#L48](../../scripts/11_1_qlora_build_dataset.py#L48) mixes `valid` into positives.
- [11_3_qlora_sidecar_eval.py#L1199](../../scripts/11_3_qlora_sidecar_eval.py#L1199) only gives credit to `truth.parquet`.
- This makes the prompt and label relationship easier than the final rerank task.

### 2. Review feature construction is too naive
- User review summary is only `recent_user_reviews=n, avg_stars=x`, see [11_1_qlora_build_dataset.py#L443](../../scripts/11_1_qlora_build_dataset.py#L443).
- Item review summary is the same pattern, see [11_1_qlora_build_dataset.py#L474](../../scripts/11_1_qlora_build_dataset.py#L474).
- Raw review text is just the latest `top2` snippets truncated to `220` chars and concatenated, see [11_1_qlora_build_dataset.py#L74](../../scripts/11_1_qlora_build_dataset.py#L74) to [11_1_qlora_build_dataset.py#L76](../../scripts/11_1_qlora_build_dataset.py#L76), [11_1_qlora_build_dataset.py#L430](../../scripts/11_1_qlora_build_dataset.py#L430) to [11_1_qlora_build_dataset.py#L441](../../scripts/11_1_qlora_build_dataset.py#L441), and [11_1_qlora_build_dataset.py#L461](../../scripts/11_1_qlora_build_dataset.py#L461) to [11_1_qlora_build_dataset.py#L472](../../scripts/11_1_qlora_build_dataset.py#L472).
- This is enough to provide some semantic hint, but not enough to support fine-grained rerank within top150.

### 3. Prompt text has high volume but limited information density
- Prompt assembly currently mixes natural language, tag lists, rank fields, and numeric features into one string, see [qlora_prompting.py#L31](../../scripts/pipeline/qlora_prompting.py#L31) and [qlora_prompting.py#L60](../../scripts/pipeline/qlora_prompting.py#L60).
- Many fields are useful, but the format is closer to serialized features than to clean preference evidence.
- Current train runs show about `430` tokens on average while supervision is almost entirely on the final `YES/NO` token, see [checkpoint700_run_meta.json](../../tmp/cloud_pull/20260308_eval/checkpoint700_run_meta.json).

### 4. Train and eval feature inconsistency exists
- The stage11 data run had `raw_review_text_enabled=true` and `short_text_summary_enabled=true`, see [stage11_1_run_meta.json](../../tmp/cloud_pull/20260308_eval/stage11_1_run_meta.json).
- The March 8 eval run skipped review text when the review table was missing, see [11_3_qlora_sidecar_eval.py#L542](../../scripts/11_3_qlora_sidecar_eval.py#L542).
- This makes it impossible to judge the true value of text features.

## Guiding Principles
- The target is not "more text". The target is "more useful evidence per token".
- User-side text should represent stable preference, not arbitrary recent text.
- Item-side text should represent stable merchant semantics, not arbitrary raw review fragments.
- Pair-level evidence should be selected for relevance and diversity, not only recency.
- Keep feature semantics aligned across train and eval.

## What Must Be Done

### Low-cost, mandatory
1. Make train and eval text features identical.
- Same review table requirement.
- Same summary toggles.
- Same snippet and topN policy.

2. Remove obvious shortcut channels.
- Stop using `valid` as the main positive label.
- Exclude candidate-business review text from user-side review evidence when it would leak direct item identity.

3. Clean existing profile text before prompt assembly.
- Normalize quoting and escaped punctuation.
- Drop malformed JSON-like fragments.
- Prefer short clean profile text over long noisy profile text.

4. Replace raw count summaries with compact aspect summaries.
- Current:
  - `recent_user_reviews=2, avg_stars=5.00`
  - `recent_item_reviews=2, avg_stars=3.00`
- New target:
  - `user_pref_summary: likes spicy food, casual lunch, good service; dislikes noisy atmosphere`
  - `item_semantic_summary: strong on dessert and coffee; weak on service consistency`

5. Reduce low-value prompt fields.
- Keep:
  - `profile_text`
  - `profile_top_pos_tags`
  - `profile_top_neg_tags`
  - `top_pos_tags`
  - `top_neg_tags`
  - `semantic_score`
  - `semantic_confidence`
  - `tower_score`
  - `seq_score`
  - one short user evidence block
  - one short item evidence block
- De-emphasize or remove from round 1 prompt text:
  - `candidate_sources`
  - multiple rank-position fields
  - weakly interpretable numeric fields that do not help language understanding

### Mid-cost, also required
1. Rebuild representative user review evidence.
- Do not use "latest two snippets" as the main rule.
- Reuse the sentence-scoring and diversity logic already present in [09_user_profile_build.py#L706](../../scripts/09_user_profile_build.py#L706) to [09_user_profile_build.py#L753](../../scripts/09_user_profile_build.py#L753).
- Reuse the sentence-level tag evidence already aggregated in [09_user_profile_build.py#L789](../../scripts/09_user_profile_build.py#L789) to [09_user_profile_build.py#L905](../../scripts/09_user_profile_build.py#L905).

2. Rebuild representative item review evidence.
- Do not use latest item snippets only.
- Build item evidence around facet stability and confidence.
- Reuse business semantic signals from [09_item_semantic_build.py#L726](../../scripts/09_item_semantic_build.py#L726) to [09_item_semantic_build.py#L751](../../scripts/09_item_semantic_build.py#L751).

3. Add pair-aware review selection.
- For each `(user, item)` prompt, choose review snippets that best explain why this item matches or mismatches the user.
- Use overlap between user positive tags and item positive tags as the primary relevance signal.
- Use overlap between user negative tags and item negative tags as mismatch evidence.
- Prefer diverse evidence over repeated lexical variants.

## How To Build Effective Reviews

### A. User review evidence
The current code already does much better work in profile building than stage11 prompt construction currently uses:
- sentence splitting and filtering, see [09_user_profile_build.py#L706](../../scripts/09_user_profile_build.py#L706) to [09_user_profile_build.py#L715](../../scripts/09_user_profile_build.py#L715)
- sentence scoring by recency and keyword hits, see [09_user_profile_build.py#L715](../../scripts/09_user_profile_build.py#L715)
- user-level dedup and diversity, see [09_user_profile_build.py#L733](../../scripts/09_user_profile_build.py#L733) to [09_user_profile_build.py#L753](../../scripts/09_user_profile_build.py#L753)
- tag polarity and confidence aggregation, see [09_user_profile_build.py#L789](../../scripts/09_user_profile_build.py#L789) to [09_user_profile_build.py#L905](../../scripts/09_user_profile_build.py#L905)

The stage11 prompt should use that output more directly.

Recommended user evidence format:
- `user_pref_summary`
  - stable positive tags
  - stable negative tags
  - confidence
- `user_evidence_snippets`
  - 2 to 3 representative sentences
  - high score
  - high diversity
  - no direct leakage from the current candidate business

### B. Item review evidence
Effective item review is not "most recent review". It is "most representative support or concern about the merchant".

Recommended item evidence rule:
- base the summary on stable business facets:
  - `top_pos_tags`
  - `top_neg_tags`
  - `semantic_score`
  - `semantic_confidence`
- optionally attach 1 to 2 short snippets only if they:
  - support top positive or top negative facets
  - are lexically distinct
  - are not generic praise/noise

Recommended item evidence format:
- `item_semantic_summary`
  - top strengths
  - top weaknesses
  - confidence/support
- `item_evidence_snippets`
  - short supporting sentences for the strongest facets

### C. Pair-aware representative review
This is the part that truly connects user profile and merchant semantics.

For each `(user, item)`:
- compute `match_tags = user_pos intersect item_pos`
- compute `conflict_tags = user_neg intersect item_pos` and `user_pos intersect item_neg`
- choose user snippets that mention `match_tags` or `conflict_tags`
- choose item snippets that mention the same tags
- if no strong overlap exists, fall back to stable summaries only

This is better than feeding raw recent reviews because it makes the evidence conditional on the current candidate.

## Recommended Stage11 Prompt Layout

### User block
- `profile_summary`
- `likes`
- `dislikes`
- `profile_confidence`
- `user_evidence_snippets`

### Candidate block
- `name`
- `city`
- `primary_category`
- `categories`
- `item_strengths`
- `item_weaknesses`
- `semantic_score`
- `semantic_confidence`
- `tower_score`
- `seq_score`
- `item_evidence_snippets`

### Remove or reduce in round 1
- raw candidate source list
- multiple rank fields
- duplicated summary and raw snippet carrying the same meaning
- long malformed profile strings

## Concrete Data Outputs To Add

### Stage11_1
- `pointwise_train.jsonl`
- `mini_rerank_eval.parquet`
- `user_evidence_table.parquet`
  - `user_idx`
  - `user_pref_summary`
  - `user_evidence_snippets`
  - `profile_confidence`
- `item_evidence_table.parquet`
  - `business_id`
  - `item_semantic_summary`
  - `item_evidence_snippets`
  - `semantic_confidence`
- `pair_evidence_audit.csv`
  - overlap counts
  - conflict counts
  - snippet source counts

## Acceptance Checks

### Data checks
- train/eval prompt fields are identical
- no candidate-business leakage from user review evidence
- `truth_only` is the main hard label
- prompt length p50/p90 decreases or stays controlled

### Text-quality checks
- malformed profile text rate drops
- duplicated review snippet rate drops
- user/item snippet overlap is intentional, not accidental label leakage
- empty evidence rate is measured and reported

### Rerank checks
- `mini_rerank_eval` improves with the same direction as full `stage11_3`
- `qlora-only` quality becomes meaningfully above random on the true rerank pool
- `blend_alpha > 0` stops consistently hurting baseline

## Rollout Order

### Phase 1
- train/eval feature parity
- `truth_only`
- prompt cleanup
- review leakage removal
- compact summary replacement

### Phase 2
- representative user evidence from `09_user_profile_build`
- representative item evidence from `09_item_semantic_build`
- pair-aware snippet selection

### Phase 3
- only after phases 1 and 2 stabilize:
  - pairwise training
  - optional dedicated reranker migration

## Current DPO Data Caveat
- The current fast DPO export path `QLORA_PAIRWISE_SOURCE_MODE=stage09_direct` is intentionally a structural audit path, not the final rich-text training path.
- It preserves `truth`, `pretrim150`, split semantics, and conservative/hard pair-selection logic, but it does not currently restore item review evidence richness for every exported pair prompt.
- In practice this means:
  - pair composition audit is valid
  - conservative vs hard pair diagnostics are valid
  - runtime is much better than the failed full-pool Spark materialization
  - but prompt richness is still below the target final DPO dataset
- Therefore:
  - use `stage09_direct` first for pair-structure audit and pilot DPO smoke tests
  - do not treat it as the final production DPO dataset until selected pairs get review evidence backfilled

## Answer To The "How Do We Give Effective Review?" Question
- Yes, this is fundamentally a user-profile and merchant-semantic optimization problem.
- A good review for rerank is not just recent, long, or emotional.
- A good review is representative, diverse, high-confidence, and relevant to the current user-item pair.
- The right solution is not to dump more raw reviews into the prompt.
- The right solution is:
  - compress user history into stable preference evidence
  - compress merchant history into stable semantic evidence
  - select a very small number of pair-relevant snippets as justification

## Out Of Scope For This Guide
- changing `stage09` candidate boundaries
- changing final metric definition
- large model-family migration in the same step as data rebuild
