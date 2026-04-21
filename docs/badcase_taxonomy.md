# Bad-Case Taxonomy

This taxonomy is for interview and release review. It maps ranking failures to the stage that should own the investigation.

| category | symptom | likely owner | first check |
| --- | --- | --- | --- |
| recall hard miss | the truth item never enters the candidate pool | Stage09 | `truth_in_pretrim150`, `hard_miss` |
| cold-start sparsity | bucket2 or `0-3` users improve less than bucket5 | Stage09 / Stage10 | cohort-specific recall and NDCG |
| candidate clutter | too many near-duplicate or generic popular items survive | Stage09 | route lane mix and pretrim composition |
| structured rank miss | candidate is in pool but XGBoost-style rerank pushes it down | Stage10 | feature trace, PreScore vs learned rank |
| reward over-rescue | Stage11 lifts a semantically interesting but irrelevant item | Stage11 | `stage11_band`, `rm_score`, final rank delta |
| coverage regression | metric improves only after dropping users or items | evaluation | surviving users, user coverage, item coverage |
| stale artifact | serving reads an old pointer or mismatched release id | release | `config/serving.yaml`, `_prod_runs/*.json` |
| service fallback | reward-rerank request returns xgboost or baseline | serving | `fallback_count`, `fallback_reason` |

## Review Workflow

1. Confirm the request id and strategy used in the serving response.
2. Check whether the item was present after Stage09 retention.
3. Compare baseline rank, Stage10 rank, and final rank.
4. If Stage11 changed the order, inspect the rescue band and reward score.
5. If the issue is cohort-specific, separate bucket2 / bucket5 / bucket10 before changing features.

## Current Known Boundaries

- Stage09 still has non-zero hard miss, so some failures are upstream recall problems.
- Bucket2 gains are weaker than bucket5, which is expected for sparse users.
- Stage11 is intentionally bounded; it should rescue shortlist misses, not replace full-list retrieval.
- The mock serving API is for local demonstration, not real production traffic.
