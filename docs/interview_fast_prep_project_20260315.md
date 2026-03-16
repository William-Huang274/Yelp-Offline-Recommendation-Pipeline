# Project Interview Fast-Prep Outline (2026-03-15)

## 1. Purpose

This document is not meant to turn someone into the person who can immediately
rewrite the whole repository.

Its purpose is narrower and more practical:

- quickly build enough system understanding, metric understanding, design-tradeoff
  language, and safe role positioning to talk about this project in interviews
  without overstating personal coding ownership

Best fit:

- rapid preparation within 3 to 7 days
- recommender, ranking, retrieval, or LLM-algorithm interview loops

## 2. Minimum Safe Interview Line

If time is limited, reach these six checkpoints first:

1. draw the main system flow
2. explain the best final solution clearly
3. memorize the key metrics
4. explain why evaluation alignment matters
5. distinguish `stage10` from `stage11`
6. explain what you personally owned and what you did not

## 3. Versions You Must Be Able To Recite

### 3.1 One-sentence version

This is an offline Yelp recommendation and reranking system for Louisiana
restaurant discovery, built around multi-route candidate retrieval, structured
reranking, and SFT/DPO-based LLM sidecar reranking.

### 3.2 Thirty-second version

The project focuses on offline recommendation reranking. Upstream stages build
merchant relabeling, user profiles, item semantics, and a multi-route candidate
pool. Downstream stages compare structured reranking with XGBoost against an
LLM sidecar path trained with SFT and DPO. The most important engineering work
was aligning all experiments under the same evaluation contract so that the
traditional model and LLM reranker could be compared fairly.

### 3.3 Ninety-second version

The project builds a complete offline recommendation-and-rerank pipeline for the
Yelp restaurant domain in Louisiana. It starts from merchant relabeling and
clustering, then builds user profiles and item semantic assets, and finally
produces a multi-route candidate pool. The ranking layer includes both a
structured XGBoost reranker and an LLM sidecar reranker trained with SFT and
DPO.

The most valuable part of the project is not simply that it fine-tuned a model.
The more important contribution is that it repaired the comparability problem
between experiments by aligning recall source, evaluation-user cohort, and
candidate truncation. Under that unified contract, `QLoRASidecar@10` reaches
`0.0678` recall, exceeding both `XGBoost` at `0.0650` and the baseline at
`0.0569`.

## 4. Metrics You Must Remember

### 4.1 Retrieval-side metrics

- `truth_in_all = 0.947208`
- `truth_in_pretrim = 0.882255`
- `hard_miss = 0.052792`

### 4.2 Ranking-side metrics

- `PreScore@10 recall = 0.056911`
- `LearnedBlendXGBCls@10 recall = 0.065041`
- `QLoRASidecar@10 recall = 0.067751`

### 4.3 Unified evaluation contract

- `source_run_09 = 20260311_005450_full_stage09_candidate_fusion`
- `eval_users = 738`
- `candidate_topn = 250`
- `top_k = 10`

## 5. System Structure You Must Be Able To Explain

Use this order in interviews:

1. raw Yelp data
2. merchant relabeling and clustering
3. user profiles
4. item semantics
5. multi-route candidate retrieval
6. structured reranking
7. LLM sidecar reranking
8. offline evaluation
9. release freeze and rollback

Do not start by listing stage numbers. Start from system functions and only map
back to `01-11` if the interviewer asks.

## 6. The Five Points You Should Emphasize Most

### 6.1 Unified evaluation contract

One of the best parts of the project is that `stage10` and `stage11` were pulled
back onto the same contract:

- same recall source
- same evaluation-user set
- same candidate truncation rule

Without this step, the traditional reranker and LLM reranker would not be
comparable.

### 6.2 Stable retrieval foundation

The `stage09` retrieval metrics are reasonably stable, which means later ranking
optimization is built on a credible candidate pool rather than on unstable
recall behavior.

### 6.3 The LLM does not replace the recommendation system end to end

In this project the LLM is a sidecar reranker placed after candidate generation,
not an end-to-end recommender.

That means the system is:

- more controllable
- easier to compare against traditional models
- easier to roll back

### 6.4 SFT and DPO are training methods inside the project, not the whole project

Do not tell the story as ?we built a DPO project.? A better framing is:

- the main recommender pipeline incorporates SFT and DPO as rerank-training
  options
- the final comparison is about the whole reranking chain, not one isolated loss

### 6.5 The project includes release closeout, not only experiments

Be ready to mention:

- release pointer
- smoke tests
- release validation
- rollback
- monitoring
- data contract

This shows that the repository is not just a pile of experiment scripts.

## 7. Safest Way To Describe Personal Role

If you were not the primary line-by-line code author, avoid saying:

- ?I independently built the entire stack?
- ?I led implementation of every script?
- ?All the training code was written by me?

Safer phrasing:

- I mainly drove problem breakdown, solution planning, evaluation-contract
  design, experiment alignment, and result closeout.
- I participated deeply in defining the reranking strategy, the SFT/DPO
  training path, and the release-freeze standards.
- My role was not just reading outputs. I focused on comparability, delivery,
  and whether the experimental results were actually usable.

Safe one-sentence version:

I mainly owned system framing, evaluation, and experiment planning, and I also
participated in the design and validation of key training and evaluation flows.

## 8. Questions You Are Most Likely To Get

### 8.1 What problem does this project actually solve?

Answer:

It puts candidate retrieval, structured reranking, and LLM reranking into the
same offline recommendation pipeline, then compares their real gains under a
unified evaluation contract.

### 8.2 Why did the project use both SFT and DPO?

Answer:

SFT teaches the model the basic reranking task in the candidate-pool setting,
while DPO further teaches relative preference between candidates. They are part
of the rerank-training stack rather than the whole system.

### 8.3 Why not let the LLM recommend directly?

Answer:

An end-to-end LLM recommender is harder to control, harder to compare, and
harder to roll back. A sidecar reranker is easier to manage operationally.

### 8.4 What was the hardest problem in the project?

Pick one of these as the main answer:

- the traditional reranker and LLM reranker originally did not share the same
  evaluation contract
- training and evaluation stability under single-machine constraints
- connecting retrieval and reranking while keeping a clean release freeze

### 8.5 Why is the observed lift worth talking about?

Answer:

Because the gain was measured under a unified contract and on top of an already
strong structured reranker. It is not a fake uplift caused by changed data or
changed evaluation boundaries.

## 9. Things You Should Know But Do Not Need To Memorize Line By Line

### 9.1 `stage09`

Know that it is the candidate-retrieval mainline and that it combines user
profiles, item semantics, and multi-route fusion with a strong focus on keeping
truth coverage high.

### 9.2 `stage10`

Know that it is the structured-rerank path, centered on XGBoost-style models and
structured features rather than generative text modeling.

### 9.3 `stage11`

Know that it is the LLM sidecar mainline:

- `11_1`: training-data build
- `11_2`: SFT / DPO training
- `11_3`: sidecar evaluation

## 10. Recommended Reading Order For Fast Prep

### Day 1

Read these first:

- [first_champion_closeout_20260313.md](./first_champion_closeout_20260313.md)
- [gl07_stage10_stage11_alignment_20260313.md](./gl07_stage10_stage11_alignment_20260313.md)
- [stage09_reaudit_20260313.md](./stage09_reaudit_20260313.md)
- [release_readiness_report_internal_pilot_v1_champion_20260313.md](./release_readiness_report_internal_pilot_v1_champion_20260313.md)

Goal:

- memorize the system story, key metrics, and final conclusion

### Day 2

Read the most important scripts and connect them to the document story:

- `scripts/09_candidate_fusion.py`
- `scripts/10_1_rank_train.py`
- `scripts/10_2_rank_infer_eval.py`
- `scripts/11_2_dpo_train.py`
- `scripts/11_3_qlora_sidecar_eval.py`

### Day 3

Practice short spoken answers and write one-page notes for:

- system flow
- evaluation alignment
- your personal role
- why the LLM is a sidecar instead of a direct recommender

## 11. Things Not To Say In Interviews

- ?This is just an LLM fine-tuning project.?
- ?We let the model directly generate recommendations.?
- ?I wrote all the code myself? when that is not true.
- ?The uplift is large because the model is better? without mentioning the
  aligned evaluation contract.

## 12. Better Ways To Say It

- ?This is an offline recommendation-and-rerank pipeline with a retrieval base,
  a structured fallback, and an LLM sidecar champion path.?
- ?One of the most valuable pieces of work was making the experiments comparable
  under the same contract.?
- ?The LLM was introduced as a controllable reranking layer, not as a full
  system replacement.?
- ?My strongest ownership was in experiment alignment, evaluation framing, and
  delivery-quality closeout.?

## 13. Final Checklist

- I can explain the system in one sentence, 30 seconds, and 90 seconds.
- I can recite the key recall and ranking metrics.
- I can explain the aligned evaluation contract.
- I can distinguish stage09, stage10, and stage11 clearly.
- I can describe my role truthfully and safely.
- I can explain why the LLM is a sidecar reranker.
- I can say why the final lift is credible.
