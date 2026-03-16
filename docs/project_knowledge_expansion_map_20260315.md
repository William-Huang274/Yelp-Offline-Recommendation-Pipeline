# Project Knowledge Expansion Map (2026-03-15)

## 1. Purpose

This document is a long-term study roadmap rather than an interview script.

Its goal is to help someone start from this repository and systematically build
knowledge in recommender systems, ranking, SFT/DPO, offline evaluation, data
pipelines, and engineering delivery.

## 2. Learning Principles

### 2.1 Learn the system before the implementation details

Understand what problem each module solves before diving into model internals or
individual script flags.

### 2.2 Learn the mainline before the side branches

Prioritize:

- recommender-system mainline
- ranking and evaluation
- SFT / DPO
- data and release engineering

Then study:

- environment-variable variations
- historical experiment branches
- low-level performance tuning details

### 2.3 Always answer two questions for every topic

1. What problem does this topic solve in the current repository?
2. What alternative approach could have been used instead?

## 3. Overall Knowledge Tree

Use the project as the center and expand into eight modules:

1. recommender-system foundations
2. candidate retrieval and multi-route fusion
3. structured reranking and learning to rank
4. offline evaluation and experiment alignment
5. SFT
6. DPO
7. data pipeline and Spark
8. freeze, validation, release, and rollback

## 4. Module 1: Recommender-System Foundations

### 4.1 Why learn it

Without recommender-system basics, it is easy to misread the repository as ?an
LLM fine-tuning project.? The real root of the project is still recommendation.

### 4.2 Where it appears in this repository

- user profiles
- item semantics
- candidate retrieval
- reranking

### 4.3 Must-learn topics

- the standard recommender chain: retrieval, coarse rank, fine rank, rerank
- collaborative filtering vs content understanding
- user features, item features, interaction features
- why a recommender rarely uses one model for the whole flow

### 4.4 Read first

- [data_contract.md](./data_contract.md)
- [09_candidate_fusion.py](../scripts/09_candidate_fusion.py)

### 4.5 You should be able to answer

- why retrieval happens before reranking
- what retrieval optimizes vs what reranking optimizes
- why the candidate pool is the foundation of the system

## 5. Module 2: Candidate Retrieval And Multi-Route Fusion

### 5.1 Why learn it

`stage09` is one of the most stable and mature parts of the project. If you do
not understand retrieval, you cannot explain why downstream ranking results are
credible.

### 5.2 Where it appears

- `09_user_profile_build.py`
- `09_item_semantic_build.py`
- `09_candidate_fusion.py`
- `09_1_recall_audit.py`

### 5.3 Must-learn topics

- what multi-route candidate retrieval means
- how user profiles participate in recall
- how item semantic signals participate in recall
- what retrieval audit metrics mean
- what `truth_in_all` and `truth_in_pretrim` tell you

### 5.4 Key evidence

- [stage09_reaudit_20260313.md](./stage09_reaudit_20260313.md)

### 5.5 You should be able to answer

- why `truth_in_pretrim` matters
- why recall cannot be judged only by final `@10`
- why multi-route fusion is better than a single-route retriever

## 6. Module 3: Structured Reranking And Learning To Rank

### 6.1 Why learn it

Without reranking knowledge, you cannot explain:

- why XGBoost is still a strong baseline
- why the LLM is not a direct replacement for structured reranking

### 6.2 Where it appears

- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`

### 6.3 Must-learn topics

- basic learning-to-rank framing
- pointwise vs pairwise vs listwise tasks
- why structured-feature reranking is strong
- why XGBoost appears so often in ranking pipelines
- the boundary between structured reranking and LLM reranking

### 6.4 You should be able to answer

- what kinds of features the XGBoost reranker consumes
- why XGBoost remains an important fallback
- why LLM gains usually build on top of a strong structured baseline

## 7. Module 4: Offline Evaluation And Experiment Alignment

### 7.1 Why learn it

This is one of the most distinctive parts of the repository.

### 7.2 Where it appears

- [gl07_stage10_stage11_alignment_20260313.md](./gl07_stage10_stage11_alignment_20260313.md)
- [release_readiness_report_internal_pilot_v1_champion_20260313.md](./release_readiness_report_internal_pilot_v1_champion_20260313.md)

### 7.3 Must-learn topics

- Recall@K
- NDCG@K
- coverage
- user cohort
- candidate truncation contract
- source lineage
- apples-to-apples comparison

### 7.4 You should be able to answer

- why results from different splits cannot be compared directly
- why a unified evaluation contract is more important than a single lucky uplift
- how this project repaired experiment comparability

## 8. Module 5: SFT

### 8.1 Why learn it

If you want to extend the project toward LLM algorithm work, SFT is the first
layer.

### 8.2 Where it appears

- `11_1_qlora_build_dataset.py`
- `11_2_qlora_train.py`
- [stage11_cloud_run_profile_20260309.md](./stage11_cloud_run_profile_20260309.md)

### 8.3 Must-learn topics

- the purpose of SFT
- why SFT can be used for reranking
- how prompts and training samples are built
- why sequence-length control matters
- why quantized training is used

### 8.4 Project-specific entry points

- `full_lite + rich_sft`
- sample-length distribution
- parameter closeout under single-card 32GB conditions

### 8.5 You should be able to answer

- what the SFT stage is learning here
- why `rich_sft` exists
- why `batch`, `grad_acc`, and `seq_len` influence stability

## 9. Module 6: DPO

### 9.1 Why learn it

DPO is one of the most likely topics that interviewers will follow up on.

### 9.2 Where it appears

- `11_2_dpo_train.py`
- [stage11_cloud_run_profile_20260309.md](./stage11_cloud_run_profile_20260309.md)

### 9.3 Must-learn topics

- difference between DPO and SFT
- what a preference pair is
- why ranking tasks benefit from preference learning
- why DPO usually builds on top of SFT
- why DPO is sensitive to memory and sequence length

### 9.4 You should be able to answer

- what pairwise learning changes compared with pointwise training
- why DPO can improve sidecar reranking
- why memory constraints dominate many design decisions

## 10. Module 7: Data Pipeline And Spark

### 10.1 Why learn it

This repository is not just about models. It is also about building data assets,
tracking manifests, and keeping a pipeline stable on local hardware.

### 10.2 Where it appears

- `scripts/pipeline/`
- stage07 through stage10 scripts
- validator tools and internal pilot helpers

### 10.3 Must-learn topics

- Spark DataFrame execution model
- joins, group-bys, windows, and shuffles
- why repeated Spark-to-Pandas conversion is risky
- why release manifests and run metadata matter

### 10.4 Suggested reading combination

Read the implementation together with:

- `AGENTS.md`
- `docs/data_contract.md`
- `docs/rollback_and_monitoring.md`

### 10.5 You should be able to answer

- why the project protects local-machine stability so aggressively
- what kind of actions are expensive in Spark
- how release pointers and manifests make results auditable

## 11. Module 8: Freeze, Validation, Release, And Rollback

### 11.1 Why learn it

This is the bridge between experiment work and real project delivery.

### 11.2 Where it appears

- release-readiness reports
- monitoring docs
- rollback docs
- `_prod_runs` pointers and manifests

### 11.3 Must-learn topics

- why freezes are necessary
- what release validation checks
- what rollback-ready thinking looks like
- why the project keeps auditable pointers

### 11.4 You should be able to answer

- why the repository is more than a modeling sandbox
- how a frozen release is documented and validated
- why rollback matters even for an offline system

## 12. Suggested Four-Week Learning Order

### Week 1: system understanding

Focus on the overall pipeline, the contract, and the release story.

### Week 2: retrieval and reranking

Study stage09 and stage10 together so that the transition from recall to rank is
clear.

### Week 3: SFT and DPO

Understand how the project turns recommendation evidence into training data and
how sidecar evaluation works.

### Week 4: engineering delivery

Study manifests, validators, rollback notes, monitoring, and release freeze
material.

## 13. Suggested Outputs After Each Module

### Output 1: one-page notes

Summarize the problem, method, and tradeoffs.

### Output 2: three interview Q&A pairs

Force yourself to state the concept as spoken language.

### Output 3: one diagram

Draw the data flow, model flow, or evaluation flow from memory.

## 14. Final Goal

The final goal is not only to ?talk about the project.? The final goal is to
understand what methods the project uses, why those methods were chosen, what
tradeoffs they imply, and what alternatives would have been possible.
