# Challenges And Trade-Offs

## 1. Scale And Data Heterogeneity

The project is not driven by one clean interaction table. It combines:

- business metadata
- review logs
- tip logs
- check-in logs
- optional photo-derived summaries

This creates the first engineering challenge: data must be ingested, normalized,
and stored in a way that is Spark-friendly and cheap enough to inspect locally.

## 2. Sparse And Noisy Preference Signals

Restaurant recommendation on Yelp is difficult because:

- user-item interaction is sparse
- star ratings are noisy
- long-tail merchants are easily buried
- the same user can exhibit mixed short-term and long-term intent

This is why the repository moved away from a single-model story and toward a
three-stage stack.

## 3. Why Stage09 Became Route-Aware

Earlier candidate generation left too much quality on the floor before reranking.
The current Stage09 line improves:

- truth retention in the pretrim candidate surface
- hard-miss rate
- downstream handoff quality for Stage10 and Stage11

This is documented in:

- [../../docs/release/stage09_stage10_stage11_upgrade_audit_20260409.md](../../docs/release/stage09_stage10_stage11_upgrade_audit_20260409.md)

## 4. Why Stage10 Stays The Global Backbone

The repository does not treat the LLM/RM path as a full replacement for the
structured ranker.

Stage10 remains the global ordering backbone because it is:

- cheaper to run at scale
- easier to validate across buckets
- easier to compare against conventional recommenders

## 5. Why Stage11 Is Bounded

The current Stage11 path intentionally avoids full-list LLM reranking.

It only reranks a bounded candidate window because this reduces:

- compute cost
- latency and demo fragility
- rollback risk
- front-rank instability

It also helps maintain clear responsibility:

- Stage10 builds the global ranking skeleton
- Stage11 performs local rescue on underweighted candidates

## 6. Leakage Control

Inference-time label leakage is explicitly avoided.

Current rules include:

- route selection uses the candidate's current rank window, not hidden truth
  labels
- shortlist reranking uses current scores and current ranks only
- truth labels are used only in training and offline evaluation

## 7. Local Resource Constraints

The repository has to remain usable on a local Windows machine, not only on a
large cloud cluster.

Important practical implications:

- parquet is preferred over repeated raw-JSON processing
- Spark parallelism should stay moderate for local runs
- review and demo flows must be available from compact checked-in assets
- Stage11 training is separated from review-only workflows

## 8. Monitoring And Rollback

The repository already contains internal pilot monitoring and rollback rules:

- [../../docs/archive/release/rollback_and_monitoring.md](../../docs/archive/release/rollback_and_monitoring.md)

This matters for teacher review because it shows that the project is not only a
model experiment. It also includes:

- release snapshots
- rollback contracts
- monitor signals
- failure criteria

## 9. Current Limits

The following are deliberate boundaries of the current freeze line:

- the tri-band Stage11 line should not yet be described as the final production
  champion
- the `61-100` expert is intentionally conservative
- the repository keeps compact release files, not full prediction dumps
- individual reflection and peer review are outside the shared code repository

## 10. Recommended Way To Explain The Trade-Offs

In proposal, demo, and report form, the engineering trade-off can be summarized
as:

1. store and process large raw Yelp data efficiently
2. build a robust global recommender first
3. use the LLM/RM path only where it adds the most value and the least risk
4. keep the release line explainable, auditable, and easy to review
