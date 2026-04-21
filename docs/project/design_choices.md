# Design Choices

[English](./design_choices.md) | [中文](./design_choices.zh-CN.md)

This page concentrates the main design decisions moved out of the root README.

## Why bounded LLM reranking instead of full-list reranking?

The project does not let the LLM or reward model rerank the full candidate
list.

It only reranks the `Stage10` output window, for four reasons:

- lower cost
- more controllable behavior
- easier rollback
- smaller front-rank disruption

That keeps `Stage10` as the global ranking backbone while `Stage11` acts as a
bounded gain layer.

## Why validate on `bucket10` before expanding to `bucket5`?

This sequencing was deliberate.

- `bucket10` is the high-interaction set with richer user semantics and cleaner
  behavioral evidence.
- It was the best place to validate whether recall routing, structured
  reranking, and rescue reranking worked as a system.
- After that direction was validated, the same design expanded to `bucket5` for
  broader coverage closer to the main outward-facing line.
- `bucket2` is kept to test whether the `Stage09 -> Stage10` path remains
  portable under colder user settings.

## How leakage is controlled

The Stage11 line is built to avoid inference-time label leakage.

- expert routing uses the candidate's current rank window, not the hidden truth band
- shortlist reranking uses current scores and current ranks only
- supervision uses true labels in training and evaluation, but those labels are not exposed at inference time

## Current Stage Responsibilities

### Recall Routing (`Stage09`)

This layer organizes candidates from multiple sources into one candidate pool
and controls recall budgets, candidate lanes, and challenger lanes.

### Structured Reranking (`Stage10`)

This layer is the global ranking backbone. It consumes match, text,
relative-cross, and group-gap features and produces the main ranked list.

### Rescue Reranking (`Stage11`)

This layer operates only on the bounded candidate window produced by `Stage10`.
The current design combines:

- segmented expert checkpoints
- shortlist reranking
- bounded gate and protection rules
