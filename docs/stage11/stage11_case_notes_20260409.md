# Stage11 Case Notes

[English](./stage11_case_notes_20260409.md) | [中文](./stage11_case_notes_20260409.zh-CN.md)

## 1. Purpose

This note expands the short case references kept in the root README.

It covers three things:

1. how the Stage11 prompts are constructed
2. why `11-30` and `31-60` truth items are suitable for reward-model rescue
3. why the `61-100` expert is still used conservatively in the freeze line

## 2. Prompt Construction Sample

The Stage11 prompt does not tell the model which candidate is the hidden truth.
Instead, it builds a local comparison task from currently visible information:

- recurring user preferences
- recent intent
- explicit avoidance cues
- candidate business evidence
- local rivals from nearby rank windows
- the candidate's current rank window

A typical `31-60` training prompt looks like this:

```text
You are a recommendation ranking judge.
Evaluate one focus candidate inside a small local rescue slate for the same user.
Current rank is context only and is not by itself the answer.

User:
- user_focus
- recent_intent
- user_avoid
- history_pattern
- user_evidence

Local ranking context:
- Focus candidate is in the mid rescue band (31-60)
- Local slate contains 4 rivals
- same_band=1, boundary=2, head_anchor=1, deep=0

Focus candidate:
- business_profile
- user_match_points
- user_conflict_points

Rivals 1-4:
- same-slice blocker / boundary blocker / head anchor
```

This matters because the model is trained to solve a local ranking question
without access to inference-time labels. The current rank window is visible;
the hidden truth band is not.

## 3. Case 1: Why an `11-30` Truth Item Can Be Rescued

Example user:

- `user_idx = 1072`
- truth `item_idx = 58`
- `learned_rank = 17`
- `blend_rank = 4`
- `final_rank = 8`
- `route_band = boundary_11_30`
- `reward_score = 13.4375`
- `sidecar_norm = 0.9474`
- `rescue_bonus = 0.7579`

This is a boundary case, not a deep recovery case.

The structured reranker already moves the truth item close to the front, but the
final top10 still contains many strong-looking boundary and `31-40` candidates.
The reward model helps because the remaining decision is highly local:

- which of a few near-front competitors is actually the best fit for this user
- whose semantic match is stronger once user preference, recent intent, and
  avoidance cues are considered together

This is where the reward model is useful. It does not replace the global ranking
backbone; it makes a finer local decision near the boundary.

## 4. Case 2: Why a `31-60` Truth Item Can Be Rescued

Example user:

- `user_idx = 1940`
- truth `item_idx = 92`
- `learned_rank = 36`
- `blend_rank = 1`
- `final_rank = 2`
- `route_band = rescue_31_40`
- `reward_score = 11.3125`
- `sidecar_norm = 1.0000`
- `rescue_bonus = 1.0350`

This user still has many `11-30` candidates in the final top10. The truth item
is not promoted because of a blind score boost. It is promoted because, inside
the local rescue slate, it is clearly stronger than nearby blockers after the
model compares:

- user-specific semantic fit
- conflict and friction signals
- local competition rather than the full ranked list

This is exactly the kind of case where a reward model is more appropriate than
trying to force a full-list rerank. The model only needs to answer a local
question: should this mid-rank candidate sit above these nearby blockers for
this user?

## 5. Case 3: Why `61-100` Is Still Used Conservatively

In the current freeze line, the `61-100` expert is trained and active, but the
policy remains conservative:

- improve deep-rank ordering first
- avoid aggressive front-rank promotion
- protect the stability of the current top ranks

This is intentional because deep-rank items are farther away from the front and
the cost of an unstable promotion policy is higher.

## 6. Current Interpretation

The current Stage11 line should be read in three parts:

1. `11-30` adds fine-grained boundary rescue near the front
2. `31-60` adds mid-rank local rescue and can surface underweighted candidates
   into the front ranks
3. `61-100` already learns useful signals, but is currently used as a
   conservative rank-uplift path rather than an aggressive promotion path
