# Qwen3.5-35B Prompt-Only User State Summary V8 Template

```text
You are analyzing one user's training-period dining behavior and text evidence.

Your job is not to imitate a rule-based profile.
Your job is to discover, extract, and summarize this user's state from the evidence.

Your output must be evidence-grounded, discriminative, and appropriately uncertain.

You must follow these rules:

1. You may output nuanced or uncommon findings if the evidence supports them.
2. Every claim must cite atomic evidence IDs only.
3. If the evidence is insufficient, mixed, or only weakly implied, use `unknowns` instead of forcing a conclusion.
4. Do not invent stable preferences from mere visits, popular merchants, convenience, or broad context alone.
5. Prefer discriminative, user-specific findings over generic head-category summaries.
6. Prefer fewer, better-supported claims over broad or repetitive lists.
7. Output only one valid JSON object. No prose before or after the JSON. No markdown code fences.

Allowed evidence reference IDs:
- review or tip evidence: `rev_*`, `tip_*`
- recent events: `event_*`
- anchor events: `anchor_pos_*`, `anchor_neg_*`, `anchor_conflict_*`

Forbidden evidence references:
- `USER_META`
- `CONTEXT_NOTES`
- `recent_event_sequence`
- `anchor_positive_events`
- `anchor_negative_events`
- any other block-level name

Atomic evidence rules:
- Use the smallest sufficient subset of atomic IDs.
- Prefer 1 to 2 atomic refs per claim.
- 3 refs is the preferred upper bound.
- `grounded_facts` and `discriminative_signals` must not cite more than 3 refs.
- `state_hypotheses` may cite 4 refs only if the claim is still narrow and concrete.
- Do not cite long runs like `event_1` through `event_8`.
- If a claim would require more than 4 refs, narrow the claim, split it, demote it to a hypothesis, or move it to `unknowns`.
- If the claim is specifically about a recent sequence pattern, prefer 2 to 3 `event_*` refs rather than the whole visible sequence.
- Prefer `rev_*`, `tip_*`, and anchor refs over many `event_*` refs when both support the same claim.
- A claim supported mainly by a wide set of weak events is too broad for `grounded_facts`; narrow it or move it out of `grounded_facts`.

Claim scope rules:
- `grounded_facts` must be narrow enough to survive on a small atomic evidence set.
- Do not trade narrower evidence discipline for a more aggressive claim.
- If the evidence only supports a broad pattern across many events, express it as a low- or medium-confidence hypothesis rather than a hard fact.
- `discriminative_signals` should cite the single strongest supporting evidence subset, not the union of all related events.
- Avoid "preference bundles" that join multiple cuisines, scenes, properties, or merchant types in one claim unless the same small evidence subset directly supports the combined claim.
- If a stable preference claim names more than one cuisine cluster or more than one merchant-style cluster, split it unless the same 1 to 3 refs support the whole bundle.
- If a claim needs many refs because it is really summarizing a theme across many visits, that is a hypothesis, not a grounded fact.

Uncertainty rules:
- Think about these major dimensions:
  - cuisine_preference
  - scene_preference
  - service_tolerance
  - recent_shift
- Add an `unknowns` entry only if that dimension remains genuinely unresolved at the user-state level.
- Do not add `unknowns` just because the evidence does not support a broader generalization beyond an already well-supported claim.
- If you already have a strong, specific state claim for a dimension, do not also mark that same dimension as unknown unless there is a clear unresolved conflict or threshold question.
- A specific supported cuisine preference is enough to resolve `cuisine_preference`; do not mark `cuisine_preference` unknown merely because the user's full cuisine range is broader or context-dependent.
- A specific supported service sensitivity claim is enough to resolve `service_tolerance`; use `tolerance_threshold` only when there is direct evidence of both tolerated and intolerable outcomes and the boundary meaningfully changes how the user should be modeled.
- If the evidence only shows general sensitivity to bad service or bad food quality, write that supported claim and omit `tolerance_threshold`.
- Use `recent_shift` unknown only when there is concrete recent evidence that could indicate either a real shift or a one-off fluctuation.
- If recent evidence is broadly consistent with the stable state, do not emit `recent_shift` unknown. Omit it.
- Prefer a narrow hypothesis over a generic unknown when the evidence supports a plausible but not fully proven explanation.
- Good `unknowns` are unresolved questions such as:
  - whether a preference is stable or context-dependent
  - whether tolerance is low in general or only for severe failures
  - whether a recent pattern is a real shift or one-off variation
- Bad `unknowns` are generic disclaimers such as:
  - “the user may like other cuisines too”
  - “the user might also like other scenes”
  - “the user's full cuisine range is unclear” when one or more specific cuisines are already well supported
  - “recent shift is unknown” when recent evidence simply continues the same pattern
  - `tolerance_threshold` unknown when the evidence only supports one-sided low tolerance

Output schema:
{
  "grounded_facts": {
    "stable_preferences": [{"claim": "", "confidence": "high|medium|low", "evidence_refs": []}],
    "avoid_signals": [{"claim": "", "confidence": "high|medium|low", "evidence_refs": []}],
    "recent_signals": [{"claim": "", "confidence": "high|medium|low", "evidence_refs": []}],
    "context_rules": [{"claim": "", "confidence": "high|medium|low", "evidence_refs": []}]
  },
  "state_hypotheses": [
    {
      "type": "conditional_preference|tolerance|shift|conflict|latent_preference|other",
      "claim": "",
      "confidence": "high|medium|low",
      "evidence_refs": []
    }
  ],
  "discriminative_signals": [
    {
      "claim": "",
      "why_not_generic": "",
      "confidence": "high|medium|low",
      "evidence_refs": []
    }
  ],
  "unknowns": [
    {
      "field": "",
      "reason": "",
      "evidence_refs": []
    }
  ],
  "confidence": {
    "overall": "high|medium|low",
    "coverage": "high|medium|low"
  }
}

List size limits:
- `stable_preferences`: at most 3
- `avoid_signals`: at most 3
- `recent_signals`: at most 2
- `context_rules`: at most 2
- `state_hypotheses`: at most 3
- `discriminative_signals`: at most 2
- `unknowns`: at most 2

Interpretation rules:

- `grounded_facts`:
  Only include claims that are directly and sufficiently supported.

- `state_hypotheses`:
  Use for plausible but not fully proven findings. These must still cite atomic evidence IDs.

- `discriminative_signals`:
  Include only findings that help distinguish this user from a generic diner. Avoid generic items like "likes dinner" unless unusually well-supported and specifically informative.

- `unknowns`:
  Use only when a meaningful user-state question remains unresolved after extracting the strongest supported claims.
  Keep `unknowns` sparse and high-signal.

- `confidence.overall`:
  Reflect how reliable the overall state summary is.

- `confidence.coverage`:
  Reflect how much of the user's behavior is actually explained by the available evidence.

Do not:
- output unsupported claims
- output evidence-free hypotheses
- cite forbidden block names
- confuse merchant popularity with user preference
- treat behavior-only weak signals as hard positive preference
- repeat the same idea across multiple sections
- emit markdown fences

Now analyze the following evidence.

[USER_META]
{USER_META_JSON}

[USER_EVIDENCE_STREAM]
{USER_EVIDENCE_STREAM_JSON}

[RECENT_EVENT_SEQUENCE]
{RECENT_EVENT_SEQUENCE_JSON}

[ANCHOR_POSITIVE_EVENTS]
{ANCHOR_POSITIVE_EVENTS_JSON}

[ANCHOR_NEGATIVE_EVENTS]
{ANCHOR_NEGATIVE_EVENTS_JSON}

[ANCHOR_CONFLICT_EVENTS]
{ANCHOR_CONFLICT_EVENTS_JSON}

[CONTEXT_NOTES]
{CONTEXT_NOTES_JSON}

Return the JSON object now.
```
