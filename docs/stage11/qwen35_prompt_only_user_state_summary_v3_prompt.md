# Qwen3.5-35B Prompt-Only User State Summary V3 Template

```text
You are analyzing one user's training-period dining behavior and text evidence.

Your job is not to imitate a rule-based profile.
Your job is to discover, extract, and summarize this user's state from the evidence.

Your output must be evidence-grounded, discriminative, and appropriately uncertain.

You must follow these rules:

1. You may output nuanced or uncommon findings if the evidence supports them.
2. Every claim must cite atomic evidence IDs only.
3. If the evidence is insufficient, mixed, or only weakly implied, you must say so in `unknowns` instead of forcing a conclusion.
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
- Prefer 1 to 4 atomic refs per claim.
- Do not cite long runs like `event_1` through `event_8` unless the claim is truly about a broad sequence pattern.
- If a claim needs more than 4 refs to survive, it is probably too broad and should be narrowed or moved to `unknowns`.

Uncertainty rules:
- For each major dimension below, decide whether the evidence is strong enough:
  - cuisine_preference
  - scene_preference
  - service_tolerance
  - recent_shift
- If a major dimension is not strongly supported, add an entry in `unknowns`.
- If the evidence is mixed or only establishment-specific, prefer `low` confidence or `unknowns`.
- It is acceptable for `unknowns` to be empty only when all major dimensions are sufficiently supported.

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
- `unknowns`: at most 4

Interpretation rules:

- `grounded_facts`:
  Only include claims that are directly and sufficiently supported.

- `state_hypotheses`:
  Use for plausible but not fully proven findings. These must still cite atomic evidence IDs.

- `discriminative_signals`:
  Include only findings that help distinguish this user from a generic diner. Avoid generic items like "likes dinner" unless unusually well-supported and specifically informative.

- `unknowns`:
  Use when cuisine preference, scene fit, tolerance, or shift cannot be safely concluded at the user-state level.
  Do not avoid `unknowns` just because a weak or broad summary sounds plausible.

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
