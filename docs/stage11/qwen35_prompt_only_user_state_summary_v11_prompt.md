# Qwen3.5-35B Prompt-Only User State Summary V11 Template

```text
You are analyzing one user's training-period dining behavior and text evidence.

Your job is not to imitate a rule-based profile.
Your job is to discover, extract, and summarize this user's state from the evidence.

Your output must be evidence-grounded, discriminative, and appropriately uncertain.

You must follow this decision order:

Step 1. Separate the evidence into support levels before writing any claim.
- Level A: direct user-authored evidence.
  Examples: review text and tip text where the user explicitly praises, dislikes, avoids, tolerates, or repeatedly mentions something.
- Level B: repeated behavioral pattern.
  Examples: a narrow repeated pattern across a small set of recent events or anchors that supports a concrete behavioral tendency.
- Level C: merchant-side context only.
  Examples: city concentration, merchant cuisines, scene tags, meal tags, property tags, merchant_state_v2, merchant popularity, or broad visit mix.

Step 2. Decide which output slots each support level may feed.
- `stable_preferences`, `avoid_signals`, and `discriminative_signals` require Level A evidence, or Level A plus a small amount of Level B support.
- `recent_signals` may use Level B evidence.
- `context_rules` may summarize narrow recurring context, but not latent identity or stable preference.
- `state_hypotheses` may use Level A plus Level B, or a narrow Level B pattern alone.
- Level C evidence alone must not produce a user preference, user trait, or discriminative signal.

Step 3. Run the promotion test before finalizing each claim.
- If all merchant tags disappeared, would this still be a valid user-state claim?
- If the answer is no, it cannot be a `stable_preference`, `avoid_signal`, or `discriminative_signal`.
- If the claim needs many refs because it is really summarizing a theme across many visits, narrow it, demote it, or omit it.

Global rules:

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
- No claim may cite more than 4 refs.
- `grounded_facts` and `discriminative_signals` must not cite more than 3 refs.
- `state_hypotheses` may cite 4 refs only if the claim is still narrow and concrete.
- Do not cite long runs like `event_1` through `event_8`.
- If a claim would require more than 4 refs, narrow the claim, split it, demote it to a hypothesis, move it to `unknowns`, or omit it.
- A claim with more than 4 refs is invalid even if the text sounds plausible.
- Prefer `rev_*`, `tip_*`, and anchor refs over many `event_*` refs when both support the same claim.

Evidence hierarchy rules:
- Merchant metadata, scene tags, meal tags, property tags, and `merchant_state_v2` are weak contextual hints, not direct user preferences.
- Do not convert merchant-side tags such as `family_friendly`, `full_bar`, `good_for_kids`, `outdoor_seating`, `table_service`, `nightlife`, `casual_vibe`, or similar tags into user `stable_preferences`, `avoid_signals`, or `discriminative_signals` unless direct user-authored text clearly supports the same finding.
- If a claim relies only on `event_*` refs and merchant-side tags, it may describe observed recent context in `recent_signals` or `context_rules`, but it must not be phrased as a stable preference or deep user trait.
- Do not infer latent identity or role labels such as "tourist", "visitor", "local", "family-oriented", "bar-seeking", or similar labels from city concentration, merchant properties, or broad sequence patterns alone.
- A user-level scene or property preference must be supported by direct user-authored text or a tight repeated cross-merchant pattern with a small evidence set.
- If the claim is mostly a description of the merchants rather than the user, it belongs in narrow context at most, and often should be omitted.

Claim scope rules:
- `grounded_facts` must be narrow enough to survive on a small atomic evidence set.
- Do not trade narrower evidence discipline for a more aggressive claim.
- If the evidence only supports a broad pattern across many events, express it as a low- or medium-confidence hypothesis rather than a hard fact.
- `discriminative_signals` should cite the single strongest supporting evidence subset, not the union of all related events.
- Avoid "preference bundles" that join multiple cuisines, scenes, properties, or merchant types in one claim unless the same small evidence subset directly supports the combined claim.
- If a stable preference claim names more than one cuisine cluster or more than one merchant-style cluster, split it unless the same 1 to 3 refs support the whole bundle.
- Prefer omission over generic filler. Not every user needs a scene claim or a context claim.

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
- A specific supported service sensitivity claim is enough to resolve `service_tolerance`.
- Use `tolerance_threshold` only when there is direct evidence of both tolerated and intolerable outcomes and the unresolved boundary materially changes how the user should be modeled.
- If the evidence only shows one-sided low tolerance or one-sided high tolerance, write the supported claim and omit `tolerance_threshold`.
- Use `recent_shift` unknown only when there is direct recent evidence pointing in a meaningfully different direction from the stable state and you still cannot tell whether it is a real shift or a one-off fluctuation.
- If recent evidence is broadly consistent with the stable state, do not emit `recent_shift` unknown. Omit it.
- Do not use `recent_shift` unknown just because recent evidence is sparse.
- Prefer a narrow hypothesis over a generic unknown when the evidence supports a plausible but not fully proven explanation.

Why-not-generic rules:
- `why_not_generic` must explain why the same claim is discriminative for this user.
- `why_not_generic` must stay semantically aligned with the `claim`.
- Do not introduce any new preference, tolerance, scene, cuisine, or user trait in `why_not_generic` that is not already supported by the claim's own evidence.
- Do not use `why_not_generic` to widen the claim, add extra merchant attributes, or smuggle in unsupported conclusions.

Examples of correct behavior:

Example 1. Merchant tag leak.
Bad:
- `stable_preferences`: "Prefers family-friendly restaurants with full bars."
- refs: `event_1`, `event_2`, `event_3`
Why bad:
- This is merchant-side context only. It is not direct user preference evidence.
Better:
- `recent_signals`: "Recent visits include restaurants tagged for group or family dining."
- refs: `event_1`, `event_3`
Best:
- Omit the claim entirely unless review or tip text directly supports it.

Example 2. Tourist or visitor inference.
Bad:
- `context_rules`: "User is likely a tourist in New Orleans."
- refs: `event_1`, `event_2`, `event_3`
Why bad:
- City concentration alone does not support a latent identity label.
Better:
- `context_rules`: "Recent activity is concentrated in New Orleans."
- refs: `event_1`, `event_2`

Example 3. Wide evidence group.
Bad:
- `stable_preferences`: "Prefers seafood and Cajun/Creole restaurants in lively settings."
- refs: `rev_5`, `rev_6`, `rev_8`, `tip_1`, `tip_2`, `tip_3`, `tip_4`
Why bad:
- Too broad, too many refs, and bundles multiple ideas.
Better:
- `stable_preferences`: "Strong preference for oysters and seafood dishes."
- refs: `rev_8`, `tip_1`
- `state_hypotheses`: "May also favor Cajun/Creole dishes."
- refs: `rev_5`, `rev_6`

Example 4. Why-not-generic widening.
Bad:
- `claim`: "Strong preference for oysters."
- `why_not_generic`: "Shows a broader preference for nightlife and family-friendly seafood venues."
Why bad:
- The explanation introduces new scene claims not supported by the same evidence.
Better:
- `claim`: "Strong preference for oysters."
- `why_not_generic`: "The user repeatedly mentions oysters directly rather than only generic seafood."

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
- confuse merchant tags with user preference
- treat behavior-only weak signals as hard positive preference
- use merchant-only context to create user traits
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
