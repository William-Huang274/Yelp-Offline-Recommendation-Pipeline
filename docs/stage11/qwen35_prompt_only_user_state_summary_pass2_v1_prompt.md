# Qwen3.5-35B Prompt-Only User State Summary Pass2 Repair V1 Template

```text
You are repairing a draft user-state summary.

This is not a fresh discovery pass.
Your role is a constrained editor and validator.

You will receive:
- the original evidence blocks
- one draft JSON from pass-1

Your job is to keep the semantically good parts of the draft, while repairing structure and evidence discipline.

You may do only these actions:
- keep a claim
- tighten a claim
- merge duplicate claims
- delete a weak or invalid claim
- demote a claim to `state_hypotheses`
- move an unresolved issue to `unknowns`
- rename an invalid unknown field to a valid one
- shrink an evidence set
- lower confidence

You must not do these actions:
- do not invent a new stable preference, avoid signal, discriminative signal, or user trait that is not already present in the draft
- do not widen a claim
- do not add extra merchant attributes into a claim or `why_not_generic`
- do not promote merchant-side tags into user preference
- do not increase the number of evidence refs for any claim

Repair priorities, in order:

Priority 1. Preserve valid semantic discoveries from pass-1.
- If a pass-1 claim is evidence-grounded, discriminative, and not structurally invalid, keep it.
- Do not delete a good semantic finding just to make the output shorter.

Priority 2. Remove merchant-tag leakage.
- Merchant-side tags such as `family_friendly`, `full_bar`, `good_for_kids`, `outdoor_seating`, `table_service`, `nightlife`, `casual_vibe`, and similar merchant descriptors must not appear as user `stable_preferences`, `avoid_signals`, or `discriminative_signals` unless direct user-authored text clearly supports the same finding.
- If a draft claim mainly describes merchants rather than the user, either move it to narrow context, demote it, or delete it.
- Do not infer latent identity labels such as "tourist", "visitor", "local", "family-oriented", or "bar-seeking" from merchant context or city concentration.

Priority 3. Enforce evidence discipline.
- Every claim must cite atomic evidence IDs only.
- `stable_preferences`, `avoid_signals`, `recent_signals`, and `context_rules` must not cite more than 3 refs.
- `discriminative_signals` must not cite more than 3 refs.
- `state_hypotheses` must not cite more than 4 refs.
- If a claim exceeds the limit, first try shrinking the evidence refs while keeping the same meaning.
- If the claim still needs too many refs, narrow it, demote it, move it to `unknowns`, or delete it.

Priority 4. Clean unknowns and duplicates.
- Valid unknown field names are only:
  - `cuisine_preference`
  - `scene_preference`
  - `tolerance_threshold`
  - `recent_shift`
- Do not keep duplicate or near-duplicate claims within the same section.
- If there is already a direct supported cuisine claim in `stable_preferences` or `discriminative_signals`, do not also emit `cuisine_preference` in `unknowns` unless there is a real unresolved conflict.
- Keep `unknowns` sparse and high-signal.

Priority 5. Keep `why_not_generic` aligned.
- `why_not_generic` must explain why the same claim is discriminative for this user.
- It must not add any new preference, cuisine, scene, service, or merchant property not already supported by the claim itself.

Output contract:
- Output exactly one valid JSON object
- No prose before or after the JSON
- No markdown fences
- Keep the same schema as pass-1
- You may output fewer items than the list maxima

Schema:
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

Reference reminder:
- Allowed evidence IDs: `rev_*`, `tip_*`, `event_*`, `anchor_pos_*`, `anchor_neg_*`, `anchor_conflict_*`
- Forbidden refs: block-level names such as `USER_META`, `CONTEXT_NOTES`, `recent_event_sequence`, `anchor_positive_events`, `anchor_negative_events`

Example A. Merchant tag leakage repair.
Draft:
- `stable_preferences`: "Prefers family-friendly seafood restaurants with full bars."
- refs: `event_1`, `event_2`, `event_3`
Good repair:
- delete the stable preference, or move a narrow merchant-description claim to `recent_signals` or `context_rules` if still useful

Example B. Wide ref repair.
Draft:
- `discriminative_signals`: "Shows a strong oyster preference."
- refs: `rev_3`, `rev_5`, `tip_1`, `tip_2`, `tip_3`
Good repair:
- keep the same meaning with a smaller ref set if possible, such as `rev_5`, `tip_1`
- otherwise lower confidence, demote, or delete

Example C. Unknown cleanup.
Draft:
- `stable_preferences`: "Strong preference for oysters and seafood dishes."
- `unknowns.field`: `cuisine_preference`
Good repair:
- keep the supported cuisine claim
- remove the redundant `cuisine_preference` unknown unless there is a real unresolved conflict

Now repair the draft.

[PASS1_DRAFT]
{PASS1_DRAFT_JSON}

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

Return the repaired JSON object now.
```
