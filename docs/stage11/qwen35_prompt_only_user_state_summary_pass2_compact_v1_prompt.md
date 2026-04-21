# Qwen3.5-35B Prompt-Only User State Summary Pass2 Compact Repair V1 Template

```text
You are repairing a pass-1 user-state draft.

This is an edit pass, not a discovery pass.

You will receive one compact JSON bundle containing:
- the pass-1 draft
- the exact evidence IDs already referenced by that draft
- a small fallback evidence slice only when the draft cited nothing from one block

Your job is to preserve good semantic discoveries while enforcing validity.

Hard rules:
- Do not invent any new stable preference, avoid signal, discriminative signal, or latent user trait.
- Do not widen a claim.
- Do not add new evidence refs that are not already shown in the compact bundle.
- Prefer delete, demote, shrink, merge, or lower confidence over rewriting into a broader claim.

Repair priorities, in order:

1. Finish with one valid JSON object.
- No prose before or after JSON.
- No markdown fences.

2. Preserve valid pass-1 semantics.
- If a claim is already specific, evidence-grounded, and structurally valid, keep it.

3. Remove merchant-tag leakage.
- Merchant-side tags like `family_friendly`, `full_bar`, `good_for_kids`, `outdoor_seating`, `table_service`, `nightlife`, `casual_vibe`, and similar descriptors must not appear as user `stable_preferences`, `avoid_signals`, or `discriminative_signals` unless direct user-authored text in the compact bundle clearly supports the same finding.
- Do not infer labels like `tourist`, `visitor`, `local`, `family-oriented`, or `bar-seeking`.

4. Enforce evidence discipline.
- `stable_preferences`, `avoid_signals`, `recent_signals`, `context_rules`, and `discriminative_signals`: at most 3 refs
- `state_hypotheses`: at most 4 refs
- If a claim still needs too many refs, shrink it, demote it, move it to `unknowns`, or delete it.

5. Clean duplicates and unknowns.
- Valid unknown field names are only:
  - `cuisine_preference`
  - `scene_preference`
  - `tolerance_threshold`
  - `recent_shift`
- Remove duplicate or near-duplicate claims within the same section.
- If a direct supported cuisine claim already exists, do not also emit `cuisine_preference` in `unknowns` unless there is a real unresolved conflict.

6. Keep `why_not_generic` narrow.
- It may explain why the claim is discriminative.
- It must not add any new cuisine, scene, service, merchant property, or user trait.

Allowed actions:
- keep
- merge duplicates
- delete
- demote to `state_hypotheses`
- move to `unknowns`
- rename an invalid unknown field
- shrink evidence refs
- lower confidence

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

List size limits:
- `stable_preferences`: at most 3
- `avoid_signals`: at most 3
- `recent_signals`: at most 2
- `context_rules`: at most 2
- `state_hypotheses`: at most 3
- `discriminative_signals`: at most 2
- `unknowns`: at most 2

Compact input:
{PASS1_DRAFT_JSON}

Return the repaired JSON object now.
```
