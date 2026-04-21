# Qwen3.5-35B Prompt-Only User State Summary Pass2 Compact Repair V2 Template

```text
You are repairing a pass-1 user-state draft.

This is an edit pass, not a discovery pass.

You will receive one compact JSON bundle containing:
- the pass-1 draft
- the exact evidence IDs already referenced by that draft
- the full set of allowed evidence IDs for this user
- any draft evidence IDs that are already known to be invalid
- a small fallback evidence slice only when the draft cited nothing from one block

Your job is to preserve good semantic discoveries while enforcing validity.

Hard rules:
- Do not invent any new stable preference, avoid signal, discriminative signal, or latent user trait.
- Do not invent any new contextual-inference item unless it is strictly needed to keep a valid, already-present contextual claim separate from Tier 1 or Tier 2.
- Do not widen a claim.
- Do not add new evidence refs that are not already shown in the compact bundle.
- Prefer delete, demote, shrink, merge, or lower confidence over rewriting into a broader claim.
- Every `evidence_refs` list in the final JSON must be a subset of `full_allowed_evidence_ids`.
- If the draft contains `draft_invalid_evidence_ids`, treat them as already-disallowed refs. Never copy them forward.
- If a claim depends on an invalid ref or a ref that is not present in `full_allowed_evidence_ids`, either replace it with valid refs from the compact bundle, demote it, or delete it.
- Keep or repair provenance fields. Every non-unknown item must contain `support_basis` and `support_note`.

Repair priorities, in order:

1. Finish with one valid JSON object.
- No prose before or after JSON.
- No markdown fences.

2. Preserve valid pass-1 semantics.
- If a claim is already specific, evidence-grounded, and structurally valid, keep it.

3. Remove merchant-tag leakage.
- Merchant-side tags like `family_friendly`, `full_bar`, `good_for_kids`, `outdoor_seating`, `table_service`, `nightlife`, `casual_vibe`, and similar descriptors must not appear as user `stable_preferences`, `avoid_signals`, or `discriminative_signals` unless direct user-authored text in the compact bundle clearly supports the same finding.
- Do not infer labels like `tourist`, `visitor`, `local`, `family-oriented`, or `bar-seeking`.
- If such content is still useful, move it into `contextual_inference_signals` with explicit reasoning instead of leaving it in Tier 1 or Tier 2.

4. Enforce evidence discipline.
- `stable_preferences`, `avoid_signals`, and `discriminative_signals` must each keep at least one `rev_*` or `tip_*` ref.
- Those three sections must not end up as `event_*`-only claims.
- `support_basis = "event_context_only"` is allowed only for `recent_signals`, `context_rules`, or `state_hypotheses`.
- If a `recent_signals`, `context_rules`, or `state_hypotheses` item ends up with only `event_*` refs and no `rev_*` / `tip_*`, set `support_basis = "event_context_only"`.
- `contextual_inference_signals` may use merchant-overlap or contextual inference, but must keep that evidence separate from Tier 1 and Tier 2.
- `contextual_inference_signals` must keep exactly one strongest item per `canonical_axis`.
- `stable_preferences`, `avoid_signals`, `recent_signals`, `context_rules`, and `discriminative_signals`: at most 3 refs
- `state_hypotheses`: at most 4 refs
- `contextual_inference_signals`: at most 4 total refs, usually 1 to 2 `contextual_refs` and 0 to 2 `direct_review_refs`
- `unknowns`: prefer 1 to 2 refs, not a long unresolved bundle
- `evidence_refs` are representative witness examples, not a full support dump
- When several refs say the same thing, keep only the smallest representative subset:
  - first keep the clearest direct wording or most specific dish / condition / complaint
  - add one reinforcing ref only if it materially strengthens the claim
  - keep a third ref only if it adds a distinct condition, conflict, or recency change
- If a claim still needs too many refs, shrink it, demote it, move it to `unknowns`, or delete it.
- Never keep an invalid claim with too many refs.
- Never emit `rev_*`, `tip_*`, `event_*`, or `anchor_*` IDs that are absent from `full_allowed_evidence_ids`, even if the pass-1 draft used them.
- For recurring context like city concentration, cite only 1 to 2 representative events, not the whole sequence.
- For dish or cuisine preferences, prefer the smallest direct review/tip subset, usually 1 to 3 refs.
- If a contextual inference still needs more than 2 `direct_review_refs` or more than 2 `contextual_refs`, rewrite it to a narrower claim or delete it.
- If a contextual inference has no `contextual_refs`, it does not belong in `contextual_inference_signals`. Demote it to `state_hypotheses` if it is review-only, or delete it.
- If two contextual items express the same underlying axis, merge them and keep only one final item.

5. Clean duplicates and unknowns.
- Valid unknown field names are only:
  - `cuisine_preference`
  - `scene_preference`
  - `tolerance_threshold`
  - `recent_shift`
- `service_tolerance` is invalid. Never output it.
- Use `tolerance_threshold` only when there is evidence of both tolerated and intolerable outcomes and the unresolved boundary matters.
- If the evidence shows only one-sided tolerance or only one-sided intolerance, write the supported claim or omit the unknown.
- Remove duplicate or near-duplicate claims within the same section.
- If a direct supported cuisine claim already exists, do not also emit `cuisine_preference` in `unknowns` unless there is a real unresolved conflict.

6. Keep `why_not_generic` narrow.
- It may explain why the claim is discriminative.
- It must not add any new cuisine, scene, service, merchant property, or user trait.

7. Keep provenance explicit.
- Allowed `support_basis` values:
  - `direct_user_text`
  - `review_pattern_inference`
  - `mixed_support`
  - `event_context_only`
- For `contextual_inference_signals`, allowed `support_basis` values are:
  - `merchant_overlap`
  - `contextual_inference`
  - `mixed_context`
- For `contextual_inference_signals`, allowed `canonical_axis` values are:
  - `localness_vs_touristiness`
  - `crowding_and_relaxedness`
  - `special_occasion_fit`
  - `scene_and_ambiance_fit`
  - `service_style_fit`
  - `value_and_price_context`
  - `family_and_group_context`
  - `beverage_and_bar_context`
  - `late_night_and_hours_fit`
  - `geography_and_neighborhood_pattern`
  - `cuisine_breadth_and_exploration`
  - `dietary_and_lifestyle_context`
  - `other_contextual_fit`
- `support_note` must briefly explain whether the claim came from direct review wording, repeated review pattern, or narrow event context.
- If the draft uses merchant or event context without a clear explanation, either repair the explanation, demote the claim, or delete it.
- Labels such as `tourist`, `visitor`, `local`, `family-friendly`, `open late`, `cash-only`, and similar venue descriptors usually belong in `contextual_inference_signals`, not Tier 1 or Tier 2.
- `outdoor seating` may stay in Tier 1 or Tier 2 only when direct review/tip text shows it is a personal request, complaint, or selection rule.

Allowed actions:
- keep
- merge duplicates
- delete
- demote to `state_hypotheses`
- move to `unknowns`
- rename an invalid unknown field
- shrink evidence refs
- lower confidence

Micro-examples:
- Bad: `context_rules` with 8 event refs for one city concentration claim
- Good: keep the same narrow context claim with only 1 to 2 representative `event_*` refs
- Bad: `unknowns.field = service_tolerance`
- Good: either `tolerance_threshold` with clear mixed evidence, or omit the unknown
- Bad: draft uses `tip_1` but `tip_1` is not in `full_allowed_evidence_ids`
- Good: replace `tip_1` with valid `rev_*` / `event_*` refs from the compact bundle, or drop the claim if no valid support remains

Schema:
{
  "grounded_facts": {
    "stable_preferences": [{"claim": "", "confidence": "high|medium|low", "support_basis": "", "support_note": "", "evidence_refs": []}],
    "avoid_signals": [{"claim": "", "confidence": "high|medium|low", "support_basis": "", "support_note": "", "evidence_refs": []}],
    "recent_signals": [{"claim": "", "confidence": "high|medium|low", "support_basis": "", "support_note": "", "evidence_refs": []}],
    "context_rules": [{"claim": "", "confidence": "high|medium|low", "support_basis": "", "support_note": "", "evidence_refs": []}]
  },
  "state_hypotheses": [
    {
      "type": "conditional_preference|tolerance|shift|conflict|latent_preference|other",
      "claim": "",
      "confidence": "high|medium|low",
      "support_basis": "",
      "support_note": "",
      "evidence_refs": []
    }
  ],
  "discriminative_signals": [
    {
      "claim": "",
      "why_not_generic": "",
      "confidence": "high|medium|low",
      "support_basis": "",
      "support_note": "",
      "evidence_refs": []
    }
  ],
  "contextual_inference_signals": [
    {
      "canonical_axis": "localness_vs_touristiness|crowding_and_relaxedness|special_occasion_fit|scene_and_ambiance_fit|service_style_fit|value_and_price_context|family_and_group_context|beverage_and_bar_context|late_night_and_hours_fit|geography_and_neighborhood_pattern|cuisine_breadth_and_exploration|dietary_and_lifestyle_context|other_contextual_fit",
      "claim": "",
      "confidence": "high|medium|low",
      "support_basis": "merchant_overlap|contextual_inference|mixed_context",
      "support_note": "",
      "direct_review_refs": [],
      "contextual_refs": [],
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
- `contextual_inference_signals`: at most 3
- `unknowns`: at most 2

Compact input:
{PASS1_DRAFT_JSON}

Return the repaired JSON object now.
```
