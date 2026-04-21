# Qwen3.5-35B Prompt-Only User State Summary V2 Template

```text
You are analyzing one user's training-period dining behavior and text evidence.

Your job is not to imitate an existing rule-based profile.
Your job is to discover, extract, and summarize this user's state from the evidence.

You must follow these rules:

1. Review text is the primary evidence. Read the user-authored review or tip `text` first and extract concrete details, conditions, edge cases, and repeated language before summarizing.
2. Merchant metadata, event metadata, cuisine labels, tags, and context aggregates are secondary context only. They may help place a visit in context, but they must not by themselves create a stable user trait.
3. Every claim must cite atomic evidence IDs only.
4. If the evidence is insufficient, conflicting, or ambiguous, you must say so in `unknowns`.
5. Do not invent stable preferences from mere visits, merchant descriptors, convenience, or broad context alone.
6. Prefer discriminative, user-specific findings over generic head-category summaries.
7. Prefer fewer, better-supported claims over broad or repetitive lists.
8. Output only one valid JSON object. No prose before or after the JSON. No markdown code fences.

Working method:
- First, mentally list the most specific review-native details you can extract from the review `text`.
- Second, separate those details into three evidence tiers before writing JSON:
  - Tier 1: repeated user-authored signals and the narrow inferences supported by those repeated signals.
  - Tier 2: personalized or latent signals visible in the user's language, even if they are not repeated enough to be Tier 1.
  - Tier 3: merchant-overlap or contextual inferences that may still be useful, but must stay separate from the main evidence.
- For Tier 3, you may internally consider multiple contextual hypotheses, but the final JSON may contain at most one item per `canonical_axis`.
- Third, use `event_*` only as narrow recency context or sequence context.
- Fourth, delete any claim that sounds like a merchant summary rather than a user trait unless it is explicitly placed into the separate contextual-inference tier.

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

If a claim depends on a sequence pattern, cite the smallest relevant `event_*` IDs instead of block names.

Grounding rules:
- Tier 1 sections are:
  - `stable_preferences`
  - `avoid_signals`
  - `discriminative_signals`
- Tier 2 section is:
  - `state_hypotheses`
- Tier 3 section is:
  - `contextual_inference_signals`
- Tier 1 sections must each cite at least one `rev_*` or `tip_*` reference.
- `event_*` refs may supplement those sections, but they must not be the only support.
- If a claim depends mainly on recent visit context or merchant context, keep it in `recent_signals`, `context_rules`, or `contextual_inference_signals`.
- If a `recent_signals`, `context_rules`, or `state_hypotheses` item cites only `event_*` refs and no `rev_*` / `tip_*`, its `support_basis` must be `event_context_only`.
- Never use cuisine labels, merchant tags, merchant quality labels, scene tags, property tags, or other structured metadata as a substitute for reading the review text.
- If merchant or event context appears in a claim, you must explicitly explain how it entered the claim:
  - `support_basis = "direct_user_text"` when the point is directly stated in review or tip text.
  - `support_basis = "review_pattern_inference"` when the point is inferred from multiple review or tip texts.
  - `support_basis = "mixed_support"` when direct review text is primary and event context is only supplementary.
  - `support_basis = "event_context_only"` only for narrow recent-context observations or hypotheses, never for stable preferences, avoid signals, or discriminative signals.
- For `contextual_inference_signals`, allowed `support_basis` values are:
  - `merchant_overlap`
  - `contextual_inference`
  - `mixed_context`
- For `contextual_inference_signals`, `canonical_axis` must be one of:
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
- `claim` may stay natural and specific, but `canonical_axis` must be normalized. If two candidate Tier 3 claims fall under the same `canonical_axis`, keep only the single strongest final item for that axis.
- `support_note` must briefly say whether the claim came from direct review wording, repeated review pattern, or narrow event-context inference.
- `unknowns.field` must be one of:
  - `cuisine_preference`
  - `scene_preference`
  - `tolerance_threshold`
  - `recent_shift`

Atomic evidence rules:
- Tier 1 and `recent_signals` / `context_rules`: prefer 1 to 2 refs; 3 refs is the hard upper bound.
- `state_hypotheses`: prefer 1 to 3 refs; 4 refs is the hard upper bound.
- `contextual_inference_signals`:
  - prefer 0 to 2 `direct_review_refs`
  - prefer 1 to 2 `contextual_refs`
  - keep `evidence_refs` equal to the union of those refs
- `unknowns`: prefer 1 to 2 refs; never dump a long unresolved evidence bundle there.
- `evidence_refs` are representative witness examples, not an exhaustive proof bundle.
- When many reviews support the same point, keep only the smallest representative subset:
  - first keep the clearest direct wording or most specific dish / condition / complaint
  - add one reinforcing ref only if it materially strengthens the claim
  - keep a third ref only if it adds a distinct condition, conflict, or recency change
- Never emit long unions like `rev_1` through `rev_30`.
- If a claim needs many refs, narrow it, split it, demote it, move it to `unknowns`, or move it to `contextual_inference_signals`.
- If a claim still seems to need more than the hard ref limit, that claim is too broad for this schema and must be rewritten or dropped.
- For `contextual_inference_signals`, if you need more than 2 `direct_review_refs` or more than 2 `contextual_refs`, the inference is too broad. Rewrite it to a narrower claim.
- For `contextual_inference_signals`, if you cannot name at least one `contextual_ref`, the point does not belong in Tier 3. Rewrite it as `state_hypotheses` if it is review-only, or drop it.
- For `contextual_inference_signals`, do not output two items with the same `canonical_axis`. Merge or choose the strongest one.
- Labels such as `tourist`, `visitor`, `local`, `family-friendly`, `open late`, `cash-only`, and similar venue descriptors usually belong in Tier 3. Do not promote them into Tier 1 or Tier 2 unless the user explicitly states them as a request, complaint, or selection rule in review/tip text.
- `outdoor seating` is only allowed in Tier 1 or Tier 2 when the user explicitly asks for it, rejects a place because of it, or clearly treats it as a personal requirement in review/tip text.

Output schema:
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
- `unknowns`: at most 3

Interpretation rules:

- `grounded_facts`:
  Only include Tier 1 claims that are directly and sufficiently supported. These should mostly come from review or tip text, not merchant metadata.

- `state_hypotheses`:
  Use for Tier 2 findings: plausible latent or personalized signals visible in user language, but not strong enough to be Tier 1. These must still cite atomic evidence IDs and explain the inference path in `support_note`.

- `discriminative_signals`:
  Include only findings that help distinguish this user from a generic diner. Avoid generic items like "likes dinner" unless unusually well-supported and specifically informative.

- `contextual_inference_signals`:
  Use for Tier 3 findings only. This is the only place where merchant-overlap or model-led contextual inference may appear. Never merge these into Tier 1 or Tier 2.
  `canonical_axis` is for normalization and dedupe, not for flattening the language. Keep the `claim` specific and natural, but collapse same-axis variants into one final item.
  If the user explicitly said something similar in review text, include `direct_review_refs`.
  If the point comes from merchant overlap, recency context, or a model inference step, include `contextual_refs` and explain the reasoning in `support_note`.

- `unknowns`:
  Use when cuisine preference, scene fit, tolerance, or shift cannot be safely concluded.

- `confidence.overall`:
  Reflect how reliable the overall state summary is.

- `confidence.coverage`:
  Reflect how much of the user's behavior is actually explained by the available evidence.

Do not:
- output unsupported claims
- output evidence-free hypotheses
- cite forbidden block names
- let merchant metadata replace review reading
- confuse merchant popularity with user preference
- let `stable_preferences`, `avoid_signals`, or `discriminative_signals` rely only on `event_*`
- merge merchant-overlap or contextual-inference claims into Tier 1 or Tier 2
- output any unknown field outside the closed list
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
