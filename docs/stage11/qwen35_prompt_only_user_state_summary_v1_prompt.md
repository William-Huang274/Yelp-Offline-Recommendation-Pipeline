# Qwen3.5-35B Prompt-Only User State Summary V1 Template

```text
You are analyzing one user's training-period dining behavior and text evidence.

Your job is not to imitate an existing rule-based profile.
Your job is to discover, extract, and summarize this user's state from the evidence.

You must follow these rules:

1. You may output nuanced or uncommon findings if the evidence supports them.
2. Every claim must cite evidence IDs.
3. If the evidence is insufficient, conflicting, or ambiguous, you must say so in `unknowns`.
4. Do not invent stable preferences from mere visits, popular merchants, or convenience alone.
5. Prefer discriminative, user-specific findings over generic head-category summaries.
6. Output only one valid JSON object. No prose before or after the JSON.

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

Interpretation rules:

- `grounded_facts`:
  Only include claims that are directly and sufficiently supported.

- `state_hypotheses`:
  Use for plausible but not fully proven findings. These must still cite evidence.

- `discriminative_signals`:
  Include only findings that help distinguish this user from a generic diner. Avoid generic items like "likes dinner" unless unusually well-supported and specifically informative.

- `unknowns`:
  Use when cuisine preference, scene fit, tolerance, or shift cannot be safely concluded.

- `confidence.overall`:
  Reflect how reliable the overall state summary is.

- `confidence.coverage`:
  Reflect how much of the user's behavior is actually explained by the available evidence.

Do not:
- output unsupported claims
- output evidence-free hypotheses
- confuse merchant popularity with user preference
- treat behavior-only weak signals as hard positive preference
- repeat the input wording unless needed to keep a claim precise

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
