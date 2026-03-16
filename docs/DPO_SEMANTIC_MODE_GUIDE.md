# DPO Semantic Mode Guide

## Important Finding

The repository supports two prompt styles for DPO data and training:

- `full`: keeps ranking-related feature text in the prompt
- `semantic`: removes ranking-heavy feature text and keeps the prompt more
  semantic and preference-oriented

## Why Semantic Mode Matters

### Problem

When ranking features are written directly into the DPO prompt, the model can be
asked to optimize against signals that already encode downstream ranking logic.
That can blur the preference-learning objective and unnecessarily inflate prompt
length.

### Semantic-mode advantage

Semantic mode shortens the prompt and reduces conflicting ranking cues. It is a
cleaner fit when the goal is preference learning instead of feature imitation.

## Prompt Length Comparison

### Current full-mode observation

The existing analysis shows that full-mode prompts are already long enough for
sequence length to become a stability bottleneck on local hardware.

### Expected semantic-mode behavior

Semantic mode should lower prompt length, reduce truncation risk, and make 512
or 640 token settings more usable.

## How To Use It

### Option 1: rebuild the dataset in semantic mode

Recommended path when you want a cleaner experiment boundary.

```bash
set QLORA_PROMPT_MODE=semantic
python scripts\11_1_qlora_build_dataset.py
```

### Option 2: strip ranking-heavy fields during training

Use this when you need a faster iteration path without rebuilding all stage11
assets.

```bash
set QLORA_PROMPT_MODE=semantic
python scripts\11_2_dpo_train.py
```

## Final Recommendation

### Fastest path

Use the existing dataset and enable semantic mode during training so that the
ranking-heavy fields are removed dynamically.

### Cleaner long-term path

Rebuild the dataset in semantic mode and keep the entire stage11 experiment
contract consistent from data generation through training and evaluation.

## Validation

After enabling semantic mode, inspect prompt examples and confirm that ranking
feature blocks were removed while preference evidence remains intact.

## Expected Improvements

- shorter prompts
- lower truncation rate
- lower VRAM pressure
- clearer DPO preference objective

## Summary

Use semantic mode when prompt length or feature leakage becomes the dominant
bottleneck. Keep the low-memory profile as the stable baseline while comparing
full vs semantic runs.
