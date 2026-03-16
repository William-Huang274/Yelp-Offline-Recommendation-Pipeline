# Final Recommendation For DPO Training

## Executive Summary

The repository is already in a workable state for pairwise DPO. The best next
step is a disciplined local-hardware workflow, not a large refactor.

## Recommended Paths

### Path A: fast start

Use this when the immediate goal is to get a stable local run.

1. configure the low-memory profile
2. launch training
3. monitor logs and GPU usage
4. run sidecar evaluation

### Path B: cleaner long-term setup

Use this after the first stable run is complete.

1. rebuild the dataset in semantic mode if needed
2. review prompt-length distribution again
3. test whether a slightly larger sequence length is still stable
4. compare evaluation results against the baseline

## Profile Comparison

| Path | Goal | Risk | Recommended when |
| --- | --- | --- | --- |
| A | stable first run | low | starting from scratch |
| B | cleaner higher-quality follow-up | medium | baseline already works |

## Practical Advice

### Step 1

Start with path A and establish a working baseline.

### Step 2

Evaluate the adapter instead of relying on training loss alone.

### Step 3

Only move to path B when the baseline has been validated and documented.

## Related Files

- `config/dpo_low_memory.env`
- `config/dpo_semantic_mode.env`
- `scripts/run_dpo_low_memory.bat`
- `scripts/11_2_dpo_train.py`
- `scripts/11_3_qlora_sidecar_eval.py`

## Checklist

- baseline config selected
- Windows page file confirmed
- training log monitored
- sidecar evaluation completed
- comparison against the baseline recorded

## Ready State

If the baseline run is reproducible and the evaluation path completes, the DPO
setup is ready for controlled follow-up experiments.
