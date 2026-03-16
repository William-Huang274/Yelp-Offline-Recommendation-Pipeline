# Final DPO Configuration Recommendation

## Data Analysis Summary

The prompt-length analysis is already sufficient to make a practical decision:
512 tokens is the correct baseline for local hardware, while 640 tokens should
be treated as an optional follow-up experiment.

## Recommended Options

### Option 1: conservative profile

Strong recommendation.

- stable on 8GB VRAM in realistic local conditions
- preserves most useful sequence coverage
- easiest to debug and reproduce

### Option 2: optimized profile

Aggressive option.

- slightly better prompt coverage
- materially higher risk of OOM or page-file pressure
- only justified after the conservative run is stable

## Recommended Workflow

1. run the conservative profile first
2. evaluate quality with `11_3_qlora_sidecar_eval.py`
3. only test the optimized profile if quality is still insufficient

## Truncation Notes

Most of the information needed for reranking remains inside the first 512 tokens
for the current dataset distribution. The samples beyond that point are the tail,
not the bulk.

## Quick Start

```bash
scripts\run_dpo_optimized.bat
# choose [1]
```

## If Problems Appear

### OOM under the 512 profile

Move to the ultra-low-memory setup.

### OOM under the 640 profile

Return to 512 immediately.

### Weak quality

Review prompt mode, pair quality, and evaluation parity before increasing
sequence length again.

## Final Recommendation

Choose option `1` first and keep it as the baseline for all later comparisons.
