# DPO Configuration Choice Guide

## Data-Length Findings

The existing prompt-length analysis indicates:

- mean length: about `432` tokens
- P95 length: about `477` tokens
- P99 length: about `525` tokens

These statistics support a conservative 512-token baseline and explain why a
640-token profile may improve coverage but can destabilize 8GB hardware.

## Config Options

### Option 1: conservative profile (recommended)

- `MAX_SEQ_LEN=512`
- safer memory behavior
- covers nearly all practical samples
- best first-run baseline

### Option 2: optimized profile (aggressive)

- `MAX_SEQ_LEN=640`
- slightly better coverage
- higher VRAM pressure
- only worth trying after the conservative run is stable

## Recommendation

### Start with option 1

Use the 512-token profile first. It provides the best balance between retention
and stability on a local Windows machine.

### Only try option 2 after stability is proven

Move to 640 only when:

- page file is already increased
- no other large GPU jobs are running
- the 512-token run is stable and quality is still insufficient

## Usage

### Conservative profile

```bash
scripts\run_dpo_optimized.bat
# choose [1]
```

### Aggressive profile

```bash
scripts\run_dpo_optimized.bat
# choose [2]
```

## Truncation Impact

At 512 tokens, only a small tail of longer prompts is truncated. That trade-off
is usually acceptable compared with the instability risk of 640 on 8GB VRAM.

## If OOM Happens

Fall back in this order:

1. 640 -> 512
2. 512 -> 384
3. reduce LoRA rank
4. reduce pairs per user

## Summary

Pick 512 first. Treat 640 as a deliberate follow-up experiment, not the default.
