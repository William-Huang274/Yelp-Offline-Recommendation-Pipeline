# DPO Pairwise Training Quick Reference

## Core Points

- the DPO script is already pairwise
- start with the low-memory profile
- sequence length is the first knob to adjust
- always evaluate the saved adapter after training

## Quick Start

### Method 1: launcher script

```bash
python tools\check_memory.py
scripts\run_dpo_low_memory.bat
```

### Method 2: manual variables

```bash
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=16
set QLORA_DPO_MAX_PAIRS=4
python scripts\11_2_dpo_train.py
```

## Config Table

| Profile | Seq Len | LoRA R | Pairs / User | VRAM |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 512 | 8 | 4 | ~7.5GB |
| Aggressive | 640 | 6 | 4 | ~8.5-9GB |
| Emergency | 384 | 4 | 2 | ~6GB |

## Parameter Cheatsheet

### Memory first

- `QLORA_MAX_SEQ_LEN`
- `QLORA_LORA_R`
- `QLORA_DPO_MAX_PAIRS`

### DPO-specific

- `QLORA_DPO_BETA`
- `QLORA_DPO_LOSS_TYPE`
- `QLORA_DPO_PREFER_EASY_NEG`
- `QLORA_DPO_FILTER_INVERTED`

## Common Problems

### CUDA OOM

Reduce `MAX_SEQ_LEN`, then `LORA_R`, then `DPO_MAX_PAIRS`.

### Windows Error 1455

Increase the page file to 40GB+.

### Training too slow

Reduce pair count or epoch count.

## File Locations

- `scripts/11_2_dpo_train.py`
- `scripts/run_dpo_low_memory.bat`
- `scripts/run_dpo_optimized.bat`
- `config/dpo_low_memory.env`
- `config/dpo_ultra_low_memory.env`
- `tools/check_memory.py`
- `tools/monitor_dpo_training.py`

## Minimal Workflow

1. Check memory.
2. Launch low-memory training.
3. Monitor logs.
4. Evaluate the adapter.
