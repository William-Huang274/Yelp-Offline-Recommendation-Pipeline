# DPO Pairwise Training Guide For Low-Memory Hardware

## Current State

The DPO path in this repository is viable on a constrained local machine, but it
must be treated as a memory-sensitive workload. The stable baseline is the 8GB
profile, not the largest possible context length.

## Memory Optimization Strategy

### Option 1: standard low-memory profile

Recommended first attempt:

```bash
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=16
set QLORA_DPO_MAX_PAIRS=4
python scripts\11_2_dpo_train.py
```

### Option 2: ultra-low-memory profile

Use this when option 1 still fails with OOM or page-file pressure:

```bash
set QLORA_MAX_SEQ_LEN=384
set QLORA_LORA_R=4
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=24
set QLORA_DPO_MAX_PAIRS=2
python scripts\11_2_dpo_train.py
```

## Parameters That Matter Most

1. `QLORA_MAX_SEQ_LEN`: biggest effect on VRAM
2. `QLORA_LORA_R`: medium effect on trainable parameter size
3. `QLORA_DPO_MAX_PAIRS`: useful when pair count pushes memory upward
4. `QLORA_GRAD_ACC`: helps recover effective batch size safely

## Pairwise vs Pointwise

| Property | Pointwise | Pairwise |
| --- | --- | --- |
| Memory | lower | higher |
| Preference modeling | limited | strong |
| Ranking alignment | weaker | stronger |
| Suitability here | fallback | preferred |

## Estimated Memory Footprint

- base model in 4-bit: about 2.5GB
- LoRA parameters: tens of MB
- activations at seq len 512: about 3 to 4GB
- gradients and optimizer state: about 1 to 2GB
- total: about 7 to 8GB

## Monitoring And Debugging

### Monitor GPU usage

```bash
nvidia-smi
python tools\monitor_dpo_training.py
```

### If you hit CUDA OOM

- reduce `MAX_SEQ_LEN`
- reduce `LORA_R`
- reduce `DPO_MAX_PAIRS`
- close other GPU processes

### If you hit Windows Error 1455

Increase the Windows page file to at least 40GB and reboot.

## Time Expectation

With the low-memory profile, expect roughly 45 to 60 minutes per epoch for the
current local-scale workflow.

## Validation

After training, run:

```bash
set INPUT_11_2_RUN_DIR=<your_run_dir>
python scripts\11_3_qlora_sidecar_eval.py
```

## Further Suggestions

- validate with a single bucket first
- stabilize the config before trying larger sequence lengths
- keep the conservative profile as the parity baseline
