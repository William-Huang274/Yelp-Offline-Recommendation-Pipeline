# DPO Pairwise Training README

## Core Conclusion

`scripts/11_2_dpo_train.py` is already a pairwise DPO implementation.
You do not need to rewrite the training logic first. The practical next step is
config tuning for a constrained single-GPU environment such as a 4060 Laptop
with 8GB VRAM.

## Quick Start

### 1. Check the environment

```bash
scripts\check_dpo_env.bat
python tools\check_memory.py
```

### 2. Start training

```bash
scripts\run_dpo_low_memory.bat
# choose [1] Standard Low Memory
```

Or launch manually:

```bash
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_MAX_PAIRS=4
python scripts\11_2_dpo_train.py
```

### 3. Monitor training

```bash
python tools\monitor_dpo_training.py
type dpo_train_log.txt
nvidia-smi
```

## Available Resources

### Config files

- `config/dpo_low_memory.env`: recommended starting point for 8GB VRAM
- `config/dpo_ultra_low_memory.env`: last-resort profile for tighter memory
- `config/dpo_safe_512.env`: conservative 512-token setup
- `config/dpo_optimized_640.env`: more aggressive 640-token setup
- `config/dpo_semantic_mode.env`: semantic prompt mode configuration

### Launch scripts

- `scripts/run_dpo_low_memory.bat`: low-memory launcher
- `scripts/run_dpo_optimized.bat`: launcher with 512 vs 640 options
- `scripts/check_dpo_env.bat`: environment validation helper

### Tools

- `tools/check_memory.py`: GPU, RAM, and disk inspection
- `tools/monitor_dpo_training.py`: live training monitor
- `tools/monitor_training.sh`: lightweight shell monitor

### Related documents

- `docs/DPO_SUMMARY.md`
- `docs/DPO_QUICK_REFERENCE.md`
- `docs/DPO_COMPLETE_GUIDE.md`
- `docs/DPO_LOW_MEMORY_GUIDE.md`
- `docs/DPO_SEMANTIC_MODE_GUIDE.md`

## Recommended 8GB Configuration

```bash
QLORA_MAX_SEQ_LEN=512
QLORA_LORA_R=8
QLORA_LORA_ALPHA=16
QLORA_BATCH_SIZE=1
QLORA_GRAD_ACC=16
QLORA_DPO_MAX_PAIRS=4
QLORA_DPO_BETA=0.1
QLORA_USE_4BIT=true
QLORA_GRADIENT_CHECKPOINTING=true
```

Expected memory footprint: about 7.5GB
Expected training time: about 45 to 60 minutes per epoch

## Configuration Comparison

| Profile | VRAM | Seq Len | LoRA R | Pairs / User | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Standard | 9.5GB | 768 | 16 | 8 | good quality, risky on 8GB |
| Low memory | 7.5GB | 512 | 8 | 4 | safest practical starting point |
| Ultra low memory | 6.0GB | 384 | 4 | 2 | emergency fallback only |

## FAQ

### What makes this pairwise DPO?

The training samples contain `chosen` and `rejected` responses as aligned pairs.
That means the model learns a relative preference rather than an isolated label.

### Why does pairwise training need more memory?

Pairwise DPO processes both the chosen and rejected branches, so activation and
optimizer memory is materially higher than a pointwise setup.

### What should I change first when CUDA OOM appears?

1. Reduce `QLORA_MAX_SEQ_LEN` from `512` to `384`.
2. Reduce `QLORA_LORA_R` from `8` to `4`.
3. Reduce `QLORA_DPO_MAX_PAIRS` from `4` to `2`.
4. Fall back to `config/dpo_ultra_low_memory.env`.

### What should I do for Windows Error 1455?

Increase the Windows page file to at least 40GB and restart the machine before
running training again.

### How do I verify that the output is usable?

Run the sidecar evaluation after training:

```bash
set INPUT_11_2_RUN_DIR=data\output\11_qlora_models\<your_run_dir>
python scripts\11_3_qlora_sidecar_eval.py
```

## Key Parameters

### Memory-sensitive parameters

1. `QLORA_MAX_SEQ_LEN`: highest impact on VRAM
2. `QLORA_LORA_R`: moderate impact on trainable parameter size
3. `QLORA_DPO_MAX_PAIRS`: smaller but still useful when memory is tight

### DPO-specific parameters

- `QLORA_DPO_BETA`: preference sharpness, default `0.1`
- `QLORA_DPO_LOSS_TYPE`: default `sigmoid`
- `QLORA_DPO_PREFER_EASY_NEG`: prioritizes easier negative pairs
- `QLORA_DPO_FILTER_INVERTED`: filters inverted or noisy preference pairs

## Training Flow

1. Validate hardware and dependencies.
2. Check whether the dataset is suitable for pairwise training.
3. Pick a memory profile.
4. Run training and monitor logs.
5. Evaluate the resulting adapter with `11_3_qlora_sidecar_eval.py`.

## Checklist

- Environment validation completed
- Sufficient page file configured on Windows
- Training config selected deliberately
- Log monitoring enabled
- Post-training evaluation run
