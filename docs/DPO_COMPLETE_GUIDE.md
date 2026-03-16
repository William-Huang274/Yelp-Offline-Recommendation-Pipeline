# DPO Pairwise Training Complete Guide

## Audience

This guide is written for the local Windows workflow in this repository,
especially when the practical target is a 4060 Laptop class GPU with 8GB VRAM.

## Core Concepts

### What is pairwise DPO?

Pairwise DPO trains on preference pairs instead of independent labels. Each
example provides a preferred response and a rejected response, which makes the
objective a better fit for reranking than plain pointwise classification.

### Why the current script already qualifies

`11_2_dpo_train.py` constructs chosen and rejected text pairs and optimizes a DPO
loss. The main engineering challenge is therefore configuration stability, not
algorithm replacement.

### Benefits

- Better alignment with ranking-style preference learning
- Cleaner comparison between stronger and weaker candidates
- More natural fit for sidecar reranking experiments

### Challenges

- Higher VRAM demand than pointwise training
- Strong sensitivity to sequence length and batch configuration
- Windows page-file issues can appear before GPU memory is fully saturated

## Environment Preparation

### 1. Check hardware

```bash
python tools/check_memory.py
```

Review GPU free VRAM, system RAM, and available disk before training.

### 2. Check dataset readiness

Confirm that the pairwise dataset contains both positive and negative evidence
per user. If pair generation is sparse, quality will degrade regardless of the
training profile.

### 3. Install dependencies

Ensure `trl`, `transformers`, `peft`, `bitsandbytes`, and the repository
requirements are present in the environment used for stage11 runs.

## Configuration Choice

### Comparison

| Option | Seq Len | LoRA R | Pairs / User | Estimated VRAM | Use case |
| --- | ---: | ---: | ---: | ---: | --- |
| Conservative | 512 | 8 | 4 | ~7.5GB | recommended default |
| Aggressive | 640 | 6 | 4 | ~8.5-9GB | only if memory is proven stable |
| Emergency | 384 | 4 | 2 | ~6GB | rescue path |

### Recommended Starting Point

Use the low-memory profile first. It is the lowest-risk path for local hardware
while still preserving almost all of the useful sequence coverage.

## Training Workflow

### Method 1: launcher script

```bash
scripts\check_dpo_env.bat
scripts\run_dpo_low_memory.bat
```

Choose `Standard Low Memory` unless you already know the machine can handle a
longer context window.

### Method 2: manual environment variables

```bash
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_MAX_PAIRS=4
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=16
python scripts\11_2_dpo_train.py
```

### Method 3: config-file driven launch

Load one of the `.env` profiles under `config/` and then run the training script.
This makes config reuse easier when iterating across several runs.

## Troubleshooting

### CUDA Out Of Memory

1. Shorten the sequence length.
2. Reduce LoRA rank.
3. Reduce pairs per user.
4. Move to the ultra-low-memory profile.

### Windows Error 1455

Raise the page file to 40GB or more. This is often necessary on Windows even
when the nominal GPU target should fit.

### Training is too slow

- reduce `QLORA_DPO_MAX_PAIRS`
- reduce the number of epochs
- restrict the run to a single bucket for debugging

### Pair generation is weak

Review the dataset build stage and confirm that each user actually contributes
useful positive and negative preference evidence.

### Model quality is weak

Review prompt mode, pair quality, and `DPO_BETA`. Poor dataset construction will
usually dominate parameter-level tuning.

## Performance Optimization

### Memory optimization

- keep `QLORA_BATCH_SIZE=1`
- use gradient accumulation instead of larger batches
- keep 4-bit quantization enabled
- keep gradient checkpointing enabled

### Throughput optimization

- reduce unnecessary monitoring overhead
- avoid very large pair counts during first-pass validation
- reuse the most stable config as the default baseline

### Data optimization

- remove obviously inverted or low-value pairs when justified
- prefer consistent preference construction over raw volume

## Monitoring

### Live monitoring

```bash
python tools\monitor_dpo_training.py
```

### Minimal shell monitoring

```bash
type dpo_train_log.txt
nvidia-smi
```

### Completion checks

- confirm the output directory exists
- confirm adapter artifacts were saved
- run `scripts/11_3_qlora_sidecar_eval.py`

## Summary

- the repository already has a real pairwise DPO path
- low-memory config is the correct default for local hardware
- sequence length is the highest-leverage stability knob
- quality validation must happen with the sidecar evaluator, not by loss alone

## Related Files

- `scripts/11_2_dpo_train.py`
- `scripts/run_dpo_low_memory.bat`
- `scripts/run_dpo_optimized.bat`
- `tools/check_memory.py`
- `tools/monitor_dpo_training.py`
- `config/dpo_low_memory.env`
- `config/dpo_ultra_low_memory.env`
