# DPO Pairwise Training Summary

## Key Finding

`scripts/11_2_dpo_train.py` is already doing pairwise DPO. The practical work is
not a training-method rewrite. The real work is stable configuration for a local
single-machine setup.

## Prepared Assets For An 8GB Laptop GPU

### Configs

1. `config/dpo_low_memory.env`: recommended profile at about 7.5GB
2. `config/dpo_ultra_low_memory.env`: emergency fallback at about 6GB

### Helper Scripts

1. `scripts/run_dpo_low_memory.bat`: one-click launcher
2. `tools/check_memory.py`: GPU and system memory check
3. `tools/monitor_dpo_training.py`: live progress monitor

### Supporting Docs

1. `docs/dpo/DPO_COMPLETE_GUIDE.md`
2. `docs/dpo/DPO_LOW_MEMORY_GUIDE.md`
3. `docs/dpo/DPO_QUICK_REFERENCE.md`

## Fast Start

```bash
python tools/check_memory.py
scripts\run_dpo_low_memory.bat
# choose [1] Standard Low Memory
```

## Recommended Baseline

```bash
QLORA_MAX_SEQ_LEN=512
QLORA_LORA_R=8
QLORA_BATCH_SIZE=1
QLORA_GRAD_ACC=16
QLORA_DPO_MAX_PAIRS=4
QLORA_DPO_BETA=0.1
```

## Common Issues

| Problem | First action |
| --- | --- |
| CUDA OOM | reduce `MAX_SEQ_LEN` from `512` to `384` |
| Windows Error 1455 | increase page file to 40GB+ |
| Training too slow | reduce `DPO_MAX_PAIRS` or epochs |
| Weak quality | test `DPO_BETA=0.2` or review pair construction |

## Pairwise vs Pointwise

| Aspect | Pointwise | Pairwise |
| --- | --- | --- |
| Objective | isolated classification | relative preference learning |
| Memory demand | lower | higher |
| Ranking fit | weaker | stronger |
| Suitability here | baseline | preferred |

## Next Step

After training finishes, point the evaluator to the produced run directory and
execute `scripts/11_3_qlora_sidecar_eval.py`.
