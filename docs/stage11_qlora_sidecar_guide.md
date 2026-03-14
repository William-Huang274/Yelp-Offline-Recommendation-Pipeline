# Stage11 QLoRA Sidecar Guide

## Scripts
- `scripts/11_1_qlora_build_dataset.py`: build QLoRA train/eval text data from stage09 candidates.
- `scripts/11_2_qlora_train.py`: train LoRA adapter (4-bit optional).
- `scripts/11_3_qlora_sidecar_eval.py`: run sidecar rerank on stage09 head candidates and compare to PreScore.

## Dependencies
Install once:

```powershell
pip install -r requirements-stage11-qlora.txt
```

If `bitsandbytes` fails on your Windows environment, keep it uninstalled and run with:

```powershell
$env:QLORA_USE_4BIT='false'
$env:QLORA_EVAL_USE_4BIT='false'
```

## Step 1: Build dataset (11_1)

```powershell
$env:PROJECT_ROOT=(Resolve-Path .).Path
$env:INPUT_09_RUN_DIR=(Join-Path $env:PROJECT_ROOT 'data/output/09_candidate_fusion/<your_run>')
$env:BUCKETS_OVERRIDE='10'
$env:QLORA_TOPN_PER_USER='120'
$env:QLORA_NEG_PER_USER='8'
$env:QLORA_INCLUDE_VALID_POS='true'
$env:QLORA_VALID_POS_WEIGHT='0.35'
python scripts/11_1_qlora_build_dataset.py
```

## Step 2: Train adapter (11_2)

```powershell
$env:PROJECT_ROOT=(Resolve-Path .).Path
$env:INPUT_11_RUN_DIR=(Join-Path $env:PROJECT_ROOT 'data/output/11_qlora_data/<your_run>')
$env:BUCKETS_OVERRIDE='10'
$env:QLORA_BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
$env:QLORA_USE_4BIT='true'
$env:QLORA_EPOCHS='1'
$env:QLORA_BATCH_SIZE='1'
$env:QLORA_GRAD_ACC='16'
python scripts/11_2_qlora_train.py
```

## Step 3: Sidecar eval (11_3)

```powershell
$env:PROJECT_ROOT=(Resolve-Path .).Path
$env:INPUT_09_RUN_DIR=(Join-Path $env:PROJECT_ROOT 'data/output/09_candidate_fusion/<your_run>')
$env:INPUT_11_2_RUN_DIR=(Join-Path $env:PROJECT_ROOT 'data/output/11_qlora_models/<your_run>')
$env:BUCKETS_OVERRIDE='10'
$env:QLORA_RERANK_TOPN='80'
$env:QLORA_BLEND_ALPHA='0.12'
python scripts/11_3_qlora_sidecar_eval.py
```

## Outputs
- Stage11 data: `data/output/11_qlora_data/<run>`
- Stage11 model: `data/output/11_qlora_models/<run>/adapter`
- Stage11 sidecar eval: `data/output/11_qlora_sidecar_eval/<run>`
- Merged metric table: `data/metrics/recsys_stage11_qlora_sidecar_results.csv`
