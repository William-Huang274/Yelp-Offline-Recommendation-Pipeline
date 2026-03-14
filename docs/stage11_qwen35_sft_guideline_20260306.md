# Stage11 Qwen3.5-4B SFT Guideline (2026-03-06)

## Goal
- Produce a defensible stage11 SFT result that can be explained externally.
- Avoid another blind 8-9 hour full run.
- Require a pilot gate before any full SFT rerun.

## Scope
- Base model: `Qwen/Qwen3.5-4B`
- Dataset stage: `stage11_1_qlora_build_dataset`
- Train stage: `stage11_2_qlora_train`
- Eval stage: `stage11_3_qlora_sidecar_eval`
- Candidate setting to keep fixed for comparison: `bucket=10`, `top150`, `prompt=semantic`, `seq_len=512`

## Current Baseline
- Full eval artifact: `/home/william/stage11_fs/sidecar/20260306_035517_stage11_3_qlora_sidecar_eval`
- Current metrics:
  - `PreScore@10`: Recall `0.0569106`, NDCG `0.0264671`
  - `QLoRASidecar@10`: Recall `0.0582656`, NDCG `0.0266519`
- Improvement is real but small:
  - Recall absolute delta: `+0.0013550`
  - NDCG absolute delta: `+0.0001849`

## Audit Summary
- `alpha` is not the main problem.
  - Offline grid on the current `bucket_10_scores.csv` shows best alpha is `0.02`.
  - `alpha=0.12` to `alpha=0.02` does not change Recall. It only slightly improves NDCG.
- `seq_len=512` is not the main problem.
  - Token audit shows clip rate around `1.2%` on train and `1.5%` on eval.
- The main issue is weak pointwise preference signal.
  - `qlora_prob` alone is weak:
    - Recall@10 `0.0338753`
    - NDCG@10 `0.0158612`
  - AUC proxy is only `0.5486`.
  - Positive and negative `qlora_prob` means are too close:
    - positive mean `0.5686`
    - negative mean `0.5549`
- The current data weighting is too negative-heavy even after rebalance.
  - Raw train rows:
    - positive `2252`
    - negative `23040`
  - After `QLORA_MAX_NEG_POS_RATIO=4.0` rebalance:
    - positive `2252`
    - negative `9008`
  - Effective weight sums after rebalance:
    - positive weight sum `1492.8`
    - negative weight sum `9803.0`
  - Effective weighted neg:pos is still about `6.6:1`
- This is consistent with the model learning a conservative score rather than a strong rerank signal.

## Evidence Files
- Alpha grid summary: [alpha_grid_stage11_3_qwen35_summary.json](/d:/5006_BDA_project/tmp/alpha_grid_stage11_3_qwen35_summary.json)
- Alpha grid table: [alpha_grid_stage11_3_qwen35.csv](/d:/5006_BDA_project/tmp/alpha_grid_stage11_3_qwen35.csv)
- Token audit: [qwen35_stage11_token_audit_20260306.json](/d:/5006_BDA_project/tmp/qwen35_stage11_token_audit_20260306.json)
- Label-source audit: [qwen35_stage11_label_source_audit_20260306.json](/d:/5006_BDA_project/tmp/qwen35_stage11_label_source_audit_20260306.json)
- Rebalance audit: [qwen35_stage11_rebalance_audit_20260306.json](/d:/5006_BDA_project/tmp/qwen35_stage11_rebalance_audit_20260306.json)
- Probability distribution audit: [qwen35_stage11_prob_distribution_20260306.json](/d:/5006_BDA_project/tmp/qwen35_stage11_prob_distribution_20260306.json)

## Execution Scripts
- Step 1 build: [stage11_qwen35_step1_build.sh](/d:/5006_BDA_project/tmp/stage11_qwen35_step1_build.sh)
- Step 2 pilot train: [stage11_qwen35_step2_pilot_train.sh](/d:/5006_BDA_project/tmp/stage11_qwen35_step2_pilot_train.sh)
- Step 3 pilot eval: [stage11_qwen35_step3_pilot_eval.sh](/d:/5006_BDA_project/tmp/stage11_qwen35_step3_pilot_eval.sh)

## Decision
- Do not jump directly to full SFT.
- First rebuild the dataset with a more ranking-friendly balance.
- Then run a medium-size pilot SFT.
- Then run a capped pilot eval.
- Only if the pilot clears the gate should we run the full 8-9 hour SFT.

## Step 1: Rebuild Stage11_1 Dataset

### Why
- Keep `semantic` prompt.
- Reduce over-emphasis on hard negatives.
- Stop under-weighting valid positives.

### Required settings
```bash
export INPUT_09_RUN_DIR=/home/william/stage11_fs/input09/20260305_002746_full_stage09_candidate_fusion
export OUTPUT_11_DATA_ROOT_DIR=/home/william/stage11_fs/input11data
export BUCKETS_OVERRIDE=10

export QLORA_PROMPT_MODE=semantic
export QLORA_TOPN_PER_USER=120
export QLORA_NEG_PER_USER=8
export QLORA_NEG_LAYERED_ENABLED=true
export QLORA_NEG_HARD_RATIO=0.35
export QLORA_NEG_NEAR_RATIO=0.35
export QLORA_NEG_HARD_WEIGHT=1.00
export QLORA_NEG_NEAR_WEIGHT=1.00
export QLORA_NEG_EASY_WEIGHT=0.90
export QLORA_NEG_FILL_WEIGHT=0.90
export QLORA_VALID_POS_WEIGHT=0.70
export QLORA_INCLUDE_VALID_POS=true
export QLORA_EVAL_USER_FRAC=0.2
```

### Run command
```bash
wsl -e bash /mnt/d/5006_BDA_project/tmp/stage11_qwen35_step1_build.sh
```

### Dataset gate
- Confirm the new run is `semantic` prompt.
- Confirm the new dataset still uses the same stage09 source run.
- Confirm train/eval rows are in the same rough order of magnitude as the current dataset.
- Confirm the effective weighting is materially healthier than the current `6.6:1` weighted neg:pos.
- Do not continue if the new dataset unexpectedly collapses row counts or candidate coverage.

## Step 2: Run a Pilot Stage11_2 SFT

### Why
- A small smoke run is too weak to predict full-run quality.
- A medium pilot is much cheaper than a blind 8-9 hour full run.
- The pilot must preserve the real training behavior as much as possible.

### Required settings
```bash
export INPUT_11_RUN_DIR=/home/william/stage11_fs/input11data/<NEW_STAGE11_1_RUN_DIR>
export OUTPUT_11_MODELS_ROOT_DIR=/home/william/stage11_fs/models
export BUCKETS_OVERRIDE=10

export QLORA_BASE_MODEL=Qwen/Qwen3.5-4B
export QLORA_REQUIRED_BASE_MODEL=Qwen/Qwen3.5-4B
export QLORA_USE_4BIT=true
export QLORA_USE_BF16=false
export QLORA_QWEN35_MAMBA_SSM_DTYPE=float16
export QLORA_CLEAR_CUDA_CACHE_BEFORE_LORA=true
export QLORA_CAST_TRAINABLE_PARAMS_TO_COMPUTE=false
export QLORA_MANUAL_FP16_AUTOCAST=false
export QLORA_MAX_SEQ_LEN=456
export QLORA_MAX_TRAIN_ROWS=6000
export QLORA_MAX_EVAL_ROWS=1500
export QLORA_MAX_NEG_POS_RATIO=2.5
export QLORA_EPOCHS=1.0
export QLORA_LR=1e-4
export QLORA_WARMUP_RATIO=0.05
export QLORA_BATCH_SIZE=1
export QLORA_GRAD_ACC=16
export QLORA_EVAL_STEPS=100
export QLORA_SAVE_STEPS=100
export QLORA_LOGGING_STEPS=20
export QLORA_TOKEN_AUDIT_ENABLED=true
```

### Preflight rule
- Before the real pilot run, do a `64-row / 32-row` cold-start smoke after `wsl --shutdown`.
- The current machine-specific finding is:
  - `4bit + bf16` load path is unstable in WSL on this GPU.
  - `4bit + fp16` loads reliably.
  - LoRA trainable params must be cast off `float32`, otherwise step-0 forward OOMs in the LoRA path.
  - `seq_len=512` and `seq_len=480` are not stable full-train settings on this machine.
  - The first numerically stable + non-OOM gate was `seq_len=456` with fp32 LoRA trainables and Trainer-native fp16.

### Run command
```bash
wsl -e bash /mnt/d/5006_BDA_project/tmp/stage11_qwen35_step2_pilot_train.sh
```

### Pilot train gate
- Confirm token audit remains near the current level. `512` should still be enough.
- Confirm the run is stable and no dtype or 4bit regressions appear.
- Record:
  - train/eval rows
  - eval loss
  - checkpoint paths
- Do not continue to full SFT yet. First run pilot eval.

## Step 3: Run a Pilot Stage11_3 Eval

### Why
- The real acceptance test is not training loss. It is whether the new model produces a stronger `qlora_prob` signal.
- A pilot eval on a capped user subset is much cheaper than a full rerun.

### Required settings
```bash
export INPUT_09_RUN_DIR=/home/william/stage11_fs/input09/20260305_002746_full_stage09_candidate_fusion
export INPUT_11_2_RUN_DIR=/home/william/stage11_fs/models/<NEW_STAGE11_2_RUN_DIR>
export INPUT_11_DATA_RUN_DIR=/home/william/stage11_fs/input11data/<NEW_STAGE11_1_RUN_DIR>
export OUTPUT_11_SIDECAR_ROOT_DIR=/home/william/stage11_fs/sidecar

export BUCKETS_OVERRIDE=10
export QLORA_PROMPT_MODE=semantic
export QLORA_RERANK_TOPN=150
export QLORA_EVAL_MAX_USERS_PER_BUCKET=200
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=30000
export QLORA_EVAL_BATCH_SIZE=4
export QLORA_EVAL_MAX_SEQ_LEN=512
export QLORA_EVAL_USE_BF16=false
export QLORA_QWEN35_MAMBA_SSM_DTYPE=float16
export QLORA_EVAL_MAX_MEMORY_CUDA=4600MiB
export QLORA_EVAL_MAX_MEMORY_CPU=64GiB
export QLORA_EVAL_OFFLOAD_FOLDER=/home/william/stage11_fs/offload
export QLORA_EVAL_OFFLOAD_STATE_DICT=true
export QLORA_EVAL_LOAD_MODEL_BEFORE_SPARK=true
export QLORA_EVAL_QWEN35_NO_THINK=true
export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS=8192
```

### Run command
```bash
wsl -e bash /mnt/d/5006_BDA_project/tmp/stage11_qwen35_step3_pilot_eval.sh
```

### Pilot eval gate
- Evaluate four things on the pilot score CSV:
  - `qlora_prob only` vs current baseline
  - blend gain vs `pre_score`
  - offline best alpha
  - improved users vs worsened users
- Recommended go/no-go thresholds:
  - AUC proxy should exceed `0.58`
  - `qlora_prob only` should materially improve over the current baseline `Recall@10=0.0339`
  - improved users should be greater than worsened users
  - best alpha should not collapse to a tiny value like `0.02`
- If the pilot does not clear these thresholds, stop. Do not start the full run.

## Step 4: Full SFT Only If the Pilot Passes

### Full train settings
- Keep the same dataset settings from Step 1.
- Keep the same train recipe from Step 2.
- Change only:
```bash
export QLORA_MAX_TRAIN_ROWS=0
export QLORA_MAX_EVAL_ROWS=0
export QLORA_EPOCHS=1.5
export QLORA_EVAL_STEPS=200
export QLORA_SAVE_STEPS=200
```

### Full eval settings
- Keep the same stable eval path from Step 3.
- Remove the pilot caps:
```bash
export QLORA_EVAL_MAX_USERS_PER_BUCKET=1200
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=200000
```

## When to Move to DPO
- Move to DPO only after SFT produces a clearly stronger pointwise signal.
- Good trigger conditions:
  - `qlora_prob` separation is materially better than the current run
  - AUC proxy exceeds `0.58`
  - blend improvement is no longer just a one-user gain
- If SFT still produces a weak signal after the Step 1 data fix, then DPO is the next aligned step.

## What Not to Change Yet
- Do not change `seq_len` before this loop completes.
- Do not change LoRA `r/alpha` first.
- Do not spend more time on alpha grid before improving `qlora_prob`.
- Do not move to a custom rank head yet.
- Do not mix old partials with different `topN` settings.

## External Narrative
- The current project should be framed as:
  - retrieval baseline
  - LLM-based preference calibration via SFT
  - ranking alignment via DPO in the next stage
  - later distillation or lightweight ranker for production
- The SFT stage is not the final production ranker. It is the preference signal foundation.

## Execution Rule
- No full 8-9 hour run without a pilot gate.
- Change one family of factors at a time:
  - first data balance
  - then training recipe
  - then DPO
- Keep eval semantics fixed while auditing training changes.
