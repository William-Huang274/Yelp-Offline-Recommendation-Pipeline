# Stage11 Cloud Run Profile (2026-03-09)

This document is the default cloud runbook for stage11 `11_1 -> 11_3`.

Scope:
- Cloud machine with large CPU/RAM and a single `RTX 5090 32G`
- Preserve business semantics
- Keep local defaults conservative; only cloud runs should opt into these overrides

## Principle

- Local defaults stay safe.
- Cloud runs should use explicit env overrides.
- `11_3` should use the fastest verified cloud prompt path, not a fixed prompt-construction backend.
- Stage lineage should follow `11_2 -> 11_1 -> 09`, not raw latest-by-mtime when a successful upstream run is already known.
- Do not change label definition, candidate boundary, split logic, or metric definition through this profile.

## Fixed Path Rule

Use stable latest pointers for stage11 on cloud:

- `data/output/_latest_runs/stage11_1_qlora_build_dataset.json`
- `data/output/_latest_runs/stage11_2_qlora_train.json`
- `data/output/_latest_runs/stage11_3_qlora_sidecar_eval.json`

Behavior:

- `11_1` writes the latest dataset pointer after a successful run.
- `11_2` prefers the `11_1` pointer before falling back to latest-by-mtime.
- `11_3` prefers the `11_2` pointer, then resolves `11_1` and `stage09` from upstream run metadata before falling back to latest-by-mtime.

Operational rule:

- Do not manually hand-edit `INPUT_11_RUN_DIR` / `INPUT_11_2_RUN_DIR` for normal cloud runs.
- Only override input run env vars when you intentionally want to rerun against an older snapshot.
- Current exception: when running the patched `tower/seq` lineage before the
  stage09 gate summary CSV is refreshed, pin the current `11_1/11_2` inputs
  explicitly and set `QLORA_ENFORCE_STAGE09_GATE=false`.

## Stage 11_1 Dataset Build

Recommended cloud env:

```bash
export PYSPARK_PYTHON=/root/miniconda3/bin/python
export PYSPARK_DRIVER_PYTHON=/root/miniconda3/bin/python
export PYTHONPATH=/root/5006_BDA_project/scripts:/root/5006_BDA_project
export PYTHONUNBUFFERED=1
export TMP=/root/autodl-tmp/tmp
export TEMP=/root/autodl-tmp/tmp
export TMPDIR=/root/autodl-tmp/tmp

export SPARK_MASTER=local[32]
export SPARK_DRIVER_MEMORY=48g
export SPARK_EXECUTOR_MEMORY=48g
export SPARK_SQL_SHUFFLE_PARTITIONS=128
export SPARK_DEFAULT_PARALLELISM=64
export SPARK_NETWORK_TIMEOUT=1200s
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL=120s
export SPARK_LOCAL_DIR=/root/autodl-tmp/spark-tmp
export SPARK_TMP_SESSION_ISOLATION=true
export SPARK_TMP_AUTOCLEAN_ENABLED=true

export QLORA_TOPN_PER_USER=150
export QLORA_EVAL_USER_FRAC=0.2
export QLORA_ENABLE_RICH_SFT_EXPORT=true
export QLORA_PROMPT_MODE=full_lite
export QLORA_INCLUDE_VALID_POS=true
export QLORA_VALID_POS_WEIGHT=0.6
export QLORA_RICH_SFT_NEG_EXPLICIT=1
export QLORA_RICH_SFT_NEG_HARD=2
export QLORA_RICH_SFT_NEG_NEAR=2
export QLORA_RICH_SFT_NEG_MID=1
export QLORA_RICH_SFT_NEG_TAIL=0
export QLORA_RICH_SFT_NEG_HARD_RANK_MAX=20
export QLORA_RICH_SFT_NEG_MID_RANK_MAX=60
export QLORA_RICH_SFT_NEG_MAX_RATING=2.5
export QLORA_RICH_SFT_ALLOW_NEG_FILL=false
export QLORA_REVIEW_TABLE_PATH=/root/autodl-tmp/project_data/data/parquet/yelp_academic_dataset_review
export QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas
```

Notes:
- The current cloud mainline is `full_lite + rich_sft`, not the older
  `semantic + banded pointwise` path.
- `TOPN_PER_USER=150` keeps data aligned with `pretrim150`, while
  `rich_sft` negatives stay in the `1 explicit + 2 hard + 2 near + 1 mid`
  shape that was used in the latest successful tower/seq-enabled build.
- Keep `QLORA_EVAL_USER_FRAC=0.2` for comparability with the current selector
  cohort.
- On cloud, prefer `QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas` so review snippet scoring runs as pandas UDF instead of per-row Python UDF.
- As of `2026-03-11`, `11_1` tail-end audit actions were reduced in code: the pre-write full `split/label` scan was replaced by a lightweight emptiness check, and summary `collect()` calls for base/rich_sft/pairwise_pool were merged where possible. Use the latest script before benchmarking `11_1`.
- A long quiet tail after the main writes is now usually the final summary/audit pass, not a stuck Spark cluster. If `train_json/eval_json/rich_sft_*` already exist and Spark still shows one last `collect`, let it finish before restarting.

## Stage 11_2 SFT Train

Recommended cloud env:

```bash
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OUTPUT_11_MODELS_ROOT_DIR=/root/autodl-tmp/stage11_fs/models

export QLORA_USE_4BIT=true
export QLORA_USE_BF16=true
export QLORA_GRADIENT_CHECKPOINTING=true

export QLORA_TRAIN_SOURCE=rich_sft
export QLORA_NEG_REBALANCE_MODE=off
export QLORA_LOSS_MODE=binary_yesno
export QLORA_MAX_SEQ_LEN=512
export QLORA_BATCH_SIZE=8
export QLORA_EVAL_BATCH_SIZE=32
export QLORA_GRAD_ACC=2
export QLORA_PAD_TO_MULTIPLE_OF=64
export QLORA_LR=2e-4
export QLORA_EPOCHS=1.0

export QLORA_MAX_NEG_POS_RATIO=0
export QLORA_EVAL_STEPS=200
export QLORA_SAVE_STEPS=200
export QLORA_SAVE_TOTAL_LIMIT=6
export QLORA_LOGGING_STEPS=25
export QLORA_FORMAT_MAP_NUM_PROC=8
export QLORA_TOKENIZE_MAP_NUM_PROC=8
export QLORA_DATALOADER_NUM_WORKERS=8
export QLORA_CHECKPOINT_AUDIT_ENABLED=true
export QLORA_CHECKPOINT_AUDIT_PROFILE=smoke
export QLORA_CHECKPOINT_AUDIT_PROMPT_MODE=full_lite
export QLORA_CHECKPOINT_AUDIT_INVERT_PROB=false
export QLORA_CHECKPOINT_AUDIT_USE_EVAL_SPLIT=true
export QLORA_CHECKPOINT_AUDIT_MAX_USERS=64
export QLORA_CHECKPOINT_AUDIT_MAX_ROWS=8192
export QLORA_TOKEN_AUDIT_ENABLED=true
```

Notes:
- The current cloud mainline trains from `rich_sft` on the `full_lite`
  prompt family and keeps trainer-side rebalance off.
- `MAX_NEG_POS_RATIO=0` keeps training aligned with the upstream rich_sft
  distribution instead of reintroducing trainer-side `1:4` rebalance.
- The current stable cloud run used `batch=8 / grad_acc=2 / eval_batch=32 /
  seq=512 / eval-save=200`.
- For RTX `5090` / Blackwell cloud runs, keep `QLORA_PAD_TO_MULTIPLE_OF=64`; this reduces first-shape compile churn without changing supervision or batch semantics.
- The current tower/seq-enabled patched-stage09 lineage may require
  `QLORA_ENFORCE_STAGE09_GATE=false` until `stage09_recall_audit_summary_latest.csv`
  includes that source run.
- `QLORA_ATTN_IMPLEMENTATION` is now supported, but there is no train-time benchmark yet; leave it unset unless a dedicated train benchmark is run.
- For future checkpoint smoke audits on the current `full_lite` mainline,
  prefer `QLORA_CHECKPOINT_AUDIT_INVERT_PROB=false`; the latest full selector
  run improved with `noinvert`, while the earlier `invert=true` result was
  specific to the previous SFT lineage.

### 9B SFT Experimental Profile

Verified `Qwen3.5-9B` cloud SFT run:

- run dir: `/root/autodl-tmp/stage11_fs/models/20260312_175852_stage11_2_qlora_train`
- base model: `/root/hf_models/Qwen3.5-9B`
- data run: `/root/autodl-tmp/stage11_fs/input11data/20260311_011112_stage11_1_qlora_build_dataset`

Verified training env:

```bash
export QLORA_BASE_MODEL=/root/hf_models/Qwen3.5-9B
export QLORA_REQUIRED_BASE_MODEL=/root/hf_models/Qwen3.5-9B
export QLORA_ENFORCE_REQUIRED_BASE_MODEL=true

export QLORA_USE_4BIT=true
export QLORA_USE_BF16=true
export QLORA_GRADIENT_CHECKPOINTING=true
export QLORA_QWEN35_SAFE_KBIT_PREP=true
export QLORA_QWEN35_FORCE_FLOAT_PARAMS_BF16=true
export QLORA_QWEN35_MAMBA_SSM_DTYPE=auto

export QLORA_TRAIN_SOURCE=rich_sft
export QLORA_NEG_REBALANCE_MODE=off
export QLORA_LOSS_MODE=binary_yesno
export QLORA_MAX_SEQ_LEN=512
export QLORA_PAD_TO_MULTIPLE_OF=64

export QLORA_BATCH_SIZE=8
export QLORA_GRAD_ACC=2
export QLORA_EVAL_BATCH_SIZE=8
export QLORA_LR=2e-4
export QLORA_EPOCHS=1.0
export QLORA_EVAL_STEPS=200
export QLORA_SAVE_STEPS=200
export QLORA_SAVE_TOTAL_LIMIT=6
export QLORA_LOGGING_STEPS=10

export QLORA_DATASET_MAP_NUM_PROC=8
export QLORA_FORMAT_MAP_NUM_PROC=8
export QLORA_TOKENIZE_MAP_NUM_PROC=8
export QLORA_DATALOADER_NUM_WORKERS=8
```

Verified result summary:

- final `eval_loss = 0.36555`
- checkpoint trajectory:
  - `ckpt-200 = 0.47031`
  - `ckpt-400 = 0.41406`
  - `ckpt-600 = 0.40154`
  - `ckpt-800 = 0.37234`
  - `ckpt-1000 = 0.36619`
  - `final = 0.36555`
- token audit:
  - `eval full_len p95=410`
  - `eval full_len p99=446`
  - `eval full_len max=482`
  - `at_cap_rate=0.00025`

Fast-path note:

- On the current `RTX 5090 32GB` cloud, `Qwen3.5-9B` was verified to use the
  `FLA + causal-conv1d` path.
- A small real-data benchmark on the same machine showed about `2.02x`
  speedup versus a forced fallback load path.

## Stage 11_2 DPO Train

Recommended cloud env:

```bash
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OUTPUT_11_MODELS_ROOT_DIR=/root/autodl-tmp/stage11_fs/models

export QLORA_BASE_MODEL=/root/hf_models/Qwen3.5-4B
export QLORA_REQUIRED_BASE_MODEL=/root/hf_models/Qwen3.5-4B
export QLORA_ENFORCE_REQUIRED_BASE_MODEL=true
export QLORA_ENFORCE_STAGE09_GATE=false

export QLORA_USE_4BIT=true
export QLORA_USE_BF16=true
export QLORA_GRADIENT_CHECKPOINTING=true
export QLORA_SFT_ADAPTER_DIR=/root/autodl-tmp/stage11_fs/models/20260311_020730_stage11_2_qlora_train/adapter
export QLORA_PAIRWISE_SOURCE_MODE=rich_sft
export QLORA_DPO_PAIR_POLICY=v2a
export QLORA_DPO_PAIR_TOPK_CUTOFF=10
export QLORA_DPO_PAIR_HIGH_RANK_CUTOFF=20

export QLORA_DPO_MAX_PAIRS=4
export QLORA_DPO_TRUE_MAX_PAIRS=2
export QLORA_DPO_VALID_MAX_PAIRS=1
export QLORA_DPO_HIST_MAX_PAIRS=1
export QLORA_DPO_ALLOW_MID_NEG=true
export QLORA_DPO_BETA=0.1
export QLORA_DPO_LOSS_TYPE=sigmoid

export QLORA_MAX_SEQ_LEN=768
export QLORA_PAD_TO_MULTIPLE_OF=64
export QLORA_BATCH_SIZE=1
export QLORA_GRAD_ACC=8
export QLORA_LR=5e-5
export QLORA_EPOCHS=0.6
export QLORA_EVAL_STEPS=100
export QLORA_SAVE_STEPS=100
export QLORA_LOGGING_STEPS=10
```

Notes:
- The current DPO mainline should warm-start from the latest healthy
  `full_lite rich_sft` SFT adapter rather than starting from the raw base.
- Keep the first DPO run conservative on batch size. `batch=1 / grad_acc=8`
  preserves an effective batch of `8` without making 32GB memory assumptions.
- Keep `QLORA_PAD_TO_MULTIPLE_OF=64` on RTX 5090 / Blackwell for the same
  reason as SFT and `11_3`: with `batch=1`, sequence-shape churn is otherwise
  especially high and the first several DPO steps spend too much time in
  compile.
- `QLORA_PAIRWISE_SOURCE_MODE=rich_sft` is required for the current mainline;
  it makes `11_2_dpo_train.py` consume `rich_sft_train_json/eval_json` and the
  same `true/valid/hist_pos > hard/near/observed_dislike` pair logic already
  audited in the DPO export step.
- For the current selector-aware DPO line, use `QLORA_DPO_PAIR_POLICY=v2a`
  with `QLORA_DPO_PAIR_TOPK_CUTOFF=10` and
  `QLORA_DPO_PAIR_HIGH_RANK_CUTOFF=20`; this shifts loser selection toward
  `top10/top20` incumbents and outranking confusers instead of generic
  `observed_dislike`.
- The current audited DPO shape is roughly `5836` train pairs and `1452` eval
  pairs on bucket10. Do not inflate `max_pairs` in the first run.
- The current `v2a` DPO line should start with the shorter schedule
  `epochs=0.6 / eval_steps=100 / save_steps=100`; the older longer schedule
  overran its best checkpoint.

## Stage 11_3 Sidecar Eval

Use one fixed base profile plus one explicit eval mode.

Cloud base env:

```bash
export PYSPARK_PYTHON=/root/miniconda3/bin/python
export PYSPARK_DRIVER_PYTHON=/root/miniconda3/bin/python
export PYTHONPATH=/root/5006_BDA_project/scripts:/root/5006_BDA_project
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TMP=/root/autodl-tmp/tmp
export TEMP=/root/autodl-tmp/tmp
export TMPDIR=/root/autodl-tmp/tmp

export SPARK_MASTER=local[32]
export SPARK_DRIVER_MEMORY=48g
export SPARK_EXECUTOR_MEMORY=48g
export SPARK_SQL_SHUFFLE_PARTITIONS=128
export SPARK_DEFAULT_PARALLELISM=64
export SPARK_NETWORK_TIMEOUT=1200s
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL=120s
export SPARK_PYTHON_WORKER_MEMORY=4g
export SPARK_LOCAL_DIR=/root/autodl-tmp/spark-tmp
export SPARK_TMP_SESSION_ISOLATION=true
export SPARK_TMP_AUTOCLEAN_ENABLED=true
export OUTPUT_11_SIDECAR_ROOT_DIR=/root/autodl-tmp/stage11_fs/sidecar_eval

export QLORA_PROMPT_MODE=full_lite
export QLORA_INVERT_PROB=false
export QLORA_EVAL_PROMPT_BUILD_MODE=driver
export QLORA_EVAL_DRIVER_PROMPT_IMPL=itertuples
export QLORA_EVAL_ARROW_TO_PANDAS=true
export QLORA_EVAL_ARROW_FALLBACK=true
export QLORA_EVAL_ATTN_IMPLEMENTATION=sdpa
export QLORA_EVAL_BATCH_SIZE=60
export QLORA_EVAL_MAX_SEQ_LEN=448
export QLORA_EVAL_PAD_TO_MULTIPLE_OF=64
export QLORA_EVAL_PROMPT_CHUNK_ROWS=8192
export QLORA_EVAL_STREAM_LOG_ROWS=4096
export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS=32768
export QLORA_EVAL_ITER_COALESCE=0
export QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK=true
export QLORA_EVAL_PIN_MEMORY=true
export QLORA_EVAL_NON_BLOCKING_H2D=true

export QLORA_RERANK_TOPN=150
export RANK_EVAL_TOP_K=10
export QLORA_BLEND_ALPHA=0.12
export QLORA_EVAL_USER_COHORT_PATH=
export QLORA_ENABLE_RAW_REVIEW_TEXT=true
export QLORA_REVIEW_TABLE_PATH=/root/autodl-tmp/project_data/data/parquet/yelp_academic_dataset_review
export QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas
export QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED=true
export QLORA_EVAL_REVIEW_BASE_CACHE_ROOT=/root/autodl-tmp/stage11_fs/tmp/review_base_cache
export QLORA_EVAL_QWEN35_NO_THINK=false
```

Choose exactly one mode below:

Official report mode:

```bash
export QLORA_EVAL_PROFILE=report
export QLORA_EVAL_USE_STAGE11_SPLIT=false
export QLORA_EVAL_MAX_USERS_PER_BUCKET=0
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=0
```

Official selector mode:

```bash
export QLORA_EVAL_PROFILE=selector
export QLORA_EVAL_USE_STAGE11_SPLIT=true
export QLORA_EVAL_MAX_USERS_PER_BUCKET=0
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=0
```

Fast smoke mode:

```bash
export QLORA_EVAL_PROFILE=smoke
export QLORA_EVAL_USE_STAGE11_SPLIT=true
export QLORA_EVAL_MAX_USERS_PER_BUCKET=200
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=30000
```

Notes:
- On cloud, the fixed `11_3` base path is `local[32] / 48g / driver-prompt / itertuples / Arrow / sdpa`.
- In the current cloud environment, Spark-side Python UDF prompt construction has a large first-yield delay on report-cohort `11_3`, even when Spark CPU is well utilized.
- For cloud driver mode, use `QLORA_EVAL_DRIVER_PROMPT_IMPL=itertuples` instead of `pandas.apply(axis=1)`; this avoids per-row Series materialization and is the preferred low-risk fast path.
- Enable `QLORA_EVAL_ARROW_TO_PANDAS=true` on cloud `driver` runs unless a specific schema issue proves otherwise.
- The current full mainline eval path is `full_lite + invert_prob=false`; the
  latest tower/seq-enabled final adapter improved under `noinvert` on the
  selector cohort.
- For the current `20260311_210350_stage11_2_dpo_train` `v2a` DPO lineage,
  offline alpha sweep on the saved selector scores peaked at
  `QLORA_BLEND_ALPHA=0.42` (`Recall@10=0.06233`, `NDCG@10=0.02807`) versus the
  official `alpha=0.12` run (`Recall@10=0.05691`, `NDCG@10=0.02644`). Treat
  `0.42` as the lineage-specific formal rerun value for that DPO adapter, not a
  new global default for all `11_3` runs.
- In `full_lite` prompt mode, `11_3` still trims driver-side payload to only
  the fields needed by `full_lite`, including history anchors and selected
  model signals, reducing `toPandas` cost without changing prompt semantics.
- On cloud, enable `QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED=true` so `11_3` reuses a preprocessed review-base parquet and does not rebuild `text_clean + substring + tag score` from raw reviews on every run.
- Keep `QLORA_EVAL_BATCH_SIZE=60` unless a specific run proves `>60` is stable on the current prompt length; GPU memory is already close to the ceiling.
- Keep `QLORA_EVAL_MAX_SEQ_LEN=448` unless a dedicated `11_3` prompt-length audit proves a smaller value is safe.
- For RTX `5090` / Blackwell cloud runs, keep `QLORA_EVAL_PAD_TO_MULTIPLE_OF=64` so different prompt lengths collapse onto fewer input shapes and avoid repeated first-batch compile stalls.
- Keep `QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK=true`,
  `QLORA_EVAL_PIN_MEMORY=true`, and `QLORA_EVAL_NON_BLOCKING_H2D=true`; they
  materially improved 5090 feed efficiency in the current eval path.
- The current patched-stage09 lineage may require
  `QLORA_ENFORCE_STAGE09_GATE=false` until the gate summary CSV catches up with
  the patched source run.
- If `INPUT_11_2_RUN_DIR` points to the current `v2a` DPO run
  (`20260311_210350_stage11_2_dpo_train`), override `QLORA_BLEND_ALPHA=0.42`
  for the formal `11_3` rerun. Keep `0.12` for the existing SFT mainline until
  a separate alpha sweep says otherwise.
- `report` and `selector` are full-cohort modes. Do not add user or row caps to them.
- Use `smoke` only for timing/debug. Its metrics are not official.

### 9B SFT Selector Eval Note

Verified `Qwen3.5-9B` SFT selector eval:

- run dir: `/root/autodl-tmp/stage11_fs/sidecar_eval/20260312_200308_stage11_3_qlora_sidecar_eval`
- model source: `/root/autodl-tmp/stage11_fs/models/20260312_175852_stage11_2_qlora_train`

Verified eval env:

```bash
export QLORA_PROMPT_MODE=full_lite
export QLORA_INVERT_PROB=false
export QLORA_BLEND_ALPHA=0.12
export QLORA_EVAL_USE_STAGE11_SPLIT=true
export QLORA_RERANK_TOPN=150
export QLORA_EVAL_BATCH_SIZE=48
export QLORA_EVAL_MAX_SEQ_LEN=512
export QLORA_EVAL_PAD_TO_MULTIPLE_OF=64
export QLORA_EVAL_PROMPT_CHUNK_ROWS=16384
export QLORA_EVAL_STREAM_LOG_ROWS=8192
export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS=65536
export QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK=true
export QLORA_EVAL_PIN_MEMORY=true
export QLORA_EVAL_NON_BLOCKING_H2D=true
export QLORA_EVAL_ATTN_IMPLEMENTATION=sdpa
export QLORA_EVAL_DRIVER_PROMPT_IMPL=itertuples
export QLORA_EVAL_ARROW_TO_PANDAS=true
export QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED=true
export QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas
```

Verified official result:

- `PreScore@10`: `Recall=0.0569106`, `NDCG=0.0264671`
- `QLoRASidecar@10`: `Recall=0.0582656`, `NDCG=0.0270444`

Offline alpha sweep on the saved selector scores:

- official run used `alpha=0.12`
- best offline point was `alpha=0.26`
- offline best metrics: `Recall=0.0623306`, `NDCG=0.0286679`

Interpretation:

- The `9B` SFT line has a stronger sidecar signal than the current `4B` SFT
  line under tuned alpha.
- The formal mainline remains the `4B DPO v3` winner until `9B DPO` is
  validated on the same full selector eval split.

## Eval Cohort Policy

Use three explicit modes:

- `report` mode:
  - `QLORA_EVAL_USE_STAGE11_SPLIT=false`
  - `QLORA_EVAL_MAX_USERS_PER_BUCKET=0`
  - `QLORA_EVAL_MAX_ROWS_PER_BUCKET=0`
  - Purpose: final metric reporting on the full current stage09 lineage cohort
- `selector` mode:
  - `QLORA_EVAL_USE_STAGE11_SPLIT=true`
  - `QLORA_EVAL_MAX_USERS_PER_BUCKET=0`
  - `QLORA_EVAL_MAX_ROWS_PER_BUCKET=0`
  - Purpose: fast iteration on the stage11 eval cohort
- `smoke` mode:
  - `QLORA_EVAL_USE_STAGE11_SPLIT=true`
  - `QLORA_EVAL_MAX_USERS_PER_BUCKET>0` and/or `QLORA_EVAL_MAX_ROWS_PER_BUCKET>0`
  - Purpose: timing/debug only

Current example:

- new stage11 pointwise dataset has `370` eval users
- with `RERANK_TOPN=150`, selector mode yields `55500` rows
- a capped report run with `MAX_USERS_PER_BUCKET=1200` yields `180000` rows
- this is a sampled report cohort, not an official report run

Do not compare selector-mode or smoke-mode `11_3` metrics directly against full-cohort report runs.

If you need comparison with an older `110700`-row historical report, reuse the same upstream stage09 run or an explicit frozen user list. Do not emulate historical parity by setting a random user cap such as `MAX_USERS_PER_BUCKET=738`.

Historical comparable report cohort:

- `data/output/11_eval_cohorts/historical_report_cohort_738_bucket10_users.csv`
- This file freezes the old `110700 = 738 * 150` user cohort and should be passed through `QLORA_EVAL_USER_COHORT_PATH` when you want historical comparability.
- Keep `QLORA_EVAL_PROFILE=report`, `QLORA_EVAL_USE_STAGE11_SPLIT=false`, and both caps at `0`; the cohort file replaces random sampling.

## Attention Backend Policy

Measured on cloud on `2026-03-09` with the current `Qwen3.5-4B + adapter` eval-like forward path, `240` real prompts, `batch=60`, `seq=448`:

- `default`: `10.983 rows/s`
- `eager`: `24.951 rows/s`
- `sdpa`: `25.263 rows/s`
- `flash_attention_2`: unavailable because `flash_attn` is not installed in `/root/miniconda3`

Default cloud policy for `11_3`:

- Use `QLORA_EVAL_ATTN_IMPLEMENTATION=sdpa`
- Do not use `eager`
- Do not assume FA2 is available; verify package installation first
- `flash-linear-attention` / `fla-core` being installed does not mean `attn_implementation=flash_attention_2` is available. They are different kernels and different code paths.

This benchmark only covers eval-style forward scoring on `240` prompt-ready samples. It does not automatically justify forcing `sdpa` for `11_2` training.

Batch-size sweep under `sdpa` on the same prompt sample:

- `batch=60`: `23.851 rows/s`, stable
- `batch=72`: `5.316 rows/s`, slower despite fitting
- `batch=84`: OOM
- `batch=96`: OOM

Default cloud policy for `11_3` remains `QLORA_EVAL_BATCH_SIZE=60`.

Throughput interpretation:

- `24-25 rows/s` is a micro-benchmark for pure model scoring on a small prompt-ready sample.
- Full `11_3` runs are slower because they also include per-batch tokenization, prompt-length variance, host-to-device transfers, and chunk warm-up effects.
- On the current `full_lite + invert_prob + pretokenize` selector path, a
  fresh `110700`-row run reached `toPandas=43.2s`, driver prompt generation
  `23.7s`, and early chunk scoring around `26.7-29.1 rows/s`.

## Why These Settings

- `11_1` is mainly Spark/review-aggregation bound; cloud should use wider Spark parallelism and larger memory.
- `11_2` is still GPU-memory sensitive, but the current 5090 cloud mainline is
  stable at `batch=8 / grad_acc=2 / seq=512 / pad64`.
- `11_3` is split between Spark preprocessing and single-GPU scoring. Cloud should accelerate the preprocessing side, but GPU batch should remain conservative near the validated ceiling.
- For the current cloud image, the best measured `11_3` path is: `full_lite`
  prompt, driver-side prompt build with `itertuples`, `pad_to_multiple_of=64`,
  chunk pretokenization, and GPU scoring with `sdpa`.

## Operational Rule

Future cloud runs for `11_1`, `11_2`, and `11_3` should use this profile unless there is a documented reason to deviate.

## Canonical Commands

Use these two commands as the only normal cloud launch paths for `11_3`.

### 11_3 Report

```bash
cd /root/5006_BDA_project
unset INPUT_09_RUN_DIR INPUT_11_DATA_RUN_DIR INPUT_11_2_RUN_DIR
export PYSPARK_PYTHON=/root/miniconda3/bin/python
export PYSPARK_DRIVER_PYTHON=/root/miniconda3/bin/python
export PYTHONPATH=/root/5006_BDA_project/scripts:/root/5006_BDA_project
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TMP=/root/autodl-tmp/tmp
export TEMP=/root/autodl-tmp/tmp
export TMPDIR=/root/autodl-tmp/tmp
export SPARK_MASTER=local[32]
export SPARK_DRIVER_MEMORY=48g
export SPARK_EXECUTOR_MEMORY=48g
export SPARK_SQL_SHUFFLE_PARTITIONS=128
export SPARK_DEFAULT_PARALLELISM=64
export SPARK_NETWORK_TIMEOUT=1200s
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL=120s
export SPARK_PYTHON_WORKER_MEMORY=4g
export SPARK_LOCAL_DIR=/root/autodl-tmp/spark-tmp
export SPARK_TMP_SESSION_ISOLATION=true
export SPARK_TMP_AUTOCLEAN_ENABLED=true
export OUTPUT_11_SIDECAR_ROOT_DIR=/root/autodl-tmp/stage11_fs/sidecar_eval
export QLORA_PROMPT_MODE=full_lite
export QLORA_INVERT_PROB=true
export QLORA_EVAL_PROMPT_BUILD_MODE=driver
export QLORA_EVAL_DRIVER_PROMPT_IMPL=itertuples
export QLORA_EVAL_ARROW_TO_PANDAS=true
export QLORA_EVAL_ARROW_FALLBACK=true
export QLORA_EVAL_ATTN_IMPLEMENTATION=sdpa
export QLORA_EVAL_BATCH_SIZE=60
export QLORA_EVAL_MAX_SEQ_LEN=448
export QLORA_EVAL_PAD_TO_MULTIPLE_OF=64
export QLORA_EVAL_PROMPT_CHUNK_ROWS=8192
export QLORA_EVAL_STREAM_LOG_ROWS=4096
export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS=32768
export QLORA_EVAL_ITER_COALESCE=0
export QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK=true
export QLORA_EVAL_PIN_MEMORY=true
export QLORA_EVAL_NON_BLOCKING_H2D=true
export QLORA_RERANK_TOPN=150
export RANK_EVAL_TOP_K=10
export QLORA_BLEND_ALPHA=0.12
export QLORA_EVAL_PROFILE=report
export QLORA_EVAL_USE_STAGE11_SPLIT=false
export QLORA_EVAL_MAX_USERS_PER_BUCKET=0
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=0
export QLORA_EVAL_USER_COHORT_PATH=
export QLORA_ENABLE_RAW_REVIEW_TEXT=true
export QLORA_REVIEW_TABLE_PATH=/root/autodl-tmp/project_data/data/parquet/yelp_academic_dataset_review
export QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas
export QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED=true
export QLORA_EVAL_REVIEW_BASE_CACHE_ROOT=/root/autodl-tmp/stage11_fs/tmp/review_base_cache
export QLORA_EVAL_QWEN35_NO_THINK=false
# Current patched-stage09/tower-seq line before gate csv refresh:
# export INPUT_11_2_RUN_DIR=/root/autodl-tmp/stage11_fs/models/20260311_020730_stage11_2_qlora_train
# export INPUT_11_DATA_RUN_DIR=/root/autodl-tmp/stage11_fs/input11data/20260311_011112_stage11_1_qlora_build_dataset
# export QLORA_ENFORCE_STAGE09_GATE=false
/root/miniconda3/bin/python /root/5006_BDA_project/scripts/11_3_qlora_sidecar_eval.py
```

Historical-comparable `11_3` report:

```bash
export QLORA_EVAL_USER_COHORT_PATH=/root/5006_BDA_project/data/output/11_eval_cohorts/historical_report_cohort_738_bucket10_users.csv
```

### 11_3 Selector

```bash
cd /root/5006_BDA_project
unset INPUT_09_RUN_DIR INPUT_11_DATA_RUN_DIR INPUT_11_2_RUN_DIR
export PYSPARK_PYTHON=/root/miniconda3/bin/python
export PYSPARK_DRIVER_PYTHON=/root/miniconda3/bin/python
export PYTHONPATH=/root/5006_BDA_project/scripts:/root/5006_BDA_project
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TMP=/root/autodl-tmp/tmp
export TEMP=/root/autodl-tmp/tmp
export TMPDIR=/root/autodl-tmp/tmp
export SPARK_MASTER=local[32]
export SPARK_DRIVER_MEMORY=48g
export SPARK_EXECUTOR_MEMORY=48g
export SPARK_SQL_SHUFFLE_PARTITIONS=128
export SPARK_DEFAULT_PARALLELISM=64
export SPARK_NETWORK_TIMEOUT=1200s
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL=120s
export SPARK_PYTHON_WORKER_MEMORY=4g
export SPARK_LOCAL_DIR=/root/autodl-tmp/spark-tmp
export SPARK_TMP_SESSION_ISOLATION=true
export SPARK_TMP_AUTOCLEAN_ENABLED=true
export OUTPUT_11_SIDECAR_ROOT_DIR=/root/autodl-tmp/stage11_fs/sidecar_eval
export QLORA_PROMPT_MODE=full_lite
export QLORA_INVERT_PROB=true
export QLORA_EVAL_PROMPT_BUILD_MODE=driver
export QLORA_EVAL_DRIVER_PROMPT_IMPL=itertuples
export QLORA_EVAL_ARROW_TO_PANDAS=true
export QLORA_EVAL_ARROW_FALLBACK=true
export QLORA_EVAL_ATTN_IMPLEMENTATION=sdpa
export QLORA_EVAL_BATCH_SIZE=60
export QLORA_EVAL_MAX_SEQ_LEN=448
export QLORA_EVAL_PAD_TO_MULTIPLE_OF=64
export QLORA_EVAL_PROMPT_CHUNK_ROWS=8192
export QLORA_EVAL_STREAM_LOG_ROWS=4096
export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS=32768
export QLORA_EVAL_ITER_COALESCE=0
export QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK=true
export QLORA_EVAL_PIN_MEMORY=true
export QLORA_EVAL_NON_BLOCKING_H2D=true
export QLORA_RERANK_TOPN=150
export RANK_EVAL_TOP_K=10
export QLORA_BLEND_ALPHA=0.12
export QLORA_EVAL_PROFILE=selector
export QLORA_EVAL_USE_STAGE11_SPLIT=true
export QLORA_EVAL_MAX_USERS_PER_BUCKET=0
export QLORA_EVAL_MAX_ROWS_PER_BUCKET=0
export QLORA_ENABLE_RAW_REVIEW_TEXT=true
export QLORA_REVIEW_TABLE_PATH=/root/autodl-tmp/project_data/data/parquet/yelp_academic_dataset_review
export QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE=pandas
export QLORA_EVAL_QWEN35_NO_THINK=false
# Current patched-stage09/tower-seq line before gate csv refresh:
# export INPUT_11_2_RUN_DIR=/root/autodl-tmp/stage11_fs/models/20260311_020730_stage11_2_qlora_train
# export INPUT_11_DATA_RUN_DIR=/root/autodl-tmp/stage11_fs/input11data/20260311_011112_stage11_1_qlora_build_dataset
# export QLORA_ENFORCE_STAGE09_GATE=false
/root/miniconda3/bin/python /root/5006_BDA_project/scripts/11_3_qlora_sidecar_eval.py
```
