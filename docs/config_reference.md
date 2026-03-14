# Config Reference

## Purpose

This file lists the main environment variables used by the active release path.

The shared path model is:

1. use the current repo root by default
2. allow explicit env override when needed
3. auto-normalize older saved roots such as `D:/5006 BDA project/...`

## Shared Root Resolution

- `BDA_PROJECT_ROOT`
- `PROJECT_ROOT`
- `REPO_ROOT`

If one of these is set, `scripts/pipeline/project_paths.py` uses it as the repo root.

## Shared Temp / Spark

- `SPARK_LOCAL_DIR`
  - default: `data/spark-tmp`
  - used by active `07` and `10` scripts after `GL-09`
- `SPARK_DRIVER_MEMORY`
- `SPARK_EXECUTOR_MEMORY`

## Stage07 Relabel / Cluster

- `LABEL_CONFIG_DIR`
  - default: `config/labeling/food_service/v1`
- `PARQUET_BASE_DIR`
  - default: `data/parquet`
- `OUTPUT_07_ROOT_DIR`
  - default: `data/output/07_embedding_cluster`

## Stage10 Train

- `INPUT_09_RUN_DIR`
  - explicit `stage09` run dir override
- `INPUT_09_ROOT_DIR`
  - default: `data/output/09_candidate_fusion`
- `OUTPUT_10_1_ROOT_DIR`
  - default: `data/output/10_rank_models`

## Stage10 Infer / Eval

- `INPUT_09_RUN_DIR`
- `INPUT_09_ROOT_DIR`
- `RANK_MODEL_ROOT_DIR`
  - default: `data/output/10_rank_models`
- `OUTPUT_10_2_ROOT_DIR`
  - default: `data/output/10_2_rank_infer_eval`
- `STAGE10_RESULTS_METRICS_PATH`
  - default: `data/metrics/recsys_stage10_results.csv`
- `RANK_EVAL_USER_COHORT_PATH`
  - explicit eval user cohort file
- `RANK_EVAL_CANDIDATE_TOPN`
  - candidate depth cap for aligned evaluation

## Stage10 Transformer Rerank Eval

- `INPUT_09_RUN_DIR`
- `INPUT_09_ROOT_DIR`
- `OUTPUT_10_5_ROOT_DIR`
  - default: `data/output/10_5_transformer_rerank_eval`
- `STAGE10_TRANSFORMER_RESULTS_METRICS_PATH`
  - default: `data/metrics/recsys_stage10_transformer_results.csv`
- `USER_PROFILE_TABLE`
  - optional explicit profile parquet/csv path
- `TF_ITEM_SEMANTIC_FEATURES`
  - optional explicit item semantic feature file
- `TRANSFORMER_MODEL`
  - optional explicit model override
- `BGE_LOCAL_MODEL_PATH`
  - optional Hugging Face local cache path

## Release Pointers

Release pointers are now separate from experiment pointers:

- experiment: `data/output/_latest_runs`
- release: `data/output/_prod_runs`

Current release pointer files:

- [stage09_release.json](../data/output/_prod_runs/stage09_release.json)
- [stage10_release.json](../data/output/_prod_runs/stage10_release.json)
- [stage11_release.json](../data/output/_prod_runs/stage11_release.json)
- [release_policy.json](../data/output/_prod_runs/release_policy.json)

## Notes

- Old saved repo-root strings under `D:/5006 BDA project/...` and
  `D:/5006_BDA_project/...` are normalized automatically by `project_paths`.
- This file is a live reference for the active release path, not a complete list of every
  historical experiment variable in the repo.
