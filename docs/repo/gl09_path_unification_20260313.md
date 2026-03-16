# GL-09 Path Unification (2026-03-13)

## Scope

This pass closes `GL-09` for the active release path and its immediate upstream prep stage.

Touched scripts:

- [scripts/10_1_rank_train.py](../../scripts/10_1_rank_train.py)
- [scripts/10_5_transformer_rerank_eval.py](../../scripts/10_5_transformer_rerank_eval.py)
- [scripts/07_relabel_then_cluster.py](../../scripts/07_relabel_then_cluster.py)
- [scripts/pipeline/project_paths.py](../../scripts/pipeline/project_paths.py)

## What Changed

1. Stage10 train now resolves repo-root paths through `project_paths` instead of hardcoded
   `D:/5006 BDA project/...` defaults.
2. Stage10 transformer rerank eval now resolves:
   - `stage09` input root
   - output root
   - metrics file
   - Spark local temp dir
   - `run_meta` paths that still use legacy repo roots
3. Stage07 relabel/cluster now resolves:
   - labeling config dir
   - parquet base dir
   - output root
   - Spark local temp dir
4. `env_or_project_path(...)` now auto-normalizes both:
   - `D:/5006 BDA project/...`
   - `D:/5006_BDA_project/...`

## Operational Meaning

The active release path now shares one consistent repo-root resolution model:

- default to current repo root
- allow explicit env override
- accept older saved paths without manual editing

This removes the highest-impact portability break between local Windows runs and synced
cloud artifacts.

## Validation

Validation run for this pass:

- `py_compile` on all touched scripts: pass
- no remaining project-root hardcoding in touched scripts

Checked command outcome:

- `scripts/pipeline/project_paths.py`
- `scripts/10_1_rank_train.py`
- `scripts/10_5_transformer_rerank_eval.py`
- `scripts/07_relabel_then_cluster.py`

## Remaining Boundary

This does not promise that every historical experiment script in the repo is clean.

It does mean:

- the current frozen release path is no longer pinned to one local absolute repo root
- future release work can use documented env overrides from
  [docs/contracts/config_reference.md](../contracts/config_reference.md)
