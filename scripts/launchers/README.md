# Launcher Interface

[English](./README.md) | [中文](./README.zh-CN.md)

This directory is the stable outward-facing launcher surface for the current
repository line.

These `.sh` launchers target Linux / cloud / WSL-style environments. On a local
Windows review machine, use the PowerShell and Python helpers under
[../../tools](../../tools), especially `run_stage09_local.ps1`,
`run_stage10_bucket5_local.ps1`, and `run_release_checks.ps1`.

The long-form shell launchers now live under:

- [../runtime_sh/](../runtime_sh)

These wrappers are the entry points that should appear in docs, demos, and
release notes.

## Naming

- `stage09_*`: Stage09 mainline and Stage11 prerequisite assets
- `stage10_*`: Stage10 train / infer mainline
- `stage11_*`: Stage11 dataset, export, train, eval, and audit entry points

## Canonical Wrappers

### Stage09

- `stage09_bucket5_mainline.sh`
- `stage09_bucket5_typed_intent_assets.sh`
- `stage09_bucket5_stage11_assets.sh`

### Stage10

- `stage10_bucket5_mainline.sh`
- `stage10_bucket2_mainline.sh`
- `stage10_bucket10_mainline.sh`

### Stage11

- `stage11_bucket5_11_1.sh`
- `stage11_bucket5_export_only.sh`
- `stage11_bucket5_train.sh`
- `stage11_bucket5_eval.sh`
- `stage11_bucket5_watch.sh`
- `stage11_bucket5_constructability_audit.sh`
- `stage11_bucket5_pool_audit.sh`

## Variable Naming Conventions

Three names are intentionally separated and should not be used interchangeably:

| name | meaning |
| --- | --- |
| `TOP_K` | final metric cutoff, for example Recall@10 or NDCG@10 |
| `RERANK_TOPN` | candidate window rescored by the model before final ranking |
| `PAIRWISE_POOL_TOPN` | learned-rank cutoff used only to mine Stage11 pairwise pool candidates |

Full definitions and stage-by-stage mappings are documented here:

- [../../docs/contracts/launcher_env_conventions.md](../../docs/contracts/launcher_env_conventions.md)

## Shared Path Contract

Active compatibility launchers share one remote path contract:

- `scripts/launchers/_path_contract.sh`

This file centralizes remote repo, output, model-cache, temp, and Python fallback
paths so that wrapper defaults stay aligned without duplicating absolute machine
paths across every launcher.

## Wrapper Rule

Wrappers should stay thin:

- forward to the active runtime shell launcher
- preserve environment variables unchanged
- avoid business logic
