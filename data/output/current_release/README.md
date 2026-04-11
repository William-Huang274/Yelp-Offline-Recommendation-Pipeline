# Current Release Result Surface

This directory is the visible result surface for the current release line.

It keeps only the small files needed to explain the active `stage09 -> stage10 -> stage11`
story:

- `stage09`: current route-aware recall result files
- `stage10`: current mainline structured rerank result files
- `stage11`: current `Qwen3.5-9B` reward-model expert summaries and freeze-baseline evaluation files
- `test_support`: local pipeline test fixture manifest

The original provenance copies remain under:

- `../_prod_runs/`

Use this directory for repository-facing references. Use `_prod_runs` when a
full frozen pack or historical manifest trace is required.
