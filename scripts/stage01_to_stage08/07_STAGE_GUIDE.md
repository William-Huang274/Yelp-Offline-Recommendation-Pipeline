# Stage 07 Script Guide

## Active entry points (use these)

- `scripts/07_relabel_only.py`
  - Run relabel only (rule + optional embedding recall + optional LLM).
- `scripts/07_cluster_only.py`
  - Run clustering from an existing `biz_relabels.csv`.
- `scripts/07_relabel_embed_only.py`
  - Debug embedding-recall behavior only.
- `scripts/07_relabel_then_cluster.py`
  - Core stage-07 implementation module (also runnable end-to-end).
- `scripts/stage07_core.py`
  - Shared wrappers for relabel-only / cluster-only flow.

## Legacy scripts (moved)

- `scripts/legacy_stage07/07_autolabel_baseline_ollama.py`
- `scripts/legacy_stage07/07_baseline_relabel_vector.py`
- `scripts/legacy_stage07/07_embedding_cluster.py`

These are not part of the active stage-07 run path.
