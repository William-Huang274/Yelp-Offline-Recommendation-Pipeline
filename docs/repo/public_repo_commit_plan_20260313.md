# Public Repo Commit Plan (2026-03-13)

## Goal

This document defines what should and should not be committed for the public
GitHub portfolio version of this project.

The target positioning is:

- public portfolio and research-delivery repository

The target positioning is not:

- full raw-data archive
- full model-weight archive
- full cloud-run mirror

## Current Recommendation

Do not use `git add .` for the first public push.

Use targeted staging only.

Reason:

- the current worktree contains valid project source files mixed with large local
  artifacts, historical recovery debris, and many previously untracked files
- the repository has only a minimal existing git history, so the first public
  commit needs a deliberate boundary

## Commit Now

These paths are appropriate for the first public GitHub version.

### Root Files

- `.gitignore`
- `README.md`
- `LICENSE`
- `requirements.txt`
- `requirements-dev.txt`
- `requirements-stage11-qlora.txt`
- `pytest.ini`

### Config and Source

- `config/`
- `scripts/`
- `tools/`
- `tests/`

### Documentation

- `docs/`

Recommended rationale:

- the docs are now part of the project value, not just internal notes
- they show audit discipline, alignment logic, rollback thinking, and release reasoning

### Small Metrics and Release Control

- `data/metrics/`
- `data/output/_prod_runs/`
- `data/output/_v1_freeze_20260313/v1_freeze_manifest.json`
- `data/output/_first_champion_freeze_20260313/first_champion_manifest.json`

Recommended rationale:

- these files are small
- they make the README claims auditable
- they preserve the frozen release contract without committing large model or parquet assets

## Do Not Commit

These paths should stay local-only.

### Raw Data and Large Local Storage

- `data/Yelp JSON/`
- `data/parquet/`
- `data/cache/`
- `data/spark-tmp/`
- `data/tmp/`
- `data/recovery_reports/`

### Large Generated Outputs

- all stage run directories under `data/output/` except:
  - `_prod_runs/*.json`
  - `_v1_freeze_20260313/v1_freeze_manifest.json`
  - `_first_champion_freeze_20260313/first_champion_manifest.json`

This means do not commit:

- full candidate parquet outputs
- full user-profile vectors
- full item semantic evidence tables
- QLoRA datasets
- checkpoint directories
- adapter weights
- cloud sync mirrors

### Local / Tooling Noise

- `.venv-wsl/`
- `tmp/`
- `_recovered_winfr/`
- `C*Usershht13/`
- `.pytest_cache/`
- `AGENTS.md`

## Public Push Order

Recommended first public push order:

1. repository metadata and README
2. docs and metrics
3. core scripts and pipeline utilities
4. tests and tools
5. optional secondary experiment scripts if you want a fuller archive

This order keeps the first public commit understandable.

## Suggested Staging Blocks

Use staging in blocks like this:

1. root files
2. `docs/`
3. `data/metrics/`
4. `data/output/_prod_runs/*.json`
5. freeze manifests
6. `scripts/`
7. `tools/`
8. `tests/`
9. `config/`

Avoid mixing large cleanup and source history reconstruction in the same commit.

## Suggested Public Commit Shape

Recommended commit structure:

1. `docs: add public-facing README, release notes, and data contract`
2. `chore: add public repo metadata and ignore rules`
3. `feat: add pipeline utilities, validators, and internal pilot runner`
4. `test: add smoke tests for pointers and artifact validation`
5. `feat: add stage10 and stage11 aligned evaluation scripts`

This does not have to match the original development order. It only needs to be
clear to a public reader.

## Recommended Minimal Public Narrative

A strong first public push should support this story:

1. problem: Yelp LA restaurant recommendation
2. retrieval: audited candidate fusion
3. rerank: structured XGB fallback
4. LLM extension: QLoRA / DPO sidecar champion
5. engineering discipline: validation, rollback, monitoring, and release pointers

## Remaining Pre-Push Checks

Before pushing publicly, verify:

1. `git diff --cached --stat` does not contain raw Yelp data or model weights
2. `git diff --cached --name-only` does not contain local scratch paths
3. `python -m pytest tests -q` still passes
4. `python tools/check_release_readiness.py` still completes
5. `README.md` renders correctly on GitHub

## Explicit Decision on Current Repo State

For this repository, the recommended public cut is:

- include most source, docs, tests, and release metadata
- exclude all heavy local data and all full run artifacts

That is enough to make the project credible as a job-search GitHub repository.
