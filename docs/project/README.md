# Project Engineering Pack

[English](./README.md) | [中文](./README.zh-CN.md)

This folder collects engineering documents for the current Yelp ranking
project: architecture notes, data lineage, reproduction paths, validation
reports, and demo entry points.

Use these documents to:

- install the local review environment
- reproduce the current `stage09 -> stage10 -> stage11` line
- run validation and demo commands
- inspect the mock-serving and release surfaces

## Quick Start

If you only need to review the frozen repository line:

1. Install the CPU review dependencies from [../../requirements.txt](../../requirements.txt).
2. Run `python tools/run_release_checks.py`.
3. Run `python tools/demo_recommend.py summary`.
4. Use `python tools/demo_recommend.py list-cases` and
   `python tools/demo_recommend.py show-case --case boundary_11_30` for case
   replay.

If you need the full training and evaluation path, start from
[reproduce_mainline.md](./reproduce_mainline.md).

## Documents

- [environment_setup.md](./environment_setup.md): review-only and full-pipeline
  environment setup.
- [data_lineage_and_storage.md](./data_lineage_and_storage.md): raw data,
  curated parquet layers, stage outputs, and frozen release surfaces.
- [reproduce_mainline.md](./reproduce_mainline.md): canonical reproduction path
  for the current mainline.
- [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md): engineering
  constraints, risk controls, and current boundaries.
- [evaluation_and_casebook.md](./evaluation_and_casebook.md): frozen metrics and
  case-study pointers for interpretation.
- [current_frozen_line.md](./current_frozen_line.md): detailed outward-facing
  scope, data scale, and frozen reference lines.
- [design_choices.md](./design_choices.md): bounded-rerank rationale, bucket
  sequencing, and leakage control.
- [repository_map.md](./repository_map.md): public entry points, checked-in
  result surface, and repository navigation.
- [demo_runbook.md](./demo_runbook.md): demo commands and supported interfaces.
- [demo_reproducibility_matrix.zh-CN.md](./demo_reproducibility_matrix.zh-CN.md):
  minimal demo test cases and interface matrix.
- [acceptance_checklist.md](./acceptance_checklist.md): validation checklist.
- [repo_navigation.md](./repo_navigation.md): which folders are canonical and
  which are historical.

## Canonical Review Commands

```bash
python tools/run_release_checks.py
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
```

For Windows PowerShell reviewers, a thin wrapper is also provided:

```powershell
.\tools\run_release_checks.ps1
```
