# Project Engineering Pack

[English](./README.md) | [中文](./README.zh-CN.md)

This folder collects the engineering-facing documents needed to review, demo,
and reproduce the current Yelp big data project in a way that matches the
practice-module briefing.

Use this pack when you need to:

- explain how the repository maps to the course requirements
- prepare a proposal, demo, or final report
- install the local review environment
- reproduce the current `stage09 -> stage10 -> stage11` line
- run reviewer-friendly validation and demo commands

## Quick Start

If you only need to review the frozen repository line:

1. Install the CPU review dependencies from [../../requirements.txt](../../requirements.txt).
2. Run `python tools/run_release_checks.py`.
3. Run `python tools/demo_recommend.py summary`.
4. Use `python tools/demo_recommend.py list-cases` and
   `python tools/demo_recommend.py show-case --case boundary_11_30` during the
   live demo.

If you need the full training and evaluation path, start from
[reproduce_mainline.md](./reproduce_mainline.md).

## Documents

- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md):
  maps the briefing requirements to repository evidence and calls out
  non-repo deliverables that still need manual submission.
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
- [demo_runbook.md](./demo_runbook.md): a teacher-facing 20-minute demo flow.
- [acceptance_checklist.md](./acceptance_checklist.md): reviewer acceptance and
  one-click validation checklist.
- [repo_navigation.md](./repo_navigation.md): which folders are canonical and
  which are historical.
- [proposal_template_content.md](./proposal_template_content.md): proposal
  structure aligned to the course briefing.
- [final_report_outline.md](./final_report_outline.md): final report structure
  aligned to the current repository assets.

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
