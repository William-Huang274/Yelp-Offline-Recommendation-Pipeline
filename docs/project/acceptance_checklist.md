# Acceptance Checklist

## Purpose

This checklist is the reviewer-facing acceptance surface for the current
repository line.

## 1. Documentation Checks

Confirm the following documents exist and are readable:

- [../../README.md](../../README.md)
- [../../docs/README.md](../../docs/README.md)
- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)
- [environment_setup.md](./environment_setup.md)
- [data_lineage_and_storage.md](./data_lineage_and_storage.md)
- [reproduce_mainline.md](./reproduce_mainline.md)
- [challenges_and_tradeoffs.md](./challenges_and_tradeoffs.md)
- [evaluation_and_casebook.md](./evaluation_and_casebook.md)
- [demo_runbook.md](./demo_runbook.md)
- [proposal_template_content.md](./proposal_template_content.md)
- [final_report_outline.md](./final_report_outline.md)

## 2. Current Release Surface Checks

Confirm the compact release surfaces exist:

- `data/output/current_release`
- `data/metrics/current_release`
- `data/output/showcase_history`
- `data/metrics/showcase_history`

## 3. One-Click Validation

Primary command:

```bash
python tools/release/run_release_checks.py
```

Expected result:

- `PASS public_surface`
- `PASS current_release`
- demo CLI smoke step passes
- batch inference demo smoke step passes
- mock serving self-test passes
- core repository tests pass

Windows PowerShell wrapper:

```powershell
.\tools\release\run_release_checks.ps1
```

## 4. Minimal Demo Checks

Run:

```bash
python tools/demo/demo_recommend.py summary
python tools/demo/demo_recommend.py list-cases
python tools/demo/demo_recommend.py show-case --case boundary_11_30
python tools/serving/batch_infer_demo.py
python tools/serving/mock_serving_api.py --self-test
```

Expected reviewer-visible outcomes:

- current Stage09 / Stage10 / Stage11 metrics are printed
- canonical case studies are available
- a mock Stage09 -> Stage10 -> Stage11 batch inference request returns top-k
- a reviewer-facing `/health` and `/rank` contract is available without extra service dependencies
- the command works without a GPU

## 5. Teacher Requirement Coverage

The repository should clearly support discussion of:

- the business problem
- ingestion
- storage
- processing
- analytics solution
- interpretation
- challenges encountered

Primary mapping document:

- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)

## 6. Non-Repo Deliverables Still Needed

These are not satisfied by the shared code repository alone and still need
manual group submission:

- final slide deck
- final report prose
- named team-member effort estimate table
- individual reflection report
- peer review

## 7. Review Exit Condition

The repository is ready for teacher-facing engineering review when:

1. the validation command passes
2. the demo commands run from frozen assets
3. the proposal and report templates have been adapted with the team's final
   names and scope wording
