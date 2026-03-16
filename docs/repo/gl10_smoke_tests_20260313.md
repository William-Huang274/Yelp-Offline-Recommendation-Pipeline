# GL-10 Minimal Smoke Tests (2026-03-13)

## Scope

This pass adds the first fast test suite for the active release path.

Added files:

- [tests/conftest.py](../../tests/conftest.py)
- [tests/test_project_paths.py](../../tests/test_project_paths.py)
- [tests/test_run_pointers.py](../../tests/test_run_pointers.py)
- [tests/test_run_validators.py](../../tests/test_run_validators.py)
- [scripts/pipeline/run_validators.py](../../scripts/pipeline/run_validators.py)
- [tools/validate_stage_artifact.py](../../tools/validate_stage_artifact.py)
- [pytest.ini](../../pytest.ini)
- [requirements-dev.txt](../../requirements-dev.txt)

## What Is Covered

1. `project_paths` root override and legacy-path normalization
2. latest/prod pointer read-write-resolve roundtrip
3. required `run_meta` field checking
4. stage09 candidate-run directory shape validation
5. stage11 dataset-run directory shape validation
6. optional smoke validation against the current local frozen runs when those artifacts exist

## Commands Run

Install test dependency:

```bash
python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest tests -q
```

The validator smoke checks for frozen local artifacts are now conditional. If the referenced local stage09/stage11 runs are absent, the test suite skips those two checks instead of hard-failing on path availability.

Validate current frozen local runs without running training:

```bash
python tools/validate_stage_artifact.py --kind stage09_candidate --run-dir data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion
python tools/validate_stage_artifact.py --kind stage11_dataset --run-dir data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset
```

## Result

- `pytest`: `9 passed`
- fixture-backed unit and validator coverage now runs without any hardcoded local repo path
- frozen local artifact smoke checks still pass when those run directories exist, and skip cleanly when they do not

## Operational Meaning

`GL-10` is now closed for the active release path.

The repo still does not have a release-readiness aggregator. That remains the purpose of
`GL-11`.
