# Serving And Release Surface

This repository does not expose a public online service, but it does expose a
real release-management surface for batch-style ranking and review workflows.

## What Counts As A Release Artifact

| surface | role |
| --- | --- |
| `data/output/current_release` | compact outward-facing result surface used in README and demo flows |
| `data/metrics/current_release` | compact metric snapshot for the active outward-facing line |
| `config/serving.yaml` | mock serving strategy, version, fallback order, and latency budget |
| `data/output/showcase_history` | selected historical run metadata kept for comparisons |
| `data/metrics/showcase_history` | selected historical metrics used for controlled comparisons |
| `data/output/_prod_runs` | release manifests, active pointers, and rollback snapshots |

The repository is intentionally built around compact release files rather than
full prediction dumps or full online-service assets.

## `_prod_runs` File Roles

Current examples already checked into the repository:

| file pattern | role |
| --- | --- |
| `release_manifest_*.json` | promoted release manifest with the stage-level run references that define the published line |
| `release_policy.json` | current policy block describing the champion, aligned fallback, and emergency baseline |
| `stage09_release.json` | pointer to the emergency baseline release run |
| `stage10_release.json` | pointer to the aligned structured-rerank fallback |
| `stage11_release.json` | pointer to the current champion rescue line |
| `rollback_snapshot_*.json` | snapshot of active pointers before a publish |
| `rollback_applied_*.json` | audit record showing a rollback was actually applied |

## Version, Checkpoint, And Config Organization

The current repository line uses a layered versioning model:

- each stage run writes a `run_meta.json` file
- compact outward-facing summaries point back to those run-level metadata files
- Stage11 training and evaluation runs keep their own adapter / checkpoint
  directories plus run metadata
- launcher wrappers and runtime shells keep environment configuration separate
  from the main Python logic

That means the project is not organized as one giant notebook or one
hand-edited manifest. The release surface can point to a specific stage run,
configuration, and metric snapshot.

## Batch Inference Path

There are two batch-style ways to interact with the current line.

### Replay-First Review Path

Use replay request ids from the frozen Stage11 pack. This is the recommended
path for README, interview, and teacher-facing review because it preserves the
`Stage09 -> Stage10 -> Stage11` request shape.

```bash
python tools/run_release_checks.py --skip-pytest
python tools/batch_infer_demo.py --input config/demo/replay_request_input.json --format json
python tools/batch_infer_demo.py --request-id stage11_b5_u000097 --strategy reward_rerank --debug --include-fallback-demo
python tools/batch_infer_demo.py --request-id stage11_b5_u000097 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay
python tools/batch_infer_demo.py --request-id stage11_b5_u000097 --strategy reward_rerank --simulate-stage11-cache-miss
python tools/mock_serving_api.py --self-test
python tools/load_test_mock_serving.py --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
python tools/demo_recommend.py
python tools/demo_recommend.py show-case --case boundary_11_30
```

The generated report is checked into:
[serving_validation_report.md](./serving_validation_report.md)

### Legacy Manual-Candidate Path

The older handwritten-candidate payload remains available for a tiny contract
smoke, but it is not the recommended serving story:

```bash
python tools/batch_infer_demo.py --input config/demo/batch_infer_demo_input.json --strategy reward_rerank
```

### Stage-Level Batch Path

Use the launcher or local wrapper surface:

- Stage09 local wrapper:
  [../tools/run_stage09_local.ps1](../tools/run_stage09_local.ps1)
- Stage10 local wrapper:
  [../tools/run_stage10_bucket5_local.ps1](../tools/run_stage10_bucket5_local.ps1)
- Stage11 cloud inventory / explicit pull helper:
  [../tools/cloud_stage11.py](../tools/cloud_stage11.py)

### Mock Serving Path

For an interview-friendly serving contract, the repository exposes lightweight
tools on top of the checked-in frozen line:

- batch inference demo:
  [../tools/batch_infer_demo.py](../tools/batch_infer_demo.py)
- HTTP mock serving surface:
  [../tools/mock_serving_api.py](../tools/mock_serving_api.py)
- local load test:
  [../tools/load_test_mock_serving.py](../tools/load_test_mock_serving.py)
- validation-report exporter:
  [../tools/export_serving_validation_report.py](../tools/export_serving_validation_report.py)

Example:

```bash
python tools/batch_infer_demo.py --request-id stage11_b5_u000097 --format json
python tools/mock_serving_api.py --host 127.0.0.1 --port 8000
```

The API surface is intentionally small:

- `GET /health`: returns the active release id and compact serving status
- `POST /rank`: accepts a replay request id or legacy mock user-profile
  request, plus an optional strategy, then returns Stage09 -> Stage10 ->
  Stage11 ranked output

Supported strategies:

- `baseline`: lightweight candidate baseline
- `xgboost`: Stage10 structured-rerank path
- `reward_rerank`: Stage11 bounded reward-model rescue path

The response reports:

- `strategy_requested`
- `strategy_used`
- `fallback_used`
- `fallback_reason`
- `serving_metrics.latency_ms`
- `serving_metrics.fallback_count`

## Portability And Endpoint Configuration

The public repository does not include raw Yelp data, full prediction dumps, or
large cloud model weights. The serving demo therefore has two layers:

- real local replay when the frozen `data/output/_prod_runs/...` or
  `data/output/cloud_stage11/...` artifacts are available
- embedded sample replay when those large artifacts are absent

The fallback sample is intentionally contract-level. It keeps request fields,
top-k shape, fallback counters, cache-miss behavior, and report generation
stable, but it is not a substitute for the frozen metric tables.

Endpoint and policy knobs are centralized in [../config/serving.yaml](../config/serving.yaml).
For cloud Stage11 verification, override these environment variables when the
temporary machine changes:

```bash
BDA_CLOUD_HOST=connect.westb.seetacloud.com
BDA_CLOUD_PORT=20804
BDA_CLOUD_USER=root
```

For the internal release-runner surface, see
[../scripts/pipeline/internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py).

## Fallback Logic

The current release model is intentionally layered:

1. `stage11_release` is the champion path.
2. `stage10_release` is the aligned fallback path.
3. `stage09_release` is the emergency baseline source.

This is not just conceptual. The active release policy already stores those
roles explicitly.

Practical meaning:

- if the rescue layer is unreadable or unstable, fall back to Stage10
- if the structured rerank artifacts cannot be trusted, fall back to Stage09
- keep rollback as a pointer change rather than a code rewrite

## Rollback And Monitoring

Rollback and monitoring are already present as an internal-pilot surface.

Primary commands:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode monitor
python scripts/pipeline/internal_pilot_runner.py --mode publish --manifest-path data/output/_first_champion_freeze_20260313/first_champion_manifest.json
python scripts/pipeline/internal_pilot_runner.py --mode rollback
```

Current monitoring checks include:

- Stage09 recall quality
- candidate row count and evaluated users
- Stage11 vs Stage10 vs PreScore ordering
- training artifact readability
- gate-state consistency

Detailed rules live in:

- [archive/release/rollback_and_monitoring.md](./archive/release/rollback_and_monitoring.md)

## Why This Matters

From an engineering perspective, this means the repository is not only an
offline scoring project. It already contains:

- versioned release pointers
- champion / fallback / baseline roles
- rollback snapshots
- monitor signals
- compact release artifacts that can be reviewed without reproducing the full
  cloud stack

That is the practical bridge between a research-style ranking stack and a
reviewable, auditable batch-release surface.
