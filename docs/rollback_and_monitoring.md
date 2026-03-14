# Rollback And Monitoring

## Purpose

This is the operational rule set for the internal-pilot release path.

It covers:

- how prod pointers are snapshotted before publish
- how to rollback prod pointers safely
- what signals must be monitored after publish

## Current Entry Point

Use the internal pilot runner:

- [internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py)
- [run_internal_pilot.bat](../scripts/run_internal_pilot.bat)

## Rollback Contract

### Rule

Never edit `_prod_runs/*.json` by hand during rollback.

Always rollback from a stored snapshot using the runner.

### Snapshot Storage

Before every `publish`, the runner writes one snapshot under:

- `data/output/_prod_runs/rollback_snapshot_<timestamp>_<release>.json`

After every applied rollback, the runner writes one audit record under:

- `data/output/_prod_runs/rollback_applied_<timestamp>_<release>.json`

### Commands

Publish with automatic snapshot:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode publish --manifest-path data/output/_first_champion_freeze_20260313/first_champion_manifest.json
```

Rollback to the latest snapshot:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode rollback
```

Rollback to one explicit snapshot:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode rollback --snapshot-path data/output/_prod_runs/rollback_snapshot_20260313_203034_internal_pilot_v1_champion_20260313.json
```

Dry-run rollback:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode rollback --snapshot-path data/output/_prod_runs/rollback_snapshot_20260313_203034_internal_pilot_v1_champion_20260313.json --dry-run
```

### Rollback Decision Rule

Rollback if any of these becomes true on the active release:

- champion recall ordering no longer holds
- candidate rows or evaluated users drift unexpectedly
- required prod pointer targets disappear
- champion evaluation artifacts become unreadable
- release monitor status becomes `FAIL`

## Monitoring Contract

### Monitor Command

```bash
python scripts/pipeline/internal_pilot_runner.py --mode monitor
```

This runs:

- [check_release_monitoring.py](../tools/check_release_monitoring.py)

and writes:

- [release_monitor_report_internal_pilot_v1_champion_20260313.md](./release_monitor_report_internal_pilot_v1_champion_20260313.md)

### Signals

The monitor currently checks:

1. `stage09 truth_in_pretrim`
2. `stage09 hard_miss`
3. users evaluated
4. candidate row count
5. champion metric drift against fallback and `PreScore`
6. stage10 training artifact readability
7. stage11 training artifact readability
8. stage11 gate flag state

### Current Thresholds

- `truth_in_pretrim >= 0.82`
- `hard_miss <= 0.10`
- `users_evaluated == release_contract.eval_users`
- `candidate_rows == eval_users * candidate_topn`
- preferred ranking order:
  - `stage11 > stage10 > PreScore`

## Current Files

Snapshot and rollback artifacts created during `GL-13`:

- [rollback_snapshot_20260313_203034_internal_pilot_v1_champion_20260313.json](../data/output/_prod_runs/rollback_snapshot_20260313_203034_internal_pilot_v1_champion_20260313.json)
- [rollback_snapshot_20260313_203053_internal_pilot_v1_champion_20260313.json](../data/output/_prod_runs/rollback_snapshot_20260313_203053_internal_pilot_v1_champion_20260313.json)
- [rollback_applied_20260313_203053_internal_pilot_v1_champion_20260313.json](../data/output/_prod_runs/rollback_applied_20260313_203053_internal_pilot_v1_champion_20260313.json)

## Current Status

The monitoring path exists and is runnable.

The current monitor report is still `WARN`, not `PASS`, because the frozen `stage11`
model lineage still records `enforce_stage09_gate=false`.
