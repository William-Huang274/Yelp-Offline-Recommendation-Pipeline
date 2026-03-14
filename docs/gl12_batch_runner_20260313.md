# GL-12 One-Click Batch Runner (2026-03-13)

## Scope

This pass adds one internal-pilot operator entry for the current frozen release path.

Added files:

- [scripts/pipeline/internal_pilot_runner.py](../scripts/pipeline/internal_pilot_runner.py)
- [scripts/run_internal_pilot.bat](../scripts/run_internal_pilot.bat)
- [scripts/run_internal_pilot.sh](../scripts/run_internal_pilot.sh)

## Supported Modes

- `validate`
  - runs [tools/check_release_readiness.py](../tools/check_release_readiness.py)
- `recall`
  - runs [scripts/09_1_recall_audit.py](../scripts/09_1_recall_audit.py)
  - resolves `stage09_release` automatically
- `rank`
  - runs [scripts/10_2_rank_infer_eval.py](../scripts/10_2_rank_infer_eval.py)
  - resolves the current `stage09_release` and `stage10_release` automatically
- `eval`
  - runs [scripts/11_3_qlora_sidecar_eval.py](../scripts/11_3_qlora_sidecar_eval.py)
  - resolves the current `stage09_release` and `stage11_release` automatically
- `publish`
  - republishes prod pointers from a frozen champion manifest
  - writes one prod release manifest under `data/output/_prod_runs`

Later extension from `GL-13`:

- `monitor`
- `rollback`

## Commands

Python entry:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode validate
python scripts/pipeline/internal_pilot_runner.py --mode recall
python scripts/pipeline/internal_pilot_runner.py --mode rank
python scripts/pipeline/internal_pilot_runner.py --mode eval
python scripts/pipeline/internal_pilot_runner.py --mode publish --manifest-path data/output/_first_champion_freeze_20260313/first_champion_manifest.json
```

Windows wrapper:

```bat
scripts\run_internal_pilot.bat --mode validate
```

Safe inspection:

```bash
python scripts/pipeline/internal_pilot_runner.py --mode rank --dry-run
python scripts/pipeline/internal_pilot_runner.py --mode eval --dry-run
```

## Verification Run

Executed during this pass:

1. `--mode validate`
2. `--mode publish` using:
   - `data/output/_first_champion_freeze_20260313/first_champion_manifest.json`
3. wrapper verification:
   - `scripts/run_internal_pilot.bat --mode validate`
4. dry-run resolution:
   - `--mode recall --dry-run`
   - `--mode rank --dry-run`
   - `--mode eval --dry-run`

## Output

Publish now writes:

- [release_manifest_internal_pilot_v1_champion_20260313.json](../data/output/_prod_runs/release_manifest_internal_pilot_v1_champion_20260313.json)

## Operational Meaning

`GL-12` is closed for internal-pilot operation.

This does not mean the project is production-ready.

The current release-readiness report still stays at `WARN`, which is expected until:

- rollback and monitoring are defined
- the `stage11` gate warning is resolved or explicitly accepted
