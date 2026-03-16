# GL-08 Production Pointer Separation (2026-03-13)

## Purpose

This note records the first separation between experiment pointers and release pointers.

Experimental pointers remain under:

- `data/output/_latest_runs`

Release pointers now live under:

- `data/output/_prod_runs`

## Files Added

- [release_policy.json](../../data/output/_prod_runs/release_policy.json)
- [stage09_release.json](../../data/output/_prod_runs/stage09_release.json)
- [stage10_release.json](../../data/output/_prod_runs/stage10_release.json)
- [stage11_release.json](../../data/output/_prod_runs/stage11_release.json)

## Current Meaning

- `stage11_release.json`
  - current frozen champion
- `stage10_release.json`
  - aligned fallback path
- `stage09_release.json`
  - emergency baseline source
- `release_policy.json`
  - tells readers which pointer currently acts as champion, fallback, and baseline

## Helper Support

The path helper now supports a second pointer namespace:

- `production_run_pointer_path`
- `read_production_run_pointer`
- `resolve_production_run_pointer`
- `write_production_run_pointer`

These live in:

- [project_paths.py](../../scripts/pipeline/project_paths.py)

## Operational Rule

From this point forward:

1. `_latest_runs` is for experiment convenience only
2. `_prod_runs` is for approved release lineage only
3. smoke runs must never overwrite `_prod_runs`
4. future release notes should reference `_prod_runs` first

## Remaining Work

This does not yet update all scripts to consume prod pointers by default.

That is intentional.

For now, the repo has:

- explicit release pointers
- explicit experiment pointers
- a clear policy split

The next repair items are:

- `GL-09` path cleanup
- `GL-10` smoke tests
- `GL-11` release validation
