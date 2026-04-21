# Repository Map

[English](./repository_map.md) | [中文](./repository_map.zh-CN.md)

This page is the deep-link map for the public repository surface.

## Main Code Paths

| path | role |
| --- | --- |
| `scripts/launchers` | outward-facing launcher entry points for stage-level runs |
| `tools` | validation, demo, cloud inventory, and mock serving helpers |
| `tests` | smoke tests for public links, release files, launchers, and metrics |
| `docs/contracts` | launcher variable and environment conventions |
| `docs/stage11` | public Stage11 design notes and case studies |
| `docs/project` | reproduction, frozen-line notes, design choices, and review pack |

## Checked-In Result Surface

| path | role |
| --- | --- |
| `data/output/current_release` | current outward-facing release outputs |
| `data/metrics/current_release` | compact metric snapshot for the active line |
| `data/output/showcase_history` | selected historical outputs kept for controlled comparison |
| `data/metrics/showcase_history` | selected historical metrics used in evaluation stories |
| `data/output/_prod_runs` | release manifests, active pointers, and rollback snapshots |

## Recommended Entry Points

- Stage09 launcher:
  [../../scripts/launchers/stage09_bucket5_mainline.sh](../../scripts/launchers/stage09_bucket5_mainline.sh)
- Stage10 launcher:
  [../../scripts/launchers/stage10_bucket5_mainline.sh](../../scripts/launchers/stage10_bucket5_mainline.sh)
- Stage11 dataset / export / train / eval:
  [../../scripts/launchers/stage11_bucket5_11_1.sh](../../scripts/launchers/stage11_bucket5_11_1.sh),
  [../../scripts/launchers/stage11_bucket5_export_only.sh](../../scripts/launchers/stage11_bucket5_export_only.sh),
  [../../scripts/launchers/stage11_bucket5_train.sh](../../scripts/launchers/stage11_bucket5_train.sh),
  [../../scripts/launchers/stage11_bucket5_eval.sh](../../scripts/launchers/stage11_bucket5_eval.sh)
- Public validation:
  [../../tools/run_release_checks.py](../../tools/run_release_checks.py)
- Demo helpers:
  [../../tools/demo_recommend.py](../../tools/demo_recommend.py),
  [../../tools/batch_infer_demo.py](../../tools/batch_infer_demo.py),
  [../../tools/mock_serving_api.py](../../tools/mock_serving_api.py)

## Notes On Historical Material

- `scripts/stage01_to_stage08` is kept as earlier reproducible project history.
- archive folders under `docs/` and `tools/` remain useful for audit trails,
  but they are not the main public entry points.
- large raw data, large cloud logs, and model weights stay outside the public
  repository surface.
