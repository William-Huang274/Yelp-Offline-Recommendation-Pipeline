## Tools

Repository utilities are grouped by execution surface. Paths below are the canonical entrypoints used by the docs and smoke checks.

### `tools/demo/`

- `demo_recommend.py`: frozen Stage09/10/11 metric summary and case replay.
- `run_stage01_11_minidemo.py`: in-memory Stage01-to-Stage11 contract smoke fixture.

### `tools/serving/`

- `batch_infer_demo.py`: single-request ranking path for `baseline`, `xgboost`, and `reward_rerank`.
- `mock_serving_api.py`: stdlib HTTP service exposing `/health` and `/rank`.
- `load_test_mock_serving.py`: short in-process or HTTP load test.
- `run_serving_engineering_load.py`: duration-based replay load runner with JSON/Markdown report export.
- `export_serving_validation_report.py`: converts load-test summaries into Markdown reports.
- `replay_store.py`, `stage10_live_local.py`, `stage11_live_remote.py`, `stage11_remote_worker.py`: serving support modules.

### `tools/release/`

- `run_release_checks.py`: release/public-surface validation entrypoint.
- `run_release_checks.ps1`: PowerShell wrapper for `run_release_checks.py`.
- `run_full_chain_smoke.py`: Stage01-to-Stage11 smoke runner.
- `validate_public_surface.py`, `validate_current_release.py`, `validate_stage_artifact.py`: validation helpers.
- `check_local_resources.py`: local CPU/GPU/RAM/disk inspection.

### `tools/stage/`

- `cloud_stage11.py`: cloud artifact inventory and pull helper.
- `run_stage09_local.ps1`, `run_stage10_bucket2_local.ps1`, `run_stage10_bucket5_local.ps1`: local Stage09/Stage10 wrappers.
- `run_stage11_model_prompt_smoke.py`: Stage11 model/prompt surface smoke check.
- `analyze_stage11_prompt_length.py`: Stage11 prompt-length diagnostics.
- `monitor_stage11_training.py` plus `.bat` / `.sh`: training log monitor.

### `tools/archive/`

Historical utilities from older release surfaces. They are retained for traceability and are not part of the current release command surface.