## Tools

This folder contains lightweight repository-side utilities aligned with the current public mainline.

- `validate_public_surface.py`: validates the public README, docs, launchers, and curated result surface
- `validate_current_release.py`: validates the current curated release summaries under `data/output/current_release`, including the three-expert Stage11 release line
- `run_release_checks.py`: reviewer-facing one-click check that verifies the project docs, validators, demo CLI, and core pytest surface
- `run_release_checks.ps1`: thin PowerShell wrapper for `run_release_checks.py`
- `run_full_chain_smoke.py`: safe `stage01 -> stage11` smoke runner covering canonical help paths, local wrappers, mini demo, serving strategies, and Stage11 checks
- `run_stage01_11_minidemo.py`: contract-level minimal sample that walks one tiny Yelp-like fixture through Stage01 -> Stage11 without Spark/GPU training
- `run_stage11_model_prompt_smoke.py`: verifies the current Stage11 Qwen3.5-9B reward-model surface and smoke-case config
- `demo_recommend.py`: frozen-release demo helper for Stage09/10/11 summaries and canonical case walkthroughs
- `batch_infer_demo.py`: mock batch inference entry that reads a small JSON request, supports `baseline` / `xgboost` / `reward_rerank`, and returns top-k plus serving metrics
- `mock_serving_api.py`: stdlib HTTP demo exposing `/health` and `/rank` on top of the mock ranking pipeline
- `load_test_mock_serving.py`: local in-process or HTTP load test reporting multi-request traffic mix, p50/p95/p99 latency, per-stage latency, success rate, cache miss, and fallback count
- `export_serving_validation_report.py`: converts a load-test JSON summary into a Markdown serving validation report
- demo request payload: `config/demo/batch_infer_demo_input.json`
- full-chain mini fixture: `config/demo/full_chain_minimal_input.json`
- serving config: `config/serving.yaml`
- `cloud_stage11.py`: cloud inventory and explicit pull helper for Stage11 artifacts and missing Stage10 source-parity prerequisites
- `run_stage09_local.ps1`: Windows PowerShell wrapper for local Stage09 debugging with conservative Spark defaults
- `run_stage10_bucket2_local.ps1`: Windows PowerShell wrapper for local Stage10 bucket2 / cold-start debugging with conservative Spark defaults
- `run_stage10_bucket5_local.ps1`: Windows PowerShell wrapper for local Stage10 bucket5 debugging with conservative Spark defaults
- `validate_stage_artifact.py`: validates local Stage09/Stage10/Stage11 run directories using repository validators
- `analyze_stage11_prompt_length.py`: reports prompt-length distribution for a Stage11 dataset run
- `monitor_stage11_training.py`: tails a Stage11 training or evaluation log and extracts recent metric lines
- `monitor_stage11_training.bat` / `monitor_stage11_training.sh`: thin wrappers for the monitor
- `check_local_resources.py`: local GPU/RAM/disk inspection before heavy Stage11 work

Legacy tools from the previous DPO-oriented surface were moved to `tools/archive/release_closeout_20260410/`.
