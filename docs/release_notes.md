# Release Notes

## release_closeout_20260409

This is the current public release surface for the Yelp offline recommendation and reranking stack.

## 2026-04-26 Serving Demo Update

- Added replay-first serving demo commands based on `request_id` rather than handwritten candidate JSON.
- Added mixed-traffic load testing with cache-miss, fallback, and per-stage latency reporting.
- Added `export_serving_validation_report.py` for turning load-test JSON into a Markdown validation report.
- Kept the old manual-candidate payload as a legacy contract smoke path.

## Scope

- Stage09: bucket5 route-aware source-parity recall surface
- Stage10: structured XGBoost-style rerank summaries across bucket2 / bucket5 / bucket10
- Stage11: frozen Qwen3.5-9B reward-model bounded rerank line
- Serving surface: mock batch inference, mock HTTP API, local load test, and release pointers

This release does not include Stage12 experiments or any A3B model line.

## Release Artifacts

| artifact | purpose |
| --- | --- |
| `config/serving.yaml` | service strategy, model version, fallback order, and latency budget |
| `data/output/current_release/manifest.json` | compact outward-facing release manifest |
| `data/output/_prod_runs/release_policy.json` | champion / fallback / baseline policy |
| `data/output/_prod_runs/stage09_release.json` | emergency recall-routing baseline pointer |
| `data/output/_prod_runs/stage10_release.json` | aligned structured-rerank fallback pointer |
| `data/output/_prod_runs/stage11_release.json` | current reward-rerank champion pointer |

## Serving Checks

```bash
python tools/serving/batch_infer_demo.py --input config/demo/replay_request_input.json --format json
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u000097 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u000097 --strategy reward_rerank --simulate-stage11-cache-miss
python tools/serving/mock_serving_api.py --self-test
python tools/serving/load_test_mock_serving.py --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/serving/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

Expected serving-level signals:

- strategy requested / strategy used
- fallback count
- latency in milliseconds
- success rate from the load-test summary

## Rollback Model

The public fallback ladder is:

1. `stage11_release`: Qwen3.5-9B reward-model bounded rerank champion
2. `stage10_release`: structured XGBoost-style fallback
3. `stage09_release`: emergency recall-routing baseline

Rollback is represented as a pointer change rather than a code rewrite.

## Validation

```bash
python tools/release/run_full_chain_smoke.py
python tools/release/run_release_checks.py --skip-pytest
pytest tests/test_demo_tools.py tests/test_public_release_surface.py
```
