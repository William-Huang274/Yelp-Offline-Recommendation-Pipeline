# Release Notes

## release_closeout_20260409

This is the current public release surface for the Yelp offline recommendation and reranking stack.

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
python tools/batch_infer_demo.py --strategy baseline
python tools/batch_infer_demo.py --strategy xgboost
python tools/batch_infer_demo.py --strategy reward_rerank
python tools/mock_serving_api.py --self-test
python tools/load_test_mock_serving.py --requests 20 --concurrency 4 --simulate-fallback-every 5
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
python tools/run_full_chain_smoke.py
python tools/run_release_checks.py --skip-pytest
pytest tests/test_demo_tools.py tests/test_public_release_surface.py
```
