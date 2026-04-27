# Demo Runbook

This runbook defines the repository-supported demo paths for the current
`Stage09 -> Stage10 -> Stage11` ranking line. It uses checked-in release
artifacts and lightweight replay fixtures; it does not retrain the full stack.

## Scope

Supported demo surfaces:

- release summary CLI
- canonical case replay
- batch inference CLI
- HTTP mock serving API
- short local load test
- contract-level Stage01-to-Stage11 smoke test

The demo does not represent production online traffic, online A/B testing,
per-request Spark recall recomputation, or online LLM inference.

## Preflight

Run from the repository root:

```bash
python tools/release/run_release_checks.py --skip-pytest
python tools/demo/demo_recommend.py summary
python tools/demo/demo_recommend.py list-cases
```

Windows PowerShell wrapper:

```powershell
.\tools\release\run_release_checks.ps1
```

## Release Summary

```bash
python tools/demo/demo_recommend.py summary
```

This command reads the compact current-release artifacts and prints the active
Stage09 recall, Stage10 rerank, and Stage11 reward-rerank summary.

## Case Replay

```bash
python tools/demo/demo_recommend.py show-case --case boundary_11_30
python tools/demo/demo_recommend.py show-case --case mid_31_40
```

These commands use frozen case notes and do not start training or remote
inference.

## Batch Inference

Legacy fixture:

```bash
python tools/serving/batch_infer_demo.py --strategy reward_rerank
```

Replay request with live lookup/scoring surfaces:

```bash
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u001072 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --include-fallback-demo
```

Stage11 cache miss path:

```bash
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u006562 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --simulate-stage11-cache-miss
```

## HTTP Mock Serving

Self-test:

```bash
python tools/serving/mock_serving_api.py --self-test
```

Start the local API:

```bash
python tools/serving/mock_serving_api.py --host 127.0.0.1 --port 18081
```

Run a short HTTP validation from another terminal:

```bash
python tools/serving/load_test_mock_serving.py --url http://127.0.0.1:18081/rank --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/serving/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

## Contract Smoke

```bash
python tools/demo/run_stage01_11_minidemo.py
```

This command walks a tiny Yelp-like fixture through the Stage01-to-Stage11
contract path. It is a contract smoke test, not a full Spark/GPU reproduction.

## Engineering Load Report

The longer HTTP validation path is documented in:

- [demo_serving_entry.zh-CN.md](./demo_serving_entry.zh-CN.md)
- [demo_reproducibility_matrix.zh-CN.md](./demo_reproducibility_matrix.zh-CN.md)
- [../serving_engineering_load_summary.zh-CN.md](../serving_engineering_load_summary.zh-CN.md)
- [../engineering_http_20260427_30min_c4_broad517_cache1024.md](../engineering_http_20260427_30min_c4_broad517_cache1024.md)
