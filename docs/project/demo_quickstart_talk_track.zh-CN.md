# Demo 快速启动

本文档记录当前仓库的最小 demo 命令。更完整的 mock serving、HTTP 压测和报告入口见 [demo_serving_entry.zh-CN.md](./demo_serving_entry.zh-CN.md)。

## 范围

当前 demo 基于冻结 release artifacts 和 replay fixtures，覆盖：

- Stage09 候选 lookup
- Stage10 XGBoost CPU rerank
- Stage11 cache-first bounded reward-rerank
- batch inference CLI
- HTTP `/rank` mock serving
- fallback 与 cache miss 路径

不覆盖：

- 生产线上服务
- 真实流量 A/B
- 每请求实时 Spark 召回重算
- 在线大模型推理

## 最小检查

```powershell
python tools/run_release_checks.py --skip-pytest
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
```

## Batch Inference

默认 fixture：

```powershell
python tools/batch_infer_demo.py --strategy reward_rerank
```

Replay request：

```powershell
python tools/batch_infer_demo.py --request-id stage11_b5_u001072 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --include-fallback-demo
```

Cache miss fallback：

```powershell
python tools/batch_infer_demo.py --request-id stage11_b5_u006562 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --simulate-stage11-cache-miss
```

## HTTP Mock Serving

Self-test：

```powershell
python tools/mock_serving_api.py --self-test
```

启动服务：

```powershell
python tools/mock_serving_api.py --host 127.0.0.1 --port 18081
```

短压测：

```powershell
python tools/load_test_mock_serving.py --url http://127.0.0.1:18081/rank --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
```

报告导出：

```powershell
python tools/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

## Contract Smoke

```powershell
python tools/run_stage01_11_minidemo.py
```

该命令运行小型内存 fixture，用于验证 Stage01-to-Stage11 的接口 contract，不启动完整 Spark/GPU 训练。

