# Demo 复现矩阵与最小用例

本文档列出当前仓库已准备的 demo 测试入口、最小输入和接口覆盖范围。

## 最小输入

| 文件 | 用途 |
| --- | --- |
| `config/demo/full_chain_minimal_input.json` | Stage01-to-Stage11 contract smoke fixture |
| `config/demo/batch_infer_demo_input.json` | legacy batch inference fixture，包含用户画像和候选列表 |
| `config/demo/online_rank_minimal_input.json` | HTTP `/rank` 最小 replay 请求，使用 `lookup_live -> xgb_live -> replay` |
| `config/demo/online_rank_cache_miss_input.json` | HTTP `/rank` cache miss 请求，验证 Stage11 fallback 到 Stage10 |
| `config/demo/replay_request_input.json` | replay/debug 请求模板 |

## 接口矩阵

| 接口 | 命令 | 覆盖内容 | 本地验证 |
| --- | --- | --- | --- |
| Release check | `python tools/run_release_checks.py --skip-pytest` | public surface、current release、smoke CLI | PASS |
| Contract smoke | `python tools/run_stage01_11_minidemo.py` | Stage01-to-Stage11 最小内存链路 | PASS |
| Release summary | `python tools/demo_recommend.py summary` | Stage09/10/11 冻结摘要 | PASS |
| Case replay | `python tools/demo_recommend.py show-case --case boundary_11_30` | Stage11 case replay | PASS |
| Batch inference | `python tools/batch_infer_demo.py --request-id stage11_b5_u001072 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay` | request-level 推理路径 | PASS |
| Cache miss inference | `python tools/batch_infer_demo.py --request-id stage11_b5_u006562 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --simulate-stage11-cache-miss` | Stage11 cache miss 与 Stage10 fallback | PASS |
| HTTP self-test | `python tools/mock_serving_api.py --self-test` | `/health` 和内置 `/rank` 样例 | PASS |
| HTTP direct POST | `Invoke-RestMethod ... -Body (Get-Content config/demo/online_rank_minimal_input.json -Raw)` | 直接调用 `/rank` 最小请求 | PASS |
| HTTP online test | `python tools/load_test_mock_serving.py --url http://127.0.0.1:18081/rank ...` | 本地 HTTP `/rank`、mixed traffic、fallback | PASS |
| Engineering load runner | `python tools/run_serving_engineering_load.py --duration-s 5 ...` | 持续压测脚本、JSON/Markdown 导出路径 | PASS |

本地验证环境：Windows PowerShell，仓库根目录 `D:\5006_BDA_project`。上述 PASS 代表命令在当前本地环境完成；长稳压测指标以 [30 分钟 HTTP 压测报告](../engineering_http_20260427_30min_c4_broad517_cache1024.md) 为准。

## HTTP API 最小测试

启动服务：

```powershell
$env:BDA_STAGE10_LIVE_USER_CACHE_SIZE="1024"
python tools/mock_serving_api.py --host 127.0.0.1 --port 18081
```

PowerShell POST：

```powershell
$payload = Get-Content config/demo/online_rank_minimal_input.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18081/rank -ContentType "application/json" -Body $payload
```

Linux/macOS shell：

```bash
curl -sS http://127.0.0.1:18081/rank \
  -H 'Content-Type: application/json' \
  --data @config/demo/online_rank_minimal_input.json
```

Cache miss fallback：

```powershell
$payload = Get-Content config/demo/online_rank_cache_miss_input.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18081/rank -ContentType "application/json" -Body $payload
```

预期关键字段：

- `strategy_used=reward_rerank`
- `execution_modes.stage09_used=lookup_live`
- `execution_modes.stage10_used=xgb_live`
- `execution_modes.stage11_used=cache_first` 或 `stage10_fallback`
- `serving_metrics.success=true`

本地直接 POST 验证结果：

- minimal payload: `success=True`, `stage11_used=cache_first`
- cache miss payload: `success=True`, `stage11_used=stage10_fallback`, `cache_status=miss_simulated`

## 短压测

```powershell
python tools/load_test_mock_serving.py --url http://127.0.0.1:18081/rank --request-sample-size 3 --warmup-requests 2 --requests 6 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/github_min_http_summary.json
```

本地短压测结果：

- mode: `http`
- requests: `6`
- success_rate: `1.0`
- stage modes: `stage09=lookup_live`, `stage10=xgb_live`, `stage11=cache_first/stage10_fallback`

## 持续压测脚本复核

```powershell
python tools/run_serving_engineering_load.py --duration-s 5 --warmup-requests 2 --concurrency 2 --request-sample-size 3 --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output-json data/output/serving_validation/github_min_engineering_load.json --output-md data/output/serving_validation/github_min_engineering_load.md
```

5 秒复核只验证脚本和报告导出路径；不作为延迟指标。公开延迟指标引用 30 分钟 HTTP 报告。

## 环境说明

| 环境 | 支持状态 | 说明 |
| --- | --- | --- |
| Windows PowerShell | 已本地验证 | 当前最小用例和 HTTP 测试均已通过 |
| Linux shell | 命令已提供 | 与云端 CPU-only 压测命令一致；30 分钟结果见工程压测报告 |
| GPU 环境 | 非最小 demo 依赖 | Stage11 在线大模型推理不在当前 mock serving 最小用例中 |
| Spark full rerun | 非最小 demo 依赖 | Stage09/10 完整重跑见 `reproduce_mainline.md` 和本地 wrapper |
