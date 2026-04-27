# Demo 与 Mock Serving 验证入口

本文档汇总当前仓库的推荐链路 demo、HTTP mock serving 运行方式和工程压测报告入口。该验证链路用于展示离线排序系统的发布模拟能力，包括 release surface、策略切换、fallback、延迟拆解和报告固化。

最小测试用例和接口验证记录见 [demo_reproducibility_matrix.zh-CN.md](./demo_reproducibility_matrix.zh-CN.md)。

## 项目范围

当前 demo 覆盖以下链路：

```text
Stage09 materialized candidate lookup
-> Stage10 XGBoost CPU rerank
-> Stage11 cache-first bounded reward-model rescue rerank
```

范围边界：

- 非生产线上服务。
- 非真实流量 A/B。
- Stage09 使用已物化候选包 lookup，不在每个请求中实时运行 Spark 召回重算。
- Stage11 使用冻结 reward-rerank 缓存策略，不执行在线大模型推理。
- 30 分钟 HTTP 压测结果不等同于通过生产 SLA。

30 分钟 HTTP mock serving 压测的主结果为：CPU-only replay 流量、27129 次请求、517 个 replay 用户、请求成功率 100%、服务侧 p95/p99 延迟约 374ms/443ms，Stage11 cache miss 或策略失败时回退 Stage10。

## 入口总览

| 场景 | 入口 | 作用 |
| --- | --- | --- |
| 冻结结果摘要 | `python tools/demo/demo_recommend.py summary` | 展示 Stage09/10/11 当前冻结指标 |
| 案例回放 | `python tools/demo/demo_recommend.py show-case --case boundary_11_30` | 展示 Stage11 bounded rescue 案例 |
| 单请求 mock inference | `python tools/serving/batch_infer_demo.py ...` | 回放 Stage09/10/11 request-level ranking path |
| HTTP mock serving | `python tools/serving/mock_serving_api.py` | 暴露本地 `/health` 和 `/rank` |
| 短压测 | `python tools/serving/load_test_mock_serving.py ...` | 快速验证 mixed traffic、cache miss 和 fallback |
| 工程压测 | `python tools/serving/run_serving_engineering_load.py ...` | 生成持续压测 JSON 与 Markdown 报告 |

## 快速验证

```powershell
cd D:\5006_BDA_project
python tools/release/run_release_checks.py --skip-pytest
python tools/demo/demo_recommend.py summary
python tools/demo/demo_recommend.py show-case --case boundary_11_30
```

## 单请求 Replay

Cache hit + Stage11 rescue：

```powershell
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u001072 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --include-fallback-demo
```

另一个 replay 用户：

```powershell
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u001940 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --include-fallback-demo
```

Cache miss + Stage10 fallback：

```powershell
python tools/serving/batch_infer_demo.py --request-id stage11_b5_u006562 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --simulate-stage11-cache-miss
```

关键字段：

- `execution_modes.stage09=lookup_live`：读取已物化候选包。
- `execution_modes.stage10=xgb_live`：CPU XGBoost backbone 打分。
- `execution_modes.stage11=cache_first`：命中冻结 Stage11 reward-rerank 缓存后应用 bounded rescue。
- `execution_modes.stage11=stage10_fallback`：Stage11 cache miss 或未覆盖时返回 Stage10 结果。
- `fallback_demo`：演示 `reward_rerank -> xgboost` 策略降级。

## HTTP Mock Serving

服务自测：

```powershell
python tools/serving/mock_serving_api.py --self-test
```

启动 `/rank`：

```powershell
python tools/serving/mock_serving_api.py --host 127.0.0.1 --port 18081
```

短压测与报告导出：

```powershell
python tools/serving/load_test_mock_serving.py --url http://127.0.0.1:18081/rank --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/serving/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

## 工程压测

517 用户广覆盖压测使用更大的 per-user lookup cache：

```powershell
$env:BDA_STAGE10_LIVE_USER_CACHE_SIZE="1024"
python tools/serving/mock_serving_api.py --host 127.0.0.1 --port 18081
```

Linux shell：

```bash
export BDA_STAGE10_LIVE_USER_CACHE_SIZE=1024
python tools/serving/mock_serving_api.py --host 127.0.0.1 --port 18081
```

30 分钟 HTTP 压测：

```bash
python tools/serving/run_serving_engineering_load.py \
  --url http://127.0.0.1:18081/rank \
  --duration-s 1800 \
  --warmup-requests 600 \
  --concurrency 4 \
  --request-sample-size 517 \
  --stage09-mode lookup_live \
  --stage10-mode xgb_live \
  --stage11-mode replay \
  --traffic-profile mixed \
  --cache-miss-rate 0.15 \
  --strategy-failure-rate 0.05 \
  --xgboost-rate 0.10 \
  --output-json data/output/serving_validation/engineering_http_20260427_30min_c4_broad517_cache1024.json \
  --output-md docs/engineering_http_20260427_30min_c4_broad517_cache1024.md
```

快速复核可将 `--duration-s` 调整为 `300`，其余参数保持一致。

## 报告入口

主报告：

- [Mock Serving 工程压测总结](../serving_engineering_load_summary.zh-CN.md)
- [30 分钟 HTTP 压测报告](../engineering_http_20260427_30min_c4_broad517_cache1024.md)

辅助报告：

- [517 用户 broad 压测，cache1024](../serving_engineering_load_report_broad517_cache1024.md)
- [517 用户 broad 压测，default256](../serving_engineering_load_report_broad517.md)
- [200 用户 hot set 压测](../serving_engineering_load_report_hot200.md)
- [短压测导出的 serving validation report](../serving_validation_report.md)

## 30 分钟 HTTP 压测摘要

| 指标 | 结果 |
| --- | ---: |
| 时长 | 30 分钟 |
| 请求数 | 27129 |
| 覆盖 replay 用户 | 517 |
| 并发 | 4 |
| 吞吐 | 15.07 req/s |
| 成功率 | 100% |
| 服务侧 p50 | 246.897 ms |
| 服务侧 p95 | 373.881 ms |
| 服务侧 p99 | 442.509 ms |
| fallback recovery | 100% |
| Stage11 policy fallback | 8171 / 30.12% |
| strategy fallback | 1354 / 4.99% |

资源使用：

- CPU-only，无 GPU 在线推理。
- 服务进程 RSS 均值约 1.45GB，峰值约 4.4GB。
- 服务端 CPU 均值约 5.2 核。
- 压测环境适配 22 核 / 110GB 云端机器。

## 外部摘要

```text
构建 replay/mock serving 压测链路，验证 Stage09 物化候选包 lookup、Stage10 XGBoost CPU 精排、Stage11 cache-first bounded rerank 的策略切换与降级路径；在 30 分钟 HTTP /rank 压测中覆盖 517 个 replay 用户、27129 次请求，请求成功率 100%，Stage11 cache miss/策略失败均可回退至 Stage10，服务侧 p95/p99 延迟约 374ms/443ms。
```

## 已知限制

- 当前 request-level serving 只接入 bucket5 Stage11 replay cohort。
- bucket2 / bucket10 以 cohort-level frozen evaluation 形式展示，未接入同一套 Stage11 request-level replay serving。
- Stage09 是物化候选包 lookup；真实线上低延迟方案应使用外部 KV/cache 或离线候选更新。
- Stage11 是 cache-first policy；未覆盖用户或 cache miss 返回 Stage10 fallback。
- `config/serving.yaml` 中的 `latency_budget_ms: 250` 属于早期 mock serving 配置字段，30 分钟 HTTP 报告按观测 p95/p99 延迟解释。
