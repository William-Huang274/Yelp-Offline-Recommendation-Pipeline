# Mock Serving 工程压测总结

压测时间：2026-04-26 / 2026-04-27

目标：验证当前 demo 在不使用 GPU 在线推理的情况下，是否能支撑更接近工程报告的本地 replay/mock-serving 压测。压测主路径为：

`Stage09 lookup_live -> Stage10 xgb_live -> Stage11 cache-first reward_rerank policy`

其中 Stage11 使用冻结 reward-rerank 分数缓存，不执行在线 Qwen/RM 推理；cache miss 或策略失败时回退 Stage10 XGBoost。

## 结论

当前可以做工程报告档，但需要分两种口径讲：

1. **517 用户广覆盖诊断口径**
   - 价值：覆盖完整 Stage11 replay 用户池，能证明 fallback/cache policy 的稳定性。
   - 初始问题：默认 256 用户 cache 会在 517 用户池里反复淘汰，p95 被 Stage09 parquet per-user lookup 拉高。
   - 修复验证：将 `BDA_STAGE10_LIVE_USER_CACHE_SIZE=1024` 后，517 用户广覆盖 p95 降到 344ms。

2. **200 用户热集 serving 口径**
   - 价值：更接近候选包已热/已缓存后的 serving policy 表现。
   - 结果：5 分钟、4450 请求、100% 成功率、p95 344ms、p99 378ms、fallback recovery 100%。
   - 简历可用，但必须写明是本地 replay/mock serving，不是真实线上流量。

3. **HTTP 30 分钟长稳口径**
   - 价值：通过本地 HTTP `/rank` 服务验证，而不是只跑 in-process 函数调用。
   - 结果：30 分钟、27129 请求、517 用户、100% 成功率、p95 374ms、p99 443ms、fallback recovery 100%。
   - 资源：服务进程 RSS 均值约 1.45GB，峰值约 4.4GB；服务端 CPU 均值约 5.2 核。

## 结果对比

| 压测 | 时长 | 请求数 | 用户数 | 并发 | 成功率 | p50 | p95 | p99 | Stage11 policy fallback | 策略 fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broad517-default256 | 300s | 902 | 517 | 4 | 100% | 1416ms | 1658ms | 1807ms | 247 | 49 |
| broad517-cache1024 | 300s | 4833 | 517 | 4 | 100% | 235ms | 344ms | 386ms | 1472 | 235 |
| hot200 | 300s | 4450 | 200 | 4 | 100% | 260ms | 344ms | 378ms | 1322 | 229 |
| http30m-cache1024 | 1800s | 27129 | 517 | 4 | 100% | 247ms | 374ms | 443ms | 8171 | 1354 |

## 关键诊断

broad517 默认配置的慢点主要在 Stage09。cache1024 后，517 用户广覆盖已经接近 hot200 表现：

| Stage | broad517-default256 p95 | broad517-cache1024 p95 | hot200 p95 |
| --- | ---: | ---: | ---: |
| request_lookup | 69ms | 31ms | 36ms |
| Stage09 lookup_live | 1302ms | 13ms | 15ms |
| Stage10 xgb_live | 210ms | 70ms | 80ms |
| Stage11 cache policy | 60ms | 33ms | 38ms |

这说明当前 demo 的主要瓶颈不是 Stage11 cache-first/fallback policy，而是默认 cache 只能容纳 256 个用户；517 用户广覆盖时会产生 cache thrash，并反复触发 Stage09 parquet filter。将用户级 cache 提高到 1024 后，Stage09 lookup p95 降到 13ms，整体 p95 降到 344ms。

## 简历建议口径

推荐写法：

> 搭建本地 replay/mock serving 压测链路，验证 `Stage09 lookup -> Stage10 XGBoost -> Stage11 cache-first rerank` 的策略切换与降级路径；在 30 分钟 HTTP `/rank` 压测中覆盖 517 个 replay 用户、27129 次请求，请求成功率 100%，Stage11 cache miss/策略失败均可回退至 Stage10，服务侧 p95 / p99 延迟约 374ms / 443ms。

更保守写法：

> 建立本地 replay/mock serving 验证链路，覆盖 Stage09 候选查表、Stage10 XGBoost CPU 打分、Stage11 缓存优先重排与失败降级；在 5 分钟持续压测中验证 100% 请求成功率、千级请求量、分阶段延迟拆解和 fallback recovery。

不建议写：

- 线上服务
- 真实流量压测
- 线上 A/B
- Stage09 实时全量召回
- Stage11 在线大模型推理

## 当前边界

- Stage09 是已物化候选包的 per-user parquet lookup，不是每个请求实时 Spark 召回重算。
- Stage10 是 bucket5 的 CPU XGBoost live scoring，bucket2/bucket10 live recompute 当前未接入 demo。
- Stage11 是冻结 reward-rerank cache policy，不跑 GPU 在线推理。
- cache miss 用模拟 miss 表示未覆盖用户/缓存未命中后的降级路径。
- `BDA_STAGE10_LIVE_USER_CACHE_SIZE=1024` 用于云端 517 用户压测；本地默认仍为 256，避免小内存机器默认占用过高。
- broad517-cache1024 报告可以作为当前简历和演示主口径；default256 报告保留为 bottleneck 诊断证据。

## Parity 检查

`BDA_STAGE10_LIVE_USER_CACHE_SIZE` 只改变 per-user lookup cache 容量，不改变候选生成、特征、模型或排序公式。已在云端抽样 20 个固定 replay 用户，对比默认 cache256 与 cache1024 的输出：

- top-k `business_id / rank / stage10_rank / stage11_rank` 一致
- Stage09 top pre candidates 一致
- Stage10 learned top candidates 一致
- execution modes 一致
- 结果：PASS，0 mismatch

## 文件

- broad517 报告：`docs/serving_engineering_load_report_broad517.md`
- broad517 cache1024 报告：`docs/serving_engineering_load_report_broad517_cache1024.md`
- hot200 报告：`docs/serving_engineering_load_report_hot200.md`
- HTTP 30 分钟报告：`docs/engineering_http_20260427_30min_c4_broad517_cache1024.md`
- HTTP 30 分钟进程监控：`data/output/serving_validation/engineering_http_20260427_30min_c4_broad517_cache1024_process_monitor.csv`
- 压测脚本：`tools/run_serving_engineering_load.py`
