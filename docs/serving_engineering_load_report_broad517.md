# Mock Serving Engineering Load Report

- Generated at: `2026-04-26 15:13:58 UTC`
- Mode: `in_process`
- Duration target / observed: `300.0s` / `300.521s`
- Requests: `902` after `40` warmup
- Concurrency: `4`
- Throughput: `3.001` req/s
- Unique replay request_ids: `517`
- Traffic profile: `mixed`
- Requested strategy default: `reward_rerank`

## SLA Signals

| Signal | Observed |
| --- | --- |
| success_rate | 100.00% |
| serving_latency_p50 | 1416.499 ms |
| serving_latency_p95 | 1658.21 ms |
| serving_latency_p99 | 1807.024 ms |
| serving_latency_max | 3542.479 ms |
| wall_latency_p95 | 1658.272 ms |
| combined_fallback_recovery_rate | 100.00% |

## Fallback And Cache

- Strategy requested counts: `reward_rerank=824, xgboost=78`
- Strategy used counts: `reward_rerank=775, xgboost=127`
- Stage mode counts: `stage09={'lookup_live': 902}`, `stage10={'xgb_live': 902}`, `stage11={'stage10_fallback': 247, 'cache_first': 655}`
- Source alignment: `stage09={'release_sourceparity': 902}`, `stage10={'release_sourceparity': 902}`
- Stage11 cache status counts: `available_not_requested=127, hit=655, miss_simulated=120`
- Stage11 policy fallback to Stage10: `247` (27.38%)
- Strategy fallback to XGBoost: `49` (5.43%)
- Backfill count: `120`

## Latency Groups

| Group | Count | p50 | p95 | p99 | avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward_rerank_cache_hit | 655 | 1412.752 | 1655.648 | 1810.522 | 1328.527 |
| stage11_policy_fallback_to_stage10 | 247 | 1423.275 | 1667.892 | 1753.588 | 1334.934 |
| strategy_fallback_to_xgboost | 49 | 1428.977 | 1727.659 | 1790.024 | 1377.742 |
| non_reward_requested | 78 | 1435.097 | 1670.454 | 2092.65 | 1387.563 |

## Per-Stage P95

| Stage | p95 | avg |
| --- | ---: | ---: |
| request_lookup | 68.772 | 18.909 |
| stage09 | 1301.837 | 1014.9 |
| stage10 | 209.636 | 122.619 |
| stage11 | 59.616 | 18.514 |
| offline_truth_audit | 37.491 | 14.111 |
| fallback_demo | 0.0 | 0.0 |

## Error Distribution

- Error counts: `-`
- First errors: `[]`

## Known Limits

- This is a CPU-only local replay/mock-serving validation, not production traffic or online A/B.
- Stage09 is represented by per-user parquet lookup from a materialized candidate pack; it does not run full Spark recall recomputation per request.
- Stage10 uses CPU XGBoost live scoring for bucket5 only in the current demo surface.
- Stage11 uses frozen reward-rerank cache policy only; no GPU reward-model online inference is executed.
- Cache misses simulate uncovered Stage11 policy and fall back to Stage10; broader bucket2/bucket10 live recompute is not wired in this report.
