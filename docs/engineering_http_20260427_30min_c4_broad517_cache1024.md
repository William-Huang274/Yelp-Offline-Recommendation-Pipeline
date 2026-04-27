# Mock Serving Engineering Load Report

- Generated at: `2026-04-27 05:27:45 UTC`
- Mode: `http`
- Duration target / observed: `1800.0s` / `1800.149s`
- Requests: `27129` after `600` warmup
- Concurrency: `4`
- Throughput: `15.07` req/s
- Unique replay request_ids: `517`
- Traffic profile: `mixed`
- Requested strategy default: `reward_rerank`

## SLA Signals

| Signal | Observed |
| --- | --- |
| success_rate | 100.00% |
| serving_latency_p50 | 246.897 ms |
| serving_latency_p95 | 373.881 ms |
| serving_latency_p99 | 442.509 ms |
| serving_latency_max | 1317.854 ms |
| wall_latency_p95 | 380.288 ms |
| combined_fallback_recovery_rate | 100.00% |

## Fallback And Cache

- Strategy requested counts: `reward_rerank=24370, xgboost=2759`
- Strategy used counts: `reward_rerank=23016, xgboost=4113`
- Stage mode counts: `stage09={'lookup_live': 27129}`, `stage10={'xgb_live': 27129}`, `stage11={'stage10_fallback': 8171, 'cache_first': 18958}`
- Source alignment: `stage09={'release_sourceparity': 27129}`, `stage10={'release_sourceparity': 27129}`
- Stage11 cache status counts: `available_not_requested=4113, hit=18958, miss_simulated=4058`
- Stage11 policy fallback to Stage10: `8171` (30.12%)
- Strategy fallback to XGBoost: `1354` (4.99%)
- Backfill count: `4058`

## Latency Groups

| Group | Count | p50 | p95 | p99 | avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward_rerank_cache_hit | 18958 | 248.075 | 373.697 | 442.765 | 259.521 |
| stage11_policy_fallback_to_stage10 | 8171 | 243.989 | 374.225 | 440.688 | 256.067 |
| strategy_fallback_to_xgboost | 1354 | 241.873 | 368.101 | 437.082 | 254.396 |
| non_reward_requested | 2759 | 241.116 | 371.455 | 437.177 | 253.677 |

## Per-Stage P95

| Stage | p95 | avg |
| --- | ---: | ---: |
| request_lookup | 36.666 | 20.646 |
| stage09 | 13.352 | 5.155 |
| stage10 | 75.824 | 51.313 |
| stage11 | 35.956 | 20.808 |
| offline_truth_audit | 30.779 | 16.51 |
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
