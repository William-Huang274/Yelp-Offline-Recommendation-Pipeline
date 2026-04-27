# Mock Serving Engineering Load Report

- Generated at: `2026-04-26 15:21:34 UTC`
- Mode: `in_process`
- Duration target / observed: `300.0s` / `300.097s`
- Requests: `4450` after `220` warmup
- Concurrency: `4`
- Throughput: `14.829` req/s
- Unique replay request_ids: `200`
- Traffic profile: `mixed`
- Requested strategy default: `reward_rerank`

## SLA Signals

| Signal | Observed |
| --- | --- |
| success_rate | 100.00% |
| serving_latency_p50 | 259.674 ms |
| serving_latency_p95 | 344.467 ms |
| serving_latency_p99 | 378.16 ms |
| serving_latency_max | 454.777 ms |
| wall_latency_p95 | 344.519 ms |
| combined_fallback_recovery_rate | 100.00% |

## Fallback And Cache

- Strategy requested counts: `reward_rerank=4021, xgboost=429`
- Strategy used counts: `reward_rerank=3792, xgboost=658`
- Stage mode counts: `stage09={'lookup_live': 4450}`, `stage10={'xgb_live': 4450}`, `stage11={'cache_first': 3128, 'stage10_fallback': 1322}`
- Source alignment: `stage09={'release_sourceparity': 4450}`, `stage10={'release_sourceparity': 4450}`
- Stage11 cache status counts: `available_not_requested=658, hit=3128, miss_simulated=664`
- Stage11 policy fallback to Stage10: `1322` (29.71%)
- Strategy fallback to XGBoost: `229` (5.15%)
- Backfill count: `664`

## Latency Groups

| Group | Count | p50 | p95 | p99 | avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward_rerank_cache_hit | 3128 | 261.307 | 345.794 | 379.74 | 270.044 |
| stage11_policy_fallback_to_stage10 | 1322 | 255.716 | 341.861 | 376.94 | 264.524 |
| strategy_fallback_to_xgboost | 229 | 254.974 | 345.045 | 374.724 | 264.155 |
| non_reward_requested | 429 | 252.195 | 336.255 | 380.453 | 261.04 |

## Per-Stage P95

| Stage | p95 | avg |
| --- | ---: | ---: |
| request_lookup | 35.6 | 19.609 |
| stage09 | 14.96 | 5.377 |
| stage10 | 80.326 | 52.362 |
| stage11 | 38.451 | 21.638 |
| offline_truth_audit | 32.998 | 17.34 |
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
