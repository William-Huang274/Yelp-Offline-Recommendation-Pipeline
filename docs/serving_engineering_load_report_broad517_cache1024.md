# Mock Serving Engineering Load Report

- Generated at: `2026-04-27 03:46:23 UTC`
- Mode: `in_process`
- Duration target / observed: `300.0s` / `300.07s`
- Requests: `4833` after `600` warmup
- Concurrency: `4`
- Throughput: `16.106` req/s
- Unique replay request_ids: `517`
- Traffic profile: `mixed`
- Requested strategy default: `reward_rerank`

## SLA Signals

| Signal | Observed |
| --- | --- |
| success_rate | 100.00% |
| serving_latency_p50 | 234.825 ms |
| serving_latency_p95 | 343.739 ms |
| serving_latency_p99 | 386.14 ms |
| serving_latency_max | 980.707 ms |
| wall_latency_p95 | 343.789 ms |
| combined_fallback_recovery_rate | 100.00% |

## Fallback And Cache

- Strategy requested counts: `reward_rerank=4348, xgboost=485`
- Strategy used counts: `reward_rerank=4113, xgboost=720`
- Stage mode counts: `stage09={'lookup_live': 4833}`, `stage10={'xgb_live': 4833}`, `stage11={'cache_first': 3361, 'stage10_fallback': 1472}`
- Source alignment: `stage09={'release_sourceparity': 4833}`, `stage10={'release_sourceparity': 4833}`
- Stage11 cache status counts: `available_not_requested=720, hit=3361, miss_simulated=752`
- Stage11 policy fallback to Stage10: `1472` (30.46%)
- Strategy fallback to XGBoost: `235` (4.86%)
- Backfill count: `752`

## Latency Groups

| Group | Count | p50 | p95 | p99 | avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward_rerank_cache_hit | 3361 | 235.634 | 345.599 | 385.318 | 248.71 |
| stage11_policy_fallback_to_stage10 | 1472 | 232.298 | 335.956 | 386.399 | 243.459 |
| strategy_fallback_to_xgboost | 235 | 229.507 | 340.119 | 388.006 | 242.685 |
| non_reward_requested | 485 | 228.775 | 335.068 | 487.181 | 244.451 |

## Per-Stage P95

| Stage | p95 | avg |
| --- | ---: | ---: |
| request_lookup | 31.204 | 18.256 |
| stage09 | 12.78 | 4.606 |
| stage10 | 70.42 | 47.58 |
| stage11 | 33.449 | 20.297 |
| offline_truth_audit | 28.936 | 16.034 |
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
