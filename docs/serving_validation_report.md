# Serving Validation Report

- Generated at: `2026-04-26 10:28:22 UTC`
- Source JSON: `data/output/serving_validation/latest_summary.json`
- Mode: `in_process`
- Requests: `20` after `5` warmup
- Concurrency: `2`
- Unique replay request_ids: `5`
- Traffic profile: `mixed`
- Requested strategy: `reward_rerank`

## SLA Gate

| Signal | Observed | Gate | Status |
| --- | --- | --- | --- |
| success_rate | 100.00% | >= 99.00% | PASS |
| serving_latency_p95 | 198.344 ms | <= 250.000 ms | PASS |
| serving_latency_p99 | 200.856 ms | <= 300.000 ms | PASS |

## Traffic Mix

- Requested strategy counts: `reward_rerank=19, xgboost=1`
- Used strategy counts: `reward_rerank=17, xgboost=3`
- Fallback rate: `10.00%`
- Cache hit rate: `70.00%`
- Cache miss count: `3`
- Backfill count: `3`
- Cache status counts: `available_not_requested=3, hit=14, miss_simulated=3`

## Latency Summary

| Metric | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| latency_ms | 133.313 ms | 198.344 ms | 200.856 ms | 201.484 ms |
| wall_latency_ms | 133.338 ms | 198.364 ms | 200.884 ms | 201.514 ms |
| audit_latency_ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |

## Per-Stage Latency

| Stage | p50 | p95 | p99 | avg |
| --- | --- | --- | --- | --- |
| request_lookup | 0.728 ms | 11.211 ms | 12.357 ms | 1.888 ms |
| stage09 | 21.998 ms | 45.069 ms | 56.314 ms | 25.986 ms |
| stage10 | 54.213 ms | 92.621 ms | 100.622 ms | 57.781 ms |
| stage11 | 1.329 ms | 2.365 ms | 3.497 ms | 1.405 ms |
| offline_truth_audit | 0.913 ms | 1.906 ms | 2.823 ms | 1.144 ms |
| fallback_demo | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |

## Stage09/10/11 Serving Scope

- Stage09 modes: `lookup_live=20`
- Stage09 source alignment: `embedded_sample_fixture=20`
- Stage10 modes: `xgb_live=20`
- Stage10 source alignment: `embedded_sample_fixture=20`
- Stage11 modes: `cache_first=14, stage10_fallback=6`

## Audit Path

- Audit latency p95: `0.000 ms`
| Audit block | p50 | p95 | avg |
| --- | --- | --- | --- |
| stage11_live | 0.000 ms | 0.000 ms | 0.000 ms |
| fallback_demo | 0.000 ms | 0.000 ms | 0.000 ms |

## Interpretation

- `serving_latency_ms` excludes offline-only audit work and is the number to compare with the mock online budget.
- `wall_latency_ms` includes local Python overhead and optional audit hooks, so it is useful for demo experience but not the serving SLA gate.
- `stage11_live` is treated as audit/backfill unless the serving policy reports `applied_to_serving=true`; cache miss should fall back to Stage10 and enqueue backfill.
- Stage09 is represented by replay lookup or live lookup mode in this validation. It does not retrain or redefine recall candidates.

## Known Limits

- This is a local replay/mock-serving validation report, not a production traffic A/B report.
- The request stream is sampled from frozen replay request_ids, so it verifies policy behavior, latency accounting, and fallback paths rather than new user distribution quality.
- Metric definitions, labels, candidate boundaries, and split logic are not changed by this report.
