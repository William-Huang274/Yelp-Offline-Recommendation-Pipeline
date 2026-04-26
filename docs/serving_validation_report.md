# Serving Validation Report

- Generated at: `2026-04-25 16:57:38 UTC`
- Source JSON: `D:/5006_BDA_project/data/output/serving_validation/latest_summary.json`
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
| serving_latency_p95 | 216.294 ms | <= 250.000 ms | PASS |
| serving_latency_p99 | 217.988 ms | <= 300.000 ms | PASS |

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
| latency_ms | 176.892 ms | 216.294 ms | 217.988 ms | 218.411 ms |
| wall_latency_ms | 176.939 ms | 216.348 ms | 218.056 ms | 218.483 ms |
| audit_latency_ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |

## Per-Stage Latency

| Stage | p50 | p95 | p99 | avg |
| --- | --- | --- | --- | --- |
| request_lookup | 2.290 ms | 5.700 ms | 26.400 ms | 3.754 ms |
| stage09 | 4.723 ms | 10.507 ms | 23.143 ms | 5.630 ms |
| stage10 | 84.689 ms | 113.674 ms | 139.812 ms | 82.831 ms |
| stage11 | 3.181 ms | 4.890 ms | 7.933 ms | 3.664 ms |
| offline_truth_audit | 1.329 ms | 3.662 ms | 3.765 ms | 1.620 ms |
| fallback_demo | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |

## Stage09/10/11 Serving Scope

- Stage09 modes: `lookup_live=20`
- Stage09 source alignment: `fallback_local_candidate_pack=20`
- Stage10 modes: `xgb_live=20`
- Stage10 source alignment: `fallback_local_candidate_pack=20`
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
