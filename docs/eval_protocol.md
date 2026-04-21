# Evaluation Protocol

This page records the current public evaluation contract. It is intentionally compact: the goal is to make the experiment governance readable without dumping every CSV.

## Frozen Inputs

| surface | path |
| --- | --- |
| release manifest | `data/output/current_release/manifest.json` |
| Stage09 summary | `data/output/current_release/stage09/bucket5_route_aware_sourceparity/summary.json` |
| Stage10 summary | `data/output/current_release/stage10/stage10_current_mainline_summary.json` |
| Stage11 summary | `data/output/current_release/stage11/eval/bucket5_tri_band_freeze_v124_alpha036/summary.json` |

## Bucket Definitions

| bucket | meaning | frozen Stage10 eval users | businesses | candidate rows |
| --- | --- | ---: | ---: | ---: |
| `bucket2` | cold-start-inclusive trainable set under leave-two-out | 5,344 | 1,798 | 3,058,600 |
| `bucket5` | mid-to-high interaction set and current main display slice | 1,935 | 1,798 | 935,160 |
| `bucket10` | high-interaction set used for earlier structural validation | 738 | 1,794 | 697,299 |

The Stage09 / Stage10 scripts also support explicit `0-3` and `4-6` cold-start user cohorts through cohort-path overrides, but those finer slices are not frozen into the headline `current_release` tables yet.

## Metrics

Primary ranking metrics:

- Recall@10
- NDCG@10

Release-quality metrics:

- Stage09 truth-in-pretrim and hard-miss rate
- user coverage
- item coverage
- tail coverage
- novelty
- Stage11 rescued users and rescued band counts

## Stage Gates

| stage | gate |
| --- | --- |
| Stage09 | candidate pool must preserve truth better than the prior route-aware baseline |
| Stage10 | learned reranker must beat PreScore on the frozen bucket view before it is promoted |
| Stage11 | reward rerank must stay bounded to shortlist rescue and preserve a clean fallback path |
| Serving | mock API must return strategy, latency, success, and fallback signals |

## Acceptance Commands

```bash
python tools/run_stage01_11_minidemo.py
python tools/run_full_chain_smoke.py
python tools/run_release_checks.py --skip-pytest
pytest tests/test_release_metrics_surface.py tests/test_demo_tools.py
```

## Non-Negotiable Boundaries

- Do not change label definitions while claiming metric parity.
- Do not change candidate boundary while comparing release metrics.
- Do not compare a filtered Stage11 subset against a full-coverage Stage10 line without reporting coverage.
- Do not publish a new model line unless release pointers, serving config, and smoke tests are updated together.
