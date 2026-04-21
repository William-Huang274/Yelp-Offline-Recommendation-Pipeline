# Evaluation

This page is the interviewer-facing evaluation summary for the current offline
ranking stack.

## Evaluation Questions

| question | current evidence |
| --- | --- |
| does recall keep the truth in the candidate pool? | Stage09 route-aware recall summary |
| does structured reranking improve ranking quality across density buckets? | Stage10 `bucket2 / bucket5 / bucket10` metrics |
| does bounded RM rescue add value without becoming a full-list reranker? | Stage11 reference lines and segmented expert evidence |
| does the stack remain interpretable under trade-offs? | bucket definitions, coverage metrics, case notes, and gated-vs-freeze comparisons |

## Baseline Vs. Current Summary

| module | baseline | current line | takeaway |
| --- | --- | --- | --- |
| `Stage09` candidate funnel | early gate: `truth_in_pretrim150 = 0.7248`, `hard_miss = 0.1616` | source-parity structural v5: `0.7451`, `0.1190` | better candidate retention with fewer hard misses |
| `Stage10` structured rerank | `PreScore@10` | `LearnedBlendXGBCls@10` | positive ranking lift across `bucket2 / bucket5 / bucket10` |
| `Stage11` rescue rerank | segmented one-band / two-band history | `v124` tri-band freeze with `v120` kept as best-known reference | rescue gains are real, but release selection still respects coverage and stability |

## Stage-By-Stage Gain Contribution

### Stage09

- `truth_in_pretrim150`: `0.7248 -> 0.7451` (`+0.0203`)
- `hard_miss`: `0.1616 -> 0.1190` (`-0.0426`)

Interpretation:

- the current route-aware line hands a stronger pool to downstream stages
- the stack still has non-trivial misses, so recall quality remains an active
  failure boundary

### Stage10

| bucket | PreScore recall / ndcg | Learned recall / ndcg | delta |
| --- | --- | --- | --- |
| `bucket2` | `0.1098 / 0.0513` | `0.1127 / 0.0522` | `+0.0028 / +0.0009` |
| `bucket5` | `0.0935 / 0.0440` | `0.1261 / 0.0581` | `+0.0326 / +0.0141` |
| `bucket10` | `0.0569 / 0.0265` | `0.0772 / 0.0341` | `+0.0203 / +0.0076` |

Interpretation:

- the largest offline lift appears on the outward-facing `bucket5` line
- `bucket2` still improves, but more modestly, which is consistent with the
  sparsity and mixed-intent difficulty of the cold-start-inclusive trainable set

### Stage11

Current showcase and freeze reference lines on the rescue cohort:

| line | recall@10 | ndcg@10 | cohort note |
| --- | ---: | ---: | --- |
| `v117 segmented` | `0.1663` | `0.0739` | early segmented rescue reference |
| `v120 two-band best-known` | `0.1973` | `0.0898` | best-known offline result on the 517-user rescue cohort |
| `v121 joint12 gate` | `0.2259` | `0.1049` | stronger score on `394` surviving users, but drops `123` users |
| `v124 tri-band freeze` | `0.1857` | `0.0838` | current frozen outward-facing line |

Interpretation:

- `v121` proves that stricter rescue gating can improve score on a filtered
  subset, but it also reduces cohort coverage
- `v124` is the current freeze line because it is easier to explain and safer to
  keep as the outward-facing baseline

## Bucketed Evaluation And Cold-Start Portability

The repository uses three user-density lines:

- `bucket2`: cold-start-inclusive trainable users under leave-two-out
- `bucket5`: mid-to-high interaction users and current outward-facing line
- `bucket10`: higher-density users used early for architecture validation

What the buckets show:

- the stack is not tuned only for one dense slice
- the Stage10 path remains portable to the colder `bucket2` line, but with
  smaller gains
- `bucket5` is the best public balance between quality lift and practical scope

The checked-in release tables currently report the aggregate public `bucket2`
line. The updated Stage09 / Stage10 scripts can also replay finer cold-start
sub-cohorts such as `0-3` or `4-6` interactions by supplying explicit cohort
CSVs through `CANDIDATE_FUSION_USER_COHORT_PATH` and
`RANK_EVAL_USER_COHORT_PATH`, but those finer slices are not yet frozen into
`current_release`.

## Coverage, Tail, And Novelty Signals

Stage10 current-release files already track more than recall / ndcg.

| bucket | model | user coverage | item coverage | tail coverage | novelty |
| --- | --- | ---: | ---: | ---: | ---: |
| `bucket2` | `PreScore@10` | `1.0000` | `0.6229` | `0.0526` | `8.1657` |
| `bucket2` | `LearnedBlendXGBCls@10` | `1.0000` | `0.6073` | `0.0484` | `8.1058` |
| `bucket5` | `PreScore@10` | `1.0000` | `0.3782` | `0.0481` | `8.3756` |
| `bucket5` | `LearnedBlendXGBCls@10` | `1.0000` | `0.3971` | `0.0506` | `8.4324` |
| `bucket10` | `PreScore@10` | `1.0000` | `0.3138` | `0.0604` | `8.7737` |
| `bucket10` | `LearnedBlendXGBCls@10` | `1.0000` | `0.3266` | `0.0709` | `8.8256` |

Interpretation:

- user coverage stays complete in the current checked-in Stage10 lines
- `bucket5` and `bucket10` improve not only ranking accuracy but also item /
  tail coverage and novelty
- `bucket2` is the main reminder that portability under sparse behavior is real
  but not free

## Ablation And Controlled Comparisons

### Stage10

The current Stage10 files allow one compact ablation story:

- `ALS@10_from_candidates`: conventional baseline on the routed candidate pool
- `PreScore@10`: candidate-order baseline
- `LearnedBlendXGBCls@10`: current structured rerank backbone

This is enough to show that the gain is not only from richer recall; the
learned reranker adds value on top of the candidate pool.

### Stage11

The current public and showcase files support a controlled rescue story:

- `v117`: segmented rescue works
- `v120`: two-band joint rescue is the best-known offline line
- `v121`: gating can push the surviving subset higher
- `v124`: freeze selection trades some peak score for a cleaner outward-facing
  line

## Known Failure Cases And Current Limits

- `Stage09` still has a `hard_miss = 0.1190` boundary on the current `bucket5`
  line, so some users never hand the truth to downstream ranking.
- `bucket2` gains are positive but small, which is consistent with sparse and
  noisy user behavior.
- the `v121` gate line improves score by filtering to `394` users and dropping
  `123`, so headline improvements must be interpreted together with coverage.
- the `61-100` expert is trained successfully but remains conservative in the
  frozen tri-band line, with no current frozen top-rank rescue count attributed
  to that band.

## Primary Artifact Pointers

- [../data/metrics/current_release/stage09/stage09_bucket5_route_aware_recall_snapshot.csv](../data/metrics/current_release/stage09/stage09_bucket5_route_aware_recall_snapshot.csv)
- [../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv](../data/metrics/current_release/stage10/stage10_current_mainline_snapshot.csv)
- [../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv](../data/metrics/current_release/stage11/stage11_bucket5_eval_reference_lines.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v117_segmented_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v117_segmented_metrics.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v120_joint12_default_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v120_joint12_default_metrics.csv)
- [../data/metrics/showcase_history/stage11/bucket5_v121_joint12_gate_metrics.csv](../data/metrics/showcase_history/stage11/bucket5_v121_joint12_gate_metrics.csv)
