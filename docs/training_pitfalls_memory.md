# Training Pitfalls Memory

This file defines the long-term training memory mechanism.

## Source of truth
- Machine-readable log: `data/metrics/training_pitfalls_memory.jsonl`
- Writer (automatic): `scripts/10_9_autopilot_xgb.py`
- Reader/report: `scripts/10_10_pitfall_memory_report.py`

## What gets logged
- Session start/end metadata.
- Round-level failures:
  - `train_fail`
  - `eval_fail`
  - `model_not_updated`
  - `no_new_metric`
  - `metric_parse_fail`
- Round-level weak outcomes:
  - `no_gain` (learned model does not beat PreScore)
- Round-level wins:
  - `positive_gain`

## Required fields
- `timestamp`
- `run_tag`
- `session_dir`
- `source_09`
- `bucket`
- `round`
- `cfg_name`
- `event`
- `severity`
- `message`
- optional extras (delta metrics, logs, run_id)

## Usage
- Append happens automatically when running autopilot.
- Quick summary:
  - `python scripts/10_10_pitfall_memory_report.py`

## Rule for future iterations
- Before new tuning: read last 30-50 pitfall records.
- If same pitfall repeats 3+ times (same event + bucket), fix pipeline logic first, then continue tuning.

