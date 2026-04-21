# Demo Runbook

## Goal

This runbook supports a 20-minute teacher-facing demonstration that matches the
practice-module expectation: show the business problem, major data flows,
application flow, and the current result story.

## 1. Demo Principle

Do not try to retrain the full stack live.

Use the frozen checked-in review surface for the live demo, and explain where
full Stage09/10/11 training would sit in the complete workflow.

## 2. Pre-Demo Checklist

Run these before the presentation:

```bash
python tools/run_release_checks.py
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
```

Optional PowerShell wrapper:

```powershell
.\tools\run_release_checks.ps1
```

## 3. Suggested 20-Minute Flow

### 0-3 min: Business Problem And Big Data Scope

Cover:

- why Yelp recommendation is a big-data problem
- why multiple data sources are needed
- why the project cannot be framed as a single-model classroom exercise

Supporting references:

- [teacher_requirement_alignment.md](./teacher_requirement_alignment.md)
- [data_lineage_and_storage.md](./data_lineage_and_storage.md)

### 3-7 min: End-To-End Pipeline

Cover:

- raw Yelp JSON and optional photo assets
- parquet layer
- Stage09 recall routing
- Stage10 structured rerank
- Stage11 bounded reward-model rescue rerank
- compact release surface

### 7-11 min: Stage09 And Stage10 Results

Show:

- Stage09 `truth_in_pretrim150` and `hard_miss`
- Stage10 `bucket2 / bucket5 / bucket10` results

CLI support:

```bash
python tools/demo_recommend.py summary
```

### 11-16 min: Stage11 Results And Interpretation

Show:

- three-expert design
- two-band best-known line
- current tri-band freeze line
- why deep rescue is conservative

CLI support:

```bash
python tools/demo_recommend.py show-case --case boundary_11_30
python tools/demo_recommend.py show-case --case mid_31_40
```

### 16-18 min: Engineering Controls

Cover:

- bounded LLM usage
- leakage control
- validation scripts
- rollback and monitoring awareness

### 18-20 min: Q&A Buffer

Keep 2 minutes free for:

- data-flow questions
- reproducibility questions
- why the current freeze line makes the claims it makes

## 4. Demo Command Set

Use this exact sequence if you need a minimal live terminal demo:

```bash
python tools/run_release_checks.py --skip-pytest
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
python tools/demo_recommend.py show-case --case mid_31_40
```

## 5. Fallback Plan

If shell wrappers or training environments are unavailable during the demo:

- rely on the checked-in current release surface
- use the CLI and the docs in `docs/project`
- do not attempt a live retrain

This is acceptable because the purpose of the demo is to explain the designed
solution and major data flows, not to benchmark a new run on the spot.
