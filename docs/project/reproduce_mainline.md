# Reproduce The Current Mainline

## Purpose

This note describes the canonical reproduction paths for the current
`stage09 -> stage10 -> stage11` line.

Two paths are supported:

1. review the frozen checked-in release surface
2. rerun the full pipeline with local or cloud compute

## 1. Fastest Path: Review The Frozen Release

This path is sufficient for inspecting the checked-in release surface and
running the lightweight demo commands.

### Step 1: Install Review Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Run Repository Validation

```bash
python tools/run_release_checks.py
python tools/run_stage11_model_prompt_smoke.py
python tools/run_full_chain_smoke.py
```

Expected result:

- public-surface validator passes
- current-release validator passes
- stage11 model / prompt smoke passes
- full chain smoke passes
- demo CLI smoke step passes

### Step 3: Inspect The Frozen Summary

```bash
python tools/demo_recommend.py summary
python tools/demo_recommend.py list-cases
python tools/demo_recommend.py show-case --case boundary_11_30
```

This path does not require a GPU.

## 2. Full Pipeline Path

Use this only when you need to rebuild stage outputs instead of citing the
checked-in release surface.

## 2.1 Inputs

You need:

- raw Yelp JSON data
- optional photo asset if you want the photo-derived path
- Spark-capable environment
- GPU-capable environment for full Stage11 training

## 2.2 Optional Raw Ingest

If the parquet layer is not already prepared:

```bash
python scripts/stage01_to_stage08/01_data\ prep.py
```

Optional photo summary build:

```bash
python scripts/archive/release_closeout_20260409/stage09_nonmainline/09_yelp_photo_summary_build.py
```

## 2.3 Canonical Wrappers

The stable wrapper surface is documented in:

- [../../scripts/launchers/README.md](../../scripts/launchers/README.md)

Current mainline wrappers:

- `scripts/launchers/stage09_bucket5_mainline.sh`
- `scripts/launchers/stage10_bucket5_mainline.sh`
- `scripts/launchers/stage11_bucket5_11_1.sh`
- `scripts/launchers/stage11_bucket5_export_only.sh`
- `scripts/launchers/stage11_bucket5_train.sh`
- `scripts/launchers/stage11_bucket5_eval.sh`

Optional cross-bucket Stage10 wrappers:

- `scripts/launchers/stage10_bucket2_mainline.sh`
- `scripts/launchers/stage10_bucket10_mainline.sh`

For finer cold-start replays under `bucket2` (for example `0-3` or `4-6`
interaction cohorts), the updated scripts now support explicit cohort scoping:

- `CANDIDATE_FUSION_USER_COHORT_PATH` for Stage09 candidate-fusion input scope
- `RANK_EVAL_USER_COHORT_PATH` for Stage10 evaluation scope
- split-aware Stage09 feature builders can also be replayed against the same
  scoped bucket directory and index maps

The checked-in headline remains aggregate `bucket2`. The fixed-eval `bucket2`
light-user diagnostic has also been frozen at
`data/metrics/current_release/stage10/stage10_bucket2_light_user_subgroups_fixedeval_20260410.csv`
for the `0-3`, `4-6`, and `7+` history bands over the same 5,344 eval users.
Treat it as a diagnostic breakdown, not a replacement for the headline table.

## 2.4 Canonical Stage Order

### Stage09

Run the current route-aware recall mainline first.

Expected outcome:

- candidate funnel artifacts
- recall audit summary
- bucket-level handoff assets for downstream stages

### Stage10

Run the structured rerank trainer and infer/eval path on top of Stage09.

Expected outcome:

- bucket-level metrics
- current structured-rerank summary

### Stage11

Run the current bounded rescue path in four steps:

1. dataset build (`11_1`)
2. pair/export stage
3. reward-model training
4. sidecar evaluation

This path is GPU-oriented and is not required for lightweight repo review.

## 3. Success Checks

When reproducing the public mainline story, compare your outputs against the
checked-in release surface.

### Stage09 `bucket5`

- `truth_in_pretrim150 ~= 0.7451`
- `hard_miss ~= 0.1190`

### Stage10

- `bucket2 ~= 0.1127 / 0.0522`
- `bucket5 ~= 0.1261 / 0.0581`
- `bucket10 ~= 0.0772 / 0.0341`

### Stage11

- two-band reference `v120 @ alpha=0.80 ~= 0.1973 / 0.0898`
- tri-band freeze `v124 @ alpha=0.36 ~= 0.1857 / 0.0838`

## 4. When Full Reproduction Is Not Needed

For the practice-module demo and report, you usually do not need to retrain the
stack during the presentation week.

Use the frozen repository line when:

- the goal is to explain architecture
- the goal is to show release metrics
- the goal is to demonstrate one or two real rescue cases

## 5. Reviewer-Friendly Minimal Command Set

```bash
python tools/run_release_checks.py
python tools/demo_recommend.py summary
python tools/demo_recommend.py show-case --case mid_31_40
```
