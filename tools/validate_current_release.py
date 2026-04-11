#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELEASE = REPO_ROOT / "data" / "output" / "current_release"
CURRENT_METRICS = REPO_ROOT / "data" / "metrics" / "current_release"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    errors: list[str] = []

    stage10_summary = read_json(CURRENT_RELEASE / "stage10" / "stage10_current_mainline_summary.json")
    buckets = {row["bucket"] for row in stage10_summary["bucket_snapshots"]}
    if buckets != {"bucket2", "bucket5", "bucket10"}:
        errors.append(f"unexpected stage10 bucket set: {sorted(buckets)}")

    expert_summary = read_json(CURRENT_RELEASE / "stage11" / "experts" / "expert_training_summary.json")
    experts = {row["expert_band"] for row in expert_summary["experts"]}
    if experts != {"11-30", "31-60", "61-100"}:
        errors.append(f"unexpected stage11 expert set: {sorted(experts)}")

    expert_rows = {row["expert_band"]: row for row in expert_summary["experts"]}
    if float(expert_rows["11-30"]["true_win_11_30"]) <= 0.95:
        errors.append("11-30 expert true-win metric is unexpectedly low")

    eval_lines = read_csv_rows(CURRENT_METRICS / "stage11" / "stage11_bucket5_eval_reference_lines.csv")
    line_names = {row["line"] for row in eval_lines}
    if {"two_band_best_known_v120", "tri_band_freeze_v124"} - line_names:
        errors.append("stage11 eval reference lines missing expected rows")

    tri_band = read_json(CURRENT_RELEASE / "stage11" / "eval" / "bucket5_tri_band_freeze_v124_alpha036" / "summary.json")
    if abs(float(tri_band["alpha"]) - 0.36) > 1e-9:
        errors.append("tri-band freeze summary alpha mismatch")

    if errors:
        print("FAIL current_release")
        for item in errors:
            print(f"- {item}")
        return 1

    print("PASS current_release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
