#!/usr/bin/env python3
"""Inspect a Stage11 QLoRA dataset run and report DPO readiness.

This tool checks whether each bucket has enough per-user positive/negative
samples to construct pairwise DPO training pairs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE11_ROOT = Path(
    os.getenv("INPUT_11_ROOT_DIR", str(PROJECT_ROOT / "data" / "output" / "11_qlora_data"))
).expanduser()
DEFAULT_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_MAX_PAIRS", "8") or 8)


def resolve_dataset_run(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser()

    if not DEFAULT_STAGE11_ROOT.exists():
        raise FileNotFoundError(f"stage11 data root not found: {DEFAULT_STAGE11_ROOT}")

    runs = [
        p
        for p in DEFAULT_STAGE11_ROOT.iterdir()
        if p.is_dir() and "stage11_1" in p.name
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(
            f"no stage11_1 dataset run found under {DEFAULT_STAGE11_ROOT}"
        )
    return runs[0]


def iter_bucket_dirs(dataset_run: Path) -> list[Path]:
    return sorted(
        [p for p in dataset_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")],
        key=lambda p: p.name,
    )


def parse_train_json(bucket_dir: Path) -> tuple[dict[int, int], dict[int, int], int]:
    pos_per_user: dict[int, int] = defaultdict(int)
    neg_per_user: dict[int, int] = defaultdict(int)
    n_rows = 0

    train_dir = bucket_dir / "train_json"
    for json_file in sorted(train_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = int(row.get("user_idx", -1))
                if uid < 0:
                    continue
                label = int(row.get("label", 0))
                if label == 1:
                    pos_per_user[uid] += 1
                else:
                    neg_per_user[uid] += 1
                n_rows += 1

    return pos_per_user, neg_per_user, n_rows


def estimate_pairs(pos_per_user: dict[int, int], neg_per_user: dict[int, int]) -> int:
    total = 0
    all_users = set(pos_per_user.keys()) | set(neg_per_user.keys())
    for uid in all_users:
        p = pos_per_user.get(uid, 0)
        n = neg_per_user.get(uid, 0)
        if p <= 0 or n <= 0:
            continue
        total += min(p * n, DEFAULT_MAX_PAIRS_PER_USER)
    return int(total)


def check_dataset_for_dpo(dataset_run: Path) -> bool:
    print("=" * 72)
    print("DPO Dataset Audit")
    print("=" * 72)
    print(f"dataset_run: {dataset_run}")

    if not dataset_run.exists():
        print("[ERROR] dataset run directory does not exist")
        return False

    bucket_dirs = iter_bucket_dirs(dataset_run)
    if not bucket_dirs:
        print("[ERROR] no bucket_* directories found")
        return False

    print(f"buckets: {[b.name for b in bucket_dirs]}")

    total_rows = 0
    total_users = 0
    total_pos = 0
    total_neg = 0
    total_users_with_both = 0
    total_est_pairs = 0

    for bucket_dir in bucket_dirs:
        train_dir = bucket_dir / "train_json"
        if not train_dir.exists():
            print(f"[WARN] {bucket_dir.name}: missing train_json/")
            continue

        pos_per_user, neg_per_user, n_rows = parse_train_json(bucket_dir)
        all_users = set(pos_per_user.keys()) | set(neg_per_user.keys())
        users_with_both = [uid for uid in all_users if pos_per_user.get(uid, 0) > 0 and neg_per_user.get(uid, 0) > 0]
        users_pos_only = [uid for uid in all_users if pos_per_user.get(uid, 0) > 0 and neg_per_user.get(uid, 0) == 0]
        users_neg_only = [uid for uid in all_users if pos_per_user.get(uid, 0) == 0 and neg_per_user.get(uid, 0) > 0]

        n_pos = int(sum(pos_per_user.values()))
        n_neg = int(sum(neg_per_user.values()))
        n_pairs = estimate_pairs(pos_per_user, neg_per_user)

        total_rows += n_rows
        total_users += len(all_users)
        total_pos += n_pos
        total_neg += n_neg
        total_users_with_both += len(users_with_both)
        total_est_pairs += n_pairs

        cover = (len(users_with_both) / max(1, len(all_users))) * 100.0
        neg_pos = n_neg / max(1, n_pos)

        print("-" * 72)
        print(bucket_dir.name)
        print(f"  rows={n_rows:,} users={len(all_users):,} pos={n_pos:,} neg={n_neg:,} neg/pos={neg_pos:.2f}")
        print(
            f"  users_with_both={len(users_with_both):,} ({cover:.1f}%) "
            f"pos_only={len(users_pos_only):,} neg_only={len(users_neg_only):,}"
        )
        print(f"  estimated_pairs(max_pairs={DEFAULT_MAX_PAIRS_PER_USER})={n_pairs:,}")

    print("=" * 72)
    print("Overall")
    print("=" * 72)
    print(f"rows={total_rows:,} users={total_users:,} pos={total_pos:,} neg={total_neg:,}")
    print(
        f"users_with_both={total_users_with_both:,} "
        f"({(total_users_with_both / max(1, total_users)) * 100.0:.1f}%)"
    )
    print(f"estimated_pairs(max_pairs={DEFAULT_MAX_PAIRS_PER_USER})={total_est_pairs:,}")

    ready = total_users_with_both > 0 and total_est_pairs > 0
    if ready:
        print("[OK] dataset is usable for DPO training")
    else:
        print("[FAIL] dataset is not usable for DPO training")
    return ready


def main() -> None:
    try:
        dataset_run = resolve_dataset_run(sys.argv)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("usage: python tools/check_dpo_dataset.py [<stage11_dataset_run_dir>]")
        raise SystemExit(1)

    ok = check_dataset_for_dpo(dataset_run)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
