#!/usr/bin/env python3
"""Tail a Stage11 training or evaluation log and extract recent metric lines."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


METRIC_PATTERNS = [
    re.compile(r"eval_listwise_win_rate", re.I),
    re.compile(r"eval_rescue_31_40_true_win_rate", re.I),
    re.compile(r"eval_rescue_41_60_true_win_rate", re.I),
    re.compile(r"eval_rescue_61_100_true_win_rate", re.I),
    re.compile(r"QLoRASidecar@10", re.I),
    re.compile(r"QLoRASidecarJoint@10", re.I),
    re.compile(r"joint_recall_at_k", re.I),
    re.compile(r"qlora_recall_at_k", re.I),
]


def tail_lines(path: Path, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-limit:]


def extract_signal(lines: list[str]) -> list[str]:
    matched: list[str] = []
    for line in lines:
        if any(p.search(line) for p in METRIC_PATTERNS):
            matched.append(line)
    return matched[-20:]


def display(path: Path) -> None:
    print("\033[2J\033[H", end="")
    print("=" * 72)
    print("Stage11 Training / Eval Monitor")
    print("=" * 72)
    print(f"log: {path}")
    print()
    if not path.exists():
        print("log file not found")
        return
    lines = tail_lines(path)
    signals = extract_signal(lines)
    if signals:
        print("Recent metric lines:")
        for line in signals:
            print(line)
    else:
        print("No tracked metric lines found in the recent log tail.")
    print()
    print("Last raw lines:")
    for line in lines[-10:]:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a Stage11 training or eval log.")
    parser.add_argument("log_path", help="Path to the log file.")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds.")
    args = parser.parse_args()

    path = Path(args.log_path).expanduser()
    try:
        while True:
            display(path)
            time.sleep(max(1, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
