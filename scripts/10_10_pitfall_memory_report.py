from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(r"D:/5006 BDA project")
PITFALLS_JSONL = PROJECT_ROOT / "data/metrics/training_pitfalls_memory.jsonl"
TOP_N = 30


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    rows = load_rows(PITFALLS_JSONL)
    print(f"[INFO] path={PITFALLS_JSONL}")
    print(f"[INFO] rows={len(rows)}")
    if not rows:
        return

    ev = Counter(str(r.get("event", "")) for r in rows)
    sev = Counter(str(r.get("severity", "")) for r in rows)
    pair = Counter((str(r.get("bucket", "")), str(r.get("event", ""))) for r in rows)

    print("\n[SUMMARY] events")
    for k, v in ev.most_common():
        print(f"  {k}: {v}")

    print("\n[SUMMARY] severity")
    for k, v in sev.most_common():
        print(f"  {k}: {v}")

    print("\n[SUMMARY] top repeated bucket-event")
    for (b, e), v in pair.most_common(12):
        print(f"  bucket={b} event={e}: {v}")

    print(f"\n[LATEST] last {min(TOP_N, len(rows))} records")
    for r in rows[-TOP_N:]:
        ts = r.get("timestamp", "")
        b = r.get("bucket", "")
        event = r.get("event", "")
        sev_v = r.get("severity", "")
        msg = r.get("message", "")
        round_id = r.get("round", "")
        cfg = r.get("cfg_name", "")
        print(f"  {ts} | b={b} | round={round_id} | {event} | {sev_v} | cfg={cfg} | {msg}")


if __name__ == "__main__":
    main()

