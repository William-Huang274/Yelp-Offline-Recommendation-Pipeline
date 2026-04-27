#!/usr/bin/env python3
"""Analyze prompt length distribution for Stage11 dataset runs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE11_ROOT = Path(
    os.getenv("INPUT_11_ROOT_DIR", str(REPO_ROOT / "data" / "output" / "11_qlora_data"))
).expanduser()
DEFAULT_TOKENIZER = os.getenv("PROMPT_LENGTH_TOKENIZER", os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-4B"))


def resolve_dataset_run(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser()
    if not DEFAULT_STAGE11_ROOT.exists():
        raise FileNotFoundError(f"stage11 data root not found: {DEFAULT_STAGE11_ROOT}")
    runs = [p for p in DEFAULT_STAGE11_ROOT.iterdir() if p.is_dir() and "stage11_1" in p.name]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no stage11_1 run found under {DEFAULT_STAGE11_ROOT}")
    return runs[0]


def iter_train_json_files(dataset_run: Path) -> list[Path]:
    files: list[Path] = []
    for bucket_dir in sorted(dataset_run.glob("bucket_*")):
        files.extend(sorted((bucket_dir / "train_json").glob("*.json")))
    return files


def load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[WARN] tokenizer load failed ({tokenizer_name}): {exc}")
        return None


def estimate_tokens_by_chars(text: str) -> int:
    return max(1, int(len(text) * 0.5))


def main() -> int:
    try:
        dataset_run = resolve_dataset_run(sys.argv)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("usage: python tools/stage/analyze_stage11_prompt_length.py [<stage11_dataset_run_dir>]")
        return 1

    tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
    token_lengths: list[int] = []
    char_lengths: list[int] = []

    for json_file in iter_train_json_files(dataset_run):
        with json_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = str(row.get("prompt", ""))
                if not prompt:
                    continue
                char_lengths.append(len(prompt))
                if tokenizer is not None:
                    token_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=True)))
                else:
                    token_lengths.append(estimate_tokens_by_chars(prompt))

    if not token_lengths:
        print("FAIL prompt_length no prompt rows found")
        return 2

    arr = np.asarray(token_lengths, dtype=np.int32)
    carr = np.asarray(char_lengths, dtype=np.int32)

    print("Prompt Length Audit")
    print(f"dataset_run: {dataset_run}")
    print(f"samples={len(arr):,}")
    print(f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f} median={np.median(arr):.1f}")
    for p in (50, 75, 90, 95, 99):
        print(f"p{p}={np.percentile(arr, p):.0f}")
    print("max_seq_len coverage truncated")
    for seq in (256, 384, 512, 640, 768, 1024, 1280):
        covered = int((arr <= seq).sum())
        print(f"{seq:>11} {covered / len(arr) * 100:>7.2f}% {len(arr) - covered:>9,}")
    print(f"char_mean={carr.mean():.1f} char_median={np.median(carr):.1f} char_max={carr.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
