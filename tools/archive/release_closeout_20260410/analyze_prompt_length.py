#!/usr/bin/env python3
"""Analyze prompt length distribution for Stage11 QLoRA datasets."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE11_ROOT = Path(
    os.getenv("INPUT_11_ROOT_DIR", str(PROJECT_ROOT / "data" / "output" / "11_qlora_data"))
).expanduser()
DEFAULT_TOKENIZER = os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-4B")


def resolve_dataset_run(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser()

    if not DEFAULT_STAGE11_ROOT.exists():
        raise FileNotFoundError(f"stage11 data root not found: {DEFAULT_STAGE11_ROOT}")

    runs = [p for p in DEFAULT_STAGE11_ROOT.iterdir() if p.is_dir() and "stage11_1" in p.name]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(
            f"no stage11_1 dataset run found under {DEFAULT_STAGE11_ROOT}"
        )
    return runs[0]


def iter_train_json_files(dataset_run: Path) -> list[Path]:
    bucket_dirs = sorted(
        [p for p in dataset_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")],
        key=lambda p: p.name,
    )
    files: list[Path] = []
    for b in bucket_dirs:
        files.extend(sorted((b / "train_json").glob("*.json")))
    return files


def load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return tok
    except Exception as exc:
        print(f"[WARN] tokenizer load failed ({tokenizer_name}): {exc}")
        return None


def estimate_tokens_by_chars(text: str) -> int:
    # Conservative fallback when tokenizer is unavailable.
    return max(1, int(len(text) * 0.5))


def analyze_prompt_lengths(dataset_run: Path, tokenizer_name: str) -> None:
    print("=" * 72)
    print("Prompt Length Audit")
    print("=" * 72)
    print(f"dataset_run: {dataset_run}")

    train_files = iter_train_json_files(dataset_run)
    if not train_files:
        raise FileNotFoundError(f"no train_json/*.json found under {dataset_run}")

    tokenizer = load_tokenizer(tokenizer_name)
    use_tokenizer = tokenizer is not None

    token_lengths: list[int] = []
    char_lengths: list[int] = []

    for jf in train_files:
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = str(row.get("prompt", ""))
                if not prompt:
                    continue

                char_lengths.append(len(prompt))
                if use_tokenizer:
                    token_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=True)))
                else:
                    token_lengths.append(estimate_tokens_by_chars(prompt))

    if not token_lengths:
        raise RuntimeError("no prompt rows found")

    arr = np.asarray(token_lengths, dtype=np.int32)
    carr = np.asarray(char_lengths, dtype=np.int32)

    print(f"samples={len(arr):,}")
    print(f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f} median={np.median(arr):.1f}")
    for p in (50, 75, 90, 95, 99):
        print(f"p{p}={np.percentile(arr, p):.0f}")

    print("-" * 72)
    print("Coverage by max_seq_len")
    print("max_seq_len  coverage  truncated")
    for seq in (256, 384, 512, 640, 768, 1024):
        covered = int((arr <= seq).sum())
        coverage = covered / len(arr) * 100.0
        truncated = len(arr) - covered
        print(f"{seq:>10}  {coverage:>7.2f}%  {truncated:>9,}")

    print("-" * 72)
    print(f"char_mean={carr.mean():.1f} char_median={np.median(carr):.1f} char_max={carr.max()}")


def main() -> None:
    try:
        dataset_run = resolve_dataset_run(sys.argv)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("usage: python tools/analyze_prompt_length.py [<stage11_dataset_run_dir>]")
        raise SystemExit(1)

    tokenizer_name = os.getenv("PROMPT_LENGTH_TOKENIZER", DEFAULT_TOKENIZER)
    analyze_prompt_lengths(dataset_run, tokenizer_name)


if __name__ == "__main__":
    main()
