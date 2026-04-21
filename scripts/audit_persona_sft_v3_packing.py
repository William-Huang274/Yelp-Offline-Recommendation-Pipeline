from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This audit script is configured by environment variables and may load tokenizer dependencies.")
    print("Set the required INPUT_/OUTPUT_/PERSONA_SFT_* environment variables, then run without --help.")
    sys.exit(0)

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from pipeline.project_paths import env_or_project_path, project_path

INPUT_ROOT = env_or_project_path("INPUT_11_PERSONA_SFT_DATA_ROOT_DIR", "data/output/11_persona_sft_data")
INPUT_RUN_DIR = os.getenv("INPUT_11_PERSONA_SFT_DATA_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_PERSONA_SFT_PACKING_AUDIT_ROOT_DIR",
    "data/output/11_persona_sft_packing_audit",
)
RUN_TAG = os.getenv("RUN_TAG", "stage11_persona_sft_v3_packing_audit").strip() or "stage11_persona_sft_v3_packing_audit"
BASE_MODEL = os.getenv("PERSONA_SFT_BASE_MODEL", "/root/hf_models/Qwen3.5-9B").strip() or "/root/hf_models/Qwen3.5-9B"

LOW_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_LOW_TOTAL_CAP", "18").strip() or 18)
LOW_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_LOW_TIP_CAP", "4").strip() or 4)
LOW_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_LOW_NEG_FLOOR", "2").strip() or 2)

MID_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_MID_TOTAL_CAP", "24").strip() or 24)
MID_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_MID_TIP_CAP", "6").strip() or 6)
MID_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_MID_NEG_FLOOR", "3").strip() or 3)

HIGH_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_HIGH_TOTAL_CAP", "36").strip() or 36)
HIGH_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_HIGH_TIP_CAP", "12").strip() or 12)
HIGH_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_HIGH_NEG_FLOOR", "4").strip() or 4)

TOKEN_BUDGETS = [int(token) for token in (os.getenv("PERSONA_SFT_PACK_TOKEN_BUDGETS", "3072,3584,4096") or "").split(",") if token.strip()]

USER_EVIDENCE_HEADER = "User evidence stream:"
BEHAVIOR_STATS_HEADER = "Behavior stats:"
SEQUENCE_HEADER = "Recent event sequence:"


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def extract_section(prompt: str, start_marker: str, end_marker: str) -> tuple[str, str, str]:
    start = prompt.find(start_marker)
    end = prompt.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        return prompt, "", ""
    prefix = prompt[:start]
    section = prompt[start:end]
    suffix = prompt[end:]
    return prefix, section, suffix


def parse_evidence_items(section: str) -> list[dict[str, Any]]:
    lines = [line for line in section.splitlines()[1:] if line.strip().startswith("- ")]
    items: list[dict[str, Any]] = []
    for line in lines:
        body = line[2:]
        parts = body.split("; text=", 1)
        if len(parts) != 2:
            continue
        meta_part, text = parts
        meta: dict[str, str] = {}
        for token in meta_part.split("; "):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            meta[key.strip()] = value.strip()
        try:
            weight = float(meta.get("weight", "0") or 0)
        except Exception:
            weight = 0.0
        items.append(
            {
                "raw_line": line,
                "id": meta.get("id", ""),
                "source": meta.get("source", "unknown"),
                "sentiment": meta.get("sentiment", "unknown"),
                "time_bucket": meta.get("time_bucket", "unknown"),
                "weight": weight,
                "text": text.strip(),
            }
        )
    return items


def format_evidence_items(items: list[dict[str, Any]]) -> str:
    lines = [USER_EVIDENCE_HEADER]
    if not items:
        lines.append("- none")
        return "\n".join(lines)
    for item in items:
        lines.append(item["raw_line"])
    return "\n".join(lines)


def density_cfg(density_band: str) -> dict[str, int]:
    if density_band == "b5_high_visible_18_plus":
        return {"total_cap": HIGH_TOTAL_CAP, "tip_cap": HIGH_TIP_CAP, "neg_floor": HIGH_NEG_FLOOR}
    if density_band == "b5_mid_visible_8_17":
        return {"total_cap": MID_TOTAL_CAP, "tip_cap": MID_TIP_CAP, "neg_floor": MID_NEG_FLOOR}
    return {"total_cap": LOW_TOTAL_CAP, "tip_cap": LOW_TIP_CAP, "neg_floor": LOW_NEG_FLOOR}


def pack_items(items: list[dict[str, Any]], density_band: str) -> list[dict[str, Any]]:
    cfg = density_cfg(density_band)
    total_cap = cfg["total_cap"]
    tip_cap = cfg["tip_cap"]
    neg_floor = cfg["neg_floor"]
    if len(items) <= total_cap:
        return items

    negatives = [item for item in items if item["sentiment"] == "negative" and item["source"] == "review"]
    negatives = negatives[:neg_floor]
    selected_ids = {item["id"] for item in negatives}
    packed: list[dict[str, Any]] = list(negatives)
    tip_count = sum(1 for item in packed if item["source"] == "tip")

    for item in items:
        if item["id"] in selected_ids:
            continue
        if len(packed) >= total_cap:
            break
        if item["source"] == "tip" and tip_count >= tip_cap:
            continue
        packed.append(item)
        selected_ids.add(item["id"])
        if item["source"] == "tip":
            tip_count += 1
    return packed


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_full_text(system_prompt: str, prompt_text: str, target_text: str, eos_token: str) -> str:
    return (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{prompt_text}\n"
        "<|assistant|>\n"
        f"{target_text}{eos_token or ''}"
    )


def apply_packing(prompt_text: str, density_band: str) -> tuple[str, int, int, int]:
    prefix, section, suffix = extract_section(prompt_text, USER_EVIDENCE_HEADER, BEHAVIOR_STATS_HEADER)
    items = parse_evidence_items(section)
    packed = pack_items(items, density_band)
    new_prompt = prefix + format_evidence_items(packed) + "\n\n" + suffix.lstrip("\n")
    return new_prompt, len(items), len(packed), sum(1 for item in packed if item["source"] == "tip")


def summarize_by_band(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for band, sub in df.groupby("density_band", sort=True):
        out.append(
            {
                "density_band": band,
                "rows": int(len(sub)),
                f"{key}_mean": float(sub[key].mean()),
                f"{key}_p50": float(sub[key].quantile(0.5)),
                f"{key}_p90": float(sub[key].quantile(0.9)),
                f"{key}_p95": float(sub[key].quantile(0.95)),
                f"{key}_max": float(sub[key].max()),
            }
        )
    return out


def main() -> None:
    input_run = resolve_run(INPUT_RUN_DIR, INPUT_ROOT, "_full_stage11_1_persona_sft_v3_build_dataset")
    train_df = pd.read_parquet(input_run / "train.parquet")
    eval_df = pd.read_parquet(input_run / "eval.parquet")
    dataset = pd.concat([train_df, eval_df], ignore_index=True)
    if dataset.empty:
        raise RuntimeError("persona sft v3 dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    eos_token = tokenizer.eos_token or ""

    rows: list[dict[str, Any]] = []
    for row in dataset.to_dict(orient="records"):
        orig_prompt = row["prompt_text"]
        packed_prompt, orig_items, packed_items, packed_tip_items = apply_packing(orig_prompt, row["density_band"])
        orig_full = build_full_text(row["system_prompt"], orig_prompt, row["target_text"], eos_token)
        packed_full = build_full_text(row["system_prompt"], packed_prompt, row["target_text"], eos_token)
        rows.append(
            {
                "user_id": row["user_id"],
                "split": row.get("split", "unknown"),
                "density_band": row["density_band"],
                "orig_prompt_tokens": token_count(tokenizer, orig_prompt),
                "orig_full_tokens": token_count(tokenizer, orig_full),
                "packed_prompt_tokens": token_count(tokenizer, packed_prompt),
                "packed_full_tokens": token_count(tokenizer, packed_full),
                "orig_evidence_items": orig_items,
                "packed_evidence_items": packed_items,
                "packed_tip_items": packed_tip_items,
                "packed_changed": int(orig_items != packed_items),
            }
        )

    out_df = pd.DataFrame(rows)
    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(run_dir / "packing_rows.parquet", index=False)

    summary = {
        "rows_total": int(len(out_df)),
        "config": {
            "low": {"total_cap": LOW_TOTAL_CAP, "tip_cap": LOW_TIP_CAP, "neg_floor": LOW_NEG_FLOOR},
            "mid": {"total_cap": MID_TOTAL_CAP, "tip_cap": MID_TIP_CAP, "neg_floor": MID_NEG_FLOOR},
            "high": {"total_cap": HIGH_TOTAL_CAP, "tip_cap": HIGH_TIP_CAP, "neg_floor": HIGH_NEG_FLOOR},
        },
        "orig_full_tokens_mean": float(out_df["orig_full_tokens"].mean()),
        "packed_full_tokens_mean": float(out_df["packed_full_tokens"].mean()),
        "orig_full_tokens_p95": float(out_df["orig_full_tokens"].quantile(0.95)),
        "packed_full_tokens_p95": float(out_df["packed_full_tokens"].quantile(0.95)),
        "orig_full_tokens_max": int(out_df["orig_full_tokens"].max()),
        "packed_full_tokens_max": int(out_df["packed_full_tokens"].max()),
        "packed_changed_rate": float(out_df["packed_changed"].mean()),
    }

    budget_check = []
    for budget in TOKEN_BUDGETS:
        budget_check.append(
            {
                "token_budget": int(budget),
                "orig_overflow_rate": float((out_df["orig_full_tokens"] > budget).mean()),
                "packed_overflow_rate": float((out_df["packed_full_tokens"] > budget).mean()),
            }
        )
    summary["budget_check"] = budget_check
    summary["orig_by_band"] = summarize_by_band(rows, "orig_full_tokens")
    summary["packed_by_band"] = summarize_by_band(rows, "packed_full_tokens")
    summary["packed_item_by_band"] = summarize_by_band(rows, "packed_evidence_items")

    gate = {
        "min_packed_items_lower": float(
            out_df.loc[out_df["density_band"].eq("b5_lower_visible_5_7"), "packed_evidence_items"].quantile(0.5)
        ),
        "min_packed_items_mid": float(
            out_df.loc[out_df["density_band"].eq("b5_mid_visible_8_17"), "packed_evidence_items"].quantile(0.5)
        ),
        "min_packed_items_high": float(
            out_df.loc[out_df["density_band"].eq("b5_high_visible_18_plus"), "packed_evidence_items"].quantile(0.5)
        ),
        "audit_pass": bool(
            float(out_df.loc[out_df["density_band"].eq("b5_lower_visible_5_7"), "packed_evidence_items"].quantile(0.5)) >= 9
            and float(out_df.loc[out_df["density_band"].eq("b5_mid_visible_8_17"), "packed_evidence_items"].quantile(0.5)) >= 12
            and float(out_df.loc[out_df["density_band"].eq("b5_high_visible_18_plus"), "packed_evidence_items"].quantile(0.5)) >= 24
        ),
    }

    safe_json_write(run_dir / "summary.json", summary)
    safe_json_write(run_dir / "gate.json", gate)
    safe_json_write(run_dir / "run_meta.json", {"input_run_dir": str(input_run), "output_run_dir": str(run_dir), "summary": summary, "gate": gate})
    print(json.dumps({"summary": summary, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
