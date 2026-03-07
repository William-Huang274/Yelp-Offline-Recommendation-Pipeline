import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_INPUT = Path(
    r"D:/5006 BDA project/data/output/07_baseline_relabel_vector/"
    r"20260209_140517_sample_baseline200/baseline_200_relabel_for_review.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:/5006 BDA project/data/output/07_baseline_relabel_vector/"
    r"20260209_140517_sample_baseline200/baseline_200_relabel_autolabeled.csv"
)
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT = 120
DEFAULT_SNIPPET_CHARS = 850

ALLOWED_L1 = ["food_service", "food_retail", "non_food", "uncertain"]
ALLOWED_L2 = [
    "restaurants_general",
    "coffee_tea",
    "bakery",
    "dessert",
    "ice_cream_frozen_yogurt",
    "juice_smoothies",
    "food_trucks",
    "fast_food",
    "pizza",
    "sandwiches",
    "burgers",
    "seafood",
    "cajun_creole",
    "mexican",
    "chinese",
    "japanese_sushi",
    "vietnamese",
    "italian",
    "bbq",
    "breakfast_brunch",
    "other_service",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label baseline rows using local Ollama.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS)
    p.add_argument("--overwrite-gold", action="store_true")
    return p.parse_args()


def clean_text(x: Any, limit: int = 1200) -> str:
    s = re.sub(r"\s+", " ", str("" if x is None else x)).strip()
    return s[:limit]


def extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def clamp_conf(v: Any) -> int:
    try:
        x = int(float(v))
    except Exception:
        x = 2
    return max(1, min(5, x))


def normalize_token(x: Any) -> str:
    return clean_text(x, 80).lower()


def build_prompt(row: pd.Series, snippet_chars: int) -> str:
    name = clean_text(row.get("name", ""), 120)
    city = clean_text(row.get("city", ""), 120)
    categories = clean_text(row.get("categories", ""), 400)
    snippet = clean_text(row.get("review_snippet", ""), snippet_chars)

    return (
        "You are a strict taxonomy labeler for Yelp businesses.\n"
        "Task: choose one L1 and one L2 label from allowed lists.\n"
        "Do not invent labels.\n"
        "Prefer category evidence first, then review evidence.\n"
        "If evidence is weak/contradictory, set can_decide='no'.\n"
        "If L1 is not food_service, set l2_gold='restaurants_general'.\n"
        "Return JSON only with keys:\n"
        "l1_gold, l2_gold, gold_reason, can_decide, confidence\n\n"
        f"Allowed L1: {', '.join(ALLOWED_L1)}\n"
        f"Allowed L2: {', '.join(ALLOWED_L2)}\n\n"
        f"Business name: {name}\n"
        f"City: {city}\n"
        f"Categories: {categories}\n"
        f"Review snippet: {snippet}\n"
    )


def call_ollama(url: str, model: str, prompt: str, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 220,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    raw = str(data.get("response") or "").strip()
    return extract_json(raw)


def validate_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    l1 = normalize_token(pred.get("l1_gold", ""))
    l2 = normalize_token(pred.get("l2_gold", ""))
    reason = clean_text(pred.get("gold_reason", ""), 450)
    can_decide = normalize_token(pred.get("can_decide", ""))
    conf = clamp_conf(pred.get("confidence", 2))

    if l1 not in ALLOWED_L1:
        l1 = "uncertain"
    if l2 not in ALLOWED_L2:
        l2 = "restaurants_general"
    if can_decide not in {"yes", "no"}:
        can_decide = "yes" if conf >= 3 else "no"
    if not reason:
        reason = "Auto-labeled by Ollama from categories and snippet."
    if l1 != "food_service":
        l2 = "restaurants_general"
    return {
        "l1_gold_auto": l1,
        "l2_gold_auto": l2,
        "gold_reason_auto": reason,
        "can_decide_auto": can_decide,
        "auto_confidence": conf,
    }


def fallback_from_row(row: pd.Series) -> dict[str, Any]:
    l1 = normalize_token(row.get("l1_label", "uncertain"))
    l1 = l1 if l1 in ALLOWED_L1 else "uncertain"
    l2 = normalize_token(row.get("l2_label_recalc", row.get("l2_label_top1", "")))
    l2 = l2 if l2 in ALLOWED_L2 else "restaurants_general"
    return {
        "l1_gold_auto": l1,
        "l2_gold_auto": l2,
        "gold_reason_auto": "Fallback from machine relabel due to Ollama parse/request failure.",
        "can_decide_auto": "no",
        "auto_confidence": 2,
    }


def merge_into_gold(df: pd.DataFrame, overwrite_gold: bool) -> pd.DataFrame:
    out = df.copy()
    if overwrite_gold:
        out["l1_gold"] = out["l1_gold_auto"]
        out["l2_gold"] = out["l2_gold_auto"]
        out["gold_reason"] = out["gold_reason_auto"]
        out["can_decide"] = out["can_decide_auto"]
        return out

    def is_blank(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().eq("")

    m1 = is_blank(out.get("l1_gold", pd.Series([""] * len(out))))
    m2 = is_blank(out.get("l2_gold", pd.Series([""] * len(out))))
    m3 = is_blank(out.get("gold_reason", pd.Series([""] * len(out))))
    m4 = is_blank(out.get("can_decide", pd.Series([""] * len(out))))

    out.loc[m1, "l1_gold"] = out.loc[m1, "l1_gold_auto"]
    out.loc[m2, "l2_gold"] = out.loc[m2, "l2_gold_auto"]
    out.loc[m3, "gold_reason"] = out.loc[m3, "gold_reason_auto"]
    out.loc[m4, "can_decide"] = out.loc[m4, "can_decide_auto"]
    return out


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_csv = Path(args.output)
    if not input_csv.exists():
        raise FileNotFoundError(f"input not found: {input_csv}")

    df = pd.read_csv(input_csv)
    rows: list[dict[str, Any]] = []
    n = len(df)
    print(f"[INFO] input rows={n}")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        prompt = build_prompt(row, args.snippet_chars)
        try:
            pred = call_ollama(args.url, args.model, prompt, args.timeout)
            rec = validate_prediction(pred)
            rec["auto_status"] = "ok"
        except Exception as exc:
            rec = fallback_from_row(row)
            rec["auto_status"] = f"fallback:{clean_text(str(exc), 220)}"
        rows.append(rec)
        if i % 20 == 0 or i == n:
            print(f"[INFO] processed {i}/{n}")

    auto_df = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), auto_df], axis=1)
    out["needs_human_review_auto"] = (
        (out["can_decide_auto"].fillna("").astype(str).str.lower() == "no")
        | (pd.to_numeric(out["auto_confidence"], errors="coerce").fillna(0) <= 2)
    )
    out = merge_into_gold(out, overwrite_gold=args.overwrite_gold)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] wrote: {output_csv}")
    print(
        "[INFO] stats: "
        f"needs_human_review_auto={int(out['needs_human_review_auto'].sum())}/{len(out)}; "
        f"auto_status_ok={int((out['auto_status'] == 'ok').sum())}/{len(out)}"
    )


if __name__ == "__main__":
    main()
