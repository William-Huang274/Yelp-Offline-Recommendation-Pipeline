from __future__ import annotations

import argparse
import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script} [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]")
    print("This stage script starts merchant-profile inference and may load tokenizer/model dependencies.")
    print("Set the required environment variables or pass explicit paths, then run without --help.")
    sys.exit(0)

import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from pipeline.project_paths import env_or_project_path, resolve_latest_run_pointer, write_latest_run_pointer


RUN_TAG = "stage12_merchant_profile_infer"
DEFAULT_INPUT_ROOT = env_or_project_path("INPUT_12_MERCHANT_PROFILE_INPUT_ROOT_DIR", "data/output/12_merchant_profile_inputs")
DEFAULT_OUTPUT_ROOT = env_or_project_path("OUTPUT_12_MERCHANT_PROFILE_INFER_ROOT_DIR", "data/output/12_merchant_profile_infer")
BASE_MODEL = os.getenv(
    "MERCHANT_PROFILE_BASE_MODEL",
    os.getenv("PROMPT_ONLY_BASE_MODEL", "/root/autodl-tmp/models/Qwen3.5-35B-A3B-Base"),
).strip()

BATCH_SIZE = int(os.getenv("MERCHANT_PROFILE_GEN_BATCH_SIZE", "8").strip() or 8)
MAX_BATCH_PROMPT_TOKENS = int(os.getenv("MERCHANT_PROFILE_MAX_BATCH_PROMPT_TOKENS", "110000").strip() or 110000)
MAX_NEW_TOKENS = int(os.getenv("MERCHANT_PROFILE_GEN_MAX_NEW_TOKENS", "3072").strip() or 3072)
SAMPLE_MAX_ROWS = int(os.getenv("MERCHANT_PROFILE_INFER_SAMPLE_MAX_ROWS", "0").strip() or 0)
CHECKPOINT_EVERY_BATCHES = int(os.getenv("MERCHANT_PROFILE_CHECKPOINT_EVERY_BATCHES", "4").strip() or 4)
RESUME_FROM_EXISTING = os.getenv("MERCHANT_PROFILE_RESUME_FROM_EXISTING", "true").strip().lower() == "true"
SORT_BY_PROMPT_LENGTH = os.getenv("MERCHANT_PROFILE_SORT_BY_PROMPT_LENGTH", "true").strip().lower() == "true"
PERSISTENT_PHASES_RAW = os.getenv("MERCHANT_PROFILE_PERSISTENT_PHASES", "").strip()

VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("MERCHANT_PROFILE_VLLM_GPU_MEMORY_UTILIZATION", "0.95").strip() or 0.95)
VLLM_MAX_MODEL_LEN = int(os.getenv("MERCHANT_PROFILE_VLLM_MAX_MODEL_LEN", "32768").strip() or 32768)
VLLM_MAX_NUM_SEQS = int(os.getenv("MERCHANT_PROFILE_VLLM_MAX_NUM_SEQS", str(BATCH_SIZE)).strip() or BATCH_SIZE)
VLLM_MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("MERCHANT_PROFILE_VLLM_MAX_NUM_BATCHED_TOKENS", str(MAX_BATCH_PROMPT_TOKENS)).strip()
    or MAX_BATCH_PROMPT_TOKENS
)
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("MERCHANT_PROFILE_VLLM_TENSOR_PARALLEL_SIZE", "1").strip() or 1)
VLLM_ENFORCE_EAGER = os.getenv("MERCHANT_PROFILE_VLLM_ENFORCE_EAGER", "false").strip().lower() == "true"
REQUIRE_GUIDED_DECODING = os.getenv("MERCHANT_PROFILE_REQUIRE_GUIDED_DECODING", "true").strip().lower() == "true"
RETRY_TRUNCATED_ROWS = os.getenv("MERCHANT_PROFILE_RETRY_TRUNCATED_ROWS", "true").strip().lower() == "true"
STRICT_SUPPORT_OVERLAP = os.getenv("MERCHANT_PROFILE_STRICT_SUPPORT_OVERLAP", "true").strip().lower() == "true"
STRICT_RISK_POLARITY = os.getenv("MERCHANT_PROFILE_STRICT_RISK_POLARITY", "true").strip().lower() == "true"
RETRY_MAX_NEW_TOKENS = int(
    os.getenv("MERCHANT_PROFILE_RETRY_MAX_NEW_TOKENS", str(MAX_NEW_TOKENS + 1024)).strip()
    or (MAX_NEW_TOKENS + 1024)
)
CANDIDATE_RECALL_MODE = os.getenv("MERCHANT_PROFILE_CANDIDATE_RECALL_MODE", "true").strip().lower() == "true"
DIRECT_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_DIRECT_MAX_ITEMS", "5" if CANDIDATE_RECALL_MODE else "5").strip()
    or 5
)
FINE_GRAINED_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_FINE_GRAINED_MAX_ITEMS", "12" if CANDIDATE_RECALL_MODE else "8").strip()
    or (12 if CANDIDATE_RECALL_MODE else 8)
)
USAGE_SCENE_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_USAGE_SCENE_MAX_ITEMS", "6" if CANDIDATE_RECALL_MODE else "4").strip()
    or (6 if CANDIDATE_RECALL_MODE else 4)
)
RISK_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_RISK_MAX_ITEMS", "6" if CANDIDATE_RECALL_MODE else "4").strip()
    or (6 if CANDIDATE_RECALL_MODE else 4)
)
CAUTIOUS_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_CAUTIOUS_MAX_ITEMS", "5" if CANDIDATE_RECALL_MODE else "3").strip()
    or (5 if CANDIDATE_RECALL_MODE else 3)
)
REFS_MAX_ITEMS = int(os.getenv("MERCHANT_PROFILE_REFS_MAX_ITEMS", "3").strip() or 3)
SNIPPETS_MAX_ITEMS = int(os.getenv("MERCHANT_PROFILE_SNIPPETS_MAX_ITEMS", "2").strip() or 2)


@dataclass
class MerchantPromptRecord:
    input_index: int
    business_id: str
    name: str
    merchant_sampling_segment: str
    selected_review_evidence_count: int
    selected_tip_evidence_count: int
    prompt_text: str
    prompt_tokens: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_input_dir(raw: Path | None) -> Path:
    if raw is not None:
        return raw
    env_raw = os.getenv("INPUT_12_MERCHANT_PROFILE_INPUT_RUN_DIR", "").strip()
    if env_raw:
        return Path(env_raw)
    pointer = resolve_latest_run_pointer("12_merchant_profile_inputs")
    if pointer is not None:
        return pointer
    runs = [path for path in DEFAULT_INPUT_ROOT.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no merchant profile input run found under {DEFAULT_INPUT_ROOT}")
    return runs[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def extract_first_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(raw)):
        ch = raw[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : pos + 1]
    return raw[start:]


def output_schema() -> dict[str, Any]:
    short_text = {"type": "string", "maxLength": 180}
    ref_array = {
        "type": "array",
        "items": {"type": "string", "maxLength": 24},
        "maxItems": REFS_MAX_ITEMS,
        "default": [],
    }
    snippet_array = {
        "type": "array",
        "items": {"type": "string", "maxLength": 180},
        "minItems": 1,
        "maxItems": SNIPPETS_MAX_ITEMS,
        "default": [],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "merchant_summary",
            "direct_evidence_signals",
            "fine_grained_attributes",
            "usage_scenes",
            "risk_or_avoidance_notes",
            "cautious_inferences",
            "audit_notes",
        ],
        "properties": {
            "merchant_summary": {"type": "string", "maxLength": 480},
            "direct_evidence_signals": {
                "type": "array",
                "maxItems": DIRECT_MAX_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["signal", "polarity", "support_basis", "evidence_refs", "evidence_snippets"],
                    "properties": {
                        "signal": short_text,
                        "polarity": {"type": "string", "enum": ["positive", "mixed", "negative", "neutral"]},
                        "support_basis": {"type": "string", "enum": ["review", "tip", "review_and_tip"]},
                        "evidence_refs": ref_array,
                        "evidence_snippets": snippet_array,
                    },
                },
            },
            "fine_grained_attributes": {
                "type": "array",
                "maxItems": FINE_GRAINED_MAX_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["attribute", "attribute_type", "evidence_refs", "evidence_snippets"],
                    "properties": {
                        "attribute": short_text,
                        "attribute_type": {
                            "type": "string",
                            "enum": ["dish", "service", "ambience", "scene", "price_value", "operations", "other"],
                        },
                        "evidence_refs": ref_array,
                        "evidence_snippets": snippet_array,
                    },
                },
            },
            "usage_scenes": {
                "type": "array",
                "maxItems": USAGE_SCENE_MAX_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scene", "fit", "evidence_refs", "evidence_snippets"],
                    "properties": {
                        "scene": short_text,
                        "fit": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                        "evidence_refs": ref_array,
                        "evidence_snippets": snippet_array,
                    },
                },
            },
            "risk_or_avoidance_notes": {
                "type": "array",
                "maxItems": RISK_MAX_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["risk", "severity", "evidence_refs", "evidence_snippets"],
                    "properties": {
                        "risk": short_text,
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence_refs": ref_array,
                        "evidence_snippets": snippet_array,
                    },
                },
            },
            "cautious_inferences": {
                "type": "array",
                "maxItems": CAUTIOUS_MAX_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["inference", "reasoning", "support_basis", "evidence_refs", "evidence_snippets"],
                    "properties": {
                        "inference": short_text,
                        "reasoning": {"type": "string", "maxLength": 260},
                        "support_basis": {
                            "type": "string",
                            "enum": ["metadata", "aggregate", "metadata_and_evidence", "aggregate_and_evidence"],
                        },
                        "evidence_refs": ref_array,
                        "evidence_snippets": snippet_array,
                    },
                },
            },
            "audit_notes": {
                "type": "object",
                "additionalProperties": False,
                "required": ["unsupported_or_weak_claims", "evidence_coverage"],
                "properties": {
                    "unsupported_or_weak_claims": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 220},
                        "maxItems": 3,
                    },
                    "evidence_coverage": {"type": "string", "maxLength": 220},
                },
            },
        },
    }


def build_sampling_params(max_tokens_override: int | None = None) -> tuple[Any, str]:
    from vllm import SamplingParams

    schema = output_schema()
    max_tokens = MAX_NEW_TOKENS if max_tokens_override is None else int(max_tokens_override)
    base_kwargs = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "skip_special_tokens": True,
    }
    attempts: list[tuple[str, dict[str, Any]]] = []
    try:
        from vllm.sampling_params import StructuredOutputsParams

        attempts.append(("structured_outputs_json_schema", {**base_kwargs, "structured_outputs": StructuredOutputsParams(json=schema)}))
        attempts.append(
            (
                "structured_outputs_json_text",
                {**base_kwargs, "structured_outputs": StructuredOutputsParams(json=json.dumps(schema, ensure_ascii=False))},
            )
        )
    except Exception:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        attempts.append(("guided_decoding_json_schema", {**base_kwargs, "guided_decoding": GuidedDecodingParams(json=schema)}))
    except Exception:
        pass
    attempts.append(("guided_json_dict", {**base_kwargs, "guided_json": schema}))
    attempts.append(("guided_json_text", {**base_kwargs, "guided_json": json.dumps(schema, ensure_ascii=False)}))

    last_error: Exception | None = None
    for mode, kwargs in attempts:
        try:
            return SamplingParams(**kwargs), mode
        except Exception as exc:
            last_error = exc
    if REQUIRE_GUIDED_DECODING:
        raise RuntimeError("vLLM guided decoding is unavailable for this vLLM build") from last_error
    return SamplingParams(**base_kwargs), "unguided"


def load_records(input_dir: Path, tokenizer: Any, sample_max_rows: int | None = None) -> tuple[list[MerchantPromptRecord], pd.DataFrame]:
    path = input_dir / "merchant_profile_inputs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"merchant_profile_inputs.parquet not found under {input_dir}")
    df = pd.read_parquet(path).reset_index(drop=True)
    if sample_max_rows is None:
        sample_max_rows = SAMPLE_MAX_ROWS
    if sample_max_rows > 0 and len(df) > sample_max_rows:
        df = df.sort_values(
            by=["selected_review_evidence_count", "selected_tip_evidence_count", "prompt_tokens_char4_est"],
            ascending=[False, False, False],
        ).head(sample_max_rows).sort_index()
    records: list[MerchantPromptRecord] = []
    prompt_tokens: list[int] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        prompt = str(row.get("prompt_text", "") or "")
        if not prompt:
            raise ValueError(f"empty prompt_text at row {idx}")
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_tokens.append(token_count)
        records.append(
            MerchantPromptRecord(
                input_index=int(idx),
                business_id=str(row.get("business_id", "") or ""),
                name=str(row.get("name", "") or ""),
                merchant_sampling_segment=str(row.get("merchant_sampling_segment", "") or ""),
                selected_review_evidence_count=int(row.get("selected_review_evidence_count", 0) or 0),
                selected_tip_evidence_count=int(row.get("selected_tip_evidence_count", 0) or 0),
                prompt_text=prompt,
                prompt_tokens=int(token_count),
            )
        )
    df = df.reset_index(drop=True).copy()
    df["input_index"] = range(len(df))
    df["prompt_tokens"] = prompt_tokens
    return records, df


def build_batches(records: list[MerchantPromptRecord]) -> list[list[MerchantPromptRecord]]:
    ordered = sorted(records, key=lambda rec: rec.prompt_tokens, reverse=True) if SORT_BY_PROMPT_LENGTH else list(records)
    batches: list[list[MerchantPromptRecord]] = []
    current: list[MerchantPromptRecord] = []
    current_tokens = 0
    for rec in ordered:
        would_exceed_count = len(current) >= BATCH_SIZE
        would_exceed_tokens = current and current_tokens + rec.prompt_tokens > MAX_BATCH_PROMPT_TOKENS
        if would_exceed_count or would_exceed_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(rec)
        current_tokens += rec.prompt_tokens
    if current:
        batches.append(current)
    return batches


def allowed_refs_from_prompt(prompt: str) -> set[str]:
    return set(re.findall(r'"evidence_id"\s*:\s*"(m(?:rev|tip)_\d+)"', prompt or ""))


def normalize_for_support_match(value: Any) -> str:
    text = str(value or "").strip().strip("\"'`“”‘’")
    return re.sub(r"\s+", " ", text.strip()).lower()


SUPPORT_STOPWORDS = {
    "about",
    "also",
    "and",
    "another",
    "been",
    "being",
    "best",
    "both",
    "claim",
    "comes",
    "common",
    "could",
    "dining",
    "dish",
    "does",
    "food",
    "from",
    "general",
    "good",
    "great",
    "have",
    "into",
    "known",
    "like",
    "meal",
    "mention",
    "noted",
    "notes",
    "option",
    "options",
    "overall",
    "place",
    "popular",
    "praise",
    "praised",
    "quality",
    "restaurant",
    "review",
    "reviews",
    "scene",
    "served",
    "some",
    "spot",
    "standout",
    "strong",
    "that",
    "their",
    "there",
    "this",
    "though",
    "with",
}


def canonical_support_text(value: Any) -> str:
    text = normalize_for_support_match(value)
    replacements = {
        "po-boys": "poboy",
        "po boy": "poboy",
        "po' boys": "poboy",
        "po'boy": "poboy",
        "po-boys": "poboy",
        "charbroiled": "chargrilled",
        "char-grilled": "chargrilled",
        "char grilled": "chargrilled",
        "ambience": "atmosphere",
        "ambiance": "atmosphere",
        "vibe": "atmosphere",
        "staff": "service",
    }
    for raw, canonical in replacements.items():
        text = text.replace(raw, canonical)
    return text


def support_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", canonical_support_text(value)):
        if len(token) < 4 or token in SUPPORT_STOPWORDS:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if token and token not in SUPPORT_STOPWORDS:
            tokens.add(token)
    return tokens


def claim_text_for_direct_item(section: str, item: dict[str, Any]) -> str:
    if section == "direct_evidence_signals":
        return str(item.get("signal") or "")
    if section == "fine_grained_attributes":
        return str(item.get("attribute") or "")
    if section == "usage_scenes":
        return str(item.get("scene") or "")
    if section == "risk_or_avoidance_notes":
        return str(item.get("risk") or "")
    return ""


def section_item_text(section: str, item: dict[str, Any]) -> str:
    if section == "direct_evidence_signals":
        return str(item.get("signal") or "")
    if section == "fine_grained_attributes":
        return str(item.get("attribute") or "")
    if section == "usage_scenes":
        return str(item.get("scene") or "")
    if section == "risk_or_avoidance_notes":
        return str(item.get("risk") or "")
    if section == "cautious_inferences":
        return str(item.get("inference") or "")
    return ""


def item_quality_score(section: str, item: dict[str, Any]) -> tuple[int, int, int]:
    refs = item.get("evidence_refs")
    snippets = item.get("evidence_snippets")
    ref_count = len(refs) if isinstance(refs, list) else 0
    snippet_count = len(snippets) if isinstance(snippets, list) else 0
    return (ref_count, snippet_count, len(section_item_text(section, item)))


def dedupe_signature(section: str, item: dict[str, Any]) -> tuple[Any, ...]:
    text = section_item_text(section, item)
    token_key = " ".join(sorted(support_tokens(text)))
    if not token_key:
        token_key = canonical_support_text(text)
    return (
        section,
        token_key,
        str(item.get("polarity") or ""),
        str(item.get("attribute_type") or ""),
        str(item.get("fit") or ""),
        str(item.get("severity") or ""),
        str(item.get("support_basis") or ""),
    )


def dedupe_structured_items(parsed: Any) -> tuple[Any, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return parsed, actions
    sections = [
        "direct_evidence_signals",
        "fine_grained_attributes",
        "usage_scenes",
        "risk_or_avoidance_notes",
        "cautious_inferences",
    ]
    for section in sections:
        items = parsed.get(section)
        if not isinstance(items, list):
            continue
        kept_items: list[dict[str, Any]] = []
        signatures: dict[tuple[Any, ...], int] = {}
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                kept_items.append(item)
                continue
            signature = dedupe_signature(section, item)
            existing_idx = signatures.get(signature)
            if existing_idx is None:
                signatures[signature] = len(kept_items)
                kept_items.append(item)
                continue
            kept = kept_items[existing_idx]
            if item_quality_score(section, item) > item_quality_score(section, kept):
                kept_items[existing_idx] = item
                actions.append(
                    {
                        "type": "dedupe_replace_item",
                        "section": section,
                        "kept_index": existing_idx,
                        "dropped_index": idx,
                        "claim": section_item_text(section, item),
                    }
                )
            else:
                actions.append(
                    {
                        "type": "dedupe_drop_item",
                        "section": section,
                        "kept_index": existing_idx,
                        "dropped_index": idx,
                        "claim": section_item_text(section, item),
                    }
                )
        parsed[section] = kept_items
    return parsed, actions


def claim_supported_by_snippets(claim: str, snippets: list[str]) -> tuple[bool, dict[str, Any]]:
    claim_tokens = support_tokens(claim)
    snippet_tokens: set[str] = set()
    for snippet in snippets:
        snippet_tokens.update(support_tokens(snippet))
    if not claim_tokens:
        return True, {"claim_tokens": [], "snippet_tokens": sorted(snippet_tokens), "overlap": []}
    overlap = claim_tokens & snippet_tokens
    if STRICT_SUPPORT_OVERLAP:
        if len(claim_tokens) <= 2:
            required = 1
        elif len(claim_tokens) <= 8:
            required = 2
        else:
            required = 3
    elif len(claim_tokens) <= 4:
        required = 1
    elif len(claim_tokens) <= 8:
        required = 2
    else:
        required = 3
    ok = len(overlap) >= required
    return ok, {
        "claim_tokens": sorted(claim_tokens),
        "snippet_tokens": sorted(snippet_tokens),
        "overlap": sorted(overlap),
        "required_overlap": required,
    }


RISK_CUE_PATTERN = re.compile(
    r"\b("
    r"bad|bland|boring|cash[-\s]?only|cold|crowded|cramped|dense|dirty|dingy|"
    r"closed|disappoint(?:ing|ed)?|difficult|dry|expensive|forgettable|greasy|hard|inconsistent|"
    r"limited|line|long|loud|mediocre|neglectful|noisy|overcook(?:ed)?|pricey|"
    r"pushy|rude|salty|shocked|slippery|slow|small|spendy|stale|tiny|trouble|"
    r"undercook(?:ed)?|underwhelming|unclean|unavailable|valet|wait|weren'?t|wasn'?t|"
    r"without|worth|wet|won'?t|wouldn'?t"
    r")\b",
    re.IGNORECASE,
)


def risk_cues(value: Any) -> set[str]:
    text = str(value or "")
    cues = {match.group(1).lower().replace("-", " ") for match in RISK_CUE_PATTERN.finditer(text)}
    lowered = text.lower()
    if re.search(r"\b(no|not|don't|doesn't|didn't|can't|cannot|couldn't|isn't|aren't)\b", lowered):
        cues.add("negation")
    if "reservation" in lowered and ("accept" in lowered or "available" in lowered or "book" in lowered):
        cues.add("reservation_constraint")
    if "parking" in lowered and any(term in lowered for term in ["spendy", "valet", "hard", "difficult", "trouble", "limited", "expensive"]):
        cues.add("parking_constraint")
    if ("price" in lowered or "$" in lowered) and any(term in lowered for term in ["shocked", "worth", "expensive", "pricey", "spendy", "high"]):
        cues.add("price_constraint")
    return cues


def risk_supported_by_snippets(claim: str, snippets: list[str]) -> bool:
    snippet_text = " ".join(str(snippet or "") for snippet in snippets)
    snippet_cues = risk_cues(snippet_text)
    if not snippet_cues:
        return False
    claim_cues = risk_cues(claim)
    if claim_cues & snippet_cues:
        return True
    overlap = support_tokens(claim) & support_tokens(snippet_text)
    return len(overlap) >= 2


def clip_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def rebuild_guarded_summary(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    direct = [clip_text(item.get("signal"), 120) for item in parsed.get("direct_evidence_signals", []) if isinstance(item, dict)]
    attrs = [clip_text(item.get("attribute"), 80) for item in parsed.get("fine_grained_attributes", []) if isinstance(item, dict)]
    risks = [clip_text(item.get("risk"), 100) for item in parsed.get("risk_or_avoidance_notes", []) if isinstance(item, dict)]
    cautious = [clip_text(item.get("inference"), 120) for item in parsed.get("cautious_inferences", []) if isinstance(item, dict)]
    sentences: list[str] = []
    if direct:
        sentences.append("Evidence-backed signals: " + "; ".join(direct[:3]) + ".")
    if attrs:
        sentences.append("Fine-grained attributes: " + "; ".join(attrs[:4]) + ".")
    if risks:
        sentences.append("Risks: " + "; ".join(risks[:2]) + ".")
    if cautious:
        sentences.append("Cautious inferences: " + "; ".join(cautious[:2]) + ".")
    if not sentences:
        sentences.append("No evidence-backed merchant profile items were retained after guardrail checks.")
    parsed["merchant_summary"] = clip_text(" ".join(sentences), 480)
    audit_notes = parsed.get("audit_notes")
    if isinstance(audit_notes, dict):
        audit_notes["evidence_coverage"] = clip_text(
            f"Guarded summary rebuilt from retained items: direct={len(direct)}, attributes={len(attrs)}, risks={len(risks)}, cautious={len(cautious)}.",
            220,
        )
    return parsed


def overlap_ratio(left: Any, right: Any) -> float:
    left_tokens = support_tokens(left)
    right_tokens = support_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def signal_looks_overly_specific(signal: Any) -> bool:
    text = canonical_support_text(signal)
    detail_markers = [
        "served with",
        "topped with",
        "made with",
        "comes with",
        "for $",
        "priced",
        "reservation",
        "parking",
        "dress code",
        "bottomless",
        "wait",
        "line",
        "noise",
        "patio",
    ]
    return bool(any(marker in text for marker in detail_markers) or re.search(r"\$\d|\d+\s*(minute|hour|course|people|person)", text))


def rebalance_direct_vs_fine(parsed: Any) -> tuple[Any, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return parsed, actions
    direct_items = parsed.get("direct_evidence_signals")
    fine_items = parsed.get("fine_grained_attributes")
    if not isinstance(direct_items, list) or not isinstance(fine_items, list):
        return parsed, actions
    filtered_direct: list[Any] = []
    for idx, item in enumerate(direct_items):
        if not isinstance(item, dict):
            filtered_direct.append(item)
            continue
        signal = section_item_text("direct_evidence_signals", item)
        signal_refs = {str(ref) for ref in item.get("evidence_refs", []) if str(ref)}
        should_drop = False
        for fine_idx, fine in enumerate(fine_items):
            if not isinstance(fine, dict):
                continue
            fine_refs = {str(ref) for ref in fine.get("evidence_refs", []) if str(ref)}
            if not signal_refs or not fine_refs or not (signal_refs & fine_refs):
                continue
            ratio = overlap_ratio(signal, section_item_text("fine_grained_attributes", fine))
            if ratio >= 0.8 and signal_looks_overly_specific(signal):
                actions.append(
                    {
                        "type": "drop_direct_duplicate_of_fine",
                        "direct_index": idx,
                        "fine_index": fine_idx,
                        "signal": signal,
                    }
                )
                should_drop = True
                break
        if not should_drop:
            filtered_direct.append(item)
    parsed["direct_evidence_signals"] = filtered_direct[:DIRECT_MAX_ITEMS]
    return parsed, actions


def evidence_text_by_ref_from_prompt(prompt: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    block_re = re.compile(
        r"(?:POSITIVE_REVIEW_EVIDENCE|MIXED_REVIEW_EVIDENCE|NEGATIVE_REVIEW_EVIDENCE|TIP_EVIDENCE)\s*"
        r"```json\s*(.*?)\s*```",
        flags=re.DOTALL,
    )
    for match in block_re.finditer(prompt or ""):
        try:
            rows = json.loads(match.group(1))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("evidence_id") or "").strip()
            if not re.match(r"^m(?:rev|tip)_\d+$", ref):
                continue
            text = row.get("text")
            if text is None:
                text = row.get("tip_text")
            evidence[ref] = normalize_for_support_match(text)
    return evidence


def collect_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_refs" and isinstance(item, list):
                refs.extend(str(ref) for ref in item if str(ref))
            else:
                refs.extend(collect_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(collect_refs(item))
    return refs


def normalize_meta_refs(value: Any, business_id: str) -> tuple[Any, list[dict[str, str]]]:
    actions: list[dict[str, str]] = []
    allowed_meta_refs = {"MERCHANT_META", "AGGREGATE_NOTES", "SAMPLING_PROFILE"}

    def normalize_ref(ref: Any) -> str:
        raw = str(ref or "").strip()
        key = raw.lower()
        if key in {"merchant_meta", "merchant meta", "meta", "business_meta"} or raw == business_id:
            actions.append({"from": raw, "to": "MERCHANT_META"})
            return "MERCHANT_META"
        if key in {"aggregate_notes", "aggregate notes", "aggregates", "aggregate"}:
            actions.append({"from": raw, "to": "AGGREGATE_NOTES"})
            return "AGGREGATE_NOTES"
        if key in {"sampling_profile", "sampling profile"}:
            actions.append({"from": raw, "to": "SAMPLING_PROFILE"})
            return "SAMPLING_PROFILE"
        return raw

    def keep_ref(ref: str) -> bool:
        return bool(ref in allowed_meta_refs or re.match(r"^m(?:rev|tip)_\d+$", ref))

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, item in node.items():
                if key == "evidence_refs" and isinstance(item, list):
                    refs: list[str] = []
                    for ref in item:
                        normalized = normalize_ref(ref)
                        if not normalized:
                            continue
                        if keep_ref(normalized):
                            refs.append(normalized)
                        else:
                            actions.append({"type": "drop_invalid_ref", "from": normalized})
                    out[key] = refs
                else:
                    out[key] = walk(item)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(value), actions


def required_direct_ref_issues(parsed: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return [{"path": "$", "issue": "parsed output is not an object"}]
    direct_sections = ["direct_evidence_signals", "fine_grained_attributes", "usage_scenes", "risk_or_avoidance_notes"]
    for section in direct_sections:
        items = parsed.get(section)
        if not isinstance(items, list):
            issues.append({"path": section, "issue": "section is not a list"})
            continue
        for idx, item in enumerate(items):
            refs = item.get("evidence_refs") if isinstance(item, dict) else None
            if not isinstance(refs, list) or not refs:
                issues.append({"path": f"{section}[{idx}].evidence_refs", "issue": "direct section item has no evidence refs"})
                continue
            bad_refs = [str(ref) for ref in refs if not re.match(r"^m(?:rev|tip)_\d+$", str(ref or ""))]
            if bad_refs:
                issues.append(
                    {
                        "path": f"{section}[{idx}].evidence_refs",
                        "issue": "direct section item cites non-review/tip refs",
                        "refs": bad_refs,
                    }
                )
    return issues


def repair_direct_snippet_refs(parsed: Any, prompt: str) -> tuple[Any, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return parsed, actions
    evidence_by_ref = evidence_text_by_ref_from_prompt(prompt)
    direct_sections = ["direct_evidence_signals", "fine_grained_attributes", "usage_scenes", "risk_or_avoidance_notes"]
    for section in direct_sections:
        items = parsed.get(section)
        if not isinstance(items, list):
            continue
        repaired_items: list[Any] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                repaired_items.append(item)
                continue
            original_refs = [
                str(ref)
                for ref in item.get("evidence_refs", [])
                if re.match(r"^m(?:rev|tip)_\d+$", str(ref or ""))
            ]
            snippets = item.get("evidence_snippets")
            if not isinstance(snippets, list):
                continue
            kept_snippets: list[str] = []
            matched_refs: list[str] = []
            for snip_idx, snippet in enumerate(snippets):
                normalized = normalize_for_support_match(snippet)
                if len(normalized) < 12:
                    actions.append(
                        {
                            "type": "drop_short_snippet",
                            "path": f"{section}[{idx}].evidence_snippets[{snip_idx}]",
                            "snippet": str(snippet),
                        }
                    )
                    continue
                cited_matches = [ref for ref in original_refs if normalized in evidence_by_ref.get(ref, "")]
                any_matches = cited_matches or [ref for ref, text in evidence_by_ref.items() if normalized in text]
                if not any_matches:
                    actions.append(
                        {
                            "type": "drop_unmatched_snippet",
                            "path": f"{section}[{idx}].evidence_snippets[{snip_idx}]",
                            "snippet": str(snippet),
                            "original_refs": original_refs,
                        }
                    )
                    continue
                kept_snippets.append(str(snippet))
                for ref in any_matches:
                    if ref not in matched_refs and len(matched_refs) < 3:
                        matched_refs.append(ref)
                if not cited_matches:
                    actions.append(
                        {
                            "type": "realign_snippet_ref",
                            "path": f"{section}[{idx}].evidence_snippets[{snip_idx}]",
                            "snippet": str(snippet),
                            "from_refs": original_refs,
                            "to_refs": any_matches[:3],
                        }
                    )
            if kept_snippets:
                claim_text = claim_text_for_direct_item(section, item)
                claim_supported, claim_support = claim_supported_by_snippets(claim_text, kept_snippets[:2])
                if not claim_supported:
                    actions.append(
                        {
                            "type": "drop_unsupported_claim_text",
                            "path": f"{section}[{idx}]",
                            "claim": claim_text,
                            **claim_support,
                        }
                    )
                    continue
                if (
                    STRICT_RISK_POLARITY
                    and section == "risk_or_avoidance_notes"
                    and not risk_supported_by_snippets(claim_text, kept_snippets[:2])
                ):
                    actions.append(
                        {
                            "type": "drop_unsupported_risk_polarity",
                            "path": f"{section}[{idx}]",
                            "claim": claim_text,
                            "snippets": kept_snippets[:2],
                        }
                    )
                    continue
                item["evidence_snippets"] = kept_snippets[:2]
                item["evidence_refs"] = matched_refs[:3]
                repaired_items.append(item)
            else:
                actions.append(
                    {
                        "type": "drop_unsupported_direct_item",
                        "path": f"{section}[{idx}]",
                        "original_refs": original_refs,
                    }
                )
        parsed[section] = repaired_items
    return parsed, actions


def required_direct_snippet_issues(parsed: Any, prompt: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return [{"path": "$", "issue": "parsed output is not an object"}]
    evidence_by_ref = evidence_text_by_ref_from_prompt(prompt)
    direct_sections = ["direct_evidence_signals", "fine_grained_attributes", "usage_scenes", "risk_or_avoidance_notes"]
    for section in direct_sections:
        items = parsed.get(section)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                issues.append({"path": f"{section}[{idx}]", "issue": "direct section item is not an object"})
                continue
            refs = [str(ref) for ref in item.get("evidence_refs", []) if re.match(r"^m(?:rev|tip)_\d+$", str(ref or ""))]
            snippets = item.get("evidence_snippets")
            if not isinstance(snippets, list) or not snippets:
                issues.append({"path": f"{section}[{idx}].evidence_snippets", "issue": "direct section item has no snippets"})
                continue
            candidate_texts = [evidence_by_ref.get(ref, "") for ref in refs]
            for snip_idx, snippet in enumerate(snippets):
                normalized = normalize_for_support_match(snippet)
                if len(normalized) < 12:
                    issues.append(
                        {
                            "path": f"{section}[{idx}].evidence_snippets[{snip_idx}]",
                            "issue": "snippet is too short to support audit",
                            "snippet": str(snippet),
                        }
                    )
                    continue
                if not any(normalized in text for text in candidate_texts if text):
                    issues.append(
                        {
                            "path": f"{section}[{idx}].evidence_snippets[{snip_idx}]",
                            "issue": "snippet is not found in the cited evidence text",
                            "snippet": str(snippet),
                            "refs": refs,
                        }
                    )
    return issues


def summarize_numbers(values: list[int]) -> dict[str, int]:
    if not values:
        return {"rows": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(int(v) for v in values)

    def pct(q: float) -> int:
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return int(round(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)))

    return {
        "rows": len(ordered),
        "min": ordered[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


def load_completed_ids(raw_jsonl_path: Path) -> set[str]:
    if not RESUME_FROM_EXISTING or not raw_jsonl_path.exists():
        return set()
    completed: set[str] = set()
    with raw_jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            bid = str(row.get("business_id", "") or "")
            if bid:
                completed.add(bid)
    return completed


def generate_outputs(llm: Any, prompts: list[str], sampling_params: Any) -> Any:
    try:
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, sampling_params=sampling_params)


def parse_candidate_output(
    *,
    rec: MerchantPromptRecord,
    raw_text: str,
    generated_token_count: int,
    guided_mode: str,
    generation_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = extract_first_json_object(raw_text)
    parsed = None
    json_valid = False
    if extracted:
        try:
            parsed = json.loads(extracted)
            json_valid = isinstance(parsed, dict)
        except Exception:
            parsed = None
    parsed_repaired = None
    repair_actions: list[dict[str, Any]] = []
    if parsed is not None:
        parsed_repaired, repair_actions = normalize_meta_refs(parsed, rec.business_id)
        parsed_repaired, snippet_repair_actions = repair_direct_snippet_refs(parsed_repaired, rec.prompt_text)
        repair_actions.extend(snippet_repair_actions)
        parsed_repaired, dedupe_actions = dedupe_structured_items(parsed_repaired)
        repair_actions.extend(dedupe_actions)
        parsed_repaired, rebalance_actions = rebalance_direct_vs_fine(parsed_repaired)
        repair_actions.extend(rebalance_actions)
        parsed_repaired = rebuild_guarded_summary(parsed_repaired)
    allowed_refs = allowed_refs_from_prompt(rec.prompt_text) | {"MERCHANT_META", "AGGREGATE_NOTES", "SAMPLING_PROFILE"}
    used_refs = collect_refs(parsed_repaired)
    invalid_refs = sorted({ref for ref in used_refs if ref not in allowed_refs})
    direct_ref_issues = required_direct_ref_issues(parsed_repaired)
    snippet_support_issues = required_direct_snippet_issues(parsed_repaired, rec.prompt_text)
    guardrails_passed = bool(json_valid and not invalid_refs and not direct_ref_issues and not snippet_support_issues)
    raw_row = {
        "business_id": rec.business_id,
        "name": rec.name,
        "input_index": rec.input_index,
        "merchant_sampling_segment": rec.merchant_sampling_segment,
        "prompt_tokens": rec.prompt_tokens,
        "selected_review_evidence_count": rec.selected_review_evidence_count,
        "selected_tip_evidence_count": rec.selected_tip_evidence_count,
        "raw_text": raw_text,
        "extracted_json_text": extracted,
        "json_valid": json_valid,
        "allowed_evidence_refs_count": len(allowed_refs),
        "used_evidence_refs": used_refs,
        "invalid_evidence_refs": invalid_refs,
        "direct_ref_issues": direct_ref_issues,
        "snippet_support_issues": snippet_support_issues,
        "repair_actions": repair_actions,
        "guardrails_passed": guardrails_passed,
        "guided_decoding_mode": guided_mode,
        "generated_token_count": generated_token_count,
        "hit_generation_limit": bool(generated_token_count >= generation_limit),
        "generation_limit": generation_limit,
        "retry_attempted": False,
        "retry_succeeded": False,
    }
    parsed_row = {
        "business_id": rec.business_id,
        "name": rec.name,
        "input_index": rec.input_index,
        "merchant_sampling_segment": rec.merchant_sampling_segment,
        "parsed_json": parsed_repaired if guardrails_passed else None,
        "parsed_json_candidate": parsed,
        "json_valid": json_valid,
        "invalid_evidence_refs": invalid_refs,
        "direct_ref_issues": direct_ref_issues,
        "snippet_support_issues": snippet_support_issues,
        "repair_actions": repair_actions,
        "guardrails_passed": guardrails_passed,
    }
    return raw_row, parsed_row


def parse_persistent_phases() -> list[int]:
    if not PERSISTENT_PHASES_RAW:
        return []
    phases: list[int] = []
    for piece in PERSISTENT_PHASES_RAW.split(","):
        raw = piece.strip().lower()
        if not raw:
            continue
        if raw in {"full", "all"}:
            phases.append(0)
            continue
        value = int(raw)
        if value < 0:
            raise ValueError(f"persistent phase must be >=0, got {value}")
        phases.append(value)
    return phases


def phase_label(sample_max_rows: int) -> str:
    return "full" if sample_max_rows <= 0 else f"sample{sample_max_rows}"


def phase_output_dir(base_output_dir: Path | None, run_stamp: str, sample_max_rows: int, persistent: bool) -> Path:
    label = phase_label(sample_max_rows)
    if not persistent:
        if base_output_dir is not None:
            return base_output_dir
        return DEFAULT_OUTPUT_ROOT / f"{run_stamp}_{label}_{RUN_TAG}"
    if base_output_dir is not None:
        return base_output_dir / label
    return DEFAULT_OUTPUT_ROOT / f"{run_stamp}_{label}_{RUN_TAG}"


def run_generation_phase(
    *,
    input_dir: Path,
    output_dir: Path,
    records: list[MerchantPromptRecord],
    audit_df: pd.DataFrame,
    llm: Any,
    sampling_params: Any,
    guided_mode: str,
    tokenizer_seconds: float,
    model_seconds: float,
    sample_max_rows: int,
    persistent_mode: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(output_dir / "merchant_prompt_token_audit.parquet", index=False)
    write_json(
        output_dir / "merchant_prompt_token_audit_summary.json",
        {
            "rows": len(records),
            "prompt_tokens": summarize_numbers([rec.prompt_tokens for rec in records]),
            "merchant_sampling_segment_counts": audit_df["merchant_sampling_segment"].value_counts(dropna=False).to_dict(),
            "sample_max_rows": sample_max_rows,
        },
    )

    raw_jsonl_path = output_dir / "merchant_profile_generation_raw.jsonl"
    parsed_jsonl_path = output_dir / "merchant_profile_generation_parsed.jsonl"
    completed_ids = load_completed_ids(raw_jsonl_path)
    remaining = [rec for rec in records if rec.business_id not in completed_ids]
    batches = build_batches(remaining)

    plan = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "base_model": BASE_MODEL,
        "rows_total": len(records),
        "rows_completed_from_resume": len(completed_ids),
        "rows_remaining": len(remaining),
        "batch_size": BATCH_SIZE,
        "max_batch_prompt_tokens": MAX_BATCH_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "guided_decoding_mode": guided_mode,
        "sample_max_rows": sample_max_rows,
        "persistent_mode": persistent_mode,
        "retry_truncated_rows": RETRY_TRUNCATED_ROWS,
        "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
        "candidate_recall_mode": CANDIDATE_RECALL_MODE,
        "output_item_caps": {
            "direct_evidence_signals": DIRECT_MAX_ITEMS,
            "fine_grained_attributes": FINE_GRAINED_MAX_ITEMS,
            "usage_scenes": USAGE_SCENE_MAX_ITEMS,
            "risk_or_avoidance_notes": RISK_MAX_ITEMS,
            "cautious_inferences": CAUTIOUS_MAX_ITEMS,
            "evidence_refs": REFS_MAX_ITEMS,
            "evidence_snippets": SNIPPETS_MAX_ITEMS,
        },
        "vllm": {
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "enforce_eager": VLLM_ENFORCE_EAGER,
        },
    }
    write_json(output_dir / "merchant_profile_infer_plan.json", plan)
    write_json(output_dir / "merchant_profile_structured_output_schema.json", output_schema())

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    generation_start = time.perf_counter()
    retry_sampling_params = None
    retry_guided_mode = guided_mode
    if RETRY_TRUNCATED_ROWS and RETRY_MAX_NEW_TOKENS > MAX_NEW_TOKENS:
        retry_sampling_params, retry_guided_mode = build_sampling_params(RETRY_MAX_NEW_TOKENS)

    def write_checkpoint(label: str) -> None:
        summary = {
            **plan,
            "checkpoint_label": label,
            "tokenizer_load_seconds": tokenizer_seconds,
            "model_load_seconds": model_seconds,
            "generation_seconds": round(time.perf_counter() - generation_start, 4),
            "raw_rows_in_memory": len(raw_rows),
            "parsed_rows_in_memory": len(parsed_rows),
            "batch_metrics_count": len(batch_metrics),
            "json_valid_count": sum(1 for row in raw_rows if row.get("json_valid")),
            "guardrails_passed_count": sum(1 for row in raw_rows if row.get("guardrails_passed")),
            "snippet_support_issue_count": sum(1 for row in raw_rows if row.get("snippet_support_issues")),
            "generation_limit_hit_count": sum(1 for row in raw_rows if row.get("hit_generation_limit")),
            "retry_attempt_count": sum(1 for row in raw_rows if row.get("retry_attempted")),
            "retry_success_count": sum(1 for row in raw_rows if row.get("retry_succeeded")),
        }
        write_json(output_dir / f"merchant_profile_generation_checkpoint_{label}.json", summary)
        write_json(output_dir / "merchant_profile_generation_batch_metrics.json", batch_metrics)

    for batch_idx, batch in enumerate(batches, start=1):
        prompts = [rec.prompt_text for rec in batch]
        batch_start = time.perf_counter()
        outputs = generate_outputs(llm, prompts, sampling_params)
        elapsed = round(time.perf_counter() - batch_start, 4)
        generated_lengths: list[int] = []
        raw_batch_rows: list[dict[str, Any]] = []
        parsed_batch_rows: list[dict[str, Any]] = []
        for rec, request_output in zip(batch, outputs):
            candidate = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
            raw_text = str(getattr(candidate, "text", "") or "") if candidate is not None else ""
            token_ids = getattr(candidate, "token_ids", None) if candidate is not None else None
            generated_token_count = len(token_ids) if isinstance(token_ids, list) else 0
            generated_lengths.append(generated_token_count)
            raw_row, parsed_row = parse_candidate_output(
                rec=rec,
                raw_text=raw_text,
                generated_token_count=generated_token_count,
                guided_mode=guided_mode,
                generation_limit=MAX_NEW_TOKENS,
            )
            needs_retry = bool(
                RETRY_TRUNCATED_ROWS
                and retry_sampling_params is not None
                and raw_row.get("hit_generation_limit")
                and not raw_row.get("guardrails_passed")
            )
            if needs_retry:
                retry_output = generate_outputs(llm, [rec.prompt_text], retry_sampling_params)[0]
                retry_candidate = retry_output.outputs[0] if getattr(retry_output, "outputs", None) else None
                retry_text = str(getattr(retry_candidate, "text", "") or "") if retry_candidate is not None else ""
                retry_token_ids = getattr(retry_candidate, "token_ids", None) if retry_candidate is not None else None
                retry_generated_count = len(retry_token_ids) if isinstance(retry_token_ids, list) else 0
                retry_raw_row, retry_parsed_row = parse_candidate_output(
                    rec=rec,
                    raw_text=retry_text,
                    generated_token_count=retry_generated_count,
                    guided_mode=retry_guided_mode,
                    generation_limit=RETRY_MAX_NEW_TOKENS,
                )
                raw_row["retry_attempted"] = True
                if retry_raw_row.get("guardrails_passed"):
                    retry_raw_row["retry_attempted"] = True
                    retry_raw_row["retry_succeeded"] = True
                    retry_raw_row["retry_from_generated_token_count"] = generated_token_count
                    retry_raw_row["retry_guided_decoding_mode"] = retry_guided_mode
                    retry_parsed_row["retry_attempted"] = True
                    retry_parsed_row["retry_succeeded"] = True
                    raw_row = retry_raw_row
                    parsed_row = retry_parsed_row
                    generated_lengths[-1] = retry_generated_count
                else:
                    raw_row["retry_succeeded"] = False
            raw_batch_rows.append(raw_row)
            parsed_batch_rows.append(parsed_row)
        raw_rows.extend(raw_batch_rows)
        parsed_rows.extend(parsed_batch_rows)
        append_jsonl(raw_jsonl_path, raw_batch_rows)
        append_jsonl(parsed_jsonl_path, parsed_batch_rows)
        metric = {
            "batch_index": batch_idx,
            "batch_size": len(batch),
            "prompt_token_sum": sum(rec.prompt_tokens for rec in batch),
            "max_prompt_tokens": max((rec.prompt_tokens for rec in batch), default=0),
            "generated_token_lengths": generated_lengths,
            "elapsed_seconds": elapsed,
            "guided_decoding_mode": guided_mode,
        }
        batch_metrics.append(metric)
        print(
            f"[BATCH] {batch_idx}/{len(batches)} size={len(batch)} max_prompt={metric['max_prompt_tokens']} "
            f"sum_prompt={metric['prompt_token_sum']} gen={generated_lengths} elapsed={elapsed:.2f}s guided={guided_mode}",
            flush=True,
        )
        if CHECKPOINT_EVERY_BATCHES > 0 and batch_idx % CHECKPOINT_EVERY_BATCHES == 0:
            write_checkpoint(f"batch_{batch_idx:05d}")

    raw_rows.sort(key=lambda row: int(row["input_index"]))
    parsed_rows.sort(key=lambda row: int(row["input_index"]))
    write_json(output_dir / "merchant_profile_generation_raw.json", raw_rows)
    write_json(output_dir / "merchant_profile_generation_parsed.json", parsed_rows)
    write_json(output_dir / "merchant_profile_generation_batch_metrics.json", batch_metrics)
    final_summary = {
        **plan,
        "tokenizer_load_seconds": tokenizer_seconds,
        "model_load_seconds": model_seconds,
        "generation_seconds": round(time.perf_counter() - generation_start, 4),
        "rows_generated_this_run": len(raw_rows),
        "json_valid_count": sum(1 for row in raw_rows if row.get("json_valid")),
        "guardrails_passed_count": sum(1 for row in raw_rows if row.get("guardrails_passed")),
        "snippet_support_issue_count": sum(1 for row in raw_rows if row.get("snippet_support_issues")),
        "generation_limit_hit_count": sum(1 for row in raw_rows if row.get("hit_generation_limit")),
        "retry_attempt_count": sum(1 for row in raw_rows if row.get("retry_attempted")),
        "retry_success_count": sum(1 for row in raw_rows if row.get("retry_succeeded")),
        "prompt_tokens": summarize_numbers([rec.prompt_tokens for rec in records]),
    }
    write_json(output_dir / "merchant_profile_generation_summary.json", final_summary)
    write_checkpoint("final")
    write_latest_run_pointer(
        "12_merchant_profile_infer",
        output_dir,
        {"run_tag": RUN_TAG, "rows": len(records), "guardrails_passed_count": final_summary["guardrails_passed_count"]},
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    return final_summary


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    persistent_phases = parse_persistent_phases()
    persistent_mode = bool(persistent_phases)
    phase_rows = persistent_phases if persistent_mode else [SAMPLE_MAX_ROWS]
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_seconds = round(time.perf_counter() - tokenizer_start, 4)

    from vllm import LLM

    sampling_params, guided_mode = build_sampling_params()
    model_start = time.perf_counter()
    llm = LLM(
        model=BASE_MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=max(VLLM_TENSOR_PARALLEL_SIZE, 1),
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
        enforce_eager=VLLM_ENFORCE_EAGER,
    )
    model_seconds = round(time.perf_counter() - model_start, 4)

    summaries: list[dict[str, Any]] = []
    for sample_max_rows in phase_rows:
        records, audit_df = load_records(input_dir, tokenizer, sample_max_rows=sample_max_rows)
        output_dir = phase_output_dir(args.output_dir, run_stamp, sample_max_rows, persistent_mode)
        summary = run_generation_phase(
            input_dir=input_dir,
            output_dir=output_dir,
            records=records,
            audit_df=audit_df,
            llm=llm,
            sampling_params=sampling_params,
            guided_mode=guided_mode,
            tokenizer_seconds=tokenizer_seconds,
            model_seconds=model_seconds,
            sample_max_rows=sample_max_rows,
            persistent_mode=persistent_mode,
        )
        summaries.append(summary)

    if persistent_mode:
        print(json.dumps({"persistent_phase_summaries": summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
