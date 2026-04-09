from __future__ import annotations

import csv
import math
import os
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Callable

from pipeline.qlora_prompting import (
    build_blocker_comparison_prompt,
    build_item_text_sft_clean,
    build_local_listwise_ranking_prompt,
    build_user_text,
)
from pipeline.stage11_text_features import clean_text, extract_user_evidence_text, merge_distinct_segments


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _maybe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _row_has_pair_prompt_inputs(row: dict[str, Any]) -> bool:
    prompt = str(row.get("prompt", "") or "").strip()
    if prompt:
        return True
    prompt_style = str(PAIR_PROMPT_STYLE or "candidate_local").strip().lower()
    if prompt_style in {"local_listwise_compare", "blocker_compare"}:
        return True
    return False


_PROMPT_SCORE_PATTERNS = {
    "tower_score": re.compile(r"(?:^|[;\n])\s*tower_score:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"),
    "seq_score": re.compile(r"(?:^|[;\n])\s*seq_score:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"),
}
_BINARY_PROMPT_RE = re.compile(
    r"^(?P<head>.*?\nUser: )(?P<user>.*?)(?P<candidate>\nCandidate: )(?P<item>.*?)(?P<answer>\nAnswer:)\s*$",
    re.S,
)
_USER_VARIANT_MARKER_RE = re.compile(r"(?:;\s*)?(history_anchors:|pair_signal:)")
_PROMPT_FIELD_RE_TEMPLATE = r"{label}:\s*([^;\n]+)"
_NO_INFO_PROMPT_VALUES = {
    "no clear direct match from available text",
    "no clear direct conflict from available text",
}
_PROMPT_LOW_SIGNAL_TOKENS = {
    "available",
    "clear",
    "atmosphere",
    "cuisine",
    "dining",
    "direct",
    "experience",
    "family",
    "flavor",
    "food",
    "friendly",
    "meal",
    "match",
    "scene",
    "service",
    "style",
    "vibe",
    "conflict",
}

DPO_PAIR_POLICY = os.getenv("QLORA_DPO_PAIR_POLICY", "v1").strip().lower() or "v1"
PAIR_PROMPT_STYLE = os.getenv("QLORA_PAIR_PROMPT_STYLE", "candidate_local").strip().lower() or "candidate_local"
DPO_PAIR_TOPK_CUTOFF = max(1, int(os.getenv("QLORA_DPO_PAIR_TOPK_CUTOFF", "10").strip() or 10))
DPO_PAIR_HIGH_RANK_CUTOFF = max(
    DPO_PAIR_TOPK_CUTOFF,
    int(os.getenv("QLORA_DPO_PAIR_HIGH_RANK_CUTOFF", "20").strip() or 20),
)
DPO_MODEL_CONFUSER_SCORES_PATH = os.getenv("QLORA_DPO_MODEL_CONFUSER_SCORES_PATH", "").strip()
DPO_MODEL_CONFUSER_SCORE_COL = (
    os.getenv("QLORA_DPO_MODEL_CONFUSER_SCORE_COL", "blend_score").strip() or "blend_score"
)
DPO_MODEL_CONFUSER_TOPN = max(
    DPO_PAIR_TOPK_CUTOFF,
    int(os.getenv("QLORA_DPO_MODEL_CONFUSER_TOPN", "20").strip() or 20),
)
PAIR_BOUNDARY_POS_RANK_MIN = max(1, int(os.getenv("QLORA_DPO_PAIR_BOUNDARY_POS_RANK_MIN", "7").strip() or 7))
PAIR_BOUNDARY_POS_RANK_MAX = max(
    PAIR_BOUNDARY_POS_RANK_MIN,
    int(os.getenv("QLORA_DPO_PAIR_BOUNDARY_POS_RANK_MAX", "12").strip() or 12),
)
PAIR_BOUNDARY_NEG_RANK_MAX = max(
    1,
    int(os.getenv("QLORA_DPO_PAIR_BOUNDARY_NEG_RANK_MAX", str(DPO_PAIR_TOPK_CUTOFF)).strip() or DPO_PAIR_TOPK_CUTOFF),
)
PAIR_STRUCTURED_POS_RANK_MIN = max(
    1,
    int(os.getenv("QLORA_DPO_PAIR_STRUCTURED_POS_RANK_MIN", "10").strip() or 10),
)
PAIR_STRUCTURED_POS_RANK_MAX = max(
    PAIR_STRUCTURED_POS_RANK_MIN,
    int(os.getenv("QLORA_DPO_PAIR_STRUCTURED_POS_RANK_MAX", "25").strip() or 25),
)
PAIR_STRUCTURED_NEG_RANK_MAX = max(
    1,
    int(os.getenv("QLORA_DPO_PAIR_STRUCTURED_NEG_RANK_MAX", str(DPO_PAIR_TOPK_CUTOFF)).strip() or DPO_PAIR_TOPK_CUTOFF),
)
PAIR_DEEP_SEMANTIC_MIN_RANK = max(
    PAIR_STRUCTURED_NEG_RANK_MAX + 1,
    int(os.getenv("QLORA_DPO_PAIR_DEEP_SEMANTIC_MIN_RANK", "20").strip() or 20),
)
PAIR_HEAD_PRESERVE_POS_RANK_MAX = max(
    1,
    int(os.getenv("QLORA_DPO_PAIR_HEAD_PRESERVE_POS_RANK_MAX", "10").strip() or 10),
)
PAIR_HEAD_PRESERVE_NEG_RANK_MAX = max(
    PAIR_HEAD_PRESERVE_POS_RANK_MAX,
    int(os.getenv("QLORA_DPO_PAIR_HEAD_PRESERVE_NEG_RANK_MAX", "25").strip() or 25),
)
PAIR_COMPETITION_STRENGTH_MIN = float(
    os.getenv("QLORA_DPO_PAIR_COMPETITION_STRENGTH_MIN", "1.5").strip() or 1.5
)
PAIR_FIT_RISK_STRENGTH_MIN = float(
    os.getenv("QLORA_DPO_PAIR_FIT_RISK_STRENGTH_MIN", "1.5").strip() or 1.5
)
PAIR_V2B_TRUE_BOUNDARY_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_TRUE_BOUNDARY_QUOTA", "3").strip() or 3))
PAIR_V2B_TRUE_STRUCTURED_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_TRUE_STRUCTURED_QUOTA", "3").strip() or 3))
PAIR_V2B_TRUE_TOPK_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_TRUE_TOPK_QUOTA", "2").strip() or 2))
PAIR_V2B_TRUE_FIT_RISK_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_TRUE_FIT_RISK_QUOTA", "1").strip() or 1))
PAIR_V2B_TRUE_SEMANTIC_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_TRUE_SEMANTIC_QUOTA", "1").strip() or 1))
PAIR_V2B_TRUE_DEEP_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2B_TRUE_DEEP_SEMANTIC_QUOTA", "1").strip() or 1),
)
PAIR_V2B_VALID_BOUNDARY_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_VALID_BOUNDARY_QUOTA", "1").strip() or 1))
PAIR_V2B_VALID_STRUCTURED_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2B_VALID_STRUCTURED_QUOTA", "1").strip() or 1),
)
PAIR_V2B_VALID_COMPETITION_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2B_VALID_COMPETITION_QUOTA", "1").strip() or 1),
)
PAIR_V2B_VALID_SEMANTIC_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2B_VALID_SEMANTIC_QUOTA", "1").strip() or 1))
PAIR_V2B_VALID_DEEP_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2B_VALID_DEEP_SEMANTIC_QUOTA", "1").strip() or 1),
)
PAIR_V2C_TRUE_BOUNDARY_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_TRUE_BOUNDARY_QUOTA", "3").strip() or 3))
PAIR_V2C_TRUE_HEAD_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_TRUE_HEAD_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_TRUE_MULTI_ROUTE_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_TRUE_MULTI_ROUTE_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_TRUE_LOW_SUPPORT_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_TRUE_LOW_SUPPORT_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_TRUE_STRUCTURED_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_TRUE_STRUCTURED_QUOTA", "3").strip() or 3))
PAIR_V2C_TRUE_TOPK_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_TRUE_TOPK_QUOTA", "2").strip() or 2))
PAIR_V2C_TRUE_FIT_RISK_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_TRUE_FIT_RISK_QUOTA", "1").strip() or 1))
PAIR_V2C_TRUE_SEMANTIC_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_TRUE_SEMANTIC_QUOTA", "1").strip() or 1))
PAIR_V2C_TRUE_DEEP_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_TRUE_DEEP_SEMANTIC_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_BOUNDARY_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_VALID_BOUNDARY_QUOTA", "1").strip() or 1))
PAIR_V2C_VALID_HEAD_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_HEAD_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_MULTI_ROUTE_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_MULTI_ROUTE_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_LOW_SUPPORT_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_LOW_SUPPORT_PRESERVE_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_STRUCTURED_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_STRUCTURED_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_COMPETITION_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_COMPETITION_QUOTA", "1").strip() or 1),
)
PAIR_V2C_VALID_SEMANTIC_QUOTA = max(0, int(os.getenv("QLORA_DPO_V2C_VALID_SEMANTIC_QUOTA", "1").strip() or 1))
PAIR_V2C_VALID_DEEP_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2C_VALID_DEEP_SEMANTIC_QUOTA", "1").strip() or 1),
)
PAIR_MULTI_ROUTE_PRESERVE_MIN_SOURCE_COUNT = max(
    1,
    int(os.getenv("QLORA_DPO_MULTI_ROUTE_PRESERVE_MIN_SOURCE_COUNT", "2").strip() or 2),
)
PAIR_MULTI_ROUTE_PRESERVE_CONTEXT_RANK_MAX = max(
    1,
    int(os.getenv("QLORA_DPO_MULTI_ROUTE_PRESERVE_CONTEXT_RANK_MAX", "20").strip() or 20),
)
PAIR_MULTI_ROUTE_PRESERVE_SOURCE_EDGE_MIN = max(
    1,
    int(os.getenv("QLORA_DPO_MULTI_ROUTE_PRESERVE_SOURCE_EDGE_MIN", "2").strip() or 2),
)
PAIR_LOW_SUPPORT_PRESERVE_MAX_SEMANTIC_SUPPORT = float(
    os.getenv("QLORA_DPO_LOW_SUPPORT_PRESERVE_MAX_SEMANTIC_SUPPORT", "500").strip() or 500
)
PAIR_LOW_SUPPORT_PRESERVE_NEG_SUPPORT_MARGIN = float(
    os.getenv("QLORA_DPO_LOW_SUPPORT_PRESERVE_NEG_SUPPORT_MARGIN", "300").strip() or 300
)
PAIR_LOW_SUPPORT_PRESERVE_MIN_STABLE_SIGNALS = max(
    1,
    int(os.getenv("QLORA_DPO_LOW_SUPPORT_PRESERVE_MIN_STABLE_SIGNALS", "2").strip() or 2),
)
PAIR_LOW_SUPPORT_PRESERVE_MAX_TAG_RICHNESS = float(
    os.getenv("QLORA_DPO_LOW_SUPPORT_PRESERVE_MAX_TAG_RICHNESS", "1").strip() or 1
)
PAIR_LOW_SUPPORT_PRESERVE_NEG_TAG_MARGIN = float(
    os.getenv("QLORA_DPO_LOW_SUPPORT_PRESERVE_NEG_TAG_MARGIN", "1").strip() or 1
)
PAIR_TRUE_HEAD_BAND_MAX_RANK = max(
    1,
    int(os.getenv("QLORA_DPO_TRUE_HEAD_BAND_MAX_RANK", "10").strip() or 10),
)
PAIR_TRUE_BOUNDARY_BAND_MIN_RANK = max(
    PAIR_TRUE_HEAD_BAND_MAX_RANK + 1,
    int(os.getenv("QLORA_DPO_TRUE_BOUNDARY_BAND_MIN_RANK", "11").strip() or 11),
)
PAIR_TRUE_BOUNDARY_BAND_MAX_RANK = max(
    PAIR_TRUE_BOUNDARY_BAND_MIN_RANK,
    int(os.getenv("QLORA_DPO_TRUE_BOUNDARY_BAND_MAX_RANK", "30").strip() or 30),
)
PAIR_TRUE_MID_BAND_MAX_RANK = max(
    PAIR_TRUE_BOUNDARY_BAND_MAX_RANK,
    int(os.getenv("QLORA_DPO_TRUE_MID_BAND_MAX_RANK", "80").strip() or 80),
)
PAIR_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX = max(
    1,
    int(os.getenv("QLORA_DPO_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX", "20").strip() or 20),
)
PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK = max(
    1,
    int(os.getenv("QLORA_DPO_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK", "7").strip() or 7),
)
PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK = max(
    PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK,
    int(os.getenv("QLORA_DPO_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK", "10").strip() or 10),
)
PAIR_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK = max(
    PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK,
    int(os.getenv("QLORA_DPO_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK", "20").strip() or 20),
)
PAIR_CLEAR_MULTI_ROUTE_RESCUE_NEG_RANK_MAX = max(
    PAIR_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX,
    int(os.getenv("QLORA_DPO_CLEAR_MULTI_ROUTE_RESCUE_NEG_RANK_MAX", "20").strip() or 20),
)
PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MIN_RANK = max(
    PAIR_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK + 1,
    int(os.getenv("QLORA_DPO_RESCUE_31_60_PRIMARY_BLOCKER_MIN_RANK", "21").strip() or 21),
)
PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MAX_RANK = max(
    PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MIN_RANK,
    int(os.getenv("QLORA_DPO_RESCUE_31_60_PRIMARY_BLOCKER_MAX_RANK", "30").strip() or 30),
)
PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK = max(
    1,
    int(os.getenv("QLORA_DPO_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK", "7").strip() or 7),
)
PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MAX_RANK = max(
    PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK,
    int(os.getenv("QLORA_DPO_RESCUE_31_60_SECONDARY_BLOCKER_MAX_RANK", "20").strip() or 20),
)
PAIR_RESCUE_31_60_NEG_RANK_MAX = max(
    PAIR_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX,
    int(os.getenv("QLORA_DPO_RESCUE_31_60_NEG_RANK_MAX", "30").strip() or 30),
)
PAIR_RESCUE_61_100_NEG_RANK_MAX = max(
    PAIR_RESCUE_31_60_NEG_RANK_MAX,
    int(os.getenv("QLORA_DPO_RESCUE_61_100_NEG_RANK_MAX", "30").strip() or 30),
)
PAIR_V2D_TRUE_HEAD_BAND_MAX_PAIRS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_HEAD_BAND_MAX_PAIRS", "3").strip() or 3),
)
PAIR_V2D_TRUE_BOUNDARY_BAND_MAX_PAIRS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_BOUNDARY_BAND_MAX_PAIRS", "4").strip() or 4),
)
PAIR_V2D_TRUE_MID_BAND_MAX_PAIRS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_BAND_MAX_PAIRS", "2").strip() or 2),
)
PAIR_V2D_TRUE_DEEP_BAND_MAX_PAIRS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_BAND_MAX_PAIRS", "1").strip() or 1),
)
PAIR_V2D_TRUE_BOUNDARY_RESCUE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_BOUNDARY_RESCUE_QUOTA", "3").strip() or 3),
)
PAIR_V2D_TRUE_HEAD_PRESERVE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_HEAD_PRESERVE_QUOTA", "3").strip() or 3),
)
PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_TOPK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_TOPK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_FIT_RISK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_FIT_RISK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_COMPETITION_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_COMPETITION_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_SEMANTIC_QUOTA", "0").strip() or 0),
)
PAIR_V2D_TRUE_DEEP_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_SEMANTIC_QUOTA", "0").strip() or 0),
)
PAIR_V2D_VALID_BOUNDARY_RESCUE_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_VALID_BOUNDARY_RESCUE_QUOTA", "0").strip() or 0),
)
PAIR_V2D_VALID_COMPETITION_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_VALID_COMPETITION_QUOTA", "0").strip() or 0),
)
PAIR_V2D_VALID_FIT_RISK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_VALID_FIT_RISK_QUOTA", "0").strip() or 0),
)
PAIR_V2D_VALID_SEMANTIC_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_VALID_SEMANTIC_QUOTA", "0").strip() or 0),
)
PAIR_V2D_TRUE_REASON_PRIMARY_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_REASON_PRIMARY_QUOTA", "2").strip() or 2),
)
PAIR_V2D_TRUE_REASON_AUX_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_REASON_AUX_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_MID_SAME_BAND_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_SAME_BAND_QUOTA", "2").strip() or 2),
)
PAIR_V2D_TRUE_DEEP_SAME_BAND_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_SAME_BAND_QUOTA", "2").strip() or 2),
)
PAIR_V2D_TRUE_DEEP_MID_BAND_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_MID_BAND_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_HEAD_MIN_PACK = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_HEAD_MIN_PACK", "2").strip() or 2),
)
PAIR_V2D_TRUE_BOUNDARY_MIN_PACK = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_BOUNDARY_MIN_PACK", "4").strip() or 4),
)
PAIR_V2D_TRUE_MID_MIN_PACK = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_MIN_PACK", "3").strip() or 3),
)
PAIR_V2D_TRUE_DEEP_MIN_PACK = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_MIN_PACK", "2").strip() or 2),
)
PAIR_LOCAL_LISTWISE_MAX_RIVALS = max(
    1,
    int(os.getenv("QLORA_DPO_LOCAL_LISTWISE_MAX_RIVALS", "4").strip() or 4),
)
PAIR_V2D_TRUE_MID_SLATE_VARIANTS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_SLATE_VARIANTS", "1").strip() or 1),
)
PAIR_V2D_TRUE_MID_31_40_SLATE_VARIANTS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_31_40_SLATE_VARIANTS", "4").strip() or 4),
)
PAIR_V2D_TRUE_MID_EXPLICIT_SLATE_TYPES = (
    str(os.getenv("QLORA_DPO_V2D_TRUE_MID_EXPLICIT_SLATE_TYPES", "false") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_EXPLICIT_SLATE_TYPES = (
    str(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_EXPLICIT_SLATE_TYPES", "false") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_SLATE_VARIANTS = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_SLATE_VARIANTS", "1").strip() or 1),
)
PAIR_V2D_TRUE_DEEP_BOUNDARY_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_BOUNDARY_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_DEEP_HEAD_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_HEAD_QUOTA", "0").strip() or 0),
)
PAIR_FREEZE_BOUNDARY_11_30_V1 = (
    str(os.getenv("QLORA_DPO_FREEZE_BOUNDARY_11_30_V1", "true") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_MID_GENERIC_FILL = (
    str(os.getenv("QLORA_DPO_V2D_MID_GENERIC_FILL", "true") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_RELAX_ACTIONABILITY = (
    str(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_RELAX_ACTIONABILITY", "false") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_RELAX_MIN_SIGNAL_COUNT = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_RELAX_MIN_SIGNAL_COUNT", "3").strip() or 3),
)
PAIR_V2D_TRUE_HEAD_FALLBACK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_HEAD_FALLBACK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_BOUNDARY_FALLBACK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_BOUNDARY_FALLBACK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_MID_FALLBACK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_MID_FALLBACK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_DEEP_FALLBACK_QUOTA = max(
    0,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_FALLBACK_QUOTA", "1").strip() or 1),
)
PAIR_V2D_TRUE_DEEP_VARIANT_FALLBACK_FILL = (
    str(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_VARIANT_FALLBACK_FILL", "true") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_ALLOW_CROSS_VARIANT_REUSE = (
    str(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_ALLOW_CROSS_VARIANT_REUSE", "true") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)
PAIR_V2D_TRUE_DEEP_CROSS_VARIANT_REUSE_CAP = max(
    1,
    int(os.getenv("QLORA_DPO_V2D_TRUE_DEEP_CROSS_VARIANT_REUSE_CAP", "2").strip() or 2),
)
_MODEL_CONFUSER_CACHE_KEY: tuple[str, str, int] | None = None
_MODEL_CONFUSER_CACHE: dict[int, dict[int, dict[str, float | int]]] | None = None


def _row_score(row: dict[str, Any], key: str) -> float:
    raw = row.get(key)
    if raw is not None and str(raw).strip() != "":
        try:
            return float(raw)
        except Exception:
            pass
    prompt = str(row.get("prompt", "") or "")
    pattern = _PROMPT_SCORE_PATTERNS.get(str(key))
    if not pattern or not prompt:
        return 0.0
    match = pattern.search(prompt)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except Exception:
        return 0.0


def _summary(values: list[float | int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    xs = sorted(float(v) for v in values)
    n = len(xs)

    def _pick(q: float) -> float:
        idx = max(0, min(n - 1, int(round((n - 1) * q))))
        return float(xs[idx])

    return {
        "count": int(n),
        "min": float(xs[0]),
        "p50": _pick(0.50),
        "mean": float(sum(xs) / n),
        "p95": _pick(0.95),
        "max": float(xs[-1]),
    }


def _band(pre_rank: Any) -> str:
    r = _safe_int(pre_rank, 999999)
    if r <= 10:
        return "001_010"
    if r <= 30:
        return "011_030"
    if r <= 80:
        return "031_080"
    if r <= 150:
        return "081_150"
    return "151_plus"


def _neg_sort_key(neg_row: dict[str, Any], pos_score: float) -> tuple[float, int]:
    score_gap = abs(pos_score - _safe_float(neg_row.get("pre_score", 0.0), 0.0))
    return (score_gap, _safe_int(neg_row.get("pre_rank", 999999), 999999))


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _category_tokens(value: Any) -> set[str]:
    raw = _norm_text(value)
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _source_tokens(value: Any) -> set[str]:
    raw = _norm_text(value)
    if not raw:
        return set()
    parts = [part.strip() for part in re.split(r"[,;/|]", raw) if part.strip()]
    return set(parts)


def _pair_evidence_signal_flags(value: Any) -> tuple[bool, bool, bool]:
    txt = str(value or "").strip().lower()
    if not txt:
        return False, False, False
    return (
        "match aspects:" in txt,
        "user-disliked aspects present:" in txt,
        "possible weakness on preferred aspects:" in txt,
    )


def _row_pair_evidence_text(row: dict[str, Any] | pd.Series) -> str:
    return str(
        row.get("pair_evidence_summary", "")
        or row.get("evidence_basis_text_v1", "")
        or ""
    ).strip()


def _selection_tier_order(tier: str) -> int:
    order = {
        "observed_dislike": 0,
        "hard": 1,
        "near": 2,
        "mid": 3,
    }
    return int(order.get(str(tier or "unknown"), 9))


def _v2a_tier_order(tier: str) -> int:
    order = {
        "hard": 0,
        "near": 1,
        "observed_dislike": 2,
        "mid": 3,
    }
    return int(order.get(str(tier or "unknown"), 9))


def _true_rank_band(pos_rank: int) -> str:
    if pos_rank <= int(PAIR_TRUE_HEAD_BAND_MAX_RANK):
        return "head"
    if int(PAIR_TRUE_BOUNDARY_BAND_MIN_RANK) <= pos_rank <= int(PAIR_TRUE_BOUNDARY_BAND_MAX_RANK):
        return "boundary"
    if pos_rank <= int(PAIR_TRUE_MID_BAND_MAX_RANK):
        return "mid"
    return "deep"


def _effective_rank(row: dict[str, Any]) -> int:
    learned_rank = _safe_int(row.get("learned_rank", -1), -1)
    if learned_rank > 0:
        return int(learned_rank)
    return max(1, _safe_int(row.get("pre_rank", 999999), 999999))


def _effective_score(row: dict[str, Any]) -> float:
    learned_score = _maybe_float(row.get("learned_blend_score"))
    if learned_score is not None:
        return float(learned_score)
    return _safe_float(row.get("pre_score", 0.0), 0.0)


def _boundary_rank_band(rank: int) -> str:
    if rank <= 10:
        return "head_guard"
    if rank <= 30:
        return "boundary_11_30"
    if rank <= 60:
        return "rescue_31_60"
    if rank <= 100:
        return "rescue_61_100"
    return "out_of_scope"


def _truth_primary_reason(row: dict[str, Any]) -> str:
    return str(row.get("primary_reason", "") or "").strip()


def _truth_reason_is_weak(row: dict[str, Any]) -> bool:
    return _truth_primary_reason(row).endswith("_weak")


def _truth_reason_bucket(row: dict[str, Any]) -> str:
    reason = _truth_primary_reason(row)
    if reason in {
        "semantic_edge_xgb_ambiguous",
        "semantic_edge_underweighted",
        "semantic_edge_xgb_ambiguous_weak",
        "semantic_edge_underweighted_weak",
    }:
        return "semantic"
    if reason in {"channel_context_underweighted", "channel_context_underweighted_weak"}:
        return "channel_context"
    if reason in {"multi_route_underweighted", "multi_route_underweighted_weak"}:
        return "multi_route"
    if reason in {"head_prior_blocked", "head_prior_blocked_weak"}:
        return "head_prior"
    return ""


def _truth_reason_actionable(row: dict[str, Any]) -> bool:
    if bool(row.get("non_actionable", False)):
        return False
    if _truth_reason_bucket(row):
        return True
    if bool(row.get("easy_but_useful", False)) or bool(row.get("hard_but_learnable", False)):
        return _boundary_rank_band(_effective_rank(row)) in {"boundary_11_30", "rescue_31_60", "rescue_61_100"}
    return False


def _truth_reason_actionable_for_pairing(row: dict[str, Any], pos_band: str) -> bool:
    if _truth_reason_actionable(row):
        return True
    if str(pos_band or "") != "rescue_61_100" or not PAIR_V2D_TRUE_DEEP_RELAX_ACTIONABILITY:
        return False
    fit_visible = bool(row.get("pair_has_visible_user_fit_v1", False))
    detail_support = bool(row.get("pair_has_detail_support_v1", False))
    conflict_visible = bool(row.get("pair_has_visible_conflict_v1", False))
    multisource_fit = bool(row.get("pair_has_multisource_fit_v1", False))
    pair_fit_fact_count = int(_safe_int(row.get("pair_fit_fact_count_v1", 0), 0))
    pair_evidence_fact_count = int(_safe_int(row.get("pair_evidence_fact_count_v1", 0), 0))
    pair_fact_signal_count = int(_safe_int(row.get("pair_fact_signal_count_v1", 0), 0))
    return bool(
        fit_visible
        and detail_support
        and (conflict_visible or multisource_fit)
        and (pair_fit_fact_count >= 2 or pair_evidence_fact_count >= 1)
        and pair_fact_signal_count >= int(PAIR_V2D_TRUE_DEEP_RELAX_MIN_SIGNAL_COUNT)
    )


def _pick_bucketed_negatives(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    bucket_specs: list[tuple[str, Any, Any, int]],
    budget: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    selected: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    local_seen: set[int] = set()
    for bucket_name, predicate, sort_key, quota in bucket_specs:
        if len(selected) >= budget:
            break
        if int(quota) <= 0:
            continue
        eligible = [
            (neg_row, feat)
            for neg_row, feat in candidates
            if predicate(feat) and _safe_int(neg_row.get("item_idx", -1), -1) not in local_seen
        ]
        eligible.sort(key=lambda pair: sort_key(pair[1]))
        picked_for_bucket = 0
        for neg_row, feat in eligible:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in local_seen:
                continue
            selected.append((neg_row, bucket_name, feat))
            local_seen.add(neg_item_idx)
            picked_for_bucket += 1
            if picked_for_bucket >= int(quota) or len(selected) >= budget:
                break
    return selected[:budget]


def _pick_bucketed_negatives_global(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    bucket_specs: list[tuple[str, Any, Any, int]],
    budget: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    # Candidates may match multiple buckets. We must allow them to fall through to
    # later buckets if an earlier bucket quota is already saturated; otherwise
    # early preserve buckets can starve the slate and collapse pair counts.
    selected: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    local_seen: set[int] = set()
    per_bucket: list[tuple[str, list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]], int]] = []

    for bucket_name, predicate, sort_key, quota in bucket_specs:
        if int(quota) <= 0:
            continue
        eligible: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
        for neg_row, feat in candidates:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0:
                continue
            if predicate(feat):
                eligible.append((tuple(sort_key(feat)), neg_row, feat))
        eligible.sort(key=lambda pair: pair[0])
        per_bucket.append((str(bucket_name), eligible, int(quota)))

    for bucket_name, eligible, quota in per_bucket:
        if len(selected) >= budget:
            break
        picked_for_bucket = 0
        for _, neg_row, feat in eligible:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in local_seen:
                continue
            selected.append((neg_row, bucket_name, feat))
            local_seen.add(neg_item_idx)
            picked_for_bucket += 1
            if picked_for_bucket >= int(quota) or len(selected) >= budget:
                break
    return selected[:budget]


def _v2d_min_pack_for_band(pos_band: str) -> int:
    band = str(pos_band or "")
    if band == "head_guard":
        return int(PAIR_V2D_TRUE_HEAD_MIN_PACK)
    if band == "boundary_11_30":
        return int(PAIR_V2D_TRUE_BOUNDARY_MIN_PACK)
    if band == "rescue_31_60":
        return int(PAIR_V2D_TRUE_MID_MIN_PACK)
    if band == "rescue_61_100":
        return int(PAIR_V2D_TRUE_DEEP_MIN_PACK)
    return 1


def _focus_band_rival_band_priority(focus_band: str, rival_rank: int) -> int:
    rank = int(max(1, rival_rank))
    band = str(focus_band or "")
    if band == "head_guard":
        if 11 <= rank <= 30:
            return 0
        if 31 <= rank <= 60:
            return 1
        if rank <= 10:
            return 2
        if 61 <= rank <= 100:
            return 3
        return 4
    if band == "boundary_11_30":
        if 11 <= rank <= 30:
            return 0
        if rank <= 10:
            return 1
        if 31 <= rank <= 60:
            return 2
        if 61 <= rank <= 100:
            return 3
        return 4
    if band == "rescue_31_60":
        if 31 <= rank <= 60:
            return 0
        if 11 <= rank <= 30:
            return 1
        if rank <= 10:
            return 2
        if 61 <= rank <= 100:
            return 3
        return 4
    if band == "rescue_61_100":
        if 61 <= rank <= 100:
            return 0
        if 31 <= rank <= 60:
            return 1
        if 11 <= rank <= 30:
            return 2
        if rank <= 10:
            return 3
        return 4
    return rank


def _v2d_pack_fallback_key(pos_band: str, feat: dict[str, Any]) -> tuple[Any, ...]:
    neg_rank = int(feat.get("effective_neg_rank", 999999) or 999999)
    pos_rank = int(feat.get("effective_pos_rank", 999999) or 999999)
    pos_edges = int(feat.get("pos_explainable_edge_count", 0) or 0)
    neg_edges = int(feat.get("neg_explainable_edge_count", 0) or 0)
    base = (
        not bool(feat.get("outranks", False)),
        not bool(feat.get("high_rank", False)),
        not bool(feat.get("same_city", False)),
        -pos_edges,
        neg_edges,
        abs(neg_rank - pos_rank),
        neg_rank,
        int(feat.get("tier_order", 99) or 99),
        -float(feat.get("hardness", 0.0) or 0.0),
    )
    band = str(pos_band or "")
    if band == "head_guard":
        return (_head_guard_blocker_priority(neg_rank),) + base
    if band in {"boundary_11_30", "rescue_31_60", "rescue_61_100"}:
        return (_focus_band_rival_band_priority(band, neg_rank),) + base
    return (neg_rank,) + base


def _ensure_v2d_min_pack(
    pos_band: str,
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
    selected: list[tuple[dict[str, Any], str, dict[str, Any]]],
    budget: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return selected[:budget]
    target = min(int(budget), int(_v2d_min_pack_for_band(pos_band)))
    if len(selected) >= target:
        return selected[:budget]
    selected_ids = {
        _safe_int(neg_row.get("item_idx", -1), -1)
        for neg_row, _, _ in selected
        if _safe_int(neg_row.get("item_idx", -1), -1) >= 0
    }
    fallback: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for neg_row, feat in candidates:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in selected_ids:
            continue
        pos_edges = int(feat.get("pos_explainable_edge_count", 0) or 0)
        neg_edges = int(feat.get("neg_explainable_edge_count", 0) or 0)
        if pos_band == "head_guard":
            keep = bool(feat.get("outranks", False)) and neg_item_idx >= 0 and pos_edges >= max(1, neg_edges)
        elif pos_band == "boundary_11_30":
            keep = bool(feat.get("outranks", False)) and pos_edges >= 1
        elif pos_band == "rescue_31_60":
            keep = bool(feat.get("outranks", False)) and pos_edges >= 1 and int(feat.get("effective_neg_rank", 999999) or 999999) <= int(PAIR_RESCUE_31_60_NEG_RANK_MAX)
        elif pos_band == "rescue_61_100":
            keep = bool(feat.get("outranks", False)) and pos_edges >= 1 and int(feat.get("effective_neg_rank", 999999) or 999999) <= int(PAIR_RESCUE_61_100_NEG_RANK_MAX)
        else:
            keep = False
        if keep:
            fallback.append((_v2d_pack_fallback_key(pos_band, feat), neg_row, feat))
    fallback.sort(key=lambda x: x[0])
    out = list(selected)
    for _, neg_row, feat in fallback:
        if len(out) >= target or len(out) >= budget:
            break
        out.append((neg_row, "fallback_blocker", feat))
    return out[:budget]


def _load_model_confuser_map() -> dict[int, dict[int, dict[str, float | int]]]:
    global _MODEL_CONFUSER_CACHE_KEY, _MODEL_CONFUSER_CACHE
    cache_key = (
        str(DPO_MODEL_CONFUSER_SCORES_PATH or ""),
        str(DPO_MODEL_CONFUSER_SCORE_COL or ""),
        int(DPO_MODEL_CONFUSER_TOPN),
    )
    if _MODEL_CONFUSER_CACHE is not None and _MODEL_CONFUSER_CACHE_KEY == cache_key:
        return _MODEL_CONFUSER_CACHE

    out: dict[int, dict[int, dict[str, float | int]]] = {}
    path = str(DPO_MODEL_CONFUSER_SCORES_PATH or "").strip()
    if not path or not os.path.exists(path):
        _MODEL_CONFUSER_CACHE_KEY = cache_key
        _MODEL_CONFUSER_CACHE = out
        return out

    user_rows: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    score_col = str(DPO_MODEL_CONFUSER_SCORE_COL or "blend_score")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                uid = _safe_int(row.get("user_idx", -1), -1)
                item_idx = _safe_int(row.get("item_idx", -1), -1)
                label_true = _safe_int(row.get("label_true", 0), 0)
                score = _safe_float(row.get(score_col, row.get("qlora_prob", 0.0)), 0.0)
                pre_rank = _safe_int(row.get("pre_rank", 999999), 999999)
            except Exception:
                continue
            if uid < 0 or item_idx < 0:
                continue
            user_rows[uid].append(
                {
                    "item_idx": int(item_idx),
                    "label_true": int(label_true),
                    "score": float(score),
                    "pre_rank": int(pre_rank),
                    "ord": int(idx),
                }
            )

    for uid, rows in user_rows.items():
        rows.sort(
            key=lambda r: (
                -float(r["score"]),
                int(r["pre_rank"]),
                int(r["ord"]),
            )
        )
        true_pos = next((i for i, row in enumerate(rows) if int(row["label_true"]) == 1), None)
        confusers: list[dict[str, float | int]] = []
        if true_pos is not None:
            for rank_idx, row in enumerate(rows, start=1):
                if int(row["label_true"]) != 0:
                    continue
                if rank_idx - 1 < int(true_pos):
                    confusers.append(
                        {
                            "item_idx": int(row["item_idx"]),
                            "rank": int(rank_idx),
                            "score": float(row["score"]),
                            "outranks_true": 1,
                        }
                    )
                if len(confusers) >= int(DPO_MODEL_CONFUSER_TOPN):
                    break
        if not confusers:
            for rank_idx, row in enumerate(rows, start=1):
                if int(row["label_true"]) != 0:
                    continue
                confusers.append(
                    {
                        "item_idx": int(row["item_idx"]),
                        "rank": int(rank_idx),
                        "score": float(row["score"]),
                        "outranks_true": 0,
                    }
                )
                if len(confusers) >= int(DPO_MODEL_CONFUSER_TOPN):
                    break
        if confusers:
            out[int(uid)] = {
                int(row["item_idx"]): {
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "outranks_true": int(row["outranks_true"]),
                }
                for row in confusers
            }

    _MODEL_CONFUSER_CACHE_KEY = cache_key
    _MODEL_CONFUSER_CACHE = out
    return out


def _v2a_neg_features(
    pos_row: dict[str, Any],
    neg_row: dict[str, Any],
    pos_score: float,
    *,
    uid: int | None = None,
) -> dict[str, Any]:
    pos_rank = max(1, _safe_int(pos_row.get("pre_rank", 999999), 999999))
    neg_rank = max(1, _safe_int(neg_row.get("pre_rank", 999999), 999999))
    neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
    tier = str(neg_row.get("neg_tier", "") or "unknown")
    pos_sources = _source_tokens(pos_row.get("source_set_text"))
    neg_sources = _source_tokens(neg_row.get("source_set_text"))
    pos_source_count = max(0, _safe_int(pos_row.get("source_count", 0), 0))
    neg_source_count = max(0, _safe_int(neg_row.get("source_count", 0), 0))
    pos_nonpopular_source_count = max(0, _safe_int(pos_row.get("nonpopular_source_count", 0), 0))
    neg_nonpopular_source_count = max(0, _safe_int(neg_row.get("nonpopular_source_count", 0), 0))
    pos_profile_cluster_source_count = max(0, _safe_int(pos_row.get("profile_cluster_source_count", 0), 0))
    neg_profile_cluster_source_count = max(0, _safe_int(neg_row.get("profile_cluster_source_count", 0), 0))
    pos_context_rank = _maybe_float(pos_row.get("context_rank"))
    neg_context_rank = _maybe_float(neg_row.get("context_rank"))
    pos_has_context = bool(_safe_int(pos_row.get("has_context", 0), 0) > 0 or "context" in pos_sources)
    neg_has_context = bool(_safe_int(neg_row.get("has_context", 0), 0) > 0 or "context" in neg_sources)
    pos_context_detail_count = max(0, _safe_int(pos_row.get("context_detail_count", 0), 0))
    neg_context_detail_count = max(0, _safe_int(neg_row.get("context_detail_count", 0), 0))
    pos_semantic_support = _maybe_float(pos_row.get("semantic_support"))
    neg_semantic_support = _maybe_float(neg_row.get("semantic_support"))
    pos_semantic_tag_richness = _maybe_float(pos_row.get("semantic_tag_richness"))
    neg_semantic_tag_richness = _maybe_float(neg_row.get("semantic_tag_richness"))
    same_city = _norm_text(pos_row.get("city")) == _norm_text(neg_row.get("city"))
    same_primary = _norm_text(pos_row.get("primary_category")) == _norm_text(neg_row.get("primary_category"))
    pos_cats = _category_tokens(pos_row.get("categories"))
    neg_cats = _category_tokens(neg_row.get("categories"))
    overlap_count = len(pos_cats & neg_cats)
    is_near = bool(neg_row.get("neg_is_near", False))
    outranks = bool(neg_rank < pos_rank or neg_score >= pos_score)
    topk_incumbent = bool(neg_rank <= DPO_PAIR_TOPK_CUTOFF)
    high_rank = bool(neg_rank <= DPO_PAIR_HIGH_RANK_CUTOFF)
    hardness, score_gap, _ = _rich_neg_hardness(neg_row, pos_score)
    tower_score = max(0.0, _row_score(neg_row, "tower_score"))
    seq_score = max(0.0, _row_score(neg_row, "seq_score"))
    tower_seq = tower_score + seq_score
    group_gap_rank_pct = _maybe_float(neg_row.get("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"))
    group_gap_to_top3 = _maybe_float(neg_row.get("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"))
    group_gap_to_top10 = _maybe_float(neg_row.get("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"))
    net_score_rank_pct = _maybe_float(neg_row.get("schema_weighted_net_score_v2_rank_pct_v3"))
    avoid_neg = _maybe_float(neg_row.get("sim_negative_avoid_neg"))
    avoid_core = _maybe_float(neg_row.get("sim_negative_avoid_core"))
    conflict_gap = _maybe_float(neg_row.get("sim_conflict_gap"))
    pos_avoid_neg = _maybe_float(pos_row.get("sim_negative_avoid_neg"))
    pos_avoid_core = _maybe_float(pos_row.get("sim_negative_avoid_core"))
    pos_conflict_gap = _maybe_float(pos_row.get("sim_conflict_gap"))
    pos_preference_core = _maybe_float(pos_row.get("channel_preference_core_v1"))
    pos_recent_intent = _maybe_float(pos_row.get("channel_recent_intent_v1"))
    pos_context_time = _maybe_float(pos_row.get("channel_context_time_v1"))
    pos_channel_conflict = _maybe_float(pos_row.get("channel_conflict_v1"))
    pos_evidence_support = _maybe_float(pos_row.get("channel_evidence_support_v1"))
    pair_match_signal, pair_disliked_signal, pair_weak_signal = _pair_evidence_signal_flags(
        _row_pair_evidence_text(neg_row)
    )
    strong_gap_rank = bool(group_gap_rank_pct is not None and group_gap_rank_pct >= 0.75)
    elite_gap_rank = bool(group_gap_rank_pct is not None and group_gap_rank_pct >= 0.90)
    near_top3 = bool(group_gap_to_top3 is not None and group_gap_to_top3 >= -0.02)
    near_top10 = bool(group_gap_to_top10 is not None and group_gap_to_top10 >= -0.02)
    strong_net_rank = bool(net_score_rank_pct is not None and net_score_rank_pct >= 0.90)
    competition_strength = float(
        (2.0 if elite_gap_rank else 0.0)
        + (1.0 if strong_gap_rank else 0.0)
        + (1.0 if near_top3 else 0.0)
        + (0.5 if near_top10 else 0.0)
        + (1.0 if strong_net_rank else 0.0)
    )
    competition_confuser = bool(outranks and competition_strength > 0.0)
    avoid_neg_good = bool(avoid_neg is not None and avoid_neg >= 0.05)
    avoid_core_good = bool(avoid_core is not None and avoid_core >= 0.05)
    conflict_good = bool(conflict_gap is not None and conflict_gap >= 0.05)
    avoid_neg_bad = bool(avoid_neg is not None and avoid_neg <= -0.05)
    avoid_core_bad = bool(avoid_core is not None and avoid_core <= -0.05)
    conflict_bad = bool(conflict_gap is not None and conflict_gap <= -0.05)
    fit_risk_strength = float(
        (2.0 if pair_match_signal else 0.0)
        + (1.0 if avoid_neg_good else 0.0)
        + (1.0 if avoid_core_good else 0.0)
        + (1.0 if conflict_good else 0.0)
        - (1.0 if pair_disliked_signal else 0.0)
        - (0.5 if pair_weak_signal else 0.0)
        - (0.5 if avoid_neg_bad else 0.0)
        - (0.5 if avoid_core_bad else 0.0)
        - (0.5 if conflict_bad else 0.0)
    )
    fit_risk_confuser = bool(
        outranks and (pair_match_signal or avoid_neg_good or avoid_core_good or conflict_good)
    )
    model_confuser = False
    model_confuser_rank = 999999
    model_confuser_score = 0.0
    model_confuser_outranks_true = False
    if uid is not None:
        confuser_meta = _load_model_confuser_map().get(int(uid), {}).get(
            _safe_int(neg_row.get("item_idx", -1), -1),
            {},
        )
        if confuser_meta:
            model_confuser = True
            model_confuser_rank = _safe_int(confuser_meta.get("rank", 999999), 999999)
            model_confuser_score = _safe_float(confuser_meta.get("score", 0.0), 0.0)
            model_confuser_outranks_true = bool(
                _safe_int(confuser_meta.get("outranks_true", 0), 0)
            )
    semantic_confuser = bool(
        same_city
        and (
            same_primary
            or overlap_count > 0
            or is_near
            or tier in {"hard", "near"}
        )
    )
    boundary_truth = bool(PAIR_BOUNDARY_POS_RANK_MIN <= pos_rank <= PAIR_BOUNDARY_POS_RANK_MAX)
    boundary_kickout_confuser = bool(
        boundary_truth
        and neg_rank <= PAIR_BOUNDARY_NEG_RANK_MAX
        and outranks
    )
    head_preserve_truth = bool(pos_rank <= PAIR_HEAD_PRESERVE_POS_RANK_MAX)
    head_preserve_confuser = bool(
        head_preserve_truth
        and neg_rank > pos_rank
        and neg_rank <= PAIR_HEAD_PRESERVE_NEG_RANK_MAX
        and (semantic_confuser or fit_risk_confuser)
    )
    top10_boundary_rescue_truth = bool(
        PAIR_TRUE_BOUNDARY_BAND_MIN_RANK <= pos_rank <= PAIR_TRUE_BOUNDARY_BAND_MAX_RANK
    )
    top10_boundary_rescue_confuser = bool(
        top10_boundary_rescue_truth
        and neg_rank <= PAIR_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX
        and outranks
        and (
            topk_incumbent
            or competition_confuser
            or fit_risk_confuser
            or semantic_confuser
        )
    )
    pos_multi_route_support = bool(
        pos_source_count >= int(PAIR_MULTI_ROUTE_PRESERVE_MIN_SOURCE_COUNT)
        and (
            pos_nonpopular_source_count >= 1
            or pos_profile_cluster_source_count >= 1
            or (pos_context_rank is not None and pos_context_rank > 0 and pos_context_rank <= PAIR_MULTI_ROUTE_PRESERVE_CONTEXT_RANK_MAX)
            or pos_has_context
        )
    )
    neg_multi_route_support = bool(
        neg_source_count >= int(PAIR_MULTI_ROUTE_PRESERVE_MIN_SOURCE_COUNT)
        and (
            neg_nonpopular_source_count >= 1
            or neg_profile_cluster_source_count >= 1
            or (neg_context_rank is not None and neg_context_rank > 0 and neg_context_rank <= PAIR_MULTI_ROUTE_PRESERVE_CONTEXT_RANK_MAX)
            or neg_has_context
        )
    )
    multi_route_preserve_route_edge = bool(
        pos_source_count >= neg_source_count + int(PAIR_MULTI_ROUTE_PRESERVE_SOURCE_EDGE_MIN)
        or pos_nonpopular_source_count > neg_nonpopular_source_count
        or pos_profile_cluster_source_count > neg_profile_cluster_source_count
        or (pos_has_context and not neg_has_context and pos_source_count >= neg_source_count + 1)
    )
    multi_route_preserve_confuser = bool(
        pos_multi_route_support
        and outranks
        and (semantic_confuser or fit_risk_confuser or competition_confuser)
        and multi_route_preserve_route_edge
    )
    clear_multi_route_rescue_confuser = bool(
        pos_rank > PAIR_TRUE_HEAD_BAND_MAX_RANK
        and neg_rank <= PAIR_CLEAR_MULTI_ROUTE_RESCUE_NEG_RANK_MAX
        and pos_multi_route_support
        and outranks
        and (topk_incumbent or competition_confuser or fit_risk_confuser or semantic_confuser)
        and multi_route_preserve_route_edge
    )
    pos_stable_signal_count = sum(
        1
        for flag in (
            pos_avoid_neg is not None and pos_avoid_neg >= 0.05,
            pos_avoid_core is not None and pos_avoid_core >= 0.05,
            pos_conflict_gap is not None and pos_conflict_gap >= 0.05,
            pos_preference_core is not None and pos_preference_core >= 0.05,
            pos_recent_intent is not None and pos_recent_intent >= 0.05,
            pos_context_time is not None and pos_context_time >= 0.05,
            pos_evidence_support is not None and pos_evidence_support >= 0.05,
            pos_context_detail_count > 0,
        )
        if flag
    )
    pos_channel_conflict_bad = bool(pos_channel_conflict is not None and pos_channel_conflict <= -0.05)
    pos_low_support_truth = bool(
        (
            (
                pos_semantic_support is not None
                and pos_semantic_support <= float(PAIR_LOW_SUPPORT_PRESERVE_MAX_SEMANTIC_SUPPORT)
            )
            or (
                pos_semantic_tag_richness is not None
                and pos_semantic_tag_richness <= float(PAIR_LOW_SUPPORT_PRESERVE_MAX_TAG_RICHNESS)
            )
        )
        and pos_stable_signal_count >= int(PAIR_LOW_SUPPORT_PRESERVE_MIN_STABLE_SIGNALS)
        and not pos_channel_conflict_bad
    )
    neg_surface_support_advantage = bool(
        (
            neg_semantic_support is not None
            and pos_semantic_support is not None
            and neg_semantic_support >= pos_semantic_support + float(PAIR_LOW_SUPPORT_PRESERVE_NEG_SUPPORT_MARGIN)
        )
        or (
            neg_semantic_tag_richness is not None
            and pos_semantic_tag_richness is not None
            and neg_semantic_tag_richness >= pos_semantic_tag_richness + float(PAIR_LOW_SUPPORT_PRESERVE_NEG_TAG_MARGIN)
        )
        or competition_confuser
        or fit_risk_confuser
    )
    low_support_preserve_confuser = bool(
        pos_low_support_truth
        and outranks
        and (semantic_confuser or fit_risk_confuser or competition_confuser)
        and neg_surface_support_advantage
    )
    structured_truth = bool(PAIR_STRUCTURED_POS_RANK_MIN <= pos_rank <= PAIR_STRUCTURED_POS_RANK_MAX)
    structured_lift_confuser = bool(
        structured_truth
        and neg_rank <= PAIR_STRUCTURED_NEG_RANK_MAX
        and outranks
        and (
            competition_strength >= float(PAIR_COMPETITION_STRENGTH_MIN)
            or fit_risk_strength >= float(PAIR_FIT_RISK_STRENGTH_MIN)
        )
    )
    deep_semantic_challenger = bool(
        semantic_confuser
        and neg_rank >= PAIR_DEEP_SEMANTIC_MIN_RANK
    )
    return {
        "neg_rank": int(neg_rank),
        "pos_rank": int(pos_rank),
        "tier": tier,
        "same_city": bool(same_city),
        "same_primary": bool(same_primary),
        "category_overlap_count": int(overlap_count),
        "is_near": bool(is_near),
        "outranks": bool(outranks),
        "topk_incumbent": bool(topk_incumbent),
        "high_rank": bool(high_rank),
        "semantic_confuser": bool(semantic_confuser),
        "boundary_kickout_confuser": bool(boundary_kickout_confuser),
        "head_preserve_confuser": bool(head_preserve_confuser),
        "top10_boundary_rescue_confuser": bool(top10_boundary_rescue_confuser),
        "multi_route_preserve_confuser": bool(multi_route_preserve_confuser),
        "clear_multi_route_rescue_confuser": bool(clear_multi_route_rescue_confuser),
        "low_support_preserve_confuser": bool(low_support_preserve_confuser),
        "structured_lift_confuser": bool(structured_lift_confuser),
        "deep_semantic_challenger": bool(deep_semantic_challenger),
        "competition_confuser": bool(competition_confuser),
        "competition_strength": float(competition_strength),
        "fit_risk_confuser": bool(fit_risk_confuser),
        "fit_risk_strength": float(fit_risk_strength),
        "pos_source_count": int(pos_source_count),
        "neg_source_count": int(neg_source_count),
        "pos_nonpopular_source_count": int(pos_nonpopular_source_count),
        "neg_nonpopular_source_count": int(neg_nonpopular_source_count),
        "pos_profile_cluster_source_count": int(pos_profile_cluster_source_count),
        "neg_profile_cluster_source_count": int(neg_profile_cluster_source_count),
        "pos_semantic_support": float(pos_semantic_support if pos_semantic_support is not None else 1.0),
        "neg_semantic_support": float(neg_semantic_support if neg_semantic_support is not None else 1.0),
        "pos_semantic_tag_richness": float(
            pos_semantic_tag_richness if pos_semantic_tag_richness is not None else 0.0
        ),
        "neg_semantic_tag_richness": float(
            neg_semantic_tag_richness if neg_semantic_tag_richness is not None else 0.0
        ),
        "pos_low_support_truth": bool(pos_low_support_truth),
        "pos_stable_signal_count": int(pos_stable_signal_count),
        "true_rank_band": str(_true_rank_band(int(pos_rank))),
        "hardness": float(hardness),
        "score_gap": float(score_gap),
        "tower_seq": float(tower_seq),
        "model_confuser": bool(model_confuser),
        "model_confuser_rank": int(model_confuser_rank),
        "model_confuser_score": float(model_confuser_score),
        "model_confuser_outranks_true": bool(model_confuser_outranks_true),
        "tier_order": int(_v2a_tier_order(tier)),
    }


def _v2d_neg_features(
    pos_row: dict[str, Any],
    neg_row: dict[str, Any],
    *,
    uid: int | None = None,
) -> dict[str, Any]:
    base = _v2a_neg_features(
        pos_row,
        neg_row,
        _safe_float(pos_row.get("pre_score", 0.0), 0.0),
        uid=uid,
    )
    signals = _comparison_signal_snapshot(pos_row, neg_row)
    pos_rank = _effective_rank(pos_row)
    neg_rank = _effective_rank(neg_row)
    pos_score = _effective_score(pos_row)
    neg_score = _effective_score(neg_row)
    outranks = bool(neg_rank < pos_rank or neg_score >= pos_score)
    topk_incumbent = bool(neg_rank <= DPO_PAIR_TOPK_CUTOFF)
    high_rank = bool(neg_rank <= DPO_PAIR_HIGH_RANK_CUTOFF)
    boundary_band = _boundary_rank_band(int(pos_rank))
    reason_bucket = _truth_reason_bucket(pos_row)
    same_theme_neighbor = bool(base["same_primary"] or int(base["category_overlap_count"]) > 0 or base["same_city"])
    head_guard_confuser = bool(
        boundary_band == "head_guard"
        and neg_rank > pos_rank
        and neg_rank <= PAIR_HEAD_PRESERVE_NEG_RANK_MAX
        and (
            int(signals["pos_explainable_edge_count"]) >= 1
            or int(signals["neg_explainable_edge_count"]) == 0
        )
        and (base["semantic_confuser"] or base["fit_risk_confuser"] or base["competition_confuser"])
    )
    boundary_rescue_11_30_confuser = bool(
        boundary_band == "boundary_11_30"
        and neg_rank <= PAIR_TOP10_BOUNDARY_RESCUE_NEG_RANK_MAX
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and not (neg_rank < PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK and int(signals["pos_explainable_edge_count"]) < 2)
        and (
            topk_incumbent
            or base["competition_confuser"]
            or base["fit_risk_confuser"]
            or base["semantic_confuser"]
        )
    )
    rescue_31_60_confuser = bool(
        boundary_band == "rescue_31_60"
        and neg_rank <= PAIR_RESCUE_31_60_NEG_RANK_MAX
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and not (neg_rank <= PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK and int(signals["pos_explainable_edge_count"]) < 2)
        and (
            base["clear_multi_route_rescue_confuser"]
            or base["competition_confuser"]
            or base["fit_risk_confuser"]
            or base["semantic_confuser"]
        )
    )
    rescue_61_100_confuser = bool(
        boundary_band == "rescue_61_100"
        and neg_rank <= PAIR_RESCUE_61_100_NEG_RANK_MAX
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and (
            base["clear_multi_route_rescue_confuser"]
            or (base["competition_confuser"] and float(base["competition_strength"]) >= 2.0)
            or (base["fit_risk_confuser"] and float(base["fit_risk_strength"]) >= 2.0)
        )
    )
    rescue_31_60_same_band_competitor = bool(
        boundary_band == "rescue_31_60"
        and 31 <= neg_rank <= 60
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and same_theme_neighbor
    )
    rescue_61_100_same_band_competitor = bool(
        boundary_band == "rescue_61_100"
        and 61 <= neg_rank <= 100
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and same_theme_neighbor
    )
    rescue_61_100_mid_band_competitor = bool(
        boundary_band == "rescue_61_100"
        and 31 <= neg_rank <= 60
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and same_theme_neighbor
    )
    semantic_reason_blocker = bool(
        reason_bucket == "semantic"
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and (signals["pos_semantic_edge"] or signals["pos_support_edge"])
        and (topk_incumbent or base["semantic_confuser"] or base["competition_confuser"] or base["fit_risk_confuser"])
    )
    channel_context_reason_blocker = bool(
        reason_bucket == "channel_context"
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and (signals["pos_channel_context_edge"] or signals["pos_stability_edge"])
        and (high_rank or base["competition_confuser"] or base["fit_risk_confuser"])
    )
    multi_route_reason_blocker = bool(
        reason_bucket == "multi_route"
        and outranks
        and int(signals["pos_explainable_edge_count"]) >= 1
        and signals["pos_route_edge"]
        and (base["clear_multi_route_rescue_confuser"] or high_rank or topk_incumbent)
    )
    head_prior_reason_blocker = bool(
        reason_bucket == "head_prior"
        and high_rank
        and int(signals["pos_explainable_edge_count"]) >= 1
        and (outranks or boundary_band == "head_guard")
    )
    out = dict(base)
    out.update(
        {
            "effective_pos_rank": int(pos_rank),
            "effective_neg_rank": int(neg_rank),
            "effective_pos_score": float(pos_score),
            "effective_neg_score": float(neg_score),
            "outranks": bool(outranks),
            "topk_incumbent": bool(topk_incumbent),
            "high_rank": bool(high_rank),
            "boundary_rank_band": str(boundary_band),
            "head_guard_confuser": bool(head_guard_confuser),
            "boundary_rescue_11_30_confuser": bool(boundary_rescue_11_30_confuser),
            "rescue_31_60_confuser": bool(rescue_31_60_confuser),
            "rescue_61_100_confuser": bool(rescue_61_100_confuser),
            "rescue_31_60_same_band_competitor": bool(rescue_31_60_same_band_competitor),
            "rescue_61_100_same_band_competitor": bool(rescue_61_100_same_band_competitor),
            "rescue_61_100_mid_band_competitor": bool(rescue_61_100_mid_band_competitor),
            "pos_route_edge": bool(signals["pos_route_edge"]),
            "neg_route_edge": bool(signals["neg_route_edge"]),
            "pos_support_edge": bool(signals["pos_support_edge"]),
            "neg_support_edge": bool(signals["neg_support_edge"]),
            "pos_semantic_edge": bool(signals["pos_semantic_edge"]),
            "neg_semantic_edge": bool(signals["neg_semantic_edge"]),
            "pos_channel_context_edge": bool(signals["pos_channel_context_edge"]),
            "neg_channel_context_edge": bool(signals["neg_channel_context_edge"]),
            "pos_stability_edge": bool(signals["pos_stability_edge"]),
            "neg_stability_edge": bool(signals["neg_stability_edge"]),
            "pos_explainable_edge_count": int(signals["pos_explainable_edge_count"]),
            "neg_explainable_edge_count": int(signals["neg_explainable_edge_count"]),
            "pos_has_explainable_edge": bool(int(signals["pos_explainable_edge_count"]) > 0),
            "chosen_primary_reason": str(_truth_primary_reason(pos_row)),
            "chosen_reason_bucket": str(reason_bucket),
            "chosen_easy_but_useful": bool(pos_row.get("easy_but_useful", False)),
            "chosen_hard_but_learnable": bool(pos_row.get("hard_but_learnable", False)),
            "chosen_non_actionable": bool(pos_row.get("non_actionable", False)),
            "semantic_reason_blocker": bool(semantic_reason_blocker),
            "channel_context_reason_blocker": bool(channel_context_reason_blocker),
            "multi_route_reason_blocker": bool(multi_route_reason_blocker),
            "head_prior_reason_blocker": bool(head_prior_reason_blocker),
        }
    )
    return out


def _pick_v2a_negatives(
    source_name: str,
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
    *,
    uid: int | None = None,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for neg_row in neg_rows:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        candidates.append((neg_row, _v2a_neg_features(pos_row, neg_row, pos_score, uid=uid)))

    if not candidates:
        return []

    if source_name == "true":
        bucket_specs = [
            (
                "topk_incumbent",
                lambda f: f["topk_incumbent"] and f["outranks"],
                lambda f: (
                    not f["topk_incumbent"],
                    not f["outranks"],
                    f["tier_order"],
                    f["neg_rank"],
                    -f["hardness"],
                ),
            ),
            (
                "competition_confuser",
                lambda f: f["competition_confuser"],
                lambda f: (
                    not f["competition_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    -f["competition_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "fit_risk_confuser",
                lambda f: f["fit_risk_confuser"],
                lambda f: (
                    not f["fit_risk_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"],
                lambda f: (
                    not f["semantic_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "towerseq_confuser",
                lambda f: f["tower_seq"] > 0.0 or f["high_rank"],
                lambda f: (
                    not f["outranks"],
                    not f["high_rank"],
                    -f["tower_seq"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
        ]
    else:
        bucket_specs = [
            (
                "competition_confuser",
                lambda f: f["competition_confuser"],
                lambda f: (
                    not f["competition_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    -f["competition_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "fit_risk_confuser",
                lambda f: f["fit_risk_confuser"],
                lambda f: (
                    not f["fit_risk_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"],
                lambda f: (
                    not f["semantic_confuser"],
                    not f["outranks"],
                    not f["high_rank"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "top20_incumbent",
                lambda f: f["high_rank"] and f["outranks"],
                lambda f: (
                    not f["high_rank"],
                    not f["outranks"],
                    f["tier_order"],
                    f["neg_rank"],
                    -f["hardness"],
                ),
            ),
            (
                "towerseq_confuser",
                lambda f: f["tower_seq"] > 0.0 or f["high_rank"],
                lambda f: (
                    not f["high_rank"],
                    not f["outranks"],
                    -f["tower_seq"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
        ]

    selected: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    local_seen: set[int] = set()
    for bucket_name, predicate, sort_key in bucket_specs:
        if len(selected) >= budget:
            break
        eligible = [
            (neg_row, feat)
            for neg_row, feat in candidates
            if predicate(feat) and _safe_int(neg_row.get("item_idx", -1), -1) not in local_seen
        ]
        eligible.sort(key=lambda pair: sort_key(pair[1]))
        for neg_row, feat in eligible:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in local_seen:
                continue
            selected.append((neg_row, bucket_name, feat))
            local_seen.add(neg_item_idx)
            break
    return selected[:budget]


def _pick_v2b_negatives(
    source_name: str,
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
    *,
    uid: int | None = None,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for neg_row in neg_rows:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        candidates.append((neg_row, _v2a_neg_features(pos_row, neg_row, pos_score, uid=uid)))

    if not candidates:
        return []

    if source_name == "true":
        bucket_specs = [
            (
                "head_preserve_confuser",
                lambda f: f["head_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    abs(int(f["neg_rank"]) - int(f["pos_rank"])),
                    not f["fit_risk_confuser"],
                    not f["same_primary"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["fit_risk_strength"],
                    -f["category_overlap_count"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_HEAD_PRESERVE_QUOTA),
            ),
            (
                "multi_route_preserve_confuser",
                lambda f: f["multi_route_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    -int(f["pos_source_count"]),
                    -int(f["pos_nonpopular_source_count"]),
                    -int(f["pos_profile_cluster_source_count"]),
                    not f["fit_risk_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_MULTI_ROUTE_PRESERVE_QUOTA),
            ),
            (
                "boundary_kickout_confuser",
                lambda f: f["boundary_kickout_confuser"],
                lambda f: (
                    not f["boundary_kickout_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    -f["hardness"],
                ),
                int(PAIR_V2B_TRUE_BOUNDARY_QUOTA),
            ),
            (
                "structured_lift_confuser",
                lambda f: f["structured_lift_confuser"],
                lambda f: (
                    not f["structured_lift_confuser"],
                    f["neg_rank"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2B_TRUE_STRUCTURED_QUOTA),
            ),
            (
                "topk_incumbent",
                lambda f: f["topk_incumbent"] and f["outranks"],
                lambda f: (
                    not f["topk_incumbent"],
                    not f["outranks"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2B_TRUE_TOPK_QUOTA),
            ),
            (
                "fit_risk_confuser",
                lambda f: f["fit_risk_confuser"],
                lambda f: (
                    not f["fit_risk_confuser"],
                    not f["outranks"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_TRUE_FIT_RISK_QUOTA),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"] and not f["deep_semantic_challenger"],
                lambda f: (
                    not f["semantic_confuser"],
                    not f["outranks"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_TRUE_SEMANTIC_QUOTA),
            ),
            (
                "deep_semantic_challenger",
                lambda f: f["deep_semantic_challenger"],
                lambda f: (
                    not f["deep_semantic_challenger"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_TRUE_DEEP_SEMANTIC_QUOTA),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                max(1, int(budget)),
            ),
        ]
    else:
        bucket_specs = [
            (
                "head_preserve_confuser",
                lambda f: f["head_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    abs(int(f["neg_rank"]) - int(f["pos_rank"])),
                    not f["fit_risk_confuser"],
                    not f["same_primary"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["fit_risk_strength"],
                    -f["category_overlap_count"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_HEAD_PRESERVE_QUOTA),
            ),
            (
                "multi_route_preserve_confuser",
                lambda f: f["multi_route_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    -int(f["pos_source_count"]),
                    -int(f["pos_nonpopular_source_count"]),
                    -int(f["pos_profile_cluster_source_count"]),
                    not f["fit_risk_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_MULTI_ROUTE_PRESERVE_QUOTA),
            ),
            (
                "boundary_kickout_confuser",
                lambda f: f["boundary_kickout_confuser"],
                lambda f: (
                    not f["boundary_kickout_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["competition_strength"],
                    -f["hardness"],
                ),
                int(PAIR_V2B_VALID_BOUNDARY_QUOTA),
            ),
            (
                "structured_lift_confuser",
                lambda f: f["structured_lift_confuser"],
                lambda f: (
                    not f["structured_lift_confuser"],
                    f["neg_rank"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2B_VALID_STRUCTURED_QUOTA),
            ),
            (
                "competition_confuser",
                lambda f: f["competition_confuser"],
                lambda f: (
                    not f["competition_confuser"],
                    not f["outranks"],
                    -f["competition_strength"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_VALID_COMPETITION_QUOTA),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"] and not f["deep_semantic_challenger"],
                lambda f: (
                    not f["semantic_confuser"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_VALID_SEMANTIC_QUOTA),
            ),
            (
                "deep_semantic_challenger",
                lambda f: f["deep_semantic_challenger"],
                lambda f: (
                    not f["deep_semantic_challenger"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                int(PAIR_V2B_VALID_DEEP_SEMANTIC_QUOTA),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                max(1, int(budget)),
            ),
        ]
    return _pick_bucketed_negatives(candidates, bucket_specs, budget)


def _pick_v2c_negatives(
    source_name: str,
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
    *,
    uid: int | None = None,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for neg_row in neg_rows:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        candidates.append((neg_row, _v2a_neg_features(pos_row, neg_row, pos_score, uid=uid)))

    if not candidates:
        return []

    if source_name == "true":
        bucket_specs = [
            (
                "head_preserve_confuser",
                lambda f: f["head_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    abs(int(f["neg_rank"]) - int(f["pos_rank"])),
                    not f["fit_risk_confuser"],
                    not f["same_primary"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["fit_risk_strength"],
                    -f["category_overlap_count"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_HEAD_PRESERVE_QUOTA),
            ),
            (
                "multi_route_preserve_confuser",
                lambda f: f["multi_route_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    -int(f["pos_source_count"]),
                    -int(f["pos_nonpopular_source_count"]),
                    -int(f["pos_profile_cluster_source_count"]),
                    not f["fit_risk_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_MULTI_ROUTE_PRESERVE_QUOTA),
            ),
            (
                "boundary_kickout_confuser",
                lambda f: f["boundary_kickout_confuser"],
                lambda f: (
                    f["neg_rank"],
                    f["tier_order"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_BOUNDARY_QUOTA),
            ),
            (
                "structured_lift_confuser",
                lambda f: f["structured_lift_confuser"],
                lambda f: (
                    f["neg_rank"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_STRUCTURED_QUOTA),
            ),
            (
                "topk_incumbent",
                lambda f: f["topk_incumbent"] and f["outranks"],
                lambda f: (
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_TOPK_QUOTA),
            ),
            (
                "fit_risk_confuser",
                lambda f: f["fit_risk_confuser"],
                lambda f: (
                    f["neg_rank"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_FIT_RISK_QUOTA),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"] and not f["deep_semantic_challenger"],
                lambda f: (
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_SEMANTIC_QUOTA),
            ),
            (
                "deep_semantic_challenger",
                lambda f: f["deep_semantic_challenger"],
                lambda f: (
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_TRUE_DEEP_SEMANTIC_QUOTA),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                max(1, int(budget)),
            ),
        ]
    else:
        bucket_specs = [
            (
                "head_preserve_confuser",
                lambda f: f["head_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    abs(int(f["neg_rank"]) - int(f["pos_rank"])),
                    not f["fit_risk_confuser"],
                    not f["same_primary"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["fit_risk_strength"],
                    -f["category_overlap_count"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_HEAD_PRESERVE_QUOTA),
            ),
            (
                "multi_route_preserve_confuser",
                lambda f: f["multi_route_preserve_confuser"],
                lambda f: (
                    -int(f["pos_stable_signal_count"]),
                    -int(f["pos_source_count"]),
                    -int(f["pos_nonpopular_source_count"]),
                    -int(f["pos_profile_cluster_source_count"]),
                    not f["fit_risk_confuser"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_MULTI_ROUTE_PRESERVE_QUOTA),
            ),
            (
                "boundary_kickout_confuser",
                lambda f: f["boundary_kickout_confuser"],
                lambda f: (
                    f["neg_rank"],
                    f["tier_order"],
                    -f["competition_strength"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_BOUNDARY_QUOTA),
            ),
            (
                "structured_lift_confuser",
                lambda f: f["structured_lift_confuser"],
                lambda f: (
                    f["neg_rank"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_STRUCTURED_QUOTA),
            ),
            (
                "competition_confuser",
                lambda f: f["competition_confuser"],
                lambda f: (
                    f["neg_rank"],
                    -f["competition_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_COMPETITION_QUOTA),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"] and not f["deep_semantic_challenger"],
                lambda f: (
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_SEMANTIC_QUOTA),
            ),
            (
                "deep_semantic_challenger",
                lambda f: f["deep_semantic_challenger"],
                lambda f: (
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2C_VALID_DEEP_SEMANTIC_QUOTA),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
                max(1, int(budget)),
            ),
        ]
    return _pick_bucketed_negatives_global(candidates, bucket_specs, budget)


def _pick_v2d_negatives(
    source_name: str,
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
    *,
    uid: int | None = None,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_rank = _effective_rank(pos_row)
    pos_band = _boundary_rank_band(int(pos_rank))
    blocker_compare_mode = PAIR_PROMPT_STYLE == "blocker_compare"
    if source_name == "true" and pos_band == "out_of_scope":
        return []
    if source_name == "true" and pos_band != "head_guard" and not _truth_reason_actionable_for_pairing(pos_row, pos_band):
        return []
    if source_name != "true" and pos_band not in {"boundary_11_30", "rescue_31_60"}:
        return []

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for neg_row in neg_rows:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        candidates.append((neg_row, _v2d_neg_features(pos_row, neg_row, uid=uid)))

    if not candidates:
        return []

    if source_name == "true":
        filtered_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for neg_row, feats in candidates:
            pos_edges = int(feats.get("pos_explainable_edge_count", 0))
            neg_edges = int(feats.get("neg_explainable_edge_count", 0))
            neg_rank = int(feats.get("effective_neg_rank", 999999))
            keep = True
            if pos_band == "head_guard":
                keep = bool(pos_edges >= neg_edges or (pos_edges >= 1 and neg_rank > 15))
            elif pos_band == "boundary_11_30":
                keep = bool(
                    pos_edges >= 1
                    and pos_edges >= neg_edges
                    and not (
                        neg_rank < PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK
                        and pos_edges < 2
                    )
                )
            elif pos_band == "rescue_31_60":
                keep = bool(
                    pos_edges >= 1
                    and pos_edges >= neg_edges
                    and not (
                        neg_rank <= PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK
                        and pos_edges < 2
                    )
                )
            elif pos_band == "rescue_61_100":
                keep = bool(pos_edges >= 1 and pos_edges >= neg_edges)
            if keep:
                filtered_candidates.append((neg_row, feats))
        candidates = filtered_candidates

    if not candidates:
        return []

    if source_name == "true":
        if pos_band == "head_guard":
            budget = min(int(budget), int(PAIR_V2D_TRUE_HEAD_BAND_MAX_PAIRS))
            if blocker_compare_mode:
                bucket_specs = [
                    (
                        "head_guard_confuser",
                        lambda f: f["head_guard_confuser"],
                        lambda f: (
                            _head_guard_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["fit_risk_confuser"],
                            not f["same_primary"],
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["fit_risk_strength"],
                            -f["category_overlap_count"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_HEAD_PRESERVE_QUOTA),
                    ),
                ] + _v2d_reason_prefix_bucket_specs(pos_row, pos_band) + [
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and f["high_rank"]
                            and (int(f["pos_explainable_edge_count"]) >= 1 or int(f["neg_explainable_edge_count"]) == 0)
                        ),
                        lambda f: (
                            _head_guard_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            -f["competition_strength"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_HEAD_FALLBACK_QUOTA),
                    ),
                ]
            else:
                bucket_specs = [
                    (
                        "head_guard_confuser",
                        lambda f: f["head_guard_confuser"],
                        lambda f: (
                            _head_guard_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["fit_risk_confuser"],
                            not f["same_primary"],
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["fit_risk_strength"],
                            -f["category_overlap_count"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_HEAD_PRESERVE_QUOTA),
                    ),
                    (
                        "topk_incumbent",
                        lambda f: f["topk_incumbent"] and f["outranks"],
                        lambda f: (
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_TOPK_QUOTA),
                    ),
                    (
                        "fit_risk_confuser",
                        lambda f: f["fit_risk_confuser"],
                        lambda f: (
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_FIT_RISK_QUOTA),
                    ),
                    (
                        "semantic_confuser",
                        lambda f: f["semantic_confuser"],
                        lambda f: (
                            not f["same_primary"],
                            -f["category_overlap_count"],
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_SEMANTIC_QUOTA),
                    ),
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and f["high_rank"]
                            and (int(f["pos_explainable_edge_count"]) >= 1 or int(f["neg_explainable_edge_count"]) == 0)
                        ),
                        lambda f: (
                            _head_guard_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            -f["competition_strength"],
                            -f["hardness"],
                        ),
                        max(1, int(budget)),
                    ),
                ]
        elif pos_band == "boundary_11_30":
            budget = min(int(budget), int(PAIR_V2D_TRUE_BOUNDARY_BAND_MAX_PAIRS))
            if blocker_compare_mode:
                bucket_specs = [
                    (
                        "boundary_rescue_11_30_confuser",
                        lambda f: f["boundary_rescue_11_30_confuser"],
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_BOUNDARY_RESCUE_QUOTA),
                    ),
                ] + _v2d_reason_prefix_bucket_specs(pos_row, pos_band) + [
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and int(f["effective_neg_rank"]) <= int(PAIR_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK)
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_BOUNDARY_FALLBACK_QUOTA),
                    ),
                ]
            else:
                bucket_specs = [
                    (
                        "boundary_rescue_11_30_confuser",
                        lambda f: f["boundary_rescue_11_30_confuser"],
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_BOUNDARY_RESCUE_QUOTA),
                    ),
                    (
                        "head_guard_confuser",
                        lambda f: f["head_guard_confuser"],
                        lambda f: (
                            _head_guard_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["fit_risk_confuser"],
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_HEAD_PRESERVE_QUOTA),
                    ),
                    (
                        "competition_confuser",
                        lambda f: f["competition_confuser"],
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_COMPETITION_QUOTA),
                    ),
                    (
                        "fit_risk_confuser",
                        lambda f: f["fit_risk_confuser"],
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_FIT_RISK_QUOTA),
                    ),
                    (
                        "semantic_confuser",
                        lambda f: f["semantic_confuser"],
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            not f["same_primary"],
                            -f["category_overlap_count"],
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_SEMANTIC_QUOTA),
                    ),
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and int(f["effective_neg_rank"]) <= int(PAIR_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK)
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _boundary_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            -f["hardness"],
                        ),
                        max(1, int(budget)),
                    ),
                ]
        elif pos_band == "rescue_31_60":
            budget = min(int(budget), int(PAIR_V2D_TRUE_MID_BAND_MAX_PAIRS))
            if blocker_compare_mode:
                bucket_specs = [
                    (
                        "same_band_rescue_31_60_competitor",
                        lambda f: f["rescue_31_60_same_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MID_SAME_BAND_QUOTA),
                    ),
                    (
                        "rescue_31_60_confuser",
                        lambda f: f["rescue_31_60_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            not f["clear_multi_route_rescue_confuser"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_BOUNDARY_RESCUE_QUOTA),
                    ),
                    (
                        "clear_multi_route_rescue_confuser",
                        lambda f: f["clear_multi_route_rescue_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -int(f["pos_source_count"]),
                            -int(f["pos_nonpopular_source_count"]),
                            -int(f["pos_profile_cluster_source_count"]),
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                ] + _v2d_reason_prefix_bucket_specs(pos_row, pos_band) + [
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and int(f["effective_neg_rank"]) <= int(PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MAX_RANK)
                            and int(f["effective_neg_rank"]) >= int(PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK)
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MID_FALLBACK_QUOTA),
                    ),
                ]
            else:
                bucket_specs = [
                    (
                        "same_band_rescue_31_60_competitor",
                        lambda f: f["rescue_31_60_same_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MID_SAME_BAND_QUOTA),
                    ),
                    (
                        "rescue_31_60_confuser",
                        lambda f: f["rescue_31_60_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            not f["clear_multi_route_rescue_confuser"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_BOUNDARY_RESCUE_QUOTA),
                    ),
                    (
                        "clear_multi_route_rescue_confuser",
                        lambda f: f["clear_multi_route_rescue_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -int(f["pos_source_count"]),
                            -int(f["pos_nonpopular_source_count"]),
                            -int(f["pos_profile_cluster_source_count"]),
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                    (
                        "competition_confuser",
                        lambda f: f["competition_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_COMPETITION_QUOTA),
                    ),
                    (
                        "fit_risk_confuser",
                        lambda f: f["fit_risk_confuser"],
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_FIT_RISK_QUOTA),
                    ),
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and int(f["effective_neg_rank"]) <= int(PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MAX_RANK)
                            and int(f["effective_neg_rank"]) >= int(PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK)
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _mid_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            -f["hardness"],
                        ),
                        max(1, int(budget)),
                    ),
                ]
        else:
            budget = min(int(budget), int(PAIR_V2D_TRUE_DEEP_BAND_MAX_PAIRS))
            if blocker_compare_mode:
                bucket_specs = [
                    (
                        "same_band_rescue_61_100_competitor",
                        lambda f: f["rescue_61_100_same_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_SAME_BAND_QUOTA),
                    ),
                    (
                        "mid_band_rescue_61_100_competitor",
                        lambda f: f["rescue_61_100_mid_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_MID_BAND_QUOTA),
                    ),
                    (
                        "boundary_blocker_for_deep",
                        lambda f: (
                            11 <= int(f["effective_neg_rank"]) <= 30
                            and f["outranks"]
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_BOUNDARY_QUOTA),
                    ),
                    (
                        "head_anchor_for_deep",
                        lambda f: (
                            int(f["effective_neg_rank"]) <= 10
                            and f["outranks"]
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_HEAD_QUOTA),
                    ),
                    (
                        "rescue_61_100_confuser",
                        lambda f: f["rescue_61_100_confuser"],
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            not f["clear_multi_route_rescue_confuser"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                    (
                        "clear_multi_route_rescue_confuser",
                        lambda f: f["clear_multi_route_rescue_confuser"],
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            -int(f["pos_source_count"]),
                            -int(f["pos_nonpopular_source_count"]),
                            -int(f["pos_profile_cluster_source_count"]),
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                ] + _v2d_reason_prefix_bucket_specs(pos_row, pos_band) + [
                    (
                        "fallback_blocker",
                        lambda f: (
                            f["outranks"]
                            and int(f["effective_neg_rank"]) <= int(PAIR_RESCUE_61_100_NEG_RANK_MAX)
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            int(f["neg_explainable_edge_count"]),
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_FALLBACK_QUOTA),
                    ),
                ]
            else:
                bucket_specs = [
                    (
                        "same_band_rescue_61_100_competitor",
                        lambda f: f["rescue_61_100_same_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_SAME_BAND_QUOTA),
                    ),
                    (
                        "mid_band_rescue_61_100_competitor",
                        lambda f: f["rescue_61_100_mid_band_competitor"],
                        lambda f: (
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_MID_BAND_QUOTA),
                    ),
                    (
                        "boundary_blocker_for_deep",
                        lambda f: (
                            11 <= int(f["effective_neg_rank"]) <= 30
                            and f["outranks"]
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            abs(int(f["effective_neg_rank"]) - int(f["effective_pos_rank"])),
                            not f["same_city"],
                            not f["same_primary"],
                            -int(f["category_overlap_count"]),
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_BOUNDARY_QUOTA),
                    ),
                    (
                        "head_anchor_for_deep",
                        lambda f: (
                            int(f["effective_neg_rank"]) <= 10
                            and f["outranks"]
                            and int(f["pos_explainable_edge_count"]) >= 1
                        ),
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            not f["same_city"],
                            -int(f["pos_explainable_edge_count"]),
                            int(f["neg_explainable_edge_count"]),
                            f["effective_neg_rank"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_DEEP_HEAD_QUOTA),
                    ),
                    (
                        "rescue_61_100_confuser",
                        lambda f: f["rescue_61_100_confuser"],
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            not f["clear_multi_route_rescue_confuser"],
                            -f["competition_strength"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                    (
                        "clear_multi_route_rescue_confuser",
                        lambda f: f["clear_multi_route_rescue_confuser"],
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            -int(f["pos_source_count"]),
                            -int(f["pos_nonpopular_source_count"]),
                            -int(f["pos_profile_cluster_source_count"]),
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_MULTI_ROUTE_RESCUE_QUOTA),
                    ),
                    (
                        "fit_risk_confuser",
                        lambda f: f["fit_risk_confuser"] and float(f["fit_risk_strength"]) >= 2.0,
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            -f["fit_risk_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_FIT_RISK_QUOTA),
                    ),
                    (
                        "competition_confuser",
                        lambda f: f["competition_confuser"] and float(f["competition_strength"]) >= 2.0,
                        lambda f: (
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["effective_neg_rank"],
                            -f["competition_strength"],
                            f["tier_order"],
                            -f["hardness"],
                        ),
                        int(PAIR_V2D_TRUE_COMPETITION_QUOTA),
                    ),
                    (
                        "fallback_competitor",
                        lambda f: True,
                        lambda f: (
                            not f["outranks"],
                            _deep_rescue_blocker_priority(int(f["effective_neg_rank"])),
                            f["tier_order"],
                            not f["high_rank"],
                            -f["hardness"],
                            f["effective_neg_rank"],
                        ),
                        max(1, int(budget)),
                    ),
                ]
    else:
        valid_total_quota = (
            int(PAIR_V2D_VALID_BOUNDARY_RESCUE_QUOTA)
            + int(PAIR_V2D_VALID_COMPETITION_QUOTA)
            + int(PAIR_V2D_VALID_FIT_RISK_QUOTA)
            + int(PAIR_V2D_VALID_SEMANTIC_QUOTA)
        )
        if valid_total_quota <= 0:
            return []
        budget = min(int(budget), max(1, int(PAIR_V2D_TRUE_MID_BAND_MAX_PAIRS)))
        bucket_specs = [
            (
                "boundary_rescue_11_30_confuser",
                lambda f: f["boundary_rescue_11_30_confuser"] or f["rescue_31_60_confuser"],
                lambda f: (
                    f["effective_neg_rank"],
                    -f["competition_strength"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2D_VALID_BOUNDARY_RESCUE_QUOTA),
            ),
            (
                "competition_confuser",
                lambda f: f["competition_confuser"],
                lambda f: (
                    f["effective_neg_rank"],
                    -f["competition_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2D_VALID_COMPETITION_QUOTA),
            ),
            (
                "fit_risk_confuser",
                lambda f: f["fit_risk_confuser"],
                lambda f: (
                    f["effective_neg_rank"],
                    -f["fit_risk_strength"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2D_VALID_FIT_RISK_QUOTA),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"],
                lambda f: (
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["effective_neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(PAIR_V2D_VALID_SEMANTIC_QUOTA),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["effective_neg_rank"],
                ),
                0,
            ),
        ]
    selected = _pick_bucketed_negatives_global(candidates, bucket_specs, budget)
    if source_name == "true":
        selected = _ensure_v2d_min_pack(pos_band, candidates, selected, budget)
    return selected


def _pick_v3_negatives(
    source_name: str,
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
    *,
    uid: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for neg_row in neg_rows:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        candidates.append((neg_row, _v2a_neg_features(pos_row, neg_row, pos_score, uid=uid)))

    if not candidates:
        return []

    if source_name == "true":
        bucket_specs = [
            (
                "model_confuser",
                lambda f: f["model_confuser"] and f["model_confuser_outranks_true"],
                lambda f: (
                    not f["model_confuser"],
                    not f["model_confuser_outranks_true"],
                    f["model_confuser_rank"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "topk_incumbent",
                lambda f: f["topk_incumbent"] and f["outranks"],
                lambda f: (
                    not f["topk_incumbent"],
                    not f["outranks"],
                    f["neg_rank"],
                    -f["hardness"],
                    -f["tower_seq"],
                ),
            ),
            (
                "towerseq_confuser",
                lambda f: (f["tower_seq"] > 0.0 and f["high_rank"]) or f["model_confuser"],
                lambda f: (
                    not f["high_rank"],
                    not f["outranks"],
                    -f["tower_seq"],
                    not f["model_confuser"],
                    f["tier_order"],
                    f["neg_rank"],
                ),
            ),
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"],
                lambda f: (
                    not f["semantic_confuser"],
                    not f["outranks"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
            (
                "fallback_competitor",
                lambda f: True,
                lambda f: (
                    not f["outranks"],
                    f["tier_order"],
                    not f["high_rank"],
                    -f["hardness"],
                    f["neg_rank"],
                ),
            ),
        ]
    else:
        bucket_specs = [
            (
                "semantic_confuser",
                lambda f: f["semantic_confuser"] or f["model_confuser"],
                lambda f: (
                    not f["model_confuser"],
                    not f["semantic_confuser"],
                    not f["outranks"],
                    not f["same_primary"],
                    -f["category_overlap_count"],
                    f["tier_order"],
                    f["neg_rank"],
                ),
            ),
            (
                "top20_incumbent",
                lambda f: f["high_rank"] and f["outranks"],
                lambda f: (
                    not f["high_rank"],
                    not f["outranks"],
                    f["neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
            ),
        ]

    selected: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    local_seen: set[int] = set()
    for bucket_name, predicate, sort_key in bucket_specs:
        if len(selected) >= budget:
            break
        eligible = [
            (neg_row, feat)
            for neg_row, feat in candidates
            if predicate(feat) and _safe_int(neg_row.get("item_idx", -1), -1) not in local_seen
        ]
        eligible.sort(key=lambda pair: sort_key(pair[1]))
        for neg_row, feat in eligible:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in local_seen:
                continue
            selected.append((neg_row, bucket_name, feat))
            local_seen.add(neg_item_idx)
            break
    return selected[:budget]


def _pick_generic_ranked_negatives(
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
    seen_neg_items: set[int],
    budget: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if budget <= 0:
        return []
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    ranked_neg: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    raw_ranked_neg = sorted(
        neg_rows,
        key=lambda neg: (
            _selection_tier_order(str(neg.get("neg_tier", "") or "unknown")),
            -_rich_neg_hardness(neg, pos_score)[0],
            _rich_neg_hardness(neg, pos_score)[1],
            _rich_neg_hardness(neg, pos_score)[2],
        ),
    )
    for neg_row in raw_ranked_neg:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
            continue
        ranked_neg.append((neg_row, str(neg_row.get("neg_tier", "") or "unknown"), {}))
        if len(ranked_neg) >= int(budget):
            break
    return ranked_neg


def _merge_ranked_negatives(
    preferred: list[tuple[dict[str, Any], str, dict[str, Any]]],
    fallback: list[tuple[dict[str, Any], str, dict[str, Any]]],
    budget: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    merged: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    seen_items: set[int] = set()
    for bucket in (preferred, fallback):
        for neg_row, neg_bucket, neg_feat in bucket:
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in seen_items:
                continue
            seen_items.add(neg_item_idx)
            merged.append((neg_row, neg_bucket, neg_feat))
            if len(merged) >= int(budget):
                return merged
    return merged


def _split_binary_prompt(prompt: str) -> dict[str, str] | None:
    match = _BINARY_PROMPT_RE.match(str(prompt or "").strip())
    if not match:
        return None
    return {
        "head": str(match.group("head")),
        "user": str(match.group("user")),
        "candidate": str(match.group("candidate")),
        "item": str(match.group("item")),
        "answer": str(match.group("answer")),
    }


def _split_user_shared_and_variant(user_text: str) -> tuple[str, str]:
    raw = str(user_text or "")
    match = _USER_VARIANT_MARKER_RE.search(raw)
    if not match:
        return raw.rstrip(" ;"), ""
    shared_end = int(match.start())
    variant_start = int(match.start(1))
    return raw[:shared_end].rstrip(" ;"), raw[variant_start:]


def _rank_band_label(rank: int) -> str:
    r = int(rank)
    if 0 < r <= 10:
        return "current rank band 1-10"
    if r <= 30:
        return "current rank band 11-30"
    if r <= 60:
        return "current rank band 31-60"
    if r <= 100:
        return "current rank band 61-100"
    return "current rank band 101+"


def _rank_cutoff_label(rank: int) -> str:
    r = int(rank)
    if 0 < r <= 10:
        return "inside the current top10"
    if r <= 30:
        return "just outside the current top10"
    if r <= 60:
        return "inside the current top60"
    if r <= 100:
        return "inside the current top100"
    return "outside the current top100"


def _comparison_ranking_context(a_rank: int, b_rank: int) -> str:
    a_rank = int(max(1, a_rank))
    b_rank = int(max(1, b_rank))
    if a_rank < b_rank:
        order_text = "Candidate A currently ranks ahead of Candidate B."
    elif b_rank < a_rank:
        order_text = "Candidate B currently ranks ahead of Candidate A."
    else:
        order_text = "Candidate A and Candidate B currently share the same rank band."
    return (
        f"Candidate A is {_rank_cutoff_label(a_rank)} ({_rank_band_label(a_rank)}). "
        f"Candidate B is {_rank_cutoff_label(b_rank)} ({_rank_band_label(b_rank)}). "
        f"{order_text}"
    )


def _head_guard_blocker_priority(rank: int) -> int:
    r = int(max(1, rank))
    if 11 <= r <= 15:
        return 0
    if 16 <= r <= 20:
        return 1
    if r <= PAIR_HEAD_PRESERVE_NEG_RANK_MAX:
        return 2
    return 3


def _boundary_blocker_priority(rank: int) -> int:
    r = int(max(1, rank))
    if PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK <= r <= PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK:
        return 0
    if (PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK + 1) <= r <= PAIR_BOUNDARY_SECONDARY_BLOCKER_MAX_RANK:
        return 1
    if 1 <= r < PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK:
        return 2
    return 3


def _mid_blocker_priority(rank: int) -> int:
    r = int(max(1, rank))
    if PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK <= r <= PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK:
        return 0
    secondary_min = max(
        int(PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MIN_RANK),
        int(PAIR_BOUNDARY_PRIMARY_BLOCKER_MAX_RANK) + 1,
    )
    secondary_max = max(
        secondary_min,
        min(int(PAIR_RESCUE_31_60_SECONDARY_BLOCKER_MAX_RANK), int(PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MIN_RANK) - 1),
    )
    if secondary_min <= r <= secondary_max:
        return 1
    if PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MIN_RANK <= r <= PAIR_RESCUE_31_60_PRIMARY_BLOCKER_MAX_RANK:
        return 2
    if 1 <= r < PAIR_BOUNDARY_PRIMARY_BLOCKER_MIN_RANK:
        return 3
    return 4


def _deep_rescue_blocker_priority(rank: int) -> int:
    r = int(max(1, rank))
    if 31 <= r <= 60:
        return 0
    if 11 <= r <= 30:
        return 1
    if 61 <= r <= 100:
        return 2
    if r <= 10:
        return 3
    return 4


def _build_boundary_user_text_from_row(row: dict[str, Any]) -> str:
    profile_text = str(row.get("profile_text", "") or "")
    profile_text_evidence = str(row.get("profile_text_evidence", "") or "")
    profile_text_short = str(row.get("profile_text_short", "") or "")
    profile_text_long = str(row.get("profile_text_long", "") or "")
    profile_pos_text = str(row.get("profile_pos_text", "") or "")
    profile_neg_text = str(row.get("profile_neg_text", "") or "")
    profile_top_pos_tags = str(row.get("profile_top_pos_tags", "") or "")
    profile_top_neg_tags = str(row.get("profile_top_neg_tags", "") or "")
    profile_top_pos_tags_by_type = str(row.get("profile_top_pos_tags_by_type", "") or "")
    profile_top_neg_tags_by_type = str(row.get("profile_top_neg_tags_by_type", "") or "")
    stable_preferences_text = str(row.get("stable_preferences_text", "") or "")
    recent_intent_text_v2 = str(row.get("recent_intent_text_v2", "") or "")
    avoidance_text_v2 = str(row.get("avoidance_text_v2", "") or "")
    history_anchor_hint_text = str(row.get("history_anchor_hint_text", "") or "")
    user_semantic_profile_text_v2 = str(row.get("user_semantic_profile_text_v2", "") or "")
    user_long_pref_text = str(row.get("user_long_pref_text", "") or "")
    user_recent_intent_text = str(row.get("user_recent_intent_text", "") or "")
    user_negative_avoid_text = str(row.get("user_negative_avoid_text", "") or "")
    user_context_text = str(row.get("user_context_text", "") or "")
    profile_confidence = _safe_float(row.get("profile_confidence", 0.0), 0.0)
    user_evidence = str(row.get("user_evidence_text", "") or "")
    history_anchor_text = str(row.get("history_anchor_text", "") or "")
    rebuilt_user_evidence = extract_user_evidence_text(
        profile_text_short or profile_text,
        profile_text_long or profile_text_evidence,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        max_chars=280,
    )
    if rebuilt_user_evidence:
        user_evidence = rebuilt_user_evidence
    return build_user_text(
        profile_text=profile_text,
        profile_text_short=profile_text_short or profile_text,
        profile_text_long=profile_text_long or profile_text_evidence,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        top_pos_tags=profile_top_pos_tags,
        top_neg_tags=profile_top_neg_tags,
        top_pos_tags_by_type=profile_top_pos_tags_by_type,
        top_neg_tags_by_type=profile_top_neg_tags_by_type,
        confidence=profile_confidence,
        evidence_snippets=user_evidence,
        history_anchors=history_anchor_text,
        stable_preferences_text=stable_preferences_text,
        recent_intent_text_v2=recent_intent_text_v2,
        avoidance_text_v2=avoidance_text_v2,
        history_anchor_hint_text=history_anchor_hint_text,
        user_semantic_profile_text_v2=user_semantic_profile_text_v2,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
    )


def _build_boundary_item_text_from_row(row: dict[str, Any]) -> str:
    profile_text = str(row.get("profile_text", "") or "")
    profile_text_evidence = str(row.get("profile_text_evidence", "") or "")
    profile_text_short = str(row.get("profile_text_short", "") or "")
    profile_text_long = str(row.get("profile_text_long", "") or "")
    profile_pos_text = str(row.get("profile_pos_text", "") or "")
    profile_neg_text = str(row.get("profile_neg_text", "") or "")
    profile_top_pos_tags = str(row.get("profile_top_pos_tags", "") or "")
    profile_top_neg_tags = str(row.get("profile_top_neg_tags", "") or "")
    profile_top_pos_tags_by_type = str(row.get("profile_top_pos_tags_by_type", "") or "")
    profile_top_neg_tags_by_type = str(row.get("profile_top_neg_tags_by_type", "") or "")
    core_offering_text = str(row.get("core_offering_text", "") or "")
    scene_fit_text = str(row.get("scene_fit_text", "") or "")
    strengths_text = str(row.get("strengths_text", "") or "")
    risk_points_text = str(row.get("risk_points_text", "") or "")
    merchant_semantic_profile_text_v2 = str(row.get("merchant_semantic_profile_text_v2", "") or "")
    fit_reasons_text_v1 = str(row.get("fit_reasons_text_v1", "") or "")
    friction_reasons_text_v1 = str(row.get("friction_reasons_text_v1", "") or "")
    evidence_basis_text_v1 = str(row.get("evidence_basis_text_v1", "") or "")
    user_long_pref_text = str(row.get("user_long_pref_text", "") or "")
    user_recent_intent_text = str(row.get("user_recent_intent_text", "") or "")
    user_negative_avoid_text = str(row.get("user_negative_avoid_text", "") or "")
    user_context_text = str(row.get("user_context_text", "") or "")
    user_evidence = str(row.get("user_evidence_text", "") or "")
    history_anchor_text = str(row.get("history_anchor_text", "") or "")
    rebuilt_user_evidence = extract_user_evidence_text(
        profile_text_short or profile_text,
        profile_text_long or profile_text_evidence,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        max_chars=280,
    )
    if rebuilt_user_evidence:
        user_evidence = rebuilt_user_evidence
    return build_item_text_sft_clean(
        name=row.get("name", ""),
        city=row.get("city", ""),
        categories=row.get("categories", ""),
        primary_category=row.get("primary_category", ""),
        top_pos_tags=row.get("top_pos_tags", ""),
        top_neg_tags=row.get("top_neg_tags", ""),
        semantic_score=row.get("semantic_score", 0.0),
        semantic_confidence=row.get("semantic_confidence", 0.0),
        cluster_label_for_recsys=row.get("cluster_label_for_recsys", ""),
        item_review_summary=row.get("item_review_summary", ""),
        item_review_snippet=row.get("item_evidence_text", "") or row.get("item_review_snippet", ""),
        user_profile_text=profile_text,
        user_top_pos_tags_by_type=profile_top_pos_tags_by_type,
        user_top_neg_tags_by_type=profile_top_neg_tags_by_type,
        user_top_pos_tags=profile_top_pos_tags,
        user_top_neg_tags=profile_top_neg_tags,
        user_evidence_text=user_evidence,
        history_anchor_text=history_anchor_text,
        user_profile_pos_text=profile_pos_text,
        user_profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        core_offering_text=core_offering_text,
        scene_fit_text=scene_fit_text,
        strengths_text=strengths_text,
        risk_points_text=risk_points_text,
        merchant_semantic_profile_text_v2=merchant_semantic_profile_text_v2,
        fit_reasons_text_v1=fit_reasons_text_v1,
        friction_reasons_text_v1=friction_reasons_text_v1,
        evidence_basis_text_v1=evidence_basis_text_v1,
    )


@lru_cache(maxsize=32768)
def _semantic_tokens_cached(raw: str) -> frozenset[str]:
    pieces = re.split(r"[|,;/>\s]+", raw)
    return frozenset(p for p in pieces if len(p) >= 2)


def _semantic_tokens(value: Any) -> set[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return set()
    return set(_semantic_tokens_cached(raw))


def _local_semantic_overlap_score(focus_row: dict[str, Any], rival_row: dict[str, Any]) -> int:
    score = 0
    focus_city = str(focus_row.get("city", "") or "").strip().lower()
    rival_city = str(rival_row.get("city", "") or "").strip().lower()
    if focus_city and focus_city == rival_city:
        score += 2
    focus_primary = str(focus_row.get("primary_category", "") or "").strip().lower()
    rival_primary = str(rival_row.get("primary_category", "") or "").strip().lower()
    if focus_primary and focus_primary == rival_primary:
        score += 3
    focus_categories = _semantic_tokens(focus_row.get("categories", ""))
    rival_categories = _semantic_tokens(rival_row.get("categories", ""))
    score += min(2, len(focus_categories & rival_categories))
    focus_pos_tags = _semantic_tokens(focus_row.get("top_pos_tags", ""))
    rival_pos_tags = _semantic_tokens(rival_row.get("top_pos_tags", ""))
    score += min(3, len(focus_pos_tags & rival_pos_tags))
    return int(score)


def _comparison_signal_snapshot(pos_row: dict[str, Any], neg_row: dict[str, Any]) -> dict[str, Any]:
    pos_rank = _effective_rank(pos_row)
    neg_rank = _effective_rank(neg_row)
    pos_route = (
        _safe_float(pos_row.get("source_count", 0.0), 0.0)
        + _safe_float(pos_row.get("nonpopular_source_count", 0.0), 0.0)
        + _safe_float(pos_row.get("profile_cluster_source_count", 0.0), 0.0)
    )
    neg_route = (
        _safe_float(neg_row.get("source_count", 0.0), 0.0)
        + _safe_float(neg_row.get("nonpopular_source_count", 0.0), 0.0)
        + _safe_float(neg_row.get("profile_cluster_source_count", 0.0), 0.0)
    )

    pos_support = _safe_float(pos_row.get("semantic_support", 0.0), 0.0) + _safe_float(
        pos_row.get("channel_evidence_support_v1", 0.0), 0.0
    )
    neg_support = _safe_float(neg_row.get("semantic_support", 0.0), 0.0) + _safe_float(
        neg_row.get("channel_evidence_support_v1", 0.0), 0.0
    )
    pos_semantic_score = _safe_float(pos_row.get("semantic_score", 0.0), 0.0)
    neg_semantic_score = _safe_float(neg_row.get("semantic_score", 0.0), 0.0)
    pos_semantic_confidence = _safe_float(pos_row.get("semantic_confidence", 0.0), 0.0)
    neg_semantic_confidence = _safe_float(neg_row.get("semantic_confidence", 0.0), 0.0)
    pos_semantic_tag_richness = _safe_float(pos_row.get("semantic_tag_richness", 0.0), 0.0)
    neg_semantic_tag_richness = _safe_float(neg_row.get("semantic_tag_richness", 0.0), 0.0)
    pos_channel_context = (
        _safe_float(pos_row.get("channel_preference_core_v1", 0.0), 0.0)
        + _safe_float(pos_row.get("channel_recent_intent_v1", 0.0), 0.0)
        + _safe_float(pos_row.get("channel_context_time_v1", 0.0), 0.0)
        - _safe_float(pos_row.get("channel_conflict_v1", 0.0), 0.0)
    )
    neg_channel_context = (
        _safe_float(neg_row.get("channel_preference_core_v1", 0.0), 0.0)
        + _safe_float(neg_row.get("channel_recent_intent_v1", 0.0), 0.0)
        + _safe_float(neg_row.get("channel_context_time_v1", 0.0), 0.0)
        - _safe_float(neg_row.get("channel_conflict_v1", 0.0), 0.0)
    )

    pos_stability = (
        _safe_float(pos_row.get("channel_preference_core_v1", 0.0), 0.0)
        + _safe_float(pos_row.get("channel_recent_intent_v1", 0.0), 0.0)
        + _safe_float(pos_row.get("channel_context_time_v1", 0.0), 0.0)
        - _safe_float(pos_row.get("sim_negative_avoid_neg", pos_row.get("avoid_neg", 0.0)), 0.0)
        - _safe_float(pos_row.get("sim_negative_avoid_core", pos_row.get("avoid_core", 0.0)), 0.0)
        - _safe_float(pos_row.get("sim_conflict_gap", pos_row.get("conflict_gap", 0.0)), 0.0)
        - _safe_float(pos_row.get("channel_conflict_v1", 0.0), 0.0)
    )
    neg_stability = (
        _safe_float(neg_row.get("channel_preference_core_v1", 0.0), 0.0)
        + _safe_float(neg_row.get("channel_recent_intent_v1", 0.0), 0.0)
        + _safe_float(neg_row.get("channel_context_time_v1", 0.0), 0.0)
        - _safe_float(neg_row.get("sim_negative_avoid_neg", neg_row.get("avoid_neg", 0.0)), 0.0)
        - _safe_float(neg_row.get("sim_negative_avoid_core", neg_row.get("avoid_core", 0.0)), 0.0)
        - _safe_float(neg_row.get("sim_conflict_gap", neg_row.get("conflict_gap", 0.0)), 0.0)
        - _safe_float(neg_row.get("channel_conflict_v1", 0.0), 0.0)
    )
    pos_route_edge = bool(pos_route >= neg_route + 1.0)
    neg_route_edge = bool(neg_route >= pos_route + 1.0)
    pos_support_edge = bool(pos_support >= neg_support + 0.25)
    neg_support_edge = bool(neg_support >= pos_support + 0.25)
    pos_semantic_edge = bool(
        pos_support_edge
        or pos_semantic_score >= neg_semantic_score + 0.02
        or pos_semantic_confidence >= neg_semantic_confidence + 0.05
        or pos_semantic_tag_richness >= neg_semantic_tag_richness + 1.0
    )
    neg_semantic_edge = bool(
        neg_support_edge
        or neg_semantic_score >= pos_semantic_score + 0.02
        or neg_semantic_confidence >= pos_semantic_confidence + 0.05
        or neg_semantic_tag_richness >= pos_semantic_tag_richness + 1.0
    )
    pos_channel_context_edge = bool(pos_channel_context >= neg_channel_context + 0.25)
    neg_channel_context_edge = bool(neg_channel_context >= pos_channel_context + 0.25)
    pos_stability_edge = bool(pos_stability >= neg_stability + 0.25)
    neg_stability_edge = bool(neg_stability >= pos_stability + 0.25)
    return {
        "pos_route_edge": pos_route_edge,
        "neg_route_edge": neg_route_edge,
        "pos_support_edge": pos_support_edge,
        "neg_support_edge": neg_support_edge,
        "pos_semantic_edge": pos_semantic_edge,
        "neg_semantic_edge": neg_semantic_edge,
        "pos_channel_context_edge": pos_channel_context_edge,
        "neg_channel_context_edge": neg_channel_context_edge,
        "pos_stability_edge": pos_stability_edge,
        "neg_stability_edge": neg_stability_edge,
        # Count the dimensions we can actually explain at inference time. Excluding
        # semantic/channel edges was suppressing many 11-100 rescue cases from training.
        "pos_explainable_edge_count": int(pos_route_edge) + int(pos_semantic_edge) + int(pos_channel_context_edge) + int(pos_stability_edge),
        "neg_explainable_edge_count": int(neg_route_edge) + int(neg_semantic_edge) + int(neg_channel_context_edge) + int(neg_stability_edge),
        "pos_rank": int(pos_rank),
        "neg_rank": int(neg_rank),
    }


def _comparison_summary_text(pos_row: dict[str, Any], neg_row: dict[str, Any]) -> str:
    pos_rank = _effective_rank(pos_row)
    neg_rank = _effective_rank(neg_row)
    same_city = int(
        str(pos_row.get("city", "") or "").strip().lower()
        and str(pos_row.get("city", "") or "").strip().lower()
        == str(neg_row.get("city", "") or "").strip().lower()
    )
    same_primary = int(
        str(pos_row.get("primary_category", "") or "").strip().lower()
        and str(pos_row.get("primary_category", "") or "").strip().lower()
        == str(neg_row.get("primary_category", "") or "").strip().lower()
    )
    pos_terms = _semantic_tokens(pos_row.get("categories", "")) | _semantic_tokens(pos_row.get("top_pos_tags", ""))
    neg_terms = _semantic_tokens(neg_row.get("categories", "")) | _semantic_tokens(neg_row.get("top_pos_tags", ""))
    shared_terms = len(pos_terms & neg_terms)
    return (
        f"candidate_a_band: {_rank_band_label(pos_rank)}; "
        f"candidate_b_band: {_rank_band_label(neg_rank)}; "
        f"local_context: same_city={same_city}; same_primary_category={same_primary}; "
        f"shared_theme_terms={min(int(shared_terms), 6)}"
    )


def _edge_count_summary(focus_wins: int, rival_wins: int, total: int) -> str:
    if total <= 0:
        return "insufficient local evidence"
    return _local_listwise_edge_tally(focus_wins, rival_wins, total)


def _local_listwise_edge_tally(focus_wins: int, rival_wins: int, total: int) -> str:
    if total <= 0:
        return "limited signal"
    if focus_wins <= 0 and rival_wins <= 0:
        return "mixed"
    if focus_wins >= max(2, rival_wins + 2):
        return "focus shows repeated edge cues"
    if rival_wins >= max(2, focus_wins + 2):
        return "rivals show repeated edge cues"
    if focus_wins > rival_wins:
        return "focus has a slight lean"
    if rival_wins > focus_wins:
        return "rivals have a slight lean"
    return "mixed"


def _local_listwise_gap_text(focus_rank: int) -> str:
    focus_rank = int(max(1, focus_rank))
    if focus_rank <= 10:
        return "inside current top10"
    if focus_rank <= 20:
        return "just outside the current top10"
    if focus_rank <= 40:
        return "inside the near-boundary rescue range"
    if focus_rank <= 60:
        return "inside the mid rescue range"
    if focus_rank <= 100:
        return "inside the deep rescue range"
    return "outside current rescue scope"


def _local_listwise_band_quota_plan(focus_rank: int) -> list[tuple[str, int]]:
    band = _boundary_rank_band(int(max(1, focus_rank)))
    if band == "head_guard":
        return [("boundary_11_30", 2), ("rescue_31_60", 2), ("head_guard", 1)]
    if band == "boundary_11_30":
        return [("boundary_11_30", 3), ("head_guard", 1), ("rescue_31_60", 1)]
    if band == "rescue_31_60":
        return [("rescue_31_60", 3), ("boundary_11_30", 2), ("head_guard", 0)]
    if band == "rescue_61_100":
        return [("rescue_61_100", 3), ("rescue_31_60", 2), ("boundary_11_30", 0)]
    return [("head_guard", 1), ("boundary_11_30", 2), ("rescue_31_60", 1)]


def _local_listwise_relation_priority(focus_rank: int, rival_rank: int) -> int:
    focus_rank = int(max(1, focus_rank))
    rival_rank = int(max(1, rival_rank))
    if focus_rank <= 10:
        return 0 if rival_rank > focus_rank else 1
    return 0 if rival_rank < focus_rank else 1


def _local_listwise_rival_sort_key(focus_row: dict[str, Any], rival_row: dict[str, Any]) -> tuple[Any, ...]:
    focus_rank = _effective_rank(focus_row)
    rival_rank = _effective_rank(rival_row)
    focus_band = _boundary_rank_band(focus_rank)
    rank_priority = _focus_band_rival_band_priority(focus_band, rival_rank)
    semantic_overlap = _local_semantic_overlap_score(focus_row, rival_row)
    return (
        _local_listwise_relation_priority(focus_rank, rival_rank),
        rank_priority,
        -semantic_overlap,
        abs(int(rival_rank) - int(focus_rank)),
        int(rival_rank),
    )


def _select_local_listwise_rivals(
    focus_row: dict[str, Any],
    rival_rows: list[dict[str, Any]],
    max_rivals: int,
) -> list[dict[str, Any]]:
    if max_rivals <= 0:
        return []
    focus_rank = _effective_rank(focus_row)
    unique_rivals: list[dict[str, Any]] = []
    seen_item_ids: set[int] = set()
    for row in rival_rows:
        item_idx = _safe_int(row.get("item_idx", -1), -1)
        if item_idx < 0 or item_idx in seen_item_ids:
            continue
        seen_item_ids.add(item_idx)
        unique_rivals.append(row)
    if not unique_rivals:
        return []
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rivals:
        bucketed[_boundary_rank_band(_effective_rank(row))].append(row)
    for band_rows in bucketed.values():
        band_rows.sort(key=lambda row: _local_listwise_rival_sort_key(focus_row, row))
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for band, quota in _local_listwise_band_quota_plan(focus_rank):
        if len(selected) >= max_rivals:
            break
        picked = 0
        for row in bucketed.get(str(band), []):
            item_idx = _safe_int(row.get("item_idx", -1), -1)
            if item_idx < 0 or item_idx in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(item_idx)
            picked += 1
            if picked >= int(quota) or len(selected) >= max_rivals:
                break
    if len(selected) < max_rivals:
        remaining = sorted(unique_rivals, key=lambda row: _local_listwise_rival_sort_key(focus_row, row))
        for row in remaining:
            item_idx = _safe_int(row.get("item_idx", -1), -1)
            if item_idx < 0 or item_idx in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(item_idx)
            if len(selected) >= max_rivals:
                break
    return selected[:max_rivals]


def _local_listwise_ranking_context(focus_row: dict[str, Any], rival_rows: list[dict[str, Any]]) -> str:
    focus_rank = _effective_rank(focus_row)
    rival_ranks = [_effective_rank(row) for row in rival_rows]
    if not rival_ranks:
        return f"Focus candidate is in the {_rank_band_label(focus_rank)}."
    head_rivals = sum(1 for r in rival_ranks if int(r) <= 10)
    boundary_rivals = sum(1 for r in rival_ranks if 11 <= int(r) <= 30)
    mid_rivals = sum(1 for r in rival_ranks if 31 <= int(r) <= 60)
    deep_rivals = sum(1 for r in rival_ranks if 61 <= int(r) <= 100)
    focus_band = _boundary_rank_band(int(focus_rank))
    if focus_band == "rescue_31_60":
        return (
            "Focus candidate is in the mid rescue band (31-60). "
            "Judge whether it should move above nearby mid-rank blockers and boundary blockers in this local rescue slate. "
            "Entering the global top10 is not required by itself. "
            f"Local slate contains {len(rival_ranks)} rivals; "
            f"same_band={mid_rivals}, boundary={boundary_rivals}, head_anchor={head_rivals}, deep={deep_rivals}."
        )
    if focus_band == "rescue_61_100":
        return (
            "Focus candidate is in the deep rescue band (61-100). "
            "Judge whether it should move above nearby deep-rank blockers, mid-rank rescue blockers, and selected boundary or head anchors in this local rescue slate. "
            "Entering the global top10 is not required by itself. "
            f"Local slate contains {len(rival_ranks)} rivals; "
            f"same_band={deep_rivals}, mid_blockers={mid_rivals}, boundary_blockers={boundary_rivals}, head_anchor={head_rivals}."
        )
    return (
        f"Focus candidate is in the {_rank_band_label(focus_rank)}. "
        f"Local slate contains {len(rival_ranks)} rivals; "
        f"head={head_rivals}, boundary={boundary_rivals}, mid={mid_rivals}, deep={deep_rivals}."
    )


def _local_listwise_slate_summary(focus_row: dict[str, Any], rival_rows: list[dict[str, Any]]) -> str:
    focus_rank = _effective_rank(focus_row)
    rival_ranks = [_effective_rank(row) for row in rival_rows]
    head_rivals = sum(1 for r in rival_ranks if r <= 10)
    boundary_rivals = sum(1 for r in rival_ranks if 11 <= r <= 30)
    mid_rivals = sum(1 for r in rival_ranks if 31 <= r <= 60)
    deep_rivals = sum(1 for r in rival_ranks if 61 <= r <= 100)
    same_city_rivals = 0
    same_primary_rivals = 0
    similar_theme_rivals = 0
    for rival_row in rival_rows:
        if (
            str(focus_row.get("city", "") or "").strip().lower()
            and str(focus_row.get("city", "") or "").strip().lower()
            == str(rival_row.get("city", "") or "").strip().lower()
        ):
            same_city_rivals += 1
        if (
            str(focus_row.get("primary_category", "") or "").strip().lower()
            and str(focus_row.get("primary_category", "") or "").strip().lower()
            == str(rival_row.get("primary_category", "") or "").strip().lower()
        ):
            same_primary_rivals += 1
        if _local_semantic_overlap_score(focus_row, rival_row) >= 3:
            similar_theme_rivals += 1
    focus_band = _boundary_rank_band(int(focus_rank))
    if focus_band == "rescue_31_60":
        focus_subband = _mid_rescue_subband(int(focus_rank))
        same_band_blockers = sum(1 for r in rival_ranks if 31 <= r <= 60)
        boundary_blockers = sum(1 for r in rival_ranks if 11 <= r <= 30)
        head_anchors = sum(1 for r in rival_ranks if r <= 10)
        deep_blockers = sum(1 for r in rival_ranks if 61 <= r <= 100)
        same_band_semantic = sum(
            1
            for rival_row in rival_rows
            if 31 <= _effective_rank(rival_row) <= 60 and _local_semantic_overlap_score(focus_row, rival_row) >= 3
        )
        boundary_semantic = sum(
            1
            for rival_row in rival_rows
            if 11 <= _effective_rank(rival_row) <= 30 and _local_semantic_overlap_score(focus_row, rival_row) >= 3
        )
        return (
            f"cutoff_context: {_local_listwise_gap_text(focus_rank)}; "
            f"rescue_slice: {focus_subband}; "
            f"blocker_types: same_band={same_band_blockers}, boundary={boundary_blockers}, "
            f"head_anchor={head_anchors}, deep={deep_blockers}; "
            f"semantic_pressure: same_band_semantic={same_band_semantic}, boundary_semantic={boundary_semantic}, "
            f"same_city={same_city_rivals}, same_primary_category={same_primary_rivals}"
        )
    if focus_band == "rescue_61_100":
        same_band_blockers = sum(1 for r in rival_ranks if 61 <= r <= 100)
        mid_blockers = sum(1 for r in rival_ranks if 31 <= r <= 60)
        boundary_blockers = sum(1 for r in rival_ranks if 11 <= r <= 30)
        head_anchors = sum(1 for r in rival_ranks if r <= 10)
        same_band_semantic = sum(
            1
            for rival_row in rival_rows
            if 61 <= _effective_rank(rival_row) <= 100 and _local_semantic_overlap_score(focus_row, rival_row) >= 3
        )
        mid_semantic = sum(
            1
            for rival_row in rival_rows
            if 31 <= _effective_rank(rival_row) <= 60 and _local_semantic_overlap_score(focus_row, rival_row) >= 3
        )
        boundary_semantic = sum(
            1
            for rival_row in rival_rows
            if 11 <= _effective_rank(rival_row) <= 30 and _local_semantic_overlap_score(focus_row, rival_row) >= 3
        )
        return (
            f"cutoff_context: {_local_listwise_gap_text(focus_rank)}; "
            f"blocker_types: same_band={same_band_blockers}, mid={mid_blockers}, "
            f"boundary={boundary_blockers}, head_anchor={head_anchors}; "
            f"semantic_pressure: same_band_semantic={same_band_semantic}, mid_semantic={mid_semantic}, "
            f"boundary_semantic={boundary_semantic}, same_city={same_city_rivals}, same_primary_category={same_primary_rivals}"
        )
    return (
        f"cutoff_context: {_local_listwise_gap_text(focus_rank)}; "
        f"rival_mix: head={head_rivals}, boundary={boundary_rivals}, mid={mid_rivals}, deep={deep_rivals}; "
        f"semantic_neighborhood: same_city={same_city_rivals}, "
        f"same_primary_category={same_primary_rivals}, similar_theme_rivals={similar_theme_rivals}"
    )


def _focus_vs_local_slate_summary(focus_row: dict[str, Any], rival_rows: list[dict[str, Any]]) -> str:
    route_focus = route_rival = 0
    semantic_focus = semantic_rival = 0
    support_focus = support_rival = 0
    channel_focus = channel_rival = 0
    stability_focus = stability_rival = 0
    for rival_row in rival_rows:
        signals = _comparison_signal_snapshot(focus_row, rival_row)
        route_focus += int(bool(signals["pos_route_edge"]) and not bool(signals["neg_route_edge"]))
        route_rival += int(bool(signals["neg_route_edge"]) and not bool(signals["pos_route_edge"]))
        semantic_focus += int(bool(signals["pos_semantic_edge"]) and not bool(signals["neg_semantic_edge"]))
        semantic_rival += int(bool(signals["neg_semantic_edge"]) and not bool(signals["pos_semantic_edge"]))
        support_focus += int(bool(signals["pos_support_edge"]) and not bool(signals["neg_support_edge"]))
        support_rival += int(bool(signals["neg_support_edge"]) and not bool(signals["pos_support_edge"]))
        channel_focus += int(bool(signals["pos_channel_context_edge"]) and not bool(signals["neg_channel_context_edge"]))
        channel_rival += int(bool(signals["neg_channel_context_edge"]) and not bool(signals["pos_channel_context_edge"]))
        stability_focus += int(bool(signals["pos_stability_edge"]) and not bool(signals["neg_stability_edge"]))
        stability_rival += int(bool(signals["neg_stability_edge"]) and not bool(signals["pos_stability_edge"]))
    focus_rank = _effective_rank(focus_row)
    head_rivals = sum(1 for row in rival_rows if _effective_rank(row) <= 10)
    if focus_rank <= 10:
        head_prior_pressure = "focus is already inside the current head set"
    else:
        if head_rivals >= 2:
            head_prior_pressure = "multiple head rivals remain ahead in this slate"
        elif head_rivals == 1:
            head_prior_pressure = "one head rival remains ahead in this slate"
        else:
            head_prior_pressure = "no head rival appears in this local slate"
    def _edge_phrase(label: str, focus_ct: int, rival_ct: int) -> str:
        if focus_ct <= 0 and rival_ct <= 0:
            return ""
        if focus_ct >= max(2, rival_ct + 1):
            return f"{label} more often leans toward the focus candidate"
        if rival_ct >= max(2, focus_ct + 1):
            return f"{label} more often leans toward the nearby rivals"
        if focus_ct > rival_ct:
            return f"{label} gives the focus candidate a slight edge"
        if rival_ct > focus_ct:
            return f"{label} gives the nearby rivals a slight edge"
        return f"{label} looks roughly balanced across this slate"

    summary_bits = [
        _edge_phrase("semantic fit", semantic_focus, semantic_rival),
        _edge_phrase("evidence support", support_focus, support_rival),
        _edge_phrase("recent context and visit pattern", channel_focus, channel_rival),
        _edge_phrase("long-run preference consistency", stability_focus, stability_rival),
        _edge_phrase("route and ranking cues", route_focus, route_rival),
    ]
    summary_bits = [bit for bit in summary_bits if bit]
    if _boundary_rank_band(int(focus_rank)) == "rescue_31_60":
        focus_subband = _mid_rescue_subband(int(focus_rank))
        under_rank_clues: list[str] = []
        if semantic_focus > semantic_rival or support_focus > support_rival:
            under_rank_clues.append("semantic evidence is stronger than the current mid-rank placement")
        if channel_focus > channel_rival or stability_focus > stability_rival:
            under_rank_clues.append("recent context and long-run preference fit both support an upward rescue")
        if route_focus > route_rival:
            under_rank_clues.append("route and ranking cues suggest the focus should sit above some local blockers")
        if focus_subband == "31_40":
            under_rank_clues.append("the focus sits in the 31-40 slice, so clearing boundary blockers can move it toward top10")
        else:
            under_rank_clues.append("the focus sits in the 41-60 slice, so the first goal is to climb toward top20 and top30")
        if under_rank_clues:
            summary_bits.insert(0, "; ".join(under_rank_clues[:2]))
    if _boundary_rank_band(int(focus_rank)) == "rescue_61_100":
        under_rank_clues = []
        if semantic_focus > semantic_rival or support_focus > support_rival:
            under_rank_clues.append("semantic evidence is stronger than the current deep-rank placement")
        if channel_focus > channel_rival or stability_focus > stability_rival:
            under_rank_clues.append("recent context and long-run preference fit support upward rescue from the deep range")
        if route_focus > route_rival:
            under_rank_clues.append("route and ranking cues suggest the focus should clear some deeper local blockers")
        if under_rank_clues:
            summary_bits.insert(0, "; ".join(under_rank_clues[:2]))
    if head_prior_pressure:
        summary_bits.append(head_prior_pressure)
    summary_bits.append(_local_listwise_gap_text(focus_rank))
    return "; ".join(summary_bits)


def _local_listwise_rival_role(focus_row: dict[str, Any], rival_row: dict[str, Any]) -> str:
    focus_band = _boundary_rank_band(_effective_rank(focus_row))
    rival_rank = _effective_rank(rival_row)
    if focus_band == "rescue_61_100":
        semantic_overlap = _local_semantic_overlap_score(focus_row, rival_row)
        if 61 <= rival_rank <= 100:
            return "same-band semantic blocker" if semantic_overlap >= 3 else "same-band blocker"
        if 31 <= rival_rank <= 60:
            return "mid-band semantic blocker" if semantic_overlap >= 3 else "mid-band blocker"
        if 11 <= rival_rank <= 30:
            return "boundary semantic blocker" if semantic_overlap >= 3 else "boundary blocker"
        if rival_rank <= 10:
            return "head anchor"
        return "outside-band blocker"
    if focus_band != "rescue_31_60":
        return _rank_band_label(rival_rank)
    semantic_overlap = _local_semantic_overlap_score(focus_row, rival_row)
    if 31 <= rival_rank <= 60:
        focus_subband = _mid_rescue_subband(_effective_rank(focus_row))
        rival_subband = _mid_rescue_subband(rival_rank)
        if focus_subband == rival_subband:
            return "same-slice semantic blocker" if semantic_overlap >= 3 else "same-slice blocker"
        return "adjacent mid semantic blocker" if semantic_overlap >= 3 else "adjacent mid blocker"
    if 11 <= rival_rank <= 30:
        return "boundary semantic blocker" if semantic_overlap >= 3 else "boundary blocker"
    if rival_rank <= 10:
        return "head anchor"
    return "deep rescue blocker"


def _mid_rescue_subband(rank: int) -> str:
    rank_int = int(_safe_int(rank, 999999))
    if 31 <= rank_int <= 40:
        return "31_40"
    if 41 <= rank_int <= 60:
        return "41_60"
    return "outside_mid"


def _deep_rescue_subband(rank: int) -> str:
    rank_int = int(_safe_int(rank, 999999))
    if 61 <= rank_int <= 80:
        return "61_80"
    if 81 <= rank_int <= 100:
        return "81_100"
    return "outside_deep"


def _ordered_ranked_neg_pool(
    ranked_neg: list[tuple[dict[str, Any], str, dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    return [entry for entry in ranked_neg if predicate(entry[0])]


def _build_mid_slate_variant(
    quotas: list[tuple[list[tuple[dict[str, Any], str, dict[str, Any]]], int]],
    fallback_pool: list[tuple[dict[str, Any], str, dict[str, Any]]],
    slate_size: int,
    rotation_seed: int = 0,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    chosen: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    seen_items: set[int] = set()

    def _rotate_pool(
        pool: list[tuple[dict[str, Any], str, dict[str, Any]]],
        offset: int,
    ) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
        if len(pool) <= 1 or offset <= 0:
            return pool
        start = int(offset) % len(pool)
        if start <= 0:
            return pool
        return list(pool[start:]) + list(pool[:start])

    def _take_from_pool(
        pool: list[tuple[dict[str, Any], str, dict[str, Any]]],
        quota: int,
    ) -> None:
        if quota <= 0:
            return
        taken = 0
        for entry in pool:
            item_idx = _safe_int(entry[0].get("item_idx", -1), -1)
            if item_idx < 0 or item_idx in seen_items:
                continue
            seen_items.add(item_idx)
            chosen.append(entry)
            taken += 1
            if taken >= int(quota) or len(chosen) >= int(slate_size):
                break

    for quota_idx, (pool, quota) in enumerate(quotas):
        _take_from_pool(_rotate_pool(pool, int(rotation_seed) + int(quota_idx)), int(quota))
        if len(chosen) >= int(slate_size):
            break

    if len(chosen) < int(slate_size):
        _take_from_pool(_rotate_pool(fallback_pool, int(rotation_seed)), int(slate_size) - len(chosen))

    return chosen[: int(slate_size)]


def _target_pairs_for_positive(
    source_name: str,
    pos_band: str,
    remaining_budget: int,
) -> int:
    target = max(0, int(remaining_budget))
    if target <= 0:
        return 0
    if str(PAIR_PROMPT_STYLE or "").strip().lower() != "local_listwise_compare":
        return int(target)
    if source_name != "true":
        return int(target)
    if pos_band == "boundary_11_30":
        return min(int(target), int(PAIR_LOCAL_LISTWISE_MAX_RIVALS))
    if pos_band == "rescue_31_60":
        return min(int(target), int(PAIR_LOCAL_LISTWISE_MAX_RIVALS) * int(PAIR_V2D_TRUE_MID_SLATE_VARIANTS))
    if pos_band == "rescue_61_100":
        return min(int(target), int(PAIR_LOCAL_LISTWISE_MAX_RIVALS) * int(PAIR_V2D_TRUE_DEEP_SLATE_VARIANTS))
    return int(target)


def _effective_pair_policy(source_name: str, pos_band: str) -> str:
    policy = str(DPO_PAIR_POLICY or "v1").strip().lower() or "v1"
    if (
        source_name == "true"
        and pos_band == "boundary_11_30"
        and str(PAIR_PROMPT_STYLE or "").strip().lower() == "local_listwise_compare"
        and PAIR_FREEZE_BOUNDARY_11_30_V1
    ):
        return "v1"
    return policy


def _build_local_listwise_slate_variants(
    pos_row: dict[str, Any],
    ranked_neg: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> list[tuple[int, list[tuple[dict[str, Any], str, dict[str, Any]]]]]:
    if not ranked_neg:
        return []
    slate_size = max(1, int(PAIR_LOCAL_LISTWISE_MAX_RIVALS))
    focus_band = _boundary_rank_band(_effective_rank(pos_row))
    if focus_band == "rescue_61_100":
        variant_count = max(1, int(PAIR_V2D_TRUE_DEEP_SLATE_VARIANTS))
        if variant_count <= 1:
            return [(0, list(ranked_neg[:slate_size]))]
        if PAIR_V2D_TRUE_DEEP_EXPLICIT_SLATE_TYPES:
            # Deep rescue uses explicit blocker templates so the model sees
            # realistic "who is ahead of me" structure: same-band peers,
            # 31-60 blockers, 11-30 boundary blockers, and occasional head anchors.
            focus_subband = _deep_rescue_subband(_effective_rank(pos_row))
            same_band_semantic = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 61 <= _effective_rank(row) <= 100 and _local_semantic_overlap_score(pos_row, row) >= 3,
            )
            same_band_other = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 61 <= _effective_rank(row) <= 100 and _local_semantic_overlap_score(pos_row, row) < 3,
            )
            mid_band_semantic = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 31 <= _effective_rank(row) <= 60 and _local_semantic_overlap_score(pos_row, row) >= 3,
            )
            mid_band_other = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 31 <= _effective_rank(row) <= 60 and _local_semantic_overlap_score(pos_row, row) < 3,
            )
            boundary_semantic = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 11 <= _effective_rank(row) <= 30 and _local_semantic_overlap_score(pos_row, row) >= 3,
            )
            boundary_other = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: 11 <= _effective_rank(row) <= 30 and _local_semantic_overlap_score(pos_row, row) < 3,
            )
            head_anchor = _ordered_ranked_neg_pool(
                ranked_neg,
                lambda row: _effective_rank(row) <= 10,
            )
            same_band_any = same_band_semantic + same_band_other
            mid_band_any = mid_band_semantic + mid_band_other
            boundary_any = boundary_semantic + boundary_other
            fallback_pool = list(ranked_neg)
            if focus_subband == "61_80":
                variant_specs = [
                    [
                        (same_band_any, 2),
                        (mid_band_any, 1),
                        (boundary_any, 1),
                    ],
                    [
                        (same_band_semantic, 1),
                        (mid_band_any, 1),
                        (boundary_any, 1),
                        (head_anchor, 1),
                    ],
                    [
                        (same_band_any, 1),
                        (mid_band_any, 2),
                        (boundary_any, 1),
                    ],
                ]
            else:
                variant_specs = [
                    [
                        (same_band_any, 2),
                        (mid_band_any, 1),
                        (boundary_any, 1),
                    ],
                    [
                        (same_band_semantic, 1),
                        (same_band_other, 1),
                        (mid_band_any, 1),
                        (head_anchor, 1),
                    ],
                    [
                        (same_band_any, 1),
                        (mid_band_any, 1),
                        (boundary_any, 1),
                        (head_anchor, 1),
                    ],
                ]
            variants: list[tuple[int, list[tuple[dict[str, Any], str, dict[str, Any]]]]] = []
            seen_variant_keys: set[tuple[int, ...]] = set()
            for variant_idx, quotas in enumerate(variant_specs[: int(variant_count)]):
                chunk = _build_mid_slate_variant(
                    quotas,
                    fallback_pool,
                    int(slate_size),
                    rotation_seed=int(variant_idx) * max(1, int(slate_size) - 1),
                )
                if len(chunk) < int(slate_size):
                    continue
                variant_key = tuple(_safe_int(entry[0].get("item_idx", -1), -1) for entry in chunk)
                if variant_key in seen_variant_keys:
                    continue
                seen_variant_keys.add(variant_key)
                variants.append((int(variant_idx), chunk))
            if variants:
                return variants
        variants: list[tuple[int, list[tuple[dict[str, Any], str, dict[str, Any]]]]] = []
        for variant_idx in range(variant_count):
            start = int(variant_idx) * int(slate_size)
            chunk = list(ranked_neg[start : start + slate_size])
            if len(chunk) < int(slate_size):
                break
            variants.append((int(variant_idx), chunk))
        if not variants:
            return [(0, list(ranked_neg[:slate_size]))]
        return variants
    if focus_band != "rescue_31_60":
        return [(0, list(ranked_neg[:slate_size]))]
    variant_count = max(1, int(PAIR_V2D_TRUE_MID_SLATE_VARIANTS))
    if variant_count <= 1:
        return [(0, list(ranked_neg[:slate_size]))]
    if PAIR_V2D_TRUE_MID_EXPLICIT_SLATE_TYPES:
        focus_subband = _mid_rescue_subband(_effective_rank(pos_row))
        same_band_semantic = _ordered_ranked_neg_pool(
            ranked_neg,
            lambda row: 31 <= _effective_rank(row) <= 60 and _local_semantic_overlap_score(pos_row, row) >= 3,
        )
        same_band_other = _ordered_ranked_neg_pool(
            ranked_neg,
            lambda row: 31 <= _effective_rank(row) <= 60 and _local_semantic_overlap_score(pos_row, row) < 3,
        )
        boundary_semantic = _ordered_ranked_neg_pool(
            ranked_neg,
            lambda row: 11 <= _effective_rank(row) <= 30 and _local_semantic_overlap_score(pos_row, row) >= 3,
        )
        boundary_other = _ordered_ranked_neg_pool(
            ranked_neg,
            lambda row: 11 <= _effective_rank(row) <= 30 and _local_semantic_overlap_score(pos_row, row) < 3,
        )
        head_anchor = _ordered_ranked_neg_pool(
            ranked_neg,
            lambda row: _effective_rank(row) <= 10,
        )
        fallback_pool = list(ranked_neg)
        if focus_subband == "31_40":
            same_band_any = same_band_semantic + same_band_other
            boundary_any = boundary_semantic + boundary_other
            variant_specs = [
                [
                    (same_band_semantic, 2),
                    (same_band_other, 1),
                    (boundary_any, 1),
                ],
                [
                    (boundary_any, 2),
                    (same_band_semantic, 1),
                    (head_anchor, 1),
                ],
                [
                    (head_anchor, 1),
                    (boundary_any, 1),
                    (same_band_any, 2),
                ],
                [
                    (boundary_any, 2),
                    (head_anchor, 1),
                    (same_band_any, 1),
                ],
            ]
            variant_limit = max(int(variant_count), int(PAIR_V2D_TRUE_MID_31_40_SLATE_VARIANTS))
        else:
            variant_specs = [
                [
                    (same_band_semantic, 2),
                    (same_band_other, 2),
                ],
                [
                    (boundary_semantic, 2),
                    (boundary_other, 1),
                    (same_band_semantic + same_band_other, 1),
                ],
                [
                    (head_anchor, 1),
                    (boundary_semantic + boundary_other, 1),
                    (same_band_semantic + same_band_other, 2),
                ],
            ]
            variant_limit = int(variant_count)
        variants: list[tuple[int, list[tuple[dict[str, Any], str, dict[str, Any]]]]] = []
        seen_variant_keys: set[tuple[int, ...]] = set()
        for variant_idx, quotas in enumerate(variant_specs[: int(variant_limit)]):
            chunk = _build_mid_slate_variant(
                quotas,
                fallback_pool,
                int(slate_size),
                rotation_seed=int(variant_idx) * max(1, int(slate_size) - 1),
            )
            if len(chunk) < int(slate_size):
                continue
            variant_key = tuple(_safe_int(entry[0].get("item_idx", -1), -1) for entry in chunk)
            if variant_key in seen_variant_keys:
                continue
            seen_variant_keys.add(variant_key)
            variants.append((int(variant_idx), chunk))
        if variants:
            return variants
    variants: list[tuple[int, list[tuple[dict[str, Any], str, dict[str, Any]]]]] = []
    for variant_idx in range(variant_count):
        start = int(variant_idx) * int(slate_size)
        chunk = list(ranked_neg[start : start + slate_size])
        if len(chunk) < int(slate_size):
            break
        variants.append((int(variant_idx), chunk))
    if not variants:
        return [(0, list(ranked_neg[:slate_size]))]
    return variants


def _deep_cross_variant_reusable_bucket(neg_bucket: str) -> bool:
    if not PAIR_V2D_TRUE_DEEP_ALLOW_CROSS_VARIANT_REUSE:
        return False
    bucket = str(neg_bucket or "").strip()
    return bucket in {
        "mid_band_rescue_61_100_competitor",
        "boundary_blocker_for_deep",
        "head_anchor_for_deep",
    }


def _deep_variant_fill_priority(neg_bucket: str) -> int:
    bucket = str(neg_bucket or "").strip()
    if bucket == "mid_band_rescue_61_100_competitor":
        return 0
    if bucket == "boundary_blocker_for_deep":
        return 1
    if bucket == "head_anchor_for_deep":
        return 2
    if bucket == "same_band_rescue_61_100_competitor":
        return 3
    if bucket in {"rescue_61_100_confuser", "clear_multi_route_rescue_confuser"}:
        return 4
    return 5


def _finalize_local_deep_variant_rows(
    ranked_neg_variant: list[tuple[dict[str, Any], str, dict[str, Any]]],
    ranked_neg_full: list[tuple[dict[str, Any], str, dict[str, Any]]],
    seen_neg_items: set[int],
    seen_neg_usage: Counter[int],
    *,
    slate_variant_id: int,
    slate_size: int,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    if not ranked_neg_variant:
        return []

    allow_cross_variant_reuse = int(slate_variant_id) > 0
    target_size = max(1, int(slate_size))
    local_seen_items: set[int] = set()
    selected: list[tuple[dict[str, Any], str, dict[str, Any]]] = []

    def _can_take_entry(
        entry: tuple[dict[str, Any], str, dict[str, Any]],
        *,
        allow_reuse: bool,
    ) -> bool:
        neg_row, neg_bucket, _ = entry
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0 or neg_item_idx in local_seen_items:
            return False
        if neg_item_idx not in seen_neg_items:
            return True
        if not allow_reuse:
            return False
        if not _deep_cross_variant_reusable_bucket(str(neg_bucket or "")):
            return False
        return int(seen_neg_usage.get(neg_item_idx, 1)) < int(PAIR_V2D_TRUE_DEEP_CROSS_VARIANT_REUSE_CAP)

    def _take_entries(
        entries: list[tuple[dict[str, Any], str, dict[str, Any]]],
        *,
        allow_reuse: bool,
    ) -> None:
        for entry in entries:
            if len(selected) >= target_size:
                break
            if not _can_take_entry(entry, allow_reuse=allow_reuse):
                continue
            neg_row, _, _ = entry
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            local_seen_items.add(neg_item_idx)
            selected.append(entry)

    _take_entries(list(ranked_neg_variant), allow_reuse=allow_cross_variant_reuse)

    if (
        allow_cross_variant_reuse
        and PAIR_V2D_TRUE_DEEP_VARIANT_FALLBACK_FILL
        and len(selected) < target_size
    ):
        refill_entries: list[tuple[int, int, int, tuple[dict[str, Any], str, dict[str, Any]]]] = []
        refill_seen: set[int] = set()
        combined_entries = list(ranked_neg_variant) + list(ranked_neg_full)
        for original_idx, entry in enumerate(combined_entries):
            neg_row, neg_bucket, _ = entry
            neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
            if neg_item_idx < 0 or neg_item_idx in refill_seen:
                continue
            refill_seen.add(neg_item_idx)
            refill_entries.append(
                (
                    int(_deep_variant_fill_priority(str(neg_bucket or ""))),
                    0 if neg_item_idx not in seen_neg_items else 1,
                    int(original_idx),
                    entry,
                )
            )
        refill_entries.sort(key=lambda x: (x[0], x[1], x[2]))
        _take_entries([entry for _, _, _, entry in refill_entries], allow_reuse=True)

    finalized = list(selected[:target_size])
    for neg_row, neg_bucket, _ in finalized:
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        if neg_item_idx < 0:
            continue
        if neg_item_idx not in seen_neg_items:
            seen_neg_items.add(neg_item_idx)
            seen_neg_usage[neg_item_idx] = 1
            continue
        if _deep_cross_variant_reusable_bucket(str(neg_bucket or "")):
            seen_neg_usage[neg_item_idx] = max(1, int(seen_neg_usage.get(neg_item_idx, 1))) + 1
    return finalized


def _prepare_local_listwise_prompt_context(
    pos_row: dict[str, Any],
    neg_rows: list[dict[str, Any]],
) -> tuple[Callable[[int], str], int] | tuple[None, int]:
    rival_rows: list[dict[str, Any]] = []
    seen_rival_items: set[int] = set()
    for row in list(neg_rows):
        item_idx = _safe_int(row.get("item_idx", -1), -1)
        if item_idx < 0 or item_idx in seen_rival_items:
            continue
        seen_rival_items.add(item_idx)
        rival_rows.append(row)
    if not rival_rows:
        return None, 0
    all_rows = [pos_row] + rival_rows
    user_text = _build_boundary_user_text_from_row(pos_row)
    rows_by_item: dict[int, dict[str, Any]] = {}
    for row in all_rows:
        item_idx = _safe_int(row.get("item_idx", -1), -1)
        if item_idx >= 0 and item_idx not in rows_by_item:
            rows_by_item[item_idx] = row
    item_texts: dict[int, str] = {}
    prompt_map: dict[int, str] = {}
    rival_count = int(min(len(rival_rows), int(PAIR_LOCAL_LISTWISE_MAX_RIVALS)))

    def _item_text(item_idx: int) -> str:
        if item_idx < 0:
            return ""
        cached = item_texts.get(item_idx)
        if cached is not None:
            return cached
        row = rows_by_item.get(item_idx)
        if row is None:
            return ""
        cached = _build_boundary_item_text_from_row(row)
        item_texts[item_idx] = cached
        return cached

    # Build prompts lazily so we only materialize focus/rival texts for
    # candidates that are actually used in emitted training pairs.
    def _prompt_for(item_idx: int) -> str:
        if item_idx < 0:
            return ""
        cached = prompt_map.get(item_idx)
        if cached is not None:
            return cached
        focus_row = rows_by_item.get(item_idx)
        if focus_row is None:
            return ""
        other_candidates = [
            row
            for row in all_rows
            if _safe_int(row.get("item_idx", -1), -1) != item_idx
        ]
        other_rows = _select_local_listwise_rivals(
            focus_row,
            other_candidates,
            int(PAIR_LOCAL_LISTWISE_MAX_RIVALS),
        )
        rival_candidates = [
            (
                _local_listwise_rival_role(focus_row, row),
                _item_text(_safe_int(row.get("item_idx", -1), -1)),
            )
            for row in other_rows
        ]
        cached = build_local_listwise_ranking_prompt(
            user_text,
            _item_text(item_idx),
            rival_candidates,
            ranking_context=_local_listwise_ranking_context(focus_row, other_rows),
            local_slate_summary=_local_listwise_slate_summary(focus_row, other_rows),
            focus_summary=_focus_vs_local_slate_summary(focus_row, other_rows),
            focus_role=_rank_band_label(_effective_rank(focus_row)),
        )
        prompt_map[item_idx] = cached
        return cached

    return _prompt_for, rival_count


def _extract_prompt_field_value(text: str, label: str) -> str:
    pattern = _PROMPT_FIELD_RE_TEMPLATE.format(label=re.escape(label))
    m = re.search(pattern, str(text or ""), flags=re.I)
    return str(m.group(1)).strip() if m else ""


def _extract_local_prompt_user_section(prompt_text: str) -> str:
    text = str(prompt_text or "")
    m = re.search(r"\nUser:\s*(.*?)\nLocal ranking context:", text, flags=re.S | re.I)
    if m:
        return str(m.group(1)).strip()
    return text


def _extract_local_prompt_focus_section(prompt_text: str) -> str:
    text = str(prompt_text or "")
    m = re.search(r"\nFocus candidate .*?:\s*(.*?)(?:\nRival 1 |\Z)", text, flags=re.S | re.I)
    if m:
        return str(m.group(1)).strip()
    return text


def _prompt_field_informative(raw: str) -> bool:
    txt = clean_text(raw, max_chars=160).lower()
    if not txt:
        return False
    txt = txt.strip(" ;,.")
    if (
        txt in _NO_INFO_PROMPT_VALUES
        or txt.startswith("no clear direct match")
        or txt.startswith("no clear direct conflict")
        or txt.startswith("no clear")
    ):
        return False
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9&+'-]{1,30}", txt.replace("_", " "))
        if len(tok) >= 3 and tok not in _PROMPT_LOW_SIGNAL_TOKENS
    }
    return bool(tokens)


def _prompt_field_present(raw: str) -> bool:
    return bool(clean_text(raw, max_chars=160).strip(" ;,."))


def _prompt_has_structured_residue(text: str) -> bool:
    txt = str(text or "")
    if not txt:
        return False
    if re.search(r'["\']?[a-z0-9_]+["\']?\s*:\s*\[[^\]]*', txt, flags=re.I):
        return True
    if re.search(r'[{}\[\]]', txt):
        return True
    if re.search(r'["\'][a-z0-9_]+["\']\s*:', txt, flags=re.I):
        return True
    return False


_PROMPT_SCAFFOLD_PATTERNS: dict[str, re.Pattern[str]] = {
    "recent_activity_points_to": re.compile(r"\bRecent activity points to\b", flags=re.I),
    "recent_choices_lean_toward": re.compile(r"\bRecent choices lean toward\b", flags=re.I),
    "longer_term_habits_favor": re.compile(r"\bLonger-term habits favor\b", flags=re.I),
    "user_consistently_goes_for": re.compile(r"\bThe user consistently goes for\b", flags=re.I),
    "recurring_preferences_include": re.compile(r"\bRecurring preferences include\b", flags=re.I),
    "other_recurring_preferences_include": re.compile(r"\bOther recurring preferences include\b", flags=re.I),
    "past_evidence_interest": re.compile(r"\bPast evidence points to recurring interest in\b", flags=re.I),
    "past_evidence_friction": re.compile(r"\bPast evidence suggests friction around\b", flags=re.I),
    "likely_friction_points_include": re.compile(r"\bLikely friction points include\b", flags=re.I),
    "likely_friction_comes_from": re.compile(r"\bLikely friction comes from\b", flags=re.I),
}


def _prompt_scaffold_hits(text: str) -> Counter[str]:
    hits: Counter[str] = Counter()
    prompt_text = str(text or "")
    if not prompt_text:
        return hits
    for name, pattern in _PROMPT_SCAFFOLD_PATTERNS.items():
        if pattern.search(prompt_text):
            hits[name] += 1
    return hits


def _local_prompt_signal_profile(prompt_text: str) -> dict[str, bool]:
    user_section = _extract_local_prompt_user_section(prompt_text)
    focus_section = _extract_local_prompt_focus_section(prompt_text)
    return {
        "user_focus": _prompt_field_informative(_extract_prompt_field_value(user_section, "user_focus")),
        "recent_intent": _prompt_field_informative(_extract_prompt_field_value(user_section, "recent_intent")),
        "user_avoid": _prompt_field_informative(_extract_prompt_field_value(user_section, "user_avoid")),
        "history_pattern": _prompt_field_informative(_extract_prompt_field_value(user_section, "history_pattern")),
        "user_evidence": _prompt_field_informative(_extract_prompt_field_value(user_section, "user_evidence")),
        "user_match_points": _prompt_field_informative(_extract_prompt_field_value(focus_section, "user_match_points")),
        "user_conflict_points": _prompt_field_informative(_extract_prompt_field_value(focus_section, "user_conflict_points")),
        "business_scene": _prompt_field_informative(_extract_prompt_field_value(focus_section, "business_scene")),
        "item_strengths": _prompt_field_informative(_extract_prompt_field_value(focus_section, "item_strengths")),
        "item_weaknesses": _prompt_field_informative(_extract_prompt_field_value(focus_section, "item_weaknesses")),
        "item_signal": _prompt_field_informative(_extract_prompt_field_value(focus_section, "item_signal")),
        "evidence_basis": _prompt_field_informative(_extract_prompt_field_value(focus_section, "evidence_basis")),
    }


def build_row_fact_signal_profile(row: dict[str, Any]) -> dict[str, bool | int]:
    user_focus = _safe_int(row.get("user_focus_fact_count_v1", 0), 0) > 0 or bool(row.get("user_has_specific_focus_v2", 0))
    recent = _safe_int(row.get("user_recent_fact_count_v1", 0), 0) > 0 or bool(row.get("user_has_specific_recent_v2", 0))
    avoid = _safe_int(row.get("user_avoid_fact_count_v1", 0), 0) > 0 or bool(row.get("user_has_avoid_signal_v2", 0))
    history = _safe_int(row.get("user_history_fact_count_v1", 0), 0) > 0 or bool(row.get("user_has_history_evidence_v2", 0))
    evidence = _safe_int(row.get("user_evidence_fact_count_v1", 0), 0) > 0 or bool(row.get("user_has_strong_clean_evidence_v2", 0))
    visible_user_semantics = user_focus and (recent or avoid or history or evidence)

    merchant_visible = _safe_int(row.get("merchant_visible_fact_count_v1", 0), 0) > 0 or _safe_int(
        row.get("merchant_profile_richness_v2", 0), 0
    ) >= 2
    fit_fact_count = _safe_int(row.get("pair_fit_fact_count_v1", 0), 0)
    conflict_visible = (
        _safe_int(row.get("pair_friction_fact_count_v1", 0), 0) > 0
        or _safe_int(row.get("pair_has_visible_conflict_v1", 0), 0) > 0
        or avoid
    )
    evidence_visible = _safe_int(row.get("pair_evidence_fact_count_v1", 0), 0) > 0
    stable_fit = _safe_int(row.get("pair_stable_fit_fact_count_v1", 0), 0) > 0
    recent_fit = _safe_int(row.get("pair_recent_fit_fact_count_v1", 0), 0) > 0
    history_fit = _safe_int(row.get("pair_history_fit_fact_count_v1", 0), 0) > 0
    practical_fit = _safe_int(row.get("pair_practical_fit_fact_count_v1", 0), 0) > 0
    fit_scope_count = _safe_int(row.get("pair_fit_scope_count_v1", 0), 0)
    detail_support = conflict_visible or recent_fit or practical_fit
    contrastive_support_raw = _safe_int(row.get("pair_has_contrastive_support_v1", -1), -1)
    contrastive_support = contrastive_support_raw > 0
    multisource_fit = fit_scope_count >= 2 or contrastive_support or (contrastive_support_raw < 0 and detail_support)
    user_fit_visible = _safe_int(row.get("pair_has_visible_user_fit_v1", 0), 0) > 0 or fit_fact_count > 0
    item_alignment = user_fit_visible and (recent_fit or history_fit or practical_fit or stable_fit)
    signal_count = sum(
        1
        for flag in [
            user_focus,
            recent,
            avoid,
            history,
            evidence,
            merchant_visible,
            user_fit_visible,
            conflict_visible,
            evidence_visible,
        ]
        if flag
    )
    has_recent_or_history = recent or history
    return {
        "user_focus": user_focus,
        "recent": recent,
        "avoid": avoid,
        "history": history,
        "evidence": evidence,
        "visible_user_semantics": visible_user_semantics,
        "merchant_visible": merchant_visible,
        "conflict_visible": conflict_visible,
        "evidence_visible": evidence_visible,
        "stable_fit": stable_fit,
        "recent_fit": recent_fit,
        "history_fit": history_fit,
        "practical_fit": practical_fit,
        "fit_scope_count": fit_scope_count,
        "detail_support": detail_support,
        "contrastive_support": contrastive_support,
        "multisource_fit": multisource_fit,
        "user_fit_visible": user_fit_visible,
        "item_alignment": item_alignment,
        "signal_count": signal_count,
        "has_focus": user_focus,
        "has_recent_or_history": has_recent_or_history,
        "has_avoid": avoid,
        "has_user_evidence": evidence,
        "has_fit": user_fit_visible,
        "has_conflict": conflict_visible,
        "has_pair_evidence": evidence_visible,
        "has_multisource_fit": multisource_fit,
        "has_contrastive_support": contrastive_support,
        "has_detail_support": detail_support,
    }


def classify_boundary_constructability(
    row: dict[str, Any],
    *,
    rival_total: Any,
    rival_head_or_boundary: Any,
) -> tuple[str, list[str]]:
    profile = build_row_fact_signal_profile(row)
    rival_total_i = _safe_int(rival_total, 0)
    rival_head_or_boundary_i = _safe_int(rival_head_or_boundary, 0)
    reasons: list[str] = []
    if not profile["has_focus"]:
        reasons.append("U_NO_FOCUS")
    if not profile["has_recent_or_history"]:
        reasons.append("U_NO_RECENT_OR_HISTORY")
    if not profile["has_avoid"]:
        reasons.append("U_NO_AVOID")
    if not profile["has_user_evidence"]:
        reasons.append("U_NO_USER_EVIDENCE")
    if not profile["has_fit"]:
        reasons.append("P_NO_VISIBLE_FIT")
    if not profile["has_conflict"]:
        reasons.append("P_NO_CONFLICT")
    if not profile["has_pair_evidence"]:
        reasons.append("P_NO_PAIR_EVIDENCE")
    if (not profile["has_multisource_fit"]) and (not profile["has_contrastive_support"]):
        reasons.append("P_SHARED_THEME_ONLY")
    if not profile["has_detail_support"]:
        reasons.append("P_NO_DETAIL_SUPPORT")
    if rival_total_i < 3:
        reasons.append("S_FEW_RIVALS")
    if rival_head_or_boundary_i < 1:
        reasons.append("S_NO_STRONG_RIVALS")

    if not profile["has_focus"] or not profile["has_recent_or_history"] or not profile["has_fit"] or rival_total_i < 3:
        return "C0_FAIL", reasons
    if (
        ((not profile["has_multisource_fit"]) and (not profile["has_contrastive_support"]))
        or ((not profile["has_avoid"]) and (not profile["has_user_evidence"]))
        or (not profile["has_pair_evidence"])
    ):
        return "C1_WEAK", reasons
    if (
        profile["has_avoid"]
        and profile["has_user_evidence"]
        and profile["has_conflict"]
        and profile["has_pair_evidence"]
        and profile["has_detail_support"]
        and rival_total_i >= 4
        and rival_head_or_boundary_i >= 2
        and (profile["has_multisource_fit"] or profile["has_contrastive_support"])
    ):
        return "C3_IDEAL", reasons
    return "C2_USABLE", reasons


def _local_prompt_is_learnable(prompt_text: str, rank_band: str) -> bool:
    profile = _local_prompt_signal_profile(prompt_text)
    score = sum(1 for val in profile.values() if val)
    has_recent_or_history = profile["recent_intent"] or profile["history_pattern"] or profile["user_evidence"]
    has_focus_history = profile["user_focus"] and has_recent_or_history
    has_user_semantics = profile["user_focus"] and (
        profile["recent_intent"] or profile["history_pattern"] or profile["user_evidence"] or profile["user_avoid"]
    )
    has_item_alignment = (
        profile["user_match_points"]
        or profile["user_conflict_points"]
        or profile["evidence_basis"]
        or profile["item_strengths"]
        or profile["business_scene"]
        or profile["item_signal"]
    )
    has_negative_or_conflict = profile["user_avoid"] or profile["user_conflict_points"] or profile["item_weaknesses"]
    has_visible_business_semantics = profile["item_strengths"] or profile["business_scene"] or profile["item_signal"]
    if rank_band in {"head_guard", "boundary_11_30"}:
        return bool(
            score >= 5
            and has_focus_history
            and has_item_alignment
            and has_visible_business_semantics
            and has_negative_or_conflict
        )
    if rank_band == "rescue_31_60":
        return bool(score >= 4 and has_focus_history and has_item_alignment and has_visible_business_semantics)
    return bool(score >= 4 and has_user_semantics and has_item_alignment and has_visible_business_semantics)


def _row_local_prompt_is_learnable(row: dict[str, Any], prompt_text: str, rank_band: str) -> bool:
    fit_fact_count = _safe_int(row.get("pair_fit_fact_count_v1", -1), -1)
    if fit_fact_count < 0:
        return _local_prompt_is_learnable(prompt_text, rank_band)

    profile = build_row_fact_signal_profile(row)
    user_focus = bool(profile["user_focus"])
    recent = bool(profile["recent"])
    avoid = bool(profile["avoid"])
    history = bool(profile["history"])
    evidence = bool(profile["evidence"])
    visible_user_semantics = bool(profile["visible_user_semantics"])
    merchant_visible = bool(profile["merchant_visible"])
    conflict_visible = bool(profile["conflict_visible"])
    evidence_visible = bool(profile["evidence_visible"])
    stable_fit = bool(profile["stable_fit"])
    recent_fit = bool(profile["recent_fit"])
    history_fit = bool(profile["history_fit"])
    practical_fit = bool(profile["practical_fit"])
    multisource_fit = bool(profile["multisource_fit"])
    user_fit_visible = bool(profile["user_fit_visible"])
    item_alignment = bool(profile["item_alignment"])
    signal_count = int(profile["signal_count"])

    if rank_band in {"head_guard", "boundary_11_30"}:
        if rank_band == "boundary_11_30":
            boundary_constructability = str(row.get("boundary_constructability_class_v1", "") or "").strip()
            if boundary_constructability in {"C0_FAIL", "C1_WEAK", "C2_USABLE", "C3_IDEAL"}:
                return boundary_constructability in {"C2_USABLE", "C3_IDEAL"}
            boundary_prompt_ready = row.get("boundary_prompt_ready_v1", None)
            if boundary_prompt_ready not in (None, ""):
                return _safe_int(boundary_prompt_ready, 0) > 0
        return bool(
            signal_count >= 5
            and visible_user_semantics
            and merchant_visible
            and item_alignment
            and conflict_visible
            and multisource_fit
            and evidence_visible
        )
    if rank_band == "rescue_31_60":
        return bool(
            signal_count >= 4
            and visible_user_semantics
            and merchant_visible
            and item_alignment
            and multisource_fit
            and (conflict_visible or history_fit or practical_fit or evidence_visible)
        )
    return bool(
        signal_count >= 4
        and visible_user_semantics
        and merchant_visible
        and user_fit_visible
        and multisource_fit
        and (history_fit or practical_fit or conflict_visible or evidence_visible)
    )


def _v2d_reason_prefix_bucket_specs(
    pos_row: dict[str, Any],
    pos_band: str,
) -> list[tuple[str, Any, Any, int]]:
    reason_bucket = _truth_reason_bucket(pos_row)
    if not reason_bucket:
        return []
    if pos_band == "head_guard":
        rank_priority = _head_guard_blocker_priority
    elif pos_band == "boundary_11_30":
        rank_priority = _boundary_blocker_priority
    elif pos_band == "rescue_31_60":
        rank_priority = _mid_blocker_priority
    else:
        rank_priority = lambda rank: int(rank)
    primary_quota = int(PAIR_V2D_TRUE_REASON_PRIMARY_QUOTA)
    aux_quota = int(PAIR_V2D_TRUE_REASON_AUX_QUOTA)
    specs: list[tuple[str, Any, Any, int]] = []
    if reason_bucket == "semantic":
        specs.append(
            (
                "semantic_reason_blocker",
                lambda f: f["semantic_reason_blocker"],
                lambda f: (
                    rank_priority(int(f["effective_neg_rank"])),
                    not f["pos_semantic_edge"],
                    not f["pos_support_edge"],
                    int(f["neg_explainable_edge_count"]),
                    f["effective_neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(primary_quota),
            )
        )
        if pos_band in {"boundary_11_30", "rescue_31_60"}:
            specs.append(
                (
                    "head_prior_reason_blocker",
                    lambda f: f["head_prior_reason_blocker"],
                    lambda f: (
                        rank_priority(int(f["effective_neg_rank"])),
                        f["effective_neg_rank"],
                        f["tier_order"],
                        -f["hardness"],
                    ),
                    int(aux_quota),
                )
            )
    elif reason_bucket == "channel_context":
        specs.append(
            (
                "channel_context_reason_blocker",
                lambda f: f["channel_context_reason_blocker"],
                lambda f: (
                    rank_priority(int(f["effective_neg_rank"])),
                    not f["pos_channel_context_edge"],
                    not f["pos_stability_edge"],
                    int(f["neg_explainable_edge_count"]),
                    f["effective_neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(primary_quota),
            )
        )
    elif reason_bucket == "multi_route":
        specs.append(
            (
                "multi_route_reason_blocker",
                lambda f: f["multi_route_reason_blocker"],
                lambda f: (
                    rank_priority(int(f["effective_neg_rank"])),
                    not f["pos_route_edge"],
                    -int(f["pos_source_count"]),
                    -int(f["pos_nonpopular_source_count"]),
                    -int(f["pos_profile_cluster_source_count"]),
                    f["effective_neg_rank"],
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(primary_quota),
            )
        )
    elif reason_bucket == "head_prior":
        specs.append(
            (
                "head_prior_reason_blocker",
                lambda f: f["head_prior_reason_blocker"],
                lambda f: (
                    rank_priority(int(f["effective_neg_rank"])),
                    f["effective_neg_rank"],
                    int(f["neg_explainable_edge_count"]),
                    f["tier_order"],
                    -f["hardness"],
                ),
                int(primary_quota),
            )
        )
    return [spec for spec in specs if int(spec[3]) > 0]


def _build_blocker_compare_prompts(
    pos_row: dict[str, Any],
    neg_row: dict[str, Any],
) -> tuple[str, str]:
    user_text = _build_boundary_user_text_from_row(pos_row)
    pos_item_text = _build_boundary_item_text_from_row(pos_row)
    neg_item_text = _build_boundary_item_text_from_row(neg_row)
    pos_rank = _effective_rank(pos_row)
    neg_rank = _effective_rank(neg_row)
    forward_prompt = build_blocker_comparison_prompt(
        user_text,
        pos_item_text,
        neg_item_text,
        ranking_context=_comparison_ranking_context(pos_rank, neg_rank),
        comparison_summary="",
        candidate_a_role=_rank_band_label(pos_rank),
        candidate_b_role=_rank_band_label(neg_rank),
    )
    reverse_prompt = build_blocker_comparison_prompt(
        user_text,
        neg_item_text,
        pos_item_text,
        ranking_context=_comparison_ranking_context(neg_rank, pos_rank),
        comparison_summary="",
        candidate_a_role=_rank_band_label(neg_rank),
        candidate_b_role=_rank_band_label(pos_rank),
    )
    return forward_prompt, reverse_prompt


def _build_pair_record(
    pos_row: dict[str, Any],
    neg_row: dict[str, Any],
    uid: int,
    pair_mode: str,
    selection_bucket: str,
    *,
    selection_slot: int = 0,
    slate_variant_id: int = 0,
    local_prompt_map: dict[int, str] | None = None,
    local_prompt_getter: Callable[[int], str] | None = None,
    local_rival_count: int = 0,
) -> dict[str, Any] | None:
    pos_prompt = str(pos_row.get("prompt", "")).strip()
    neg_prompt = str(neg_row.get("prompt", "")).strip()
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
    pos_learned_score = _effective_score(pos_row)
    neg_learned_score = _effective_score(neg_row)
    pos_learned_rank = _effective_rank(pos_row)
    neg_learned_rank = _effective_rank(neg_row)
    common_prompt = ""
    chosen_text = pos_prompt
    rejected_text = neg_prompt
    prompt_style = str(PAIR_PROMPT_STYLE or "candidate_local").strip().lower()
    if prompt_style == "local_listwise_compare":
        pos_item_idx = _safe_int(pos_row.get("item_idx", -1), -1)
        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
        chosen_local = ""
        rejected_local = ""
        if local_prompt_getter is not None:
            chosen_local = str(local_prompt_getter(pos_item_idx) or "").strip()
            rejected_local = str(local_prompt_getter(neg_item_idx) or "").strip()
        elif local_prompt_map and pos_item_idx in local_prompt_map and neg_item_idx in local_prompt_map:
            chosen_local = str(local_prompt_map.get(pos_item_idx, "") or "").strip()
            rejected_local = str(local_prompt_map.get(neg_item_idx, "") or "").strip()
        if chosen_local and rejected_local:
            chosen_text = chosen_local
            rejected_text = rejected_local
            common_prompt = ""
        else:
            prompt_style = "blocker_compare"
    if prompt_style == "blocker_compare":
        try:
            chosen_text, rejected_text = _build_blocker_compare_prompts(pos_row, neg_row)
            common_prompt = ""
        except Exception:
            prompt_style = "candidate_local"
    if prompt_style != "blocker_compare":
        pos_parts = _split_binary_prompt(pos_prompt)
        neg_parts = _split_binary_prompt(neg_prompt)
        if pos_parts and neg_parts and pos_parts["head"] == neg_parts["head"]:
            pos_user_base, pos_user_variant = _split_user_shared_and_variant(pos_parts["user"])
            neg_user_base, neg_user_variant = _split_user_shared_and_variant(neg_parts["user"])
            if pos_user_base == neg_user_base:
                # Keep the shared user context in `prompt` and move candidate-specific
                # user details (history anchors / pair signal) into the completion.
                shared_variant_sep = "; " if (pos_user_variant or neg_user_variant) else ""
                common_prompt = pos_parts["head"] + pos_user_base + shared_variant_sep
                chosen_text = (
                    pos_user_variant
                    + pos_parts["candidate"]
                    + pos_parts["item"]
                    + pos_parts["answer"]
                    + " YES"
                )
                rejected_text = (
                    neg_user_variant
                    + neg_parts["candidate"]
                    + neg_parts["item"]
                    + neg_parts["answer"]
                    + " YES"
                )
    if prompt_style == "local_listwise_compare":
        pos_band = _boundary_rank_band(int(pos_learned_rank))
        if not _row_local_prompt_is_learnable(pos_row, chosen_text, pos_band):
            return None
    return {
        "user_idx": uid,
        "split": str(pos_row.get("split", "") or neg_row.get("split", "") or "train"),
        "pair_mode": pair_mode,
        "selection_bucket": selection_bucket,
        "selection_slot": int(selection_slot),
        "slate_variant_id": int(slate_variant_id),
        "prompt": common_prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_business_id": str(pos_row.get("business_id", "") or ""),
        "rejected_business_id": str(neg_row.get("business_id", "") or ""),
        "chosen_item_idx": _safe_int(pos_row.get("item_idx", -1), -1),
        "rejected_item_idx": _safe_int(neg_row.get("item_idx", -1), -1),
        "chosen_label_source": str(pos_row.get("label_source", "") or ""),
        "rejected_label_source": str(neg_row.get("label_source", "") or ""),
        "rejected_neg_tier": str(neg_row.get("neg_tier", "") or ""),
        "chosen_pre_rank": _safe_int(pos_row.get("pre_rank", -1), -1),
        "rejected_pre_rank": _safe_int(neg_row.get("pre_rank", -1), -1),
        "chosen_learned_rank": int(pos_learned_rank),
        "rejected_learned_rank": int(neg_learned_rank),
        "chosen_learned_rank_band": _boundary_rank_band(int(pos_learned_rank)),
        "rejected_learned_rank_band": _boundary_rank_band(int(neg_learned_rank)),
        "chosen_primary_reason": str(pos_row.get("primary_reason", "") or ""),
        "chosen_easy_but_useful": bool(pos_row.get("easy_but_useful", False)),
        "chosen_hard_but_learnable": bool(pos_row.get("hard_but_learnable", False)),
        "chosen_non_actionable": bool(pos_row.get("non_actionable", False)),
        "chosen_pre_score": pos_score,
        "rejected_pre_score": neg_score,
        "chosen_learned_blend_score": float(pos_learned_score),
        "rejected_learned_blend_score": float(neg_learned_score),
        "score_gap": float(pos_score - neg_score),
        "rank_gap": int(_safe_int(neg_row.get("pre_rank", -1), -1) - _safe_int(pos_row.get("pre_rank", -1), -1)),
        "learned_score_gap": float(pos_learned_score - neg_learned_score),
        "learned_rank_gap": int(int(neg_learned_rank) - int(pos_learned_rank)),
        "chosen_has_user_evidence": _has_text(pos_row.get("user_evidence_text")),
        "chosen_has_item_evidence": _has_text(pos_row.get("item_evidence_text")),
        "chosen_has_pair_evidence": _has_text(_row_pair_evidence_text(pos_row)),
        "rejected_has_user_evidence": _has_text(neg_row.get("user_evidence_text")),
        "rejected_has_item_evidence": _has_text(neg_row.get("item_evidence_text")),
        "rejected_has_pair_evidence": _has_text(_row_pair_evidence_text(neg_row)),
        "prompt_style": prompt_style,
        "local_listwise_rival_count": (
            int(local_rival_count)
            if int(local_rival_count) > 0
            else (
                int(min(max(len(local_prompt_map or {}) - 1, 0), int(PAIR_LOCAL_LISTWISE_MAX_RIVALS)))
                if local_prompt_map
                else 0
            )
        ),
    }


def build_pointwise_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_label_counts: Counter[tuple[str, int]] = Counter()
    label_source_counts: Counter[str] = Counter()
    neg_tier_counts: Counter[str] = Counter()
    rank_band_label_counts: Counter[tuple[str, int]] = Counter()
    user_state: dict[int, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    pos_per_user: list[int] = []
    neg_per_user: list[int] = []
    user_evidence_counts: Counter[str] = Counter()
    item_evidence_counts: Counter[str] = Counter()
    pair_evidence_counts: Counter[str] = Counter()

    for row in rows:
        split = str(row.get("split", "") or "unknown")
        label = _safe_int(row.get("label", 0), 0)
        uid = _safe_int(row.get("user_idx", -1), -1)
        split_label_counts[(split, label)] += 1
        label_source_counts[str(row.get("label_source", "") or "unknown")] += 1
        neg_tier = str(row.get("neg_tier", "") or "none")
        if label == 0:
            neg_tier_counts[neg_tier] += 1
        rank_band_label_counts[(_band(row.get("pre_rank")), label)] += 1

        if uid >= 0:
            if label == 1:
                user_state[uid]["pos"] += 1
            else:
                user_state[uid]["neg"] += 1

        if _has_text(row.get("user_evidence_text")):
            user_evidence_counts[str(label)] += 1
        if _has_text(row.get("item_evidence_text")):
            item_evidence_counts[str(label)] += 1
        if _has_text(_row_pair_evidence_text(row)):
            pair_evidence_counts[str(label)] += 1

    users_total = len(user_state)
    users_with_pos = sum(1 for v in user_state.values() if v["pos"] > 0)
    users_with_neg = sum(1 for v in user_state.values() if v["neg"] > 0)
    users_with_both = sum(1 for v in user_state.values() if v["pos"] > 0 and v["neg"] > 0)
    users_positive_only = sum(1 for v in user_state.values() if v["pos"] > 0 and v["neg"] == 0)
    users_negative_only = sum(1 for v in user_state.values() if v["pos"] == 0 and v["neg"] > 0)
    pos_per_user = [v["pos"] for v in user_state.values() if v["pos"] > 0]
    neg_per_user = [v["neg"] for v in user_state.values() if v["neg"] > 0]

    return {
        "rows_total": int(len(rows)),
        "split_label_counts": [
            {"split": split, "label": int(label), "count": int(count)}
            for (split, label), count in sorted(split_label_counts.items())
        ],
        "users_total": int(users_total),
        "users_with_positive": int(users_with_pos),
        "users_with_negative": int(users_with_neg),
        "users_with_both": int(users_with_both),
        "users_positive_only": int(users_positive_only),
        "users_negative_only": int(users_negative_only),
        "positive_per_user": _summary(pos_per_user),
        "negative_per_user": _summary(neg_per_user),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "neg_tier_counts": dict(sorted(neg_tier_counts.items())),
        "pre_rank_band_counts": [
            {"pre_rank_band": band, "label": int(label), "count": int(count)}
            for (band, label), count in sorted(rank_band_label_counts.items())
        ],
        "evidence_coverage": {
            "user_evidence_by_label": {
                "label_0": int(user_evidence_counts.get("0", 0)),
                "label_1": int(user_evidence_counts.get("1", 0)),
            },
            "item_evidence_by_label": {
                "label_0": int(item_evidence_counts.get("0", 0)),
                "label_1": int(item_evidence_counts.get("1", 0)),
            },
            "pair_evidence_by_label": {
                "label_0": int(pair_evidence_counts.get("0", 0)),
                "label_1": int(pair_evidence_counts.get("1", 0)),
            },
        },
    }


def build_dpo_pairs(
    rows: list[dict[str, Any]],
    max_pairs_per_user: int,
    seed: int,
    prefer_easy_neg: bool = True,
    filter_inverted: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    user_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    user_neg: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        uid = _safe_int(row.get("user_idx", -1), -1)
        label = _safe_int(row.get("label", 0), 0)
        prompt = str(row.get("prompt", "")).strip()
        if uid < 0 or not prompt:
            continue
        if label == 1:
            user_pos[uid].append(row)
        else:
            user_neg[uid].append(row)

    pairs: list[dict[str, Any]] = []
    users_with_pairs = 0
    users_skipped_no_neg = 0
    n_filtered_inverted = 0
    pairs_per_user: list[int] = []
    selected_neg_tiers: Counter[str] = Counter()
    score_gaps: list[float] = []
    rank_gaps: list[int] = []

    for uid in sorted(user_pos.keys()):
        pos_rows = user_pos[uid]
        neg_rows = user_neg.get(uid, [])
        if not neg_rows:
            users_skipped_no_neg += 1
            continue

        if prefer_easy_neg:
            easy_near = [r for r in neg_rows if str(r.get("neg_tier", "")) in ("easy", "near", "fill")]
            hard = [r for r in neg_rows if str(r.get("neg_tier", "")) == "hard"]
            ordered_neg = easy_near + hard
        else:
            ordered_neg = list(neg_rows)
            rng.shuffle(ordered_neg)

        all_combos: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for pos_row in pos_rows:
            for neg_row in ordered_neg:
                all_combos.append((pos_row, neg_row))

        if not prefer_easy_neg:
            rng.shuffle(all_combos)

        selected = all_combos[:max_pairs_per_user]
        built_for_user = 0

        for pos_row, neg_row in selected:
            pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
            neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
            if filter_inverted and neg_score > pos_score:
                n_filtered_inverted += 1
                continue

            pos_prompt = str(pos_row.get("prompt", "")).strip()
            neg_prompt = str(neg_row.get("prompt", "")).strip()
            if not pos_prompt or not neg_prompt:
                continue
            pair = _build_pair_record(pos_row, neg_row, uid, pair_mode="pointwise", selection_bucket=str(neg_row.get("neg_tier", "") or "unknown"))
            pairs.append(pair)
            built_for_user += 1
            selected_neg_tiers[pair["rejected_neg_tier"]] += 1
            score_gaps.append(pair["score_gap"])
            rank_gaps.append(pair["rank_gap"])

        if built_for_user > 0:
            users_with_pairs += 1
            pairs_per_user.append(built_for_user)

    rng.shuffle(pairs)
    chosen_item_evidence = [1.0 if bool(p["chosen_has_item_evidence"]) else 0.0 for p in pairs]
    rejected_item_evidence = [1.0 if bool(p["rejected_has_item_evidence"]) else 0.0 for p in pairs]
    chosen_pair_evidence = [1.0 if bool(p["chosen_has_pair_evidence"]) else 0.0 for p in pairs]
    rejected_pair_evidence = [1.0 if bool(p["rejected_has_pair_evidence"]) else 0.0 for p in pairs]
    chosen_prompt_field_present: Counter[str] = Counter()
    chosen_prompt_field_informative: Counter[str] = Counter()
    rejected_prompt_field_present: Counter[str] = Counter()
    rejected_prompt_field_informative: Counter[str] = Counter()
    chosen_prompt_scaffold_counts: Counter[str] = Counter()
    rejected_prompt_scaffold_counts: Counter[str] = Counter()
    chosen_structured_residue = 0
    rejected_structured_residue = 0
    chosen_prompt_chars: list[int] = []
    rejected_prompt_chars: list[int] = []
    prompt_char_gap: list[int] = []
    for pair in pairs:
        chosen_prompt = str(
            pair.get("chosen_prompt")
            or pair.get("chosen")
            or pair.get("prompt")
            or ""
        )
        rejected_prompt = str(
            pair.get("rejected_prompt")
            or pair.get("rejected")
            or pair.get("prompt")
            or ""
        )
        chosen_prompt_chars.append(len(chosen_prompt))
        rejected_prompt_chars.append(len(rejected_prompt))
        prompt_char_gap.append(len(chosen_prompt) - len(rejected_prompt))
        if _prompt_has_structured_residue(chosen_prompt):
            chosen_structured_residue += 1
        if _prompt_has_structured_residue(rejected_prompt):
            rejected_structured_residue += 1
        chosen_prompt_scaffold_counts.update(_prompt_scaffold_hits(chosen_prompt))
        rejected_prompt_scaffold_counts.update(_prompt_scaffold_hits(rejected_prompt))
        chosen_user_section = _extract_local_prompt_user_section(chosen_prompt)
        chosen_focus_section = _extract_local_prompt_focus_section(chosen_prompt)
        rejected_user_section = _extract_local_prompt_user_section(rejected_prompt)
        rejected_focus_section = _extract_local_prompt_focus_section(rejected_prompt)
        for field_name, section in (
            ("user_focus", chosen_user_section),
            ("recent_intent", chosen_user_section),
            ("user_avoid", chosen_user_section),
            ("history_pattern", chosen_user_section),
            ("user_evidence", chosen_user_section),
            ("user_match_points", chosen_focus_section),
            ("user_conflict_points", chosen_focus_section),
            ("business_scene", chosen_focus_section),
            ("item_strengths", chosen_focus_section),
            ("item_weaknesses", chosen_focus_section),
            ("item_signal", chosen_focus_section),
            ("evidence_basis", chosen_focus_section),
        ):
            raw = _extract_prompt_field_value(section, field_name)
            if _prompt_field_present(raw):
                chosen_prompt_field_present[field_name] += 1
            if _prompt_field_informative(raw):
                chosen_prompt_field_informative[field_name] += 1
        for field_name, section in (
            ("user_focus", rejected_user_section),
            ("recent_intent", rejected_user_section),
            ("user_avoid", rejected_user_section),
            ("history_pattern", rejected_user_section),
            ("user_evidence", rejected_user_section),
            ("user_match_points", rejected_focus_section),
            ("user_conflict_points", rejected_focus_section),
            ("business_scene", rejected_focus_section),
            ("item_strengths", rejected_focus_section),
            ("item_weaknesses", rejected_focus_section),
            ("item_signal", rejected_focus_section),
            ("evidence_basis", rejected_focus_section),
        ):
            raw = _extract_prompt_field_value(section, field_name)
            if _prompt_field_present(raw):
                rejected_prompt_field_present[field_name] += 1
            if _prompt_field_informative(raw):
                rejected_prompt_field_informative[field_name] += 1

    audit = {
        "users_with_pairs": int(users_with_pairs),
        "users_skipped_no_neg": int(users_skipped_no_neg),
        "filtered_inverted": int(n_filtered_inverted),
        "total_pairs": int(len(pairs)),
        "max_pairs_per_user": int(max_pairs_per_user),
        "prefer_easy_neg": bool(prefer_easy_neg),
        "filter_inverted": bool(filter_inverted),
        "pairs_per_user": _summary(pairs_per_user),
        "selected_neg_tier_counts": dict(sorted(selected_neg_tiers.items())),
        "score_gap": _summary(score_gaps),
        "rank_gap": _summary(rank_gaps),
        "coverage": {
            "chosen_item_evidence_rate": float(sum(chosen_item_evidence) / max(len(chosen_item_evidence), 1)),
            "rejected_item_evidence_rate": float(sum(rejected_item_evidence) / max(len(rejected_item_evidence), 1)),
            "chosen_pair_evidence_rate": float(sum(chosen_pair_evidence) / max(len(chosen_pair_evidence), 1)),
            "rejected_pair_evidence_rate": float(sum(rejected_pair_evidence) / max(len(rejected_pair_evidence), 1)),
        },
        "prompt_quality": {
            "chosen_prompt_chars": _summary(chosen_prompt_chars),
            "rejected_prompt_chars": _summary(rejected_prompt_chars),
            "prompt_char_gap_chosen_minus_rejected": _summary(prompt_char_gap),
            "chosen_scaffold_phrase_count": dict(sorted(chosen_prompt_scaffold_counts.items())),
            "rejected_scaffold_phrase_count": dict(sorted(rejected_prompt_scaffold_counts.items())),
            "chosen_scaffold_phrase_rate": {
                key: float(chosen_prompt_scaffold_counts.get(key, 0) / max(len(pairs), 1))
                for key in sorted(_PROMPT_SCAFFOLD_PATTERNS.keys())
            },
            "rejected_scaffold_phrase_rate": {
                key: float(rejected_prompt_scaffold_counts.get(key, 0) / max(len(pairs), 1))
                for key in sorted(_PROMPT_SCAFFOLD_PATTERNS.keys())
            },
        },
    }
    return pairs, audit


def _rich_neg_hardness(neg_row: dict[str, Any], pos_score: float) -> tuple[float, float, int]:
    neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
    tower_score = max(0.0, _row_score(neg_row, "tower_score"))
    seq_score = max(0.0, _row_score(neg_row, "seq_score"))
    pre_rank = max(1, _safe_int(neg_row.get("pre_rank", 999999), 999999))
    pair_bonus = 0.03 if _has_text(_row_pair_evidence_text(neg_row)) else 0.0
    item_bonus = 0.02 if _has_text(neg_row.get("item_evidence_text")) else 0.0
    hardness = (
        neg_score
        + (0.35 * tower_score)
        + (0.35 * seq_score)
        + (0.10 / float(pre_rank))
        + pair_bonus
        + item_bonus
    )
    return (
        float(hardness),
        float(abs(pos_score - neg_score)),
        int(pre_rank),
    )


def build_rich_sft_dpo_pairs(
    rows: list[dict[str, Any]],
    max_pairs_per_user: int,
    seed: int,
    *,
    true_max_pairs_per_user: int = 2,
    valid_max_pairs_per_user: int = 1,
    hist_max_pairs_per_user: int = 1,
    allow_mid_neg: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    user_pos: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    user_neg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped_missing_prompt_inputs = 0

    for row in rows:
        uid = _safe_int(row.get("user_idx", -1), -1)
        label = _safe_int(row.get("label", 0), 0)
        if uid < 0:
            continue
        if not _row_has_pair_prompt_inputs(row):
            skipped_missing_prompt_inputs += 1
            continue
        if label == 1:
            label_source = str(row.get("label_source", "") or "unknown")
            user_pos[uid][label_source].append(row)
        else:
            tier = str(row.get("neg_tier", "") or "unknown")
            if tier in {"observed_dislike", "hard", "near"} or (allow_mid_neg and tier == "mid"):
                user_neg[uid].append(row)

    source_limits = {
        "true": max(0, int(true_max_pairs_per_user)),
        "valid": max(0, int(valid_max_pairs_per_user)),
        "hist_pos": max(0, int(hist_max_pairs_per_user)),
    }
    pairs: list[dict[str, Any]] = []
    users_with_pairs = 0
    users_skipped_no_neg = 0
    pairs_per_user: list[int] = []
    chosen_label_source_counts: Counter[str] = Counter()
    rejected_neg_tier_counts: Counter[str] = Counter()
    selection_bucket_counts: Counter[str] = Counter()
    score_gaps: list[float] = []
    rank_gaps: list[int] = []
    chosen_tower_nonzero = 0
    rejected_tower_nonzero = 0
    chosen_seq_nonzero = 0
    rejected_seq_nonzero = 0
    rejected_topk_incumbent = 0
    rejected_high_rank = 0
    rejected_outranks_chosen = 0
    rejected_same_city = 0
    rejected_same_primary = 0
    rejected_category_overlap = 0
    rejected_model_confuser = 0
    rejected_competition_confuser = 0
    rejected_fit_risk_confuser = 0
    rejected_boundary_kickout = 0
    rejected_head_preserve = 0
    rejected_multi_route_preserve = 0
    rejected_low_support_preserve = 0
    rejected_structured_lift = 0
    rejected_deep_semantic = 0
    chosen_totals_by_band: Counter[str] = Counter()
    chosen_pack_sizes_by_band: dict[str, list[int]] = defaultdict(list)
    chosen_pack_sizes_by_reason: dict[str, list[int]] = defaultdict(list)
    chosen_slate_variants_by_band: dict[str, list[int]] = defaultdict(list)
    prompt_style_counts: Counter[str] = Counter()
    skipped_low_signal_prompt = 0
    chosen_prompt_field_present: Counter[str] = Counter()
    chosen_prompt_field_informative: Counter[str] = Counter()
    rejected_prompt_field_present: Counter[str] = Counter()
    rejected_prompt_field_informative: Counter[str] = Counter()
    chosen_prompt_scaffold_counts: Counter[str] = Counter()
    rejected_prompt_scaffold_counts: Counter[str] = Counter()
    chosen_structured_residue = 0
    rejected_structured_residue = 0
    chosen_prompt_chars: list[int] = []
    rejected_prompt_chars: list[int] = []
    prompt_char_gap: list[int] = []

    for uid in sorted(user_pos.keys()):
        neg_rows = user_neg.get(uid, [])
        if not neg_rows:
            users_skipped_no_neg += 1
            continue

        built_for_user = 0
        # De-duplicate negative items across variants for the same user-level
        # positive so export thickness does not come from repeated copies of one
        # blocker. Variant-specific refill logic handles underfilled slates later.
        seen_neg_items: set[int] = set()
        seen_neg_usage: Counter[int] = Counter()
        pos_groups = user_pos[uid]
        true_rows = pos_groups.get("true", [])
        has_true_source = bool(true_rows)

        for source_name in ("true", "valid", "hist_pos"):
            if DPO_PAIR_POLICY == "v3":
                if source_name == "true":
                    source_budget = min(int(max_pairs_per_user), max(0, int(max_pairs_per_user) - built_for_user))
                else:
                    if has_true_source and built_for_user > 0:
                        continue
                    source_budget = min(
                        int(source_limits.get(source_name, 0)),
                        max(0, int(max_pairs_per_user) - built_for_user),
                    )
            else:
                source_budget = min(int(source_limits.get(source_name, 0)), max(0, int(max_pairs_per_user) - built_for_user))
            if source_budget <= 0:
                continue
            pos_rows = pos_groups.get(source_name, [])
            if not pos_rows:
                continue
            pos_rows = sorted(
                pos_rows,
                key=lambda r: (
                    -_safe_float(r.get("sample_weight", 1.0), 1.0),
                    _safe_int(r.get("pre_rank", 999999), 999999),
                    -_safe_float(r.get("pre_score", 0.0), 0.0),
                ),
            )

            built_for_source = 0
            for pos_row in pos_rows:
                if built_for_source >= source_budget or built_for_user >= int(max_pairs_per_user):
                    break
                pos_band = _boundary_rank_band(_effective_rank(pos_row))
                if source_name == "true":
                    chosen_totals_by_band[pos_band] += 1
                selection_budget = min(
                    source_budget - built_for_source,
                    int(max_pairs_per_user) - built_for_user,
                )
                selection_budget = min(
                    int(selection_budget),
                    int(_target_pairs_for_positive(source_name, pos_band, int(selection_budget))),
                )
                effective_policy = _effective_pair_policy(source_name, pos_band)
                prompt_budget = int(selection_budget)
                if str(PAIR_PROMPT_STYLE or "").strip().lower() == "local_listwise_compare":
                    prompt_budget = max(
                        int(selection_budget),
                        int(PAIR_LOCAL_LISTWISE_MAX_RIVALS),
                        int(_v2d_min_pack_for_band(pos_band)),
                    )
                if effective_policy == "v3":
                    ranked_neg = _pick_v3_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                        uid=uid,
                    )
                elif effective_policy == "v2a":
                    ranked_neg = _pick_v2a_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                        uid=uid,
                    )
                elif effective_policy == "v2b":
                    ranked_neg = _pick_v2b_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                        uid=uid,
                    )
                elif effective_policy == "v2c":
                    ranked_neg = _pick_v2c_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                        uid=uid,
                    )
                elif effective_policy == "v2d":
                    ranked_neg = _pick_v2d_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                        uid=uid,
                    )
                    if pos_band == "rescue_31_60" and PAIR_V2D_MID_GENERIC_FILL and len(ranked_neg) < int(prompt_budget):
                        generic_ranked = _pick_generic_ranked_negatives(
                            pos_row,
                            neg_rows,
                            seen_neg_items,
                            int(prompt_budget),
                        )
                        ranked_neg = _merge_ranked_negatives(ranked_neg, generic_ranked, int(prompt_budget))
                else:
                    ranked_neg = _pick_generic_ranked_negatives(
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        int(prompt_budget),
                    )

                slate_variants = _build_local_listwise_slate_variants(pos_row, ranked_neg)
                if not slate_variants:
                    slate_variants = [(0, list(ranked_neg))]

                built_for_pos = 0
                used_slate_variants = 0
                for slate_variant_id, ranked_neg_variant in slate_variants:
                    if built_for_pos >= int(selection_budget) or built_for_source >= source_budget or built_for_user >= int(max_pairs_per_user):
                        break
                    effective_ranked_neg_variant = list(ranked_neg_variant)
                    if pos_band == "rescue_61_100" and effective_policy == "v2d":
                        effective_ranked_neg_variant = _finalize_local_deep_variant_rows(
                            effective_ranked_neg_variant,
                            list(ranked_neg),
                            seen_neg_items,
                            seen_neg_usage,
                            slate_variant_id=int(slate_variant_id),
                            slate_size=int(PAIR_LOCAL_LISTWISE_MAX_RIVALS),
                        )
                    if not effective_ranked_neg_variant:
                        continue
                    local_prompt_map: dict[int, str] | None = None
                    local_prompt_getter: Callable[[int], str] | None = None
                    local_rival_count = 0
                    if str(PAIR_PROMPT_STYLE or "").strip().lower() == "local_listwise_compare" and effective_ranked_neg_variant:
                        local_prompt_getter, local_rival_count = _prepare_local_listwise_prompt_context(
                            pos_row,
                            [neg_row for neg_row, _, _ in effective_ranked_neg_variant],
                        )
                    built_in_variant = 0
                    for neg_slot, (neg_row, neg_bucket, neg_feat) in enumerate(effective_ranked_neg_variant):
                        if built_for_pos >= int(selection_budget) or built_for_source >= source_budget or built_for_user >= int(max_pairs_per_user):
                            break
                        neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
                        if neg_item_idx < 0:
                            continue
                        if pos_band != "rescue_61_100" or effective_policy != "v2d":
                            if neg_item_idx in seen_neg_items:
                                continue
                            seen_neg_items.add(neg_item_idx)
                            seen_neg_usage[neg_item_idx] = max(1, int(seen_neg_usage.get(neg_item_idx, 0)))
                        if effective_policy in {"v2a", "v2b", "v2c", "v2d", "v3"}:
                            selection_bucket = f"{source_name}_gt_{neg_bucket}"
                        else:
                            selection_bucket = f"{source_name}_gt_{str(neg_row.get('neg_tier', '') or 'unknown')}"
                        pair = _build_pair_record(
                            pos_row,
                            neg_row,
                            uid,
                            pair_mode="rich_sft",
                            selection_bucket=selection_bucket,
                            selection_slot=int(neg_slot),
                            slate_variant_id=int(slate_variant_id),
                            local_prompt_map=local_prompt_map,
                            local_prompt_getter=local_prompt_getter,
                            local_rival_count=int(local_rival_count),
                        )
                        if pair is None:
                            skipped_low_signal_prompt += 1
                            continue
                        pairs.append(pair)
                        built_for_source += 1
                        built_for_user += 1
                        built_for_pos += 1
                        built_in_variant += 1
                        prompt_style_counts[str(pair.get("prompt_style", "") or "unknown")] += 1
                        chosen_prompt = str(
                            pair.get("chosen_prompt")
                            or pair.get("chosen")
                            or pair.get("prompt")
                            or ""
                        )
                        rejected_prompt = str(
                            pair.get("rejected_prompt")
                            or pair.get("rejected")
                            or pair.get("prompt")
                            or ""
                        )
                        chosen_prompt_chars.append(len(chosen_prompt))
                        rejected_prompt_chars.append(len(rejected_prompt))
                        prompt_char_gap.append(len(chosen_prompt) - len(rejected_prompt))
                        if _prompt_has_structured_residue(chosen_prompt):
                            chosen_structured_residue += 1
                        if _prompt_has_structured_residue(rejected_prompt):
                            rejected_structured_residue += 1
                        chosen_prompt_scaffold_counts.update(_prompt_scaffold_hits(chosen_prompt))
                        rejected_prompt_scaffold_counts.update(_prompt_scaffold_hits(rejected_prompt))
                        chosen_user_section = _extract_local_prompt_user_section(chosen_prompt)
                        chosen_focus_section = _extract_local_prompt_focus_section(chosen_prompt)
                        rejected_user_section = _extract_local_prompt_user_section(rejected_prompt)
                        rejected_focus_section = _extract_local_prompt_focus_section(rejected_prompt)
                        for field_name, section in (
                            ("user_focus", chosen_user_section),
                            ("recent_intent", chosen_user_section),
                            ("user_avoid", chosen_user_section),
                            ("history_pattern", chosen_user_section),
                            ("user_evidence", chosen_user_section),
                            ("user_match_points", chosen_focus_section),
                            ("user_conflict_points", chosen_focus_section),
                            ("business_scene", chosen_focus_section),
                            ("item_strengths", chosen_focus_section),
                            ("item_weaknesses", chosen_focus_section),
                            ("item_signal", chosen_focus_section),
                            ("evidence_basis", chosen_focus_section),
                        ):
                            raw = _extract_prompt_field_value(section, field_name)
                            if _prompt_field_present(raw):
                                chosen_prompt_field_present[field_name] += 1
                            if _prompt_field_informative(raw):
                                chosen_prompt_field_informative[field_name] += 1
                        for field_name, section in (
                            ("user_focus", rejected_user_section),
                            ("recent_intent", rejected_user_section),
                            ("user_avoid", rejected_user_section),
                            ("history_pattern", rejected_user_section),
                            ("user_evidence", rejected_user_section),
                            ("user_match_points", rejected_focus_section),
                            ("user_conflict_points", rejected_focus_section),
                            ("business_scene", rejected_focus_section),
                            ("item_strengths", rejected_focus_section),
                            ("item_weaknesses", rejected_focus_section),
                            ("item_signal", rejected_focus_section),
                            ("evidence_basis", rejected_focus_section),
                        ):
                            raw = _extract_prompt_field_value(section, field_name)
                            if _prompt_field_present(raw):
                                rejected_prompt_field_present[field_name] += 1
                            if _prompt_field_informative(raw):
                                rejected_prompt_field_informative[field_name] += 1
                        chosen_label_source_counts[str(pair.get("chosen_label_source", "") or "unknown")] += 1
                        rejected_neg_tier_counts[str(pair.get("rejected_neg_tier", "") or "unknown")] += 1
                        selection_bucket_counts[selection_bucket] += 1
                        score_gaps.append(float(pair["score_gap"]))
                        rank_gaps.append(int(pair["rank_gap"]))
                        if abs(_row_score(pos_row, "tower_score")) > 1e-9:
                            chosen_tower_nonzero += 1
                        if abs(_row_score(neg_row, "tower_score")) > 1e-9:
                            rejected_tower_nonzero += 1
                        if abs(_row_score(pos_row, "seq_score")) > 1e-9:
                            chosen_seq_nonzero += 1
                        if abs(_row_score(neg_row, "seq_score")) > 1e-9:
                            rejected_seq_nonzero += 1
                        if neg_feat:
                            if bool(neg_feat.get("topk_incumbent", False)):
                                rejected_topk_incumbent += 1
                            if bool(neg_feat.get("high_rank", False)):
                                rejected_high_rank += 1
                            if bool(neg_feat.get("outranks", False)):
                                rejected_outranks_chosen += 1
                            if bool(neg_feat.get("same_city", False)):
                                rejected_same_city += 1
                            if bool(neg_feat.get("same_primary", False)):
                                rejected_same_primary += 1
                            if int(neg_feat.get("category_overlap_count", 0)) > 0:
                                rejected_category_overlap += 1
                            if bool(neg_feat.get("model_confuser", False)):
                                rejected_model_confuser += 1
                            if bool(neg_feat.get("competition_confuser", False)):
                                rejected_competition_confuser += 1
                            if bool(neg_feat.get("fit_risk_confuser", False)):
                                rejected_fit_risk_confuser += 1
                            if bool(neg_feat.get("boundary_kickout_confuser", False)):
                                rejected_boundary_kickout += 1
                            if bool(neg_feat.get("head_preserve_confuser", False)):
                                rejected_head_preserve += 1
                            if bool(neg_feat.get("multi_route_preserve_confuser", False)):
                                rejected_multi_route_preserve += 1
                            if bool(neg_feat.get("low_support_preserve_confuser", False)):
                                rejected_low_support_preserve += 1
                            if bool(neg_feat.get("structured_lift_confuser", False)):
                                rejected_structured_lift += 1
                            if bool(neg_feat.get("deep_semantic_challenger", False)):
                                rejected_deep_semantic += 1
                    if built_in_variant > 0:
                        used_slate_variants += 1
                if source_name == "true" and built_for_pos > 0:
                    chosen_pack_sizes_by_band[pos_band].append(int(built_for_pos))
                    chosen_slate_variants_by_band[pos_band].append(int(max(1, used_slate_variants)))
                    chosen_reason = str(pos_row.get("primary_reason", "") or "").strip() or "unknown"
                    chosen_pack_sizes_by_reason[chosen_reason].append(int(built_for_pos))

        if built_for_user > 0:
            users_with_pairs += 1
            pairs_per_user.append(built_for_user)

    rng.shuffle(pairs)
    chosen_item_evidence = [1.0 if bool(p["chosen_has_item_evidence"]) else 0.0 for p in pairs]
    rejected_item_evidence = [1.0 if bool(p["rejected_has_item_evidence"]) else 0.0 for p in pairs]
    chosen_pair_evidence = [1.0 if bool(p["chosen_has_pair_evidence"]) else 0.0 for p in pairs]
    rejected_pair_evidence = [1.0 if bool(p["rejected_has_pair_evidence"]) else 0.0 for p in pairs]

    audit = {
        "users_with_pairs": int(users_with_pairs),
        "users_skipped_no_neg": int(users_skipped_no_neg),
        "rows_skipped_missing_prompt_inputs": int(skipped_missing_prompt_inputs),
        "pairs_skipped_low_signal_prompt": int(skipped_low_signal_prompt),
        "total_pairs": int(len(pairs)),
        "max_pairs_per_user": int(max_pairs_per_user),
        "true_max_pairs_per_user": int(true_max_pairs_per_user),
        "valid_max_pairs_per_user": int(valid_max_pairs_per_user),
        "hist_max_pairs_per_user": int(hist_max_pairs_per_user),
        "allow_mid_neg": bool(allow_mid_neg),
        "pair_policy": DPO_PAIR_POLICY,
        "topk_cutoff": int(DPO_PAIR_TOPK_CUTOFF),
        "high_rank_cutoff": int(DPO_PAIR_HIGH_RANK_CUTOFF),
        "pairs_per_user": _summary(pairs_per_user),
        "prompt_style_counts": dict(sorted(prompt_style_counts.items())),
        "chosen_label_source_counts": dict(sorted(chosen_label_source_counts.items())),
        "selected_neg_tier_counts": dict(sorted(rejected_neg_tier_counts.items())),
        "selection_bucket_counts": dict(sorted(selection_bucket_counts.items())),
        "score_gap": _summary(score_gaps),
        "rank_gap": _summary(rank_gaps),
        "coverage": {
            "chosen_item_evidence_rate": float(sum(chosen_item_evidence) / max(len(chosen_item_evidence), 1)),
            "rejected_item_evidence_rate": float(sum(rejected_item_evidence) / max(len(rejected_item_evidence), 1)),
            "chosen_pair_evidence_rate": float(sum(chosen_pair_evidence) / max(len(chosen_pair_evidence), 1)),
            "rejected_pair_evidence_rate": float(sum(rejected_pair_evidence) / max(len(rejected_pair_evidence), 1)),
            "chosen_tower_nonzero_rate": float(chosen_tower_nonzero / max(len(pairs), 1)),
            "rejected_tower_nonzero_rate": float(rejected_tower_nonzero / max(len(pairs), 1)),
            "chosen_seq_nonzero_rate": float(chosen_seq_nonzero / max(len(pairs), 1)),
            "rejected_seq_nonzero_rate": float(rejected_seq_nonzero / max(len(pairs), 1)),
            "chosen_true_rate": float(chosen_label_source_counts.get("true", 0) / max(len(pairs), 1)),
            "rejected_topk_incumbent_rate": float(rejected_topk_incumbent / max(len(pairs), 1)),
            "rejected_high_rank_rate": float(rejected_high_rank / max(len(pairs), 1)),
            "rejected_outranks_chosen_rate": float(rejected_outranks_chosen / max(len(pairs), 1)),
            "rejected_same_city_rate": float(rejected_same_city / max(len(pairs), 1)),
            "rejected_same_primary_category_rate": float(rejected_same_primary / max(len(pairs), 1)),
            "rejected_category_overlap_rate": float(rejected_category_overlap / max(len(pairs), 1)),
            "rejected_model_confuser_rate": float(rejected_model_confuser / max(len(pairs), 1)),
            "rejected_competition_confuser_rate": float(rejected_competition_confuser / max(len(pairs), 1)),
            "rejected_fit_risk_confuser_rate": float(rejected_fit_risk_confuser / max(len(pairs), 1)),
            "rejected_boundary_kickout_rate": float(rejected_boundary_kickout / max(len(pairs), 1)),
            "rejected_head_preserve_rate": float(rejected_head_preserve / max(len(pairs), 1)),
            "rejected_multi_route_preserve_rate": float(rejected_multi_route_preserve / max(len(pairs), 1)),
            "rejected_low_support_preserve_rate": float(rejected_low_support_preserve / max(len(pairs), 1)),
            "rejected_structured_lift_rate": float(rejected_structured_lift / max(len(pairs), 1)),
            "rejected_deep_semantic_rate": float(rejected_deep_semantic / max(len(pairs), 1)),
            "local_listwise_compare_rate": float(prompt_style_counts.get("local_listwise_compare", 0) / max(len(pairs), 1)),
            "blocker_compare_rate": float(prompt_style_counts.get("blocker_compare", 0) / max(len(pairs), 1)),
            "chosen_structured_residue_rate": float(chosen_structured_residue / max(len(pairs), 1)),
            "rejected_structured_residue_rate": float(rejected_structured_residue / max(len(pairs), 1)),
        },
        "prompt_quality": {
            "chosen_prompt_chars": _summary(chosen_prompt_chars),
            "rejected_prompt_chars": _summary(rejected_prompt_chars),
            "prompt_char_gap_chosen_minus_rejected": _summary(prompt_char_gap),
            "chosen_field_present_rate": {
                key: float(chosen_prompt_field_present.get(key, 0) / max(len(pairs), 1))
                for key in (
                    "user_focus",
                    "user_avoid",
                    "history_pattern",
                    "user_evidence",
                    "user_match_points",
                    "user_conflict_points",
                )
            },
            "chosen_field_informative_rate": {
                key: float(chosen_prompt_field_informative.get(key, 0) / max(len(pairs), 1))
                for key in (
                    "user_focus",
                    "user_avoid",
                    "history_pattern",
                    "user_evidence",
                    "user_match_points",
                    "user_conflict_points",
                )
            },
            "rejected_field_present_rate": {
                key: float(rejected_prompt_field_present.get(key, 0) / max(len(pairs), 1))
                for key in (
                    "user_focus",
                    "user_avoid",
                    "history_pattern",
                    "user_evidence",
                    "user_match_points",
                    "user_conflict_points",
                )
            },
            "rejected_field_informative_rate": {
                key: float(rejected_prompt_field_informative.get(key, 0) / max(len(pairs), 1))
                for key in (
                    "user_focus",
                    "user_avoid",
                    "history_pattern",
                    "user_evidence",
                    "user_match_points",
                    "user_conflict_points",
                )
            },
            "chosen_scaffold_phrase_count": dict(sorted(chosen_prompt_scaffold_counts.items())),
            "rejected_scaffold_phrase_count": dict(sorted(rejected_prompt_scaffold_counts.items())),
            "chosen_scaffold_phrase_rate": {
                key: float(chosen_prompt_scaffold_counts.get(key, 0) / max(len(pairs), 1))
                for key in sorted(_PROMPT_SCAFFOLD_PATTERNS.keys())
            },
            "rejected_scaffold_phrase_rate": {
                key: float(rejected_prompt_scaffold_counts.get(key, 0) / max(len(pairs), 1))
                for key in sorted(_PROMPT_SCAFFOLD_PATTERNS.keys())
            },
        },
        "chosen_pack_by_band": {
            str(band): {
                "chosen_total": int(chosen_totals_by_band.get(band, 0)),
                "chosen_with_pairs": int(len(vals)),
                "coverage_rate": float(len(vals) / max(int(chosen_totals_by_band.get(band, 0)), 1)),
                "pairs_per_chosen": _summary(vals),
            }
            for band, vals in sorted(chosen_pack_sizes_by_band.items())
        },
        "chosen_pack_by_reason": {
            str(reason): {
                "chosen_with_pairs": int(len(vals)),
                "pairs_per_chosen": _summary(vals),
            }
            for reason, vals in sorted(chosen_pack_sizes_by_reason.items())
        },
        "chosen_slates_by_band": {
            str(band): {
                "chosen_with_pairs": int(len(vals)),
                "slates_per_chosen": _summary(vals),
            }
            for band, vals in sorted(chosen_slate_variants_by_band.items())
        },
    }
    return pairs, audit


def build_rerank_pool_pairs(
    rows: list[dict[str, Any]],
    max_pairs_per_user: int,
    seed: int,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    user_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    user_neg: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        uid = _safe_int(row.get("user_idx", -1), -1)
        label = _safe_int(row.get("label", 0), 0)
        if uid < 0:
            continue
        if label == 1:
            user_pos[uid].append(row)
        else:
            user_neg[uid].append(row)

    pairs: list[dict[str, Any]] = []
    users_with_pairs = 0
    users_skipped = 0
    pairs_per_user: list[int] = []
    selected_buckets: Counter[str] = Counter()
    selected_neg_tiers: Counter[str] = Counter()
    score_gaps: list[float] = []
    rank_gaps: list[int] = []
    rejected_competition_confuser = 0
    rejected_fit_risk_confuser = 0
    skipped_low_signal_prompt = 0

    for uid in sorted(user_pos.keys()):
        pos_rows = user_pos[uid]
        neg_rows = user_neg.get(uid, [])
        if not neg_rows:
            users_skipped += 1
            continue

        built_for_user = 0
        for pos_row in pos_rows:
            pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
            safe_near: list[dict[str, Any]] = []
            safe_bands: dict[str, list[dict[str, Any]]] = defaultdict(list)
            inv_near: list[dict[str, Any]] = []
            inv_bands: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for neg_row in neg_rows:
                neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
                band = str(neg_row.get("pre_rank_band", "") or _band(neg_row.get("pre_rank")))
                is_near = bool(neg_row.get("neg_is_near", False))
                if pos_score >= neg_score:
                    if is_near:
                        safe_near.append(neg_row)
                    safe_bands[band].append(neg_row)
                else:
                    if is_near:
                        inv_near.append(neg_row)
                    inv_bands[band].append(neg_row)

            for rows_list in safe_bands.values():
                rows_list.sort(key=lambda r: _neg_sort_key(r, pos_score))
            for rows_list in inv_bands.values():
                rows_list.sort(key=lambda r: (_safe_float(r.get("pre_score", 0.0), 0.0) - pos_score, _safe_int(r.get("pre_rank", 999999), 999999)), reverse=True)
            safe_near.sort(key=lambda r: _neg_sort_key(r, pos_score))
            inv_near.sort(key=lambda r: (_safe_float(r.get("pre_score", 0.0), 0.0) - pos_score, _safe_int(r.get("pre_rank", 999999), 999999)), reverse=True)

            selected: list[tuple[dict[str, Any], str]] = []
            seen_items: set[int] = set()

            def _pick_from(candidates: list[dict[str, Any]], bucket: str, limit: int) -> None:
                if limit <= 0:
                    return
                for row in candidates:
                    item_idx = _safe_int(row.get("item_idx", -1), -1)
                    if item_idx in seen_items:
                        continue
                    seen_items.add(item_idx)
                    selected.append((row, bucket))
                    if len([1 for _, b in selected if b == bucket]) >= limit:
                        break

            if mode == "conservative":
                _pick_from(safe_near, "near_safe", 1)
                _pick_from(safe_bands.get("011_030", []), "band_011_030_safe", 1)
                _pick_from(safe_bands.get("031_080", []), "band_031_080_safe", 1)
                _pick_from(safe_bands.get("081_150", []), "band_081_150_safe", 1)
                fill_order = (
                    safe_bands.get("001_010", [])
                    + safe_bands.get("011_030", [])
                    + safe_bands.get("031_080", [])
                    + safe_bands.get("081_150", [])
                    + safe_near
                )
            elif mode == "hard":
                _pick_from(inv_near, "near_inverted", 1)
                _pick_from(inv_bands.get("001_010", []), "band_001_010_inverted", 1)
                fill_order = (
                    inv_bands.get("001_010", [])
                    + inv_bands.get("011_030", [])
                    + inv_near
                    + inv_bands.get("031_080", [])
                    + inv_bands.get("081_150", [])
                )
            else:
                raise ValueError(f"unsupported rerank pair mode: {mode}")

            for neg_row in fill_order:
                if len(selected) >= int(max_pairs_per_user):
                    break
                item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
                if item_idx in seen_items:
                    continue
                seen_items.add(item_idx)
                band = str(neg_row.get("pre_rank_band", "") or _band(neg_row.get("pre_rank")))
                bucket = f"{band}_{mode}"
                selected.append((neg_row, bucket))

            local_prompt_map: dict[int, str] | None = None
            local_prompt_getter: Callable[[int], str] | None = None
            local_rival_count = 0
            if str(PAIR_PROMPT_STYLE or "").strip().lower() == "local_listwise_compare" and selected:
                local_prompt_getter, local_rival_count = _prepare_local_listwise_prompt_context(
                    pos_row,
                    [neg_row for neg_row, _ in selected],
                )

            for neg_row, bucket in selected[: max_pairs_per_user]:
                neg_feat = _v2a_neg_features(pos_row, neg_row, pos_score, uid=uid)
                pair = _build_pair_record(
                    pos_row,
                    neg_row,
                    uid,
                    pair_mode=mode,
                    selection_bucket=bucket,
                    local_prompt_map=local_prompt_map,
                    local_prompt_getter=local_prompt_getter,
                    local_rival_count=int(local_rival_count),
                )
                if pair is None:
                    skipped_low_signal_prompt += 1
                    continue
                pairs.append(pair)
                built_for_user += 1
                selected_buckets[bucket] += 1
                selected_neg_tiers[pair["rejected_neg_tier"]] += 1
                score_gaps.append(pair["score_gap"])
                rank_gaps.append(pair["rank_gap"])
                if bool(neg_feat.get("competition_confuser", False)):
                    rejected_competition_confuser += 1
                if bool(neg_feat.get("fit_risk_confuser", False)):
                    rejected_fit_risk_confuser += 1

        if built_for_user > 0:
            users_with_pairs += 1
            pairs_per_user.append(built_for_user)

    rng.shuffle(pairs)
    chosen_item_evidence = [1.0 if bool(p["chosen_has_item_evidence"]) else 0.0 for p in pairs]
    rejected_item_evidence = [1.0 if bool(p["rejected_has_item_evidence"]) else 0.0 for p in pairs]
    chosen_pair_evidence = [1.0 if bool(p["chosen_has_pair_evidence"]) else 0.0 for p in pairs]
    rejected_pair_evidence = [1.0 if bool(p["rejected_has_pair_evidence"]) else 0.0 for p in pairs]
    neg_score_gt_pos = [1.0 if float(p["score_gap"]) < 0 else 0.0 for p in pairs]

    audit = {
        "mode": mode,
        "users_with_pairs": int(users_with_pairs),
        "users_skipped_no_neg": int(users_skipped),
        "pairs_skipped_low_signal_prompt": int(skipped_low_signal_prompt),
        "total_pairs": int(len(pairs)),
        "max_pairs_per_user": int(max_pairs_per_user),
        "pairs_per_user": _summary(pairs_per_user),
        "selected_bucket_counts": dict(sorted(selected_buckets.items())),
        "selected_neg_tier_counts": dict(sorted(selected_neg_tiers.items())),
        "score_gap": _summary(score_gaps),
        "rank_gap": _summary(rank_gaps),
        "neg_score_gt_pos_rate": float(sum(neg_score_gt_pos) / max(len(neg_score_gt_pos), 1)),
        "coverage": {
            "chosen_item_evidence_rate": float(sum(chosen_item_evidence) / max(len(chosen_item_evidence), 1)),
            "rejected_item_evidence_rate": float(sum(rejected_item_evidence) / max(len(rejected_item_evidence), 1)),
            "chosen_pair_evidence_rate": float(sum(chosen_pair_evidence) / max(len(chosen_pair_evidence), 1)),
            "rejected_pair_evidence_rate": float(sum(rejected_pair_evidence) / max(len(rejected_pair_evidence), 1)),
        },
        "bucket5_confuser_rates": {
            "rejected_competition_confuser_rate": float(rejected_competition_confuser / max(len(pairs), 1)),
            "rejected_fit_risk_confuser_rate": float(rejected_fit_risk_confuser / max(len(pairs), 1)),
        },
    }
    return pairs, audit


def pair_records_for_training(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"prompt": str(p["prompt"]), "chosen": str(p["chosen"]), "rejected": str(p["rejected"])} for p in pairs]


def pair_records_for_reward_training(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        shared_prompt = str(pair.get("prompt", "") or "")
        chosen = str(pair.get("chosen", "") or "")
        rejected = str(pair.get("rejected", "") or "")
        split = str(pair.get("split", "") or "train")
        user_idx = _safe_int(pair.get("user_idx", -1), -1)
        chosen_item_idx = _safe_int(pair.get("chosen_item_idx", -1), -1)
        chosen_label_source = str(pair.get("chosen_label_source", "") or "")
        slate_variant_id = _safe_int(pair.get("slate_variant_id", 0), 0)
        group_key = f"{split}:{user_idx}:{chosen_label_source}:{chosen_item_idx}"
        if slate_variant_id > 0:
            group_key = f"{group_key}:sv{slate_variant_id}"
        records.append(
            {
                "text_chosen": shared_prompt + chosen,
                "text_rejected": shared_prompt + rejected,
                "selection_bucket": str(pair.get("selection_bucket", "") or ""),
                "selection_slot": _safe_int(pair.get("selection_slot", 0), 0),
                "slate_variant_id": int(slate_variant_id),
                "chosen_label_source": chosen_label_source,
                "split": split,
                "group_key": group_key,
                "user_idx": user_idx,
                "chosen_item_idx": chosen_item_idx,
                "rejected_item_idx": _safe_int(pair.get("rejected_item_idx", -1), -1),
                "chosen_business_id": str(pair.get("chosen_business_id", "") or ""),
                "rejected_business_id": str(pair.get("rejected_business_id", "") or ""),
                "score_gap": _safe_float(pair.get("score_gap", 0.0), 0.0),
                "learned_score_gap": _safe_float(pair.get("learned_score_gap", 0.0), 0.0),
                "rank_gap": _safe_int(pair.get("rank_gap", 0), 0),
                "learned_rank_gap": _safe_int(pair.get("learned_rank_gap", 0), 0),
                "chosen_pre_rank": _safe_int(pair.get("chosen_pre_rank", -1), -1),
                "rejected_pre_rank": _safe_int(pair.get("rejected_pre_rank", -1), -1),
                "chosen_learned_rank": _safe_int(pair.get("chosen_learned_rank", -1), -1),
                "rejected_learned_rank": _safe_int(pair.get("rejected_learned_rank", -1), -1),
                "chosen_learned_rank_band": str(pair.get("chosen_learned_rank_band", "") or ""),
                "rejected_learned_rank_band": str(pair.get("rejected_learned_rank_band", "") or ""),
                "chosen_primary_reason": str(pair.get("chosen_primary_reason", "") or ""),
                "chosen_easy_but_useful": bool(pair.get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(pair.get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(pair.get("chosen_non_actionable", False)),
                "chosen_learned_blend_score": _safe_float(pair.get("chosen_learned_blend_score", 0.0), 0.0),
                "rejected_learned_blend_score": _safe_float(pair.get("rejected_learned_blend_score", 0.0), 0.0),
                "chosen_has_item_evidence": bool(pair.get("chosen_has_item_evidence", False)),
                "rejected_has_item_evidence": bool(pair.get("rejected_has_item_evidence", False)),
                "chosen_has_pair_evidence": bool(pair.get("chosen_has_pair_evidence", False)),
                "rejected_has_pair_evidence": bool(pair.get("rejected_has_pair_evidence", False)),
            }
        )
    return records
