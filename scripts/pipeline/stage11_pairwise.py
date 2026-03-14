from __future__ import annotations

import csv
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any


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


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


_PROMPT_SCORE_PATTERNS = {
    "tower_score": re.compile(r"(?:^|[;\n])\s*tower_score:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"),
    "seq_score": re.compile(r"(?:^|[;\n])\s*seq_score:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"),
}
_BINARY_PROMPT_RE = re.compile(
    r"^(?P<head>.*?\nUser: )(?P<user>.*?)(?P<candidate>\nCandidate: )(?P<item>.*?)(?P<answer>\nAnswer:)\s*$",
    re.S,
)
_USER_VARIANT_MARKER_RE = re.compile(r"(?:;\s*)?(history_anchors:|pair_signal:)")

DPO_PAIR_POLICY = os.getenv("QLORA_DPO_PAIR_POLICY", "v1").strip().lower() or "v1"
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
        "hardness": float(hardness),
        "score_gap": float(score_gap),
        "tower_seq": float(tower_seq),
        "model_confuser": bool(model_confuser),
        "model_confuser_rank": int(model_confuser_rank),
        "model_confuser_score": float(model_confuser_score),
        "model_confuser_outranks_true": bool(model_confuser_outranks_true),
        "tier_order": int(_v2a_tier_order(tier)),
    }


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


def _build_pair_record(
    pos_row: dict[str, Any],
    neg_row: dict[str, Any],
    uid: int,
    pair_mode: str,
    selection_bucket: str,
) -> dict[str, Any]:
    pos_prompt = str(pos_row.get("prompt", "")).strip()
    neg_prompt = str(neg_row.get("prompt", "")).strip()
    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
    neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
    common_prompt = ""
    chosen_text = (pos_prompt + " YES") if pos_prompt else ""
    rejected_text = (neg_prompt + " YES") if neg_prompt else ""
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
    return {
        "user_idx": uid,
        "split": str(pos_row.get("split", "") or neg_row.get("split", "") or "train"),
        "pair_mode": pair_mode,
        "selection_bucket": selection_bucket,
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
        "chosen_pre_score": pos_score,
        "rejected_pre_score": neg_score,
        "score_gap": float(pos_score - neg_score),
        "rank_gap": int(_safe_int(neg_row.get("pre_rank", -1), -1) - _safe_int(pos_row.get("pre_rank", -1), -1)),
        "chosen_has_user_evidence": _has_text(pos_row.get("user_evidence_text")),
        "chosen_has_item_evidence": _has_text(pos_row.get("item_evidence_text")),
        "chosen_has_pair_evidence": _has_text(pos_row.get("pair_evidence_summary")),
        "rejected_has_user_evidence": _has_text(neg_row.get("user_evidence_text")),
        "rejected_has_item_evidence": _has_text(neg_row.get("item_evidence_text")),
        "rejected_has_pair_evidence": _has_text(neg_row.get("pair_evidence_summary")),
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
        if _has_text(row.get("pair_evidence_summary")):
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
    }
    return pairs, audit


def _rich_neg_hardness(neg_row: dict[str, Any], pos_score: float) -> tuple[float, float, int]:
    neg_score = _safe_float(neg_row.get("pre_score", 0.0), 0.0)
    tower_score = max(0.0, _row_score(neg_row, "tower_score"))
    seq_score = max(0.0, _row_score(neg_row, "seq_score"))
    pre_rank = max(1, _safe_int(neg_row.get("pre_rank", 999999), 999999))
    pair_bonus = 0.03 if _has_text(neg_row.get("pair_evidence_summary")) else 0.0
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

    for row in rows:
        uid = _safe_int(row.get("user_idx", -1), -1)
        label = _safe_int(row.get("label", 0), 0)
        prompt = str(row.get("prompt", "")).strip()
        if uid < 0 or not prompt:
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

    for uid in sorted(user_pos.keys()):
        neg_rows = user_neg.get(uid, [])
        if not neg_rows:
            users_skipped_no_neg += 1
            continue

        built_for_user = 0
        seen_neg_items: set[int] = set()
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
                if DPO_PAIR_POLICY == "v3":
                    ranked_neg = _pick_v3_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        min(source_budget - built_for_source, int(max_pairs_per_user) - built_for_user),
                        uid=uid,
                    )
                elif DPO_PAIR_POLICY == "v2a":
                    ranked_neg = _pick_v2a_negatives(
                        source_name,
                        pos_row,
                        neg_rows,
                        seen_neg_items,
                        min(source_budget - built_for_source, int(max_pairs_per_user) - built_for_user),
                        uid=uid,
                    )
                else:
                    pos_score = _safe_float(pos_row.get("pre_score", 0.0), 0.0)
                    ranked_neg = []
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
                        ranked_neg.append((neg_row, str(neg_row.get("neg_tier", "") or "unknown"), {}))

                for neg_row, neg_bucket, neg_feat in ranked_neg:
                    if built_for_source >= source_budget or built_for_user >= int(max_pairs_per_user):
                        break
                    neg_item_idx = _safe_int(neg_row.get("item_idx", -1), -1)
                    if neg_item_idx < 0 or neg_item_idx in seen_neg_items:
                        continue
                    seen_neg_items.add(neg_item_idx)
                    if DPO_PAIR_POLICY in {"v2a", "v3"}:
                        selection_bucket = f"{source_name}_gt_{neg_bucket}"
                    else:
                        selection_bucket = f"{source_name}_gt_{str(neg_row.get('neg_tier', '') or 'unknown')}"
                    pair = _build_pair_record(
                        pos_row,
                        neg_row,
                        uid,
                        pair_mode="rich_sft",
                        selection_bucket=selection_bucket,
                    )
                    pairs.append(pair)
                    built_for_source += 1
                    built_for_user += 1
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

            for neg_row, bucket in selected[: max_pairs_per_user]:
                pair = _build_pair_record(pos_row, neg_row, uid, pair_mode=mode, selection_bucket=bucket)
                pairs.append(pair)
                built_for_user += 1
                selected_buckets[bucket] += 1
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
    neg_score_gt_pos = [1.0 if float(p["score_gap"]) < 0 else 0.0 for p in pairs]

    audit = {
        "mode": mode,
        "users_with_pairs": int(users_with_pairs),
        "users_skipped_no_neg": int(users_skipped),
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
    }
    return pairs, audit


def pair_records_for_training(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"prompt": str(p["prompt"]), "chosen": str(p["chosen"]), "rejected": str(p["rejected"])} for p in pairs]
