import argparse
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd


SOURCE_07_ROOT = Path(r"D:/5006 BDA project/data/output/07_embedding_cluster")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/07_baseline_relabel_vector")
DEFAULT_PROFILE = "sample"

# Recompute/correction heuristics
MIXED_CLUSTER_THRESHOLD = 0.55
L2_AMBIGUOUS_GAP_MAX = 0.8
L2_LOW_SIGNAL_MIN_SCORE = 1.25
CLUSTER_DOMINANT_SHARE_STRONG = 0.85

# Baseline sample quotas
BASELINE_SIZE = 200
QUOTA_LOW_CONF = 80
QUOTA_GENERAL = 60
QUOTA_MIXED = 40
QUOTA_CONTROL = 20

# Vector settings
PROFILE_FEATURE_WEIGHT = 0.45
CITY_TOPK = 8
RANDOM_SEED = 42

SPECIFIC_L2_BLOCKLIST = {"", "other_service", "restaurants_general"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 200-baseline relabel set and semantic+profile vectors."
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["sample", "full"])
    parser.add_argument(
        "--source-run-dir",
        default="",
        help="Optional absolute/relative path to one 07 run dir.",
    )
    parser.add_argument(
        "--embedding-cache",
        default="",
        help="Optional path to embedding cache npz. If empty, auto-pick cache for profile.",
    )
    parser.add_argument("--baseline-size", type=int, default=BASELINE_SIZE)
    return parser.parse_args()


def clean_text(s: object, limit: int = 800) -> str:
    x = re.sub(r"\s+", " ", str("" if s is None else s)).strip()
    return x[:limit]


def normalize_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in {"sample", "full"}:
        raise ValueError(f"profile must be sample/full, got: {profile!r}")
    return p


def run_name_matches_profile(run_name: str, profile: str) -> bool:
    return re.search(rf"(^|_){re.escape(profile)}($|_)", run_name.lower()) is not None


def pick_source_run_dir(profile: str, override: str) -> Path:
    if override.strip():
        run_dir = Path(override.strip())
        if not run_dir.exists():
            raise FileNotFoundError(f"source run dir not found: {run_dir}")
        return run_dir

    runs = [p for p in SOURCE_07_ROOT.iterdir() if p.is_dir() and p.name != "_cache"]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    profile_runs = [p for p in runs if run_name_matches_profile(p.name, profile)]
    if not profile_runs:
        raise FileNotFoundError(f"No 07 run found for profile={profile} under {SOURCE_07_ROOT}")
    return profile_runs[0]


def parse_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def trigger_flags(text: object) -> set[str]:
    s = str("" if text is None else text).strip().lower()
    if not s:
        return set()
    out: set[str] = set()
    for token in s.split(","):
        t = token.strip()
        if t:
            out.add(t)
    return out


def is_specific_l2(label: object) -> bool:
    token = str("" if label is None else label).strip().lower()
    return token not in SPECIFIC_L2_BLOCKLIST


def build_cluster_stats(assign_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster, g in assign_df.groupby("cluster"):
        counts = g["l2_label_top1"].fillna("").astype(str).str.lower().value_counts()
        if counts.empty:
            dominant = ""
            share = 0.0
        else:
            dominant = str(counts.index[0])
            share = float(counts.iloc[0]) / float(counts.sum())
        rows.append(
            {
                "cluster": int(cluster),
                "cluster_dominant_l2": dominant,
                "cluster_dominant_share": share,
                "is_mixed_cluster": bool(share < MIXED_CLUSTER_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


def recalc_one(row: pd.Series) -> tuple[str, int, str, str]:
    l1 = str(row.get("l1_label", "")).strip().lower()
    top1 = str(row.get("l2_label_top1", "")).strip().lower()
    top2 = str(row.get("l2_label_top2", "")).strip().lower()
    score1 = parse_float(row.get("l2_score_top1", 0.0), 0.0)
    score2 = parse_float(row.get("l2_score_top2", 0.0), 0.0)
    gap = parse_float(row.get("l2_score_gap", 99.0), 99.0)
    conf = max(1, min(5, parse_int(row.get("label_confidence", 2), 2)))
    cluster_l2 = str(row.get("cluster_dominant_l2", "")).strip().lower()
    cluster_share = parse_float(row.get("cluster_dominant_share", 0.0), 0.0)
    mixed = bool(row.get("is_mixed_cluster", False))
    flags = trigger_flags(row.get("llm_trigger_reason", ""))

    if l1 != "food_service":
        return top1, conf, "keep_non_food_service", "l1 not food_service; keep original"

    recalc = top1
    action = "keep"
    reason = "no correction rule hit"

    if top1 == "other_service" and is_specific_l2(top2) and score2 >= 0.8:
        recalc = top2
        action = "other_to_top2"
        reason = f"top1=other_service; top2={top2} score2={score2:.2f}"
    elif top1 == "restaurants_general":
        if is_specific_l2(top2) and score2 >= 0.9:
            recalc = top2
            action = "general_to_top2"
            reason = f"top1=general; top2={top2} score2={score2:.2f}"
        elif (
            is_specific_l2(cluster_l2)
            and cluster_share >= CLUSTER_DOMINANT_SHARE_STRONG
            and not mixed
        ):
            recalc = cluster_l2
            action = "general_to_cluster"
            reason = (
                f"top1=general; cluster_dominant={cluster_l2}; "
                f"share={cluster_share:.3f}"
            )
    else:
        ambiguous = gap <= L2_AMBIGUOUS_GAP_MAX or ("l2_ambiguous" in flags)
        low_signal = score1 < L2_LOW_SIGNAL_MIN_SCORE or ("low_l2_signal" in flags)
        if (
            (ambiguous or low_signal)
            and is_specific_l2(cluster_l2)
            and cluster_share >= CLUSTER_DOMINANT_SHARE_STRONG
            and not mixed
            and cluster_l2 != top1
        ):
            recalc = cluster_l2
            action = "ambiguous_to_cluster"
            reason = (
                f"top1={top1}; cluster_dominant={cluster_l2}; "
                f"share={cluster_share:.3f}; ambiguous={ambiguous}; low_signal={low_signal}"
            )

    conf_new = conf
    if action in {"other_to_top2", "general_to_top2"}:
        conf_new = max(conf_new, 3)
    elif action in {"general_to_cluster", "ambiguous_to_cluster"}:
        conf_new = max(conf_new, 4 if cluster_share >= 0.95 else 3)

    return recalc, conf_new, action, reason


def add_recalc_labels(df: pd.DataFrame) -> pd.DataFrame:
    recalcs = df.apply(recalc_one, axis=1, result_type="expand")
    recalcs.columns = [
        "l2_label_recalc",
        "label_confidence_recalc",
        "label_correction_action",
        "label_correction_reason",
    ]
    out = df.copy()
    out[recalcs.columns] = recalcs
    out["label_changed"] = out["l2_label_recalc"] != out["l2_label_top1"].fillna("").astype(str).str.lower()
    return out


def stratified_baseline_sample(df: pd.DataFrame, baseline_size: int) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_SEED)
    work = df.copy()
    work = work[work["l1_label"].astype(str).str.lower() == "food_service"].copy()

    conf = pd.to_numeric(work["label_confidence"], errors="coerce").fillna(2.0)
    gap = pd.to_numeric(work["l2_score_gap"], errors="coerce").fillna(99.0)
    trig = work["llm_trigger_reason"].fillna("").astype(str).str.lower()

    mask_low = (conf <= 2.0) | (gap <= L2_AMBIGUOUS_GAP_MAX) | trig.str.contains("l2_ambiguous|low_l2_signal")
    mask_general = work["l2_label_top1"].fillna("").astype(str).str.lower().eq("restaurants_general")
    mask_mixed = work["is_mixed_cluster"].fillna(False).astype(bool)
    mask_control = (conf >= 4.0) & (~mask_low) & (~mask_general) & (~mask_mixed)

    selected_ids: set[str] = set()
    sampled_parts: list[pd.DataFrame] = []

    def sample_bucket(mask: pd.Series, quota: int, bucket: str) -> None:
        nonlocal sampled_parts
        cand = work[mask & (~work["business_id"].isin(selected_ids))]
        if cand.empty or quota <= 0:
            return
        n = min(quota, len(cand))
        idx = rng.choice(cand.index.to_numpy(), size=n, replace=False)
        part = cand.loc[idx].copy()
        part["baseline_bucket"] = bucket
        sampled_parts.append(part)
        selected_ids.update(part["business_id"].astype(str).tolist())

    sample_bucket(mask_low, QUOTA_LOW_CONF, "low_conf_or_ambiguous")
    sample_bucket(mask_general, QUOTA_GENERAL, "general_label_focus")
    sample_bucket(mask_mixed, QUOTA_MIXED, "mixed_cluster_focus")
    sample_bucket(mask_control, QUOTA_CONTROL, "high_conf_control")

    current = pd.concat(sampled_parts, axis=0) if sampled_parts else work.iloc[:0].copy()
    need = max(0, baseline_size - len(current))
    if need > 0:
        remain = work[~work["business_id"].isin(selected_ids)]
        if not remain.empty:
            n = min(need, len(remain))
            idx = rng.choice(remain.index.to_numpy(), size=n, replace=False)
            extra = remain.loc[idx].copy()
            extra["baseline_bucket"] = "fill_random"
            current = pd.concat([current, extra], axis=0)

    current = current.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    current["review_snippet"] = current["text"].map(lambda x: clean_text(x, 700))
    current["l1_gold"] = ""
    current["l2_gold"] = ""
    current["gold_reason"] = ""
    current["can_decide"] = ""
    return current


def pick_embedding_cache(profile: str, override: str) -> Path:
    if override.strip():
        p = Path(override.strip())
        if not p.exists():
            raise FileNotFoundError(f"embedding cache not found: {p}")
        return p

    cache_dir = SOURCE_07_ROOT / "_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")

    pattern = "*_relabel1.npz" if profile == "sample" else "*.npz"
    cands = [p for p in cache_dir.glob(pattern) if p.is_file()]
    if profile == "sample":
        cands = [p for p in cands if "_s0.08_" in p.name]
    else:
        cands = [p for p in cands if "_s0.0_" in p.name]
    if not cands:
        raise FileNotFoundError(f"No embedding cache found for profile={profile} in {cache_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def zscore(col: np.ndarray) -> np.ndarray:
    x = col.astype(np.float32)
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    if sigma < 1e-6:
        sigma = 1.0
    return ((x - mu) / sigma).astype(np.float32)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_profile_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = len(df)
    features: list[np.ndarray] = []
    names: list[str] = []

    num_cols = {
        "num_stars": pd.to_numeric(df["stars"], errors="coerce").fillna(0.0).to_numpy(np.float32),
        "num_review_count_log1p": np.log1p(
            pd.to_numeric(df["review_count"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        ),
        "num_avg_review_stars": pd.to_numeric(df["avg_review_stars"], errors="coerce").fillna(0.0).to_numpy(np.float32),
        "num_n_reviews_log1p": np.log1p(
            pd.to_numeric(df["n_reviews"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        ),
        "num_label_confidence_recalc": pd.to_numeric(df["label_confidence_recalc"], errors="coerce").fillna(0.0).to_numpy(np.float32),
    }
    for k, v in num_cols.items():
        features.append(zscore(v).reshape(n, 1))
        names.append(k)

    l2_tokens = df["l2_label_recalc"].fillna("").astype(str).str.lower().str.strip()
    l2_dummies = pd.get_dummies(l2_tokens, prefix="l2")
    for col in sorted(l2_dummies.columns.tolist()):
        features.append(l2_dummies[col].to_numpy(np.float32).reshape(n, 1))
        names.append(col)

    city = df["city"].fillna("").astype(str).str.strip().str.lower()
    top_cities = city.value_counts().head(CITY_TOPK).index.tolist()
    city_norm = city.where(city.isin(top_cities), "other_city")
    city_dummies = pd.get_dummies(city_norm, prefix="city")
    for col in sorted(city_dummies.columns.tolist()):
        features.append(city_dummies[col].to_numpy(np.float32).reshape(n, 1))
        names.append(col)

    cat_text = (
        df["categories"].fillna("").astype(str).str.lower()
        + " "
        + df["name"].fillna("").astype(str).str.lower()
    )
    indicator_terms = {
        "scene_nightlife": ["nightlife", "bars", "bar", "cocktail", "beer", "music", "lounges"],
        "scene_morning": ["breakfast", "brunch", "coffee", "tea", "cafe", "espresso", "latte"],
        "scene_dessert": ["dessert", "bakery", "pastry", "cake", "ice cream", "frozen yogurt", "donut"],
        "scene_quick_service": ["fast food", "drive thru", "drive-thru", "burger", "sandwich", "pizza"],
        "scene_seafood": ["seafood", "oyster", "shrimp", "crawfish", "fish"],
        "scene_cajun_creole": ["cajun", "creole", "gumbo", "po boy", "po' boy"],
        "scene_asian": ["chinese", "japanese", "sushi", "ramen", "vietnamese", "pho", "thai", "korean"],
        "scene_latin": ["mexican", "taco", "burrito", "latin", "tex-mex"],
        "scene_italian": ["italian", "pasta", "pizzeria"],
        "scene_retail": ["grocery", "supermarket", "convenience store", "pharmacy", "drugstore", "market"],
    }
    for fname, terms in indicator_terms.items():
        escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in terms]
        pattern = "|".join([rf"(?<![a-z0-9]){e}(?![a-z0-9])" for e in escaped])
        vec = cat_text.str.contains(pattern, regex=True).astype(np.float32).to_numpy().reshape(n, 1)
        features.append(vec)
        names.append(fname)

    mat = np.concatenate(features, axis=1).astype(np.float32)
    return mat, names


def create_vectors(corrected_df: pd.DataFrame, embedding_cache: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    data = np.load(embedding_cache, allow_pickle=False)
    emb_ids = pd.Series(data["business_id"].astype(str), name="business_id")
    emb_mat = np.asarray(data["embeddings"], dtype=np.float32)
    emb_df = pd.DataFrame({"business_id": emb_ids, "_emb_idx": np.arange(len(emb_ids))})

    work = corrected_df.copy()
    work = work[work["l1_label"].astype(str).str.lower() == "food_service"].copy()
    work = work.merge(emb_df, on="business_id", how="inner")
    work = work.reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No food_service rows matched embedding cache.")

    semantic = emb_mat[work["_emb_idx"].to_numpy(dtype=int)]
    semantic = normalize_rows(semantic.astype(np.float32))

    profile_mat, profile_names = build_profile_feature_matrix(work)
    profile_mat = normalize_rows(profile_mat)

    fused = np.concatenate([semantic, profile_mat * np.float32(PROFILE_FEATURE_WEIGHT)], axis=1)
    meta = {
        "n_rows": int(len(work)),
        "semantic_dim": int(semantic.shape[1]),
        "profile_dim": int(profile_mat.shape[1]),
        "fused_dim": int(fused.shape[1]),
        "profile_feature_weight": float(PROFILE_FEATURE_WEIGHT),
    }
    return work, fused.astype(np.float32), profile_names, meta


def main() -> None:
    args = parse_args()
    profile = normalize_profile(args.profile)
    run_dir = pick_source_run_dir(profile, args.source_run_dir)
    emb_cache = pick_embedding_cache(profile, args.embedding_cache)

    relabel_csv = run_dir / "biz_relabels.csv"
    assign_csv = run_dir / "biz_cluster_assignments.csv"
    if not relabel_csv.exists() or not assign_csv.exists():
        raise FileNotFoundError(f"Required files missing in run dir: {run_dir}")

    relabel_df = pd.read_csv(relabel_csv)
    assign_df = pd.read_csv(assign_csv)
    cluster_stats = build_cluster_stats(assign_df)

    merged = relabel_df.merge(
        assign_df[["business_id", "cluster"]].drop_duplicates(),
        on="business_id",
        how="left",
    ).merge(cluster_stats, on="cluster", how="left")
    merged["cluster_dominant_l2"] = merged["cluster_dominant_l2"].fillna("")
    merged["cluster_dominant_share"] = pd.to_numeric(
        merged["cluster_dominant_share"], errors="coerce"
    ).fillna(0.0)
    merged["is_mixed_cluster"] = merged["is_mixed_cluster"].fillna(False).astype(bool)

    corrected = add_recalc_labels(merged)
    baseline = stratified_baseline_sample(corrected, int(args.baseline_size))

    vector_df, fused_vectors, profile_names, vec_meta = create_vectors(corrected, emb_cache)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{profile}_baseline{int(args.baseline_size)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corrected_csv = out_dir / "relabel_corrected_full.csv"
    baseline_csv = out_dir / "baseline_200_relabel_for_review.csv"
    vector_meta_csv = out_dir / "vector_input_corrected_food_service.csv"
    profile_schema_csv = out_dir / "profile_feature_schema.csv"
    vectors_npz = out_dir / "semantic_profile_vectors.npz"
    run_meta_csv = out_dir / "run_meta.csv"

    corrected.to_csv(corrected_csv, index=False, encoding="utf-8-sig")
    baseline_cols = [
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
        "avg_review_stars",
        "n_reviews",
        "review_snippet",
        "cluster",
        "cluster_dominant_l2",
        "cluster_dominant_share",
        "is_mixed_cluster",
        "baseline_bucket",
        "l1_label",
        "l2_label_top1",
        "l2_label_top2",
        "label_confidence",
        "llm_trigger_reason",
        "label_source",
        "l2_score_top1",
        "l2_score_top2",
        "l2_score_gap",
        "l2_label_recalc",
        "label_confidence_recalc",
        "label_changed",
        "label_correction_action",
        "label_correction_reason",
        "l1_gold",
        "l2_gold",
        "gold_reason",
        "can_decide",
    ]
    baseline[baseline_cols].to_csv(baseline_csv, index=False, encoding="utf-8-sig")

    vector_export_cols = [
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
        "avg_review_stars",
        "n_reviews",
        "cluster",
        "l2_label_top1",
        "l2_label_recalc",
        "label_changed",
        "label_correction_action",
    ]
    vector_df[vector_export_cols].to_csv(vector_meta_csv, index=False, encoding="utf-8-sig")

    schema_df = pd.DataFrame(
        {
            "profile_feature_index": np.arange(len(profile_names), dtype=int),
            "profile_feature_name": profile_names,
        }
    )
    schema_df.to_csv(profile_schema_csv, index=False, encoding="utf-8-sig")

    np.savez(
        vectors_npz,
        business_id=vector_df["business_id"].astype(str).to_numpy(),
        fused_vectors=fused_vectors.astype(np.float32),
        profile_feature_names=np.asarray(profile_names, dtype="<U64"),
    )

    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "profile": profile,
                "source_run_dir": str(run_dir),
                "embedding_cache": str(emb_cache),
                "rows_total": int(len(corrected)),
                "rows_food_service": int((corrected["l1_label"] == "food_service").sum()),
                "rows_label_changed": int(corrected["label_changed"].sum()),
                "baseline_rows": int(len(baseline)),
                "baseline_size_target": int(args.baseline_size),
                "vector_rows": int(vec_meta["n_rows"]),
                "semantic_dim": int(vec_meta["semantic_dim"]),
                "profile_dim": int(vec_meta["profile_dim"]),
                "fused_dim": int(vec_meta["fused_dim"]),
                "profile_feature_weight": float(vec_meta["profile_feature_weight"]),
                "mixed_cluster_threshold": float(MIXED_CLUSTER_THRESHOLD),
            }
        ]
    ).to_csv(run_meta_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] source run: {run_dir}")
    print(f"[INFO] embedding cache: {emb_cache}")
    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] corrected rows: {len(corrected)}")
    print(f"[INFO] label changed rows: {int(corrected['label_changed'].sum())}")
    print(f"[INFO] baseline rows: {len(baseline)}")
    print(
        "[INFO] vectors shape: "
        f"{int(vec_meta['n_rows'])} x {int(vec_meta['fused_dim'])} "
        f"(semantic={int(vec_meta['semantic_dim'])}, profile={int(vec_meta['profile_dim'])})"
    )


if __name__ == "__main__":
    main()
