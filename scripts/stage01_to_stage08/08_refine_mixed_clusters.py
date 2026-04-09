from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from stage07_core import load_stage07_module


# Source from step 08
SOURCE_08_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels")
SOURCE_08_RUN_DIR = ""  # optional override
RUN_PROFILE = "full"  # "sample" | "full"

# Output
RUN_TAG = "mixed_refine"

# Selection
MIXED_SHARE_THRESHOLD = 0.55
USE_MIXED_FLAG = True
MIN_CLUSTER_SIZE = 60

# Model/feature
TOP_TERMS_PER_SUBCLUSTER = 12
TFIDF_MAX_FEATURES = 12000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.85
TFIDF_NGRAM_RANGE = (1, 2)
USE_HYBRID_L2_FEATURES = True
HYBRID_L2_WEIGHT = 0.45


def normalize_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in {"sample", "full"}:
        raise ValueError(f"RUN_PROFILE must be 'sample' or 'full', got: {profile!r}")
    return p


def run_name_matches_profile(run_name: str, profile: str) -> bool:
    return f"_{profile}_" in run_name.lower() or run_name.lower().endswith(f"_{profile}")


def pick_source_08_run_dir(profile: str) -> Path:
    if SOURCE_08_RUN_DIR.strip():
        p = Path(SOURCE_08_RUN_DIR.strip())
        if not p.exists():
            raise FileNotFoundError(f"SOURCE_08_RUN_DIR not found: {p}")
        return p
    base = SOURCE_08_ROOT / profile
    if not base.exists():
        raise FileNotFoundError(f"08 profile dir not found: {base}")
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        if not run_name_matches_profile(p.name, profile):
            continue
        has_files = (
            (p / "cluster_labels.csv").exists()
            and (p / "biz_cluster_assignments_labeled.csv").exists()
            and (p / "run_meta.csv").exists()
        )
        if has_files:
            return p
    raise FileNotFoundError(f"No 08 '{profile}' run found under {base}")


def require_08_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "cluster_labels": run_dir / "cluster_labels.csv",
        "assignments": run_dir / "biz_cluster_assignments_labeled.csv",
        "meta": run_dir / "run_meta.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing 08 files:\n" + "\n".join(missing))
    return files


def pick_subcluster_k(n_businesses: int, dominant_share: float) -> int:
    k = 2
    if n_businesses >= 180:
        k = 3
    if n_businesses >= 320:
        k = 4
    if dominant_share < 0.22 and n_businesses >= 160:
        k = max(k, 4)
    elif dominant_share < 0.35 and n_businesses >= 120:
        k = max(k, 3)
    k = min(k, max(2, n_businesses // 45))
    return max(2, int(k))


def get_embedding_map(assign_df: pd.DataFrame, source_07_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    run_meta = pd.read_csv(source_07_dir / "run_meta.csv")
    cache_file = Path(str(run_meta.loc[0, "cache_file"]))
    model_used = str(run_meta.loc[0, "model_used"])

    source_input = source_07_dir / "biz_cluster_input.csv"
    if not source_input.exists():
        return {}, model_used
    if not cache_file.exists():
        return {}, model_used

    input_df = pd.read_csv(source_input, usecols=["business_id"])
    input_ids = input_df["business_id"].astype(str).tolist()
    npz = np.load(cache_file, allow_pickle=False)
    cached_ids = npz["business_id"].astype(str).tolist()
    if input_ids != cached_ids:
        return {}, model_used

    embs = np.asarray(npz["embeddings"], dtype=np.float32)
    emb_map = {bid: embs[i] for i, bid in enumerate(cached_ids)}
    # Keep only assignment ids to reduce memory.
    use_ids = set(assign_df["business_id"].astype(str).tolist())
    emb_map = {k: v for k, v in emb_map.items() if k in use_ids}
    return emb_map, model_used


def compute_missing_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    m07 = load_stage07_module()
    embs, _, _ = m07.compute_embeddings(texts, model_name)
    return np.asarray(embs, dtype=np.float32)


def main() -> None:
    profile = normalize_profile(RUN_PROFILE)
    src08 = pick_source_08_run_dir(profile)
    files = require_08_files(src08)

    labels_df = pd.read_csv(files["cluster_labels"])
    assign_df = pd.read_csv(files["assignments"])
    meta08 = pd.read_csv(files["meta"])
    source_07_dir = Path(str(meta08.loc[0, "source_07_dir"]))

    if "cluster" not in assign_df.columns:
        raise RuntimeError("biz_cluster_assignments_labeled.csv missing 'cluster' column.")
    if "text" not in assign_df.columns:
        meta07 = pd.read_csv(source_07_dir / "run_meta.csv")
        source_relabels_csv = Path(str(meta07.loc[0, "source_relabels_csv"]))
        if not source_relabels_csv.exists():
            raise RuntimeError(
                "biz_cluster_assignments_labeled.csv missing 'text' and source_relabels_csv not found."
            )
        relabel_text_df = pd.read_csv(
            source_relabels_csv, usecols=["business_id", "text"]
        ).copy()
        relabel_text_df["business_id"] = relabel_text_df["business_id"].astype(str)
        assign_df = assign_df.copy()
        assign_df["business_id"] = assign_df["business_id"].astype(str)
        assign_df = assign_df.merge(
            relabel_text_df, on="business_id", how="left", suffixes=("", "_src")
        )
        if "text" not in assign_df.columns and "text_src" in assign_df.columns:
            assign_df["text"] = assign_df["text_src"]
        elif "text_src" in assign_df.columns:
            assign_df["text"] = assign_df["text"].fillna(assign_df["text_src"])
        assign_df.drop(columns=["text_src"], errors="ignore", inplace=True)
        if "text" not in assign_df.columns:
            raise RuntimeError("Failed to recover text column for mixed refinement.")

    labels_df["dominant_l2_share_num"] = pd.to_numeric(
        labels_df.get("dominant_l2_share", pd.Series(np.nan, index=labels_df.index)),
        errors="coerce",
    ).fillna(0.0)
    labels_df["is_mixed_bool"] = (
        labels_df.get("is_mixed_cluster", pd.Series(False, index=labels_df.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )

    select_mask = labels_df["dominant_l2_share_num"] < MIXED_SHARE_THRESHOLD
    if USE_MIXED_FLAG and "is_mixed_bool" in labels_df.columns:
        select_mask = select_mask | labels_df["is_mixed_bool"]
    cand = labels_df.loc[select_mask].copy()
    cand = cand[pd.to_numeric(cand["n_businesses"], errors="coerce").fillna(0).astype(int) >= MIN_CLUSTER_SIZE]
    cand = cand.sort_values(["dominant_l2_share_num", "n_businesses"], ascending=[True, False]).reset_index(drop=True)
    if cand.empty:
        raise RuntimeError("No mixed clusters selected for refinement.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SOURCE_08_ROOT / profile / f"{run_id}_{profile}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand.to_csv(out_dir / "mixed_clusters_selected.csv", index=False)
    print(f"[CONFIG] source_08={src08}")
    print(f"[CONFIG] source_07={source_07_dir}")
    print(f"[COUNT] mixed_clusters_selected={len(cand)}")

    emb_map, model_name = get_embedding_map(assign_df, source_07_dir)
    print(f"[INFO] embedding_map_size={len(emb_map)} model={model_name}")

    assign_df["business_id"] = assign_df["business_id"].astype(str)
    all_assign_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    all_kw_rows: list[dict] = []

    for _, crow in cand.iterrows():
        parent_cluster = int(crow["cluster"])
        parent_df = assign_df[assign_df["cluster"] == parent_cluster].copy().reset_index(drop=True)
        if len(parent_df) < 2:
            continue

        dom_share = float(crow["dominant_l2_share_num"])
        sub_k = pick_subcluster_k(len(parent_df), dom_share)
        sub_k = min(sub_k, len(parent_df))
        if sub_k < 2:
            continue

        rows = parent_df.to_dict("records")
        x = []
        missing_texts = []
        missing_idx = []
        for i, r in enumerate(rows):
            bid = str(r["business_id"])
            emb = emb_map.get(bid)
            if emb is None:
                missing_idx.append(i)
                missing_texts.append(str(r.get("text", "")))
                x.append(None)
            else:
                x.append(emb)

        if missing_texts:
            miss_embs = compute_missing_embeddings(missing_texts, model_name)
            for j, i in enumerate(missing_idx):
                x[i] = miss_embs[j]

        X = np.vstack([np.asarray(v, dtype=np.float32) for v in x]).astype(np.float32)
        if USE_HYBRID_L2_FEATURES:
            m07 = load_stage07_module()
            l2_mat, _ = m07.build_l2_feature_matrix(parent_df)
            if int(l2_mat.shape[1]) > 0:
                X = np.hstack([X, np.asarray(l2_mat, dtype=np.float32) * float(HYBRID_L2_WEIGHT)]).astype(
                    np.float32
                )
        kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(X)
        parent_df["subcluster"] = sub_labels
        parent_df["parent_cluster"] = parent_cluster
        parent_df["subcluster_id"] = parent_df["parent_cluster"].astype(str) + "_" + parent_df["subcluster"].astype(str)

        # keyword extraction within parent cluster.
        vectorizer = TfidfVectorizer(
            max_features=int(TFIDF_MAX_FEATURES),
            min_df=int(TFIDF_MIN_DF),
            max_df=float(TFIDF_MAX_DF),
            ngram_range=tuple(TFIDF_NGRAM_RANGE),
            stop_words="english",
        )
        X_tfidf = vectorizer.fit_transform(parent_df["text"].fillna("").astype(str).tolist())
        terms = vectorizer.get_feature_names_out()

        for s in sorted(parent_df["subcluster"].unique().tolist()):
            sub_df = parent_df[parent_df["subcluster"] == s]
            idx = sub_df.index.to_numpy()
            mean_tfidf = X_tfidf[idx].mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:TOP_TERMS_PER_SUBCLUSTER]
            for rank, term_idx in enumerate(top_idx, start=1):
                score = float(mean_tfidf[term_idx])
                if score <= 0:
                    continue
                all_kw_rows.append(
                    {
                        "parent_cluster": parent_cluster,
                        "subcluster": int(s),
                        "subcluster_id": f"{parent_cluster}_{int(s)}",
                        "rank": rank,
                        "term": terms[term_idx],
                        "score": score,
                        "n_businesses": int(len(sub_df)),
                    }
                )

            vc = (
                sub_df["final_l2_label_top1"]
                .fillna("NA")
                .astype(str)
                .value_counts()
            )
            top_l2 = str(vc.index[0])
            top_l2_share = float(vc.iloc[0] / max(1, len(sub_df)))
            all_summary_rows.append(
                {
                    "parent_cluster": parent_cluster,
                    "subcluster": int(s),
                    "subcluster_id": f"{parent_cluster}_{int(s)}",
                    "n_businesses": int(len(sub_df)),
                    "top_l2": top_l2,
                    "top_l2_share": top_l2_share,
                    "avg_review_stars": float(pd.to_numeric(sub_df["avg_review_stars"], errors="coerce").mean()),
                    "avg_business_stars": float(pd.to_numeric(sub_df["stars"], errors="coerce").mean()),
                    "avg_n_reviews": float(pd.to_numeric(sub_df["n_reviews"], errors="coerce").mean()),
                }
            )

        all_assign_rows.extend(parent_df.to_dict("records"))
        print(
            f"[INFO] parent_cluster={parent_cluster} n={len(parent_df)} "
            f"dominant_share={dom_share:.3f} -> sub_k={sub_k}"
        )

    assign_out = pd.DataFrame(all_assign_rows)
    summary_out = pd.DataFrame(all_summary_rows).sort_values(
        ["parent_cluster", "subcluster"], ascending=[True, True]
    )
    kw_out = pd.DataFrame(all_kw_rows).sort_values(
        ["parent_cluster", "subcluster", "rank"], ascending=[True, True, True]
    )

    assign_out.to_csv(out_dir / "mixed_refine_assignments.csv", index=False)
    summary_out.to_csv(out_dir / "mixed_refine_summary.csv", index=False)
    kw_out.to_csv(out_dir / "mixed_refine_keywords.csv", index=False)

    run_meta = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_profile": profile,
                "source_08_dir": str(src08),
                "source_07_dir": str(source_07_dir),
                "mixed_share_threshold": float(MIXED_SHARE_THRESHOLD),
                "use_mixed_flag": bool(USE_MIXED_FLAG),
                "min_cluster_size": int(MIN_CLUSTER_SIZE),
                "n_mixed_clusters_selected": int(len(cand)),
                "n_rows_refined": int(len(assign_out)),
                "n_subclusters_total": int(len(summary_out)),
                "embedding_model": str(model_name),
                "output_dir": str(out_dir),
            }
        ]
    )
    run_meta.to_csv(out_dir / "run_meta.csv", index=False)

    print(f"[INFO] wrote {out_dir / 'mixed_clusters_selected.csv'}")
    print(f"[INFO] wrote {out_dir / 'mixed_refine_assignments.csv'}")
    print(f"[INFO] wrote {out_dir / 'mixed_refine_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'mixed_refine_keywords.csv'}")
    print(f"[INFO] wrote {out_dir / 'run_meta.csv'}")


if __name__ == "__main__":
    main()
