from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from stage07_core import load_stage07_module


SOURCE_08_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels")
SOURCE_MIXED_REFINE_RUN_DIR = ""  # optional override
RUN_PROFILE = "full"  # "sample" | "full"

RUN_TAG = "target_refine"

# Target groups:
# 1) all records from parent cluster 9
# 2) only records from parent cluster 19, subcluster 2
TARGET_GROUPS = [
    {"group_id": "p9_all", "parent_cluster": 9, "subcluster": None, "k": 4},
    {"group_id": "p19_2", "parent_cluster": 19, "subcluster": 2, "k": 4},
]

TOP_TERMS_PER_CLUSTER = 12
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


def pick_source_mixed_refine_dir(profile: str) -> Path:
    if SOURCE_MIXED_REFINE_RUN_DIR.strip():
        p = Path(SOURCE_MIXED_REFINE_RUN_DIR.strip())
        if not p.exists():
            raise FileNotFoundError(f"SOURCE_MIXED_REFINE_RUN_DIR not found: {p}")
        return p
    base = SOURCE_08_ROOT / profile
    if not base.exists():
        raise FileNotFoundError(f"08 profile dir not found: {base}")
    runs = [p for p in base.iterdir() if p.is_dir() and "mixed_refine" in p.name.lower()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        if not run_name_matches_profile(p.name, profile):
            continue
        has_files = (
            (p / "mixed_refine_assignments.csv").exists()
            and (p / "mixed_refine_summary.csv").exists()
            and (p / "run_meta.csv").exists()
        )
        if has_files:
            return p
    raise FileNotFoundError(f"No mixed_refine run found under: {base}")


def require_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "assignments": run_dir / "mixed_refine_assignments.csv",
        "summary": run_dir / "mixed_refine_summary.csv",
        "meta": run_dir / "run_meta.csv",
    }
    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing mixed_refine files:\n" + "\n".join(missing))
    return files


def get_embedding_map(assign_df: pd.DataFrame, source_07_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    run_meta = pd.read_csv(source_07_dir / "run_meta.csv")
    cache_file = Path(str(run_meta.loc[0, "cache_file"]))
    model_used = str(run_meta.loc[0, "model_used"])

    source_input = source_07_dir / "biz_cluster_input.csv"
    if (not source_input.exists()) or (not cache_file.exists()):
        return {}, model_used

    input_df = pd.read_csv(source_input, usecols=["business_id"])
    input_ids = input_df["business_id"].astype(str).tolist()
    npz = np.load(cache_file, allow_pickle=False)
    cached_ids = npz["business_id"].astype(str).tolist()
    if input_ids != cached_ids:
        return {}, model_used

    embs = np.asarray(npz["embeddings"], dtype=np.float32)
    emb_map = {bid: embs[i] for i, bid in enumerate(cached_ids)}
    use_ids = set(assign_df["business_id"].astype(str).tolist())
    emb_map = {k: v for k, v in emb_map.items() if k in use_ids}
    return emb_map, model_used


def compute_missing_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    m07 = load_stage07_module()
    embs, _, _ = m07.compute_embeddings(texts, model_name)
    return np.asarray(embs, dtype=np.float32)


def filter_group(df: pd.DataFrame, parent_cluster: int, subcluster: int | None) -> pd.DataFrame:
    out = df[pd.to_numeric(df["parent_cluster"], errors="coerce").fillna(-1).astype(int) == int(parent_cluster)].copy()
    if subcluster is not None:
        out = out[pd.to_numeric(out["subcluster"], errors="coerce").fillna(-1).astype(int) == int(subcluster)].copy()
    return out.reset_index(drop=True)


def main() -> None:
    profile = normalize_profile(RUN_PROFILE)
    src_dir = pick_source_mixed_refine_dir(profile)
    files = require_files(src_dir)
    assign_df = pd.read_csv(files["assignments"])
    meta_df = pd.read_csv(files["meta"])
    source_07_dir = Path(str(meta_df.loc[0, "source_07_dir"]))

    if "text" not in assign_df.columns:
        raise RuntimeError("mixed_refine_assignments.csv missing text column.")

    assign_df["business_id"] = assign_df["business_id"].astype(str)
    emb_map, model_name = get_embedding_map(assign_df, source_07_dir)
    print(f"[CONFIG] source_mixed_refine={src_dir}")
    print(f"[CONFIG] source_07={source_07_dir}")
    print(f"[INFO] embedding_map_size={len(emb_map)} model={model_name}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SOURCE_08_ROOT / profile / f"{run_id}_{profile}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_assign_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    all_kw_rows: list[dict] = []
    selected_rows: list[dict] = []

    m07 = load_stage07_module()

    for spec in TARGET_GROUPS:
        gid = str(spec["group_id"])
        parent_cluster = int(spec["parent_cluster"])
        subcluster = spec.get("subcluster")
        k = int(spec.get("k", 2))

        gdf = filter_group(assign_df, parent_cluster=parent_cluster, subcluster=subcluster)
        if gdf.empty:
            print(f"[WARN] skip {gid}: no rows")
            continue
        n = len(gdf)
        k = min(max(2, k), n)
        if k < 2:
            print(f"[WARN] skip {gid}: n={n} too small")
            continue

        selected_rows.extend(gdf.to_dict("records"))

        rows = gdf.to_dict("records")
        x = []
        missing_idx = []
        missing_texts = []
        for i, r in enumerate(rows):
            bid = str(r["business_id"])
            emb = emb_map.get(bid)
            if emb is None:
                x.append(None)
                missing_idx.append(i)
                missing_texts.append(str(r.get("text", "")))
            else:
                x.append(emb)
        if missing_texts:
            miss_emb = compute_missing_embeddings(missing_texts, model_name)
            for j, i in enumerate(missing_idx):
                x[i] = miss_emb[j]

        X = np.vstack([np.asarray(v, dtype=np.float32) for v in x]).astype(np.float32)
        if USE_HYBRID_L2_FEATURES:
            l2_mat, _ = m07.build_l2_feature_matrix(gdf)
            if int(l2_mat.shape[1]) > 0:
                X = np.hstack([X, np.asarray(l2_mat, dtype=np.float32) * float(HYBRID_L2_WEIGHT)]).astype(np.float32)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        gdf["refine_group"] = gid
        gdf["refine2_cluster"] = labels
        gdf["refine2_cluster_id"] = gdf["refine_group"].astype(str) + "_" + gdf["refine2_cluster"].astype(str)

        vectorizer = TfidfVectorizer(
            max_features=int(TFIDF_MAX_FEATURES),
            min_df=int(TFIDF_MIN_DF),
            max_df=float(TFIDF_MAX_DF),
            ngram_range=tuple(TFIDF_NGRAM_RANGE),
            stop_words="english",
        )
        X_tfidf = vectorizer.fit_transform(gdf["text"].fillna("").astype(str).tolist())
        terms = vectorizer.get_feature_names_out()

        for s in sorted(gdf["refine2_cluster"].unique().tolist()):
            sdf = gdf[gdf["refine2_cluster"] == s]
            idx = sdf.index.to_numpy()
            mean_tfidf = X_tfidf[idx].mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
            for rank, term_idx in enumerate(top_idx, start=1):
                score = float(mean_tfidf[term_idx])
                if score <= 0:
                    continue
                all_kw_rows.append(
                    {
                        "refine_group": gid,
                        "parent_cluster": parent_cluster,
                        "parent_subcluster": subcluster if subcluster is not None else "",
                        "refine2_cluster": int(s),
                        "refine2_cluster_id": f"{gid}_{int(s)}",
                        "rank": rank,
                        "term": terms[term_idx],
                        "score": score,
                        "n_businesses": int(len(sdf)),
                    }
                )

            vc = sdf["final_l2_label_top1"].fillna("NA").astype(str).value_counts()
            top_l2 = str(vc.index[0])
            top_share = float(vc.iloc[0] / max(1, len(sdf)))
            all_summary_rows.append(
                {
                    "refine_group": gid,
                    "parent_cluster": parent_cluster,
                    "parent_subcluster": subcluster if subcluster is not None else "",
                    "refine2_cluster": int(s),
                    "refine2_cluster_id": f"{gid}_{int(s)}",
                    "n_businesses": int(len(sdf)),
                    "top_l2": top_l2,
                    "top_l2_share": top_share,
                    "avg_review_stars": float(pd.to_numeric(sdf["avg_review_stars"], errors="coerce").mean()),
                    "avg_business_stars": float(pd.to_numeric(sdf["stars"], errors="coerce").mean()),
                    "avg_n_reviews": float(pd.to_numeric(sdf["n_reviews"], errors="coerce").mean()),
                }
            )

        all_assign_rows.extend(gdf.to_dict("records"))
        print(f"[INFO] refine_group={gid} n={n} k={k}")

    selected_df = pd.DataFrame(selected_rows)
    assign_out = pd.DataFrame(all_assign_rows)
    summary_out = pd.DataFrame(all_summary_rows).sort_values(
        ["refine_group", "refine2_cluster"], ascending=[True, True]
    )
    kw_out = pd.DataFrame(all_kw_rows).sort_values(
        ["refine_group", "refine2_cluster", "rank"], ascending=[True, True, True]
    )

    selected_df.to_csv(out_dir / "target_selected_rows.csv", index=False)
    assign_out.to_csv(out_dir / "target_refine_assignments.csv", index=False)
    summary_out.to_csv(out_dir / "target_refine_summary.csv", index=False)
    kw_out.to_csv(out_dir / "target_refine_keywords.csv", index=False)

    run_meta = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_profile": profile,
                "source_mixed_refine_dir": str(src_dir),
                "source_07_dir": str(source_07_dir),
                "n_target_groups": int(len(TARGET_GROUPS)),
                "n_selected_rows": int(len(selected_df)),
                "n_refine2_rows": int(len(assign_out)),
                "n_refine2_clusters": int(len(summary_out)),
                "embedding_model": str(model_name),
                "use_hybrid_l2_features": bool(USE_HYBRID_L2_FEATURES),
                "hybrid_l2_weight": float(HYBRID_L2_WEIGHT),
                "output_dir": str(out_dir),
            }
        ]
    )
    run_meta.to_csv(out_dir / "run_meta.csv", index=False)

    print(f"[INFO] wrote {out_dir / 'target_selected_rows.csv'}")
    print(f"[INFO] wrote {out_dir / 'target_refine_assignments.csv'}")
    print(f"[INFO] wrote {out_dir / 'target_refine_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'target_refine_keywords.csv'}")
    print(f"[INFO] wrote {out_dir / 'run_meta.csv'}")


if __name__ == "__main__":
    main()
