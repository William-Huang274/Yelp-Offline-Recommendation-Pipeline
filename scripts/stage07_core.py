import importlib.util
import re
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


STAGE07_ENTRY_FILE = "07_relabel_then_cluster.py"


def load_stage07_module() -> ModuleType:
    script_path = Path(__file__).resolve().with_name(STAGE07_ENTRY_FILE)
    spec = importlib.util.spec_from_file_location("stage07_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stage07 module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_stage07(
    m07: ModuleType,
    run_profile_override: str = "",
    use_bge_m3_override: str = "",
    relabel_strategy_override: str = "",
    relabel_use_embed_recall_override: str = "",
    relabel_use_llm_override: str = "",
    ollama_model_override: str = "",
) -> None:
    if run_profile_override.strip():
        m07.RUN_PROFILE = run_profile_override.strip().lower()
    if use_bge_m3_override.strip():
        m07.USE_BGE_M3 = use_bge_m3_override.strip().lower() in {"1", "true", "yes", "y"}
    if relabel_strategy_override.strip():
        m07.RELABEL_EXPERIMENT_STRATEGY = relabel_strategy_override.strip().lower()
    if relabel_use_embed_recall_override.strip():
        m07.RELABEL_USE_EMBED_RECALL = (
            relabel_use_embed_recall_override.strip().lower() in {"1", "true", "yes", "y"}
        )
    if relabel_use_llm_override.strip():
        m07.RELABEL_USE_LLM = relabel_use_llm_override.strip().lower() in {"1", "true", "yes", "y"}
    if ollama_model_override.strip():
        m07.OLLAMA_MODEL = ollama_model_override.strip()


def run_relabel_only(
    run_profile_override: str = "",
    run_tag_suffix: str = "relabel_only",
    use_bge_m3_override: str = "",
    relabel_strategy_override: str = "",
    relabel_use_embed_recall_override: str = "",
    relabel_use_llm_override: str = "",
    ollama_model_override: str = "",
) -> None:
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    m07 = load_stage07_module()
    configure_stage07(
        m07,
        run_profile_override=run_profile_override,
        use_bge_m3_override=use_bge_m3_override,
        relabel_strategy_override=relabel_strategy_override,
        relabel_use_embed_recall_override=relabel_use_embed_recall_override,
        relabel_use_llm_override=relabel_use_llm_override,
        ollama_model_override=ollama_model_override,
    )

    orig_tag = m07.RUN_TAG
    m07.RUN_TAG = f"{orig_tag}_{run_tag_suffix}" if orig_tag else run_tag_suffix

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = m07.resolve_profile()
    paths = m07.build_output_paths(run_id, cfg["profile"])
    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] use_bge_m3={m07.USE_BGE_M3}")
    print(f"[CONFIG] ollama_model={getattr(m07, 'OLLAMA_MODEL', '')}")
    print(f"[CONFIG] relabel_use_llm={getattr(m07, 'RELABEL_USE_LLM', False)}")
    print(f"[CONFIG] relabel_use_embed_recall={getattr(m07, 'RELABEL_USE_EMBED_RECALL', False)}")
    if hasattr(m07, "get_relabel_embed_strategy"):
        print(f"[CONFIG] relabel_embed_strategy={m07.get_relabel_embed_strategy()}")
    print(f"[CONFIG] output_dir={paths['run_dir']}")

    spark = m07.build_spark()
    m07.set_log_level(spark)

    print("[STEP] load + filter businesses")
    biz = m07.load_business_filtered(spark).persist(StorageLevel.DISK_ONLY)
    if biz.limit(1).count() == 0:
        print("[ERROR] no businesses after filtering")
        spark.stop()
        return

    print("[STEP] load + filter reviews")
    reviews = m07.load_reviews(spark, biz, cfg["sample_fraction"]).persist(StorageLevel.DISK_ONLY)
    if reviews.limit(1).count() == 0:
        print("[ERROR] no reviews after filtering")
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] build business texts")
    biz_text = m07.build_business_texts(
        reviews,
        cfg["min_reviews_per_business"],
        cfg["max_reviews_per_business"],
    ).persist(StorageLevel.DISK_ONLY)
    if biz_text.limit(1).count() == 0:
        print("[ERROR] no businesses after review aggregation")
        reviews.unpersist()
        biz.unpersist()
        spark.stop()
        return

    biz_joined = biz_text.join(biz, on="business_id", how="left")
    if cfg["max_businesses"] and cfg["max_businesses"] > 0:
        biz_joined = (
            biz_joined.orderBy(F.desc("n_reviews"), F.desc("review_count"))
            .limit(cfg["max_businesses"])
        )

    pdf = biz_joined.select(
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
        "avg_review_stars",
        "n_reviews",
        "text",
    ).toPandas()
    print(f"[COUNT] businesses_before_relabel={len(pdf)}")
    if len(pdf) == 0:
        print("[ERROR] no businesses collected")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] relabel businesses (rule + llm)")
    relabeled_pdf, relabel_stats = m07.relabel_businesses(
        pdf,
        profile=cfg["profile"],
        progress_csv_path=paths["relabels_inprogress_csv"],
        cache_dir=paths["cache_dir"],
    )
    relabeled_pdf = m07.add_final_label_alias_columns(relabeled_pdf)
    relabeled_pdf.to_csv(paths["relabels_csv"], index=False)

    labels_final_df = relabeled_pdf
    labels_review_df = relabeled_pdf.iloc[0:0].copy()
    labels_audit_df = relabeled_pdf
    if m07.WRITE_LAYERED_LABEL_OUTPUTS:
        labels_final_df, labels_review_df, labels_audit_df = m07.build_layered_label_outputs(relabeled_pdf)
        labels_final_df.to_csv(paths["labels_final_csv"], index=False)
        labels_review_df.to_csv(paths["labels_review_csv"], index=False)
        labels_audit_df.to_csv(paths["labels_audit_csv"], index=False)

    label_stats = (
        relabeled_pdf.groupby(["l1_label", "label_source"], dropna=False)
        .size()
        .reset_index(name="n_businesses")
        .sort_values("n_businesses", ascending=False)
    )
    label_stats.to_csv(paths["label_stats_csv"], index=False)

    relabel_n_total = int(len(relabeled_pdf))
    relabel_uncertain_rate = float(
        (relabeled_pdf["l1_label"].fillna("").astype(str).str.lower() == "uncertain").mean()
    ) if relabel_n_total > 0 else 0.0
    relabel_review_rate = float(int(len(labels_review_df)) / max(1, relabel_n_total))
    llm_action_series = labels_audit_df.get("llm_action", pd.Series("NOT_CALLED", index=labels_audit_df.index))
    llm_called_mask = llm_action_series.fillna("").astype(str).str.upper() != "NOT_CALLED"
    llm_called_n = int(llm_called_mask.sum())
    llm_from_series = labels_audit_df.get("from_label", pd.Series("", index=labels_audit_df.index)).fillna("").astype(str)
    llm_to_series = labels_audit_df.get("to_label", pd.Series("", index=labels_audit_df.index)).fillna("").astype(str)
    llm_nochange_n = int((llm_called_mask & (llm_from_series == llm_to_series)).sum())

    run_meta = {
        "run_id": run_id,
        "profile": cfg["profile"],
        "stage_mode": "relabel_only",
        "n_businesses_before_relabel": int(len(relabeled_pdf)),
        "relabel_use_llm": bool(m07.RELABEL_USE_LLM),
        "relabel_embed_recall_enabled": bool(getattr(m07, "RELABEL_USE_EMBED_RECALL", False)),
        "relabel_embed_strategy": (
            m07.get_relabel_embed_strategy()
            if hasattr(m07, "get_relabel_embed_strategy")
            else str(getattr(m07, "RELABEL_EXPERIMENT_STRATEGY", "selective")).lower()
        ),
        "relabel_embed_candidate_pool": int(relabel_stats.get("n_embed_candidate_pool", 0)),
        "relabel_embed_called": int(relabel_stats.get("n_embed_called", 0)),
        "relabel_embed_applied": int(relabel_stats.get("n_embed_applied", 0)),
        "relabel_embed_llm_skipped": int(relabel_stats.get("n_embed_llm_skipped", 0)),
        "relabel_embed_forced_llm": int(relabel_stats.get("n_embed_forced_llm", 0)),
        "relabel_embed_cache_miss": int(relabel_stats.get("n_embed_cache_miss", 0)),
        "relabel_llm_called": int(relabel_stats.get("n_llm_called", 0)),
        "relabel_llm_validated": int(relabel_stats.get("n_llm_validated", 0)),
        "relabel_llm_applied": int(relabel_stats.get("n_llm_applied", 0)),
        "relabel_llm_action_keep": int(relabel_stats.get("n_llm_action_keep", 0)),
        "relabel_llm_action_modify": int(relabel_stats.get("n_llm_action_modify", 0)),
        "relabel_llm_action_add": int(relabel_stats.get("n_llm_action_add", 0)),
        "relabel_llm_action_reject": int(relabel_stats.get("n_llm_action_reject", 0)),
        "relabel_llm_parse_fail": int(relabel_stats.get("n_llm_parse_fail", 0)),
        "relabel_llm_schema_fail": int(relabel_stats.get("n_llm_schema_fail", 0)),
        "relabel_llm_http_error": int(relabel_stats.get("n_llm_http_error", 0)),
        "relabel_llm_parallel_error": int(relabel_stats.get("n_llm_parallel_error", 0)),
        "relabel_llm_fallback_keep": int(relabel_stats.get("n_llm_fallback_keep", 0)),
        "relabel_llm_valid_rate": float(
            int(relabel_stats.get("n_llm_validated", 0)) / max(1, int(relabel_stats.get("n_llm_called", 0)))
        ),
        "relabel_llm_nochange": llm_nochange_n,
        "relabel_llm_nochange_rate": float(llm_nochange_n / max(1, llm_called_n)),
        "relabel_review_queue": int(len(labels_review_df)),
        "relabel_review_rate": relabel_review_rate,
        "relabel_uncertain_rate": relabel_uncertain_rate,
        "cluster_stage_skipped": True,
    }
    m07.pd.DataFrame([run_meta]).to_csv(paths["run_meta_csv"], index=False)

    print(f"[INFO] wrote {paths['relabels_csv']}")
    if m07.RELABEL_WRITE_IN_PROGRESS:
        print(f"[INFO] in-progress relabel snapshot path: {paths['relabels_inprogress_csv']}")
    if m07.WRITE_LAYERED_LABEL_OUTPUTS:
        print(f"[INFO] wrote {paths['labels_final_csv']}")
        print(f"[INFO] wrote {paths['labels_review_csv']}")
        print(f"[INFO] wrote {paths['labels_audit_csv']}")
    print(f"[INFO] wrote {paths['label_stats_csv']}")
    print(f"[INFO] wrote {paths['run_meta_csv']}")

    reviews.unpersist()
    biz_text.unpersist()
    biz.unpersist()
    spark.stop()
    print(f"[INFO] run_id={run_id}")


def _run_name_matches_profile(run_name: str, profile: str) -> bool:
    return re.search(rf"(^|_){re.escape(profile)}($|_)", run_name.lower()) is not None


def pick_source_relabel_run_dir(output_root: Path, profile: str, source_relabel_run_dir: str = "") -> Path:
    if source_relabel_run_dir.strip():
        p = Path(source_relabel_run_dir.strip())
        if not p.exists():
            raise FileNotFoundError(f"SOURCE_RELABEL_RUN_DIR not found: {p}")
        return p
    runs = [p for p in output_root.iterdir() if p.is_dir() and p.name != "_cache"]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        if not _run_name_matches_profile(p.name, profile):
            continue
        if (p / "biz_relabels.csv").exists():
            return p
    raise FileNotFoundError(f"No relabel run found under {output_root} for profile={profile}")


def run_cluster_only(
    run_profile_override: str = "",
    source_relabel_run_dir: str = "",
    run_tag_suffix: str = "cluster_only",
    cluster_k_override: int = 0,
) -> None:
    m07 = load_stage07_module()
    configure_stage07(m07, run_profile_override=run_profile_override)

    cfg = m07.resolve_profile()
    if int(cluster_k_override) > 0:
        cfg["cluster_k"] = int(cluster_k_override)
        print(f"[CONFIG] cluster_k_override={int(cluster_k_override)}")
    source_dir = pick_source_relabel_run_dir(
        m07.OUTPUT_ROOT, cfg["profile"], source_relabel_run_dir=source_relabel_run_dir
    )
    source_relabels_csv = source_dir / "biz_relabels.csv"
    if not source_relabels_csv.exists():
        raise FileNotFoundError(f"Missing source relabel csv: {source_relabels_csv}")

    relabeled_pdf = pd.read_csv(source_relabels_csv)
    relabeled_pdf = m07.add_final_label_alias_columns(relabeled_pdf)
    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] source_relabels_csv={source_relabels_csv}")
    print(f"[COUNT] businesses_from_relabels={len(relabeled_pdf)}")
    if relabeled_pdf.empty:
        raise RuntimeError("Source relabel csv is empty.")

    orig_tag = m07.RUN_TAG
    m07.RUN_TAG = f"{orig_tag}_{run_tag_suffix}" if orig_tag else run_tag_suffix
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = m07.build_output_paths(run_id, cfg["profile"])
    model_name, model_source = m07.choose_model_name()
    print(f"[CONFIG] model={model_name}, model_source={model_source}")
    print(f"[CONFIG] output_dir={paths['run_dir']}")

    if hasattr(m07, "build_cluster_input"):
        cluster_pdf, cluster_input_stats = m07.build_cluster_input(relabeled_pdf)
    else:
        cluster_pdf = relabeled_pdf.copy()
        if m07.KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER:
            cluster_pdf = cluster_pdf[cluster_pdf["l1_label"] == "food_service"].copy()
        cluster_pdf = cluster_pdf.reset_index(drop=True)
        cluster_input_stats = {
            "cluster_input_mode": "food_service",
            "n_source": int(len(relabeled_pdf)),
            "n_after_food_service_filter": int(len(cluster_pdf)),
            "n_after_conf_filter": int(len(cluster_pdf)),
            "n_after_general_filter": int(len(cluster_pdf)),
            "n_after_review_filter": int(len(cluster_pdf)),
        }
    print(
        "[CONFIG] cluster_input_mode="
        f"{cluster_input_stats.get('cluster_input_mode', 'strict')}"
        f", keep_only_food_service={m07.KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER}"
    )
    print(
        "[COUNT] cluster input:"
        f" source={cluster_input_stats.get('n_source', 0)}"
        f" -> food_service={cluster_input_stats.get('n_after_food_service_filter', 0)}"
        f" -> conf={cluster_input_stats.get('n_after_conf_filter', 0)}"
        f" -> non_general={cluster_input_stats.get('n_after_general_filter', 0)}"
        f" -> no_review={cluster_input_stats.get('n_after_review_filter', 0)}"
    )
    print(f"[COUNT] businesses_for_cluster={len(cluster_pdf)}")
    if len(cluster_pdf) < 2:
        raise RuntimeError("Not enough businesses for clustering.")
    cluster_pdf.drop(columns=["text"], errors="ignore").to_csv(paths["cluster_input_csv"], index=False)

    cache_file = m07.make_cache_file(paths["cache_dir"], model_name, cfg)
    business_ids = cluster_pdf["business_id"].astype(str).tolist()
    embeddings = None
    actual_model_name = model_name
    actual_batch_size = m07.MANUAL_BATCH_SIZE
    if m07.REUSE_EMBEDDINGS:
        embeddings = m07.try_load_cached_embeddings(cache_file, business_ids)

    if embeddings is None:
        print("[STEP] compute embeddings")
        embeddings, actual_model_name, actual_batch_size = m07.compute_embeddings(
            cluster_pdf["text"].astype(str).tolist(), model_name
        )
        m07.save_cache_embeddings(cache_file, business_ids, embeddings)
        print(f"[INFO] saved embeddings cache to {cache_file}")

    print("[STEP] kmeans clustering")
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_k = min(int(cfg["cluster_k"]), len(cluster_pdf))
    if cluster_k < 2:
        raise RuntimeError("Not enough businesses for clustering.")

    cluster_features = np.asarray(embeddings, dtype=np.float32)
    hybrid_l2_feature_dim = 0
    if m07.USE_HYBRID_CLUSTER_FEATURES:
        l2_mat, l2_feature_labels = m07.build_l2_feature_matrix(cluster_pdf)
        hybrid_l2_feature_dim = int(l2_mat.shape[1])
        if hybrid_l2_feature_dim > 0:
            cluster_features = np.hstack([cluster_features, l2_mat]).astype(np.float32)
            print(
                "[INFO] hybrid cluster features enabled: "
                f"l2_dim={hybrid_l2_feature_dim}, l2_weight={m07.HYBRID_L2_FEATURE_WEIGHT}, "
                f"l2_top2={m07.HYBRID_USE_L2_TOP2}"
            )
            print(f"[INFO] hybrid l2 labels: {', '.join(l2_feature_labels)}")
        else:
            print("[INFO] hybrid cluster features skipped: no usable l2 labels")

    kmeans = KMeans(n_clusters=cluster_k, random_state=m07.RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(cluster_features)
    cluster_pdf["cluster"] = labels

    print("[STEP] tf-idf keywords per cluster")
    vectorizer = TfidfVectorizer(
        max_features=int(cfg["tfidf_max_features"]),
        min_df=int(cfg["tfidf_min_df"]),
        max_df=float(cfg["tfidf_max_df"]),
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        stop_words="english",
    )
    X = vectorizer.fit_transform(cluster_pdf["text"].astype(str).tolist())
    terms = vectorizer.get_feature_names_out()

    keyword_rows = []
    summary_rows = []
    example_rows = []
    top_terms = int(cfg["top_terms_per_cluster"])
    for c in range(cluster_k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        cluster_df = cluster_pdf.iloc[idx]
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_terms]

        for rank, term_idx in enumerate(top_idx, start=1):
            score = float(mean_tfidf[term_idx])
            if score <= 0:
                continue
            keyword_rows.append(
                {
                    "cluster": c,
                    "rank": rank,
                    "term": terms[term_idx],
                    "score": score,
                    "n_businesses": int(idx.size),
                }
            )

        summary_rows.append(
            {
                "cluster": c,
                "n_businesses": int(idx.size),
                "avg_review_stars": float(cluster_df["avg_review_stars"].mean()),
                "avg_business_stars": float(cluster_df["stars"].mean()),
                "avg_n_reviews": float(cluster_df["n_reviews"].mean()),
            }
        )

        examples = cluster_df.sort_values("n_reviews", ascending=False).head(5)
        for _, row in examples.iterrows():
            example_rows.append(
                {
                    "cluster": c,
                    "business_id": row["business_id"],
                    "name": row["name"],
                    "city": row["city"],
                    "categories": row["categories"],
                    "l1_label": row["l1_label"],
                    "l2_label_top1": row["l2_label_top1"],
                    "n_reviews": int(row["n_reviews"]),
                    "avg_review_stars": float(row["avg_review_stars"]),
                }
            )

    cluster_pdf.drop(columns=["text"]).to_csv(paths["assignments_csv"], index=False)
    pd.DataFrame(keyword_rows).to_csv(paths["cluster_keywords_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["cluster_summary_csv"], index=False)
    pd.DataFrame(example_rows).to_csv(paths["cluster_examples_csv"], index=False)

    run_meta = {
        "run_id": run_id,
        "profile": cfg["profile"],
        "stage_mode": "cluster_only",
        "source_relabel_run_dir": str(source_dir),
        "source_relabels_csv": str(source_relabels_csv),
        "model_requested": model_name,
        "model_source_requested": model_source,
        "model_used": actual_model_name,
        "batch_size_used": actual_batch_size,
        "cluster_k_used": int(cluster_k),
        "n_businesses_before_relabel": int(len(relabeled_pdf)),
        "n_businesses_after_relabel_filter": int(len(cluster_pdf)),
        "keep_only_food_service_for_cluster": bool(m07.KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER),
        "cluster_input_mode": str(cluster_input_stats.get("cluster_input_mode", "strict")),
        "cluster_strict_min_confidence": float(getattr(m07, "CLUSTER_STRICT_MIN_CONFIDENCE", 0.0)),
        "cluster_strict_exclude_review_queue": bool(
            getattr(m07, "CLUSTER_STRICT_EXCLUDE_REVIEW_QUEUE", False)
        ),
        "cluster_strict_exclude_general": bool(getattr(m07, "CLUSTER_STRICT_EXCLUDE_GENERAL", False)),
        "cluster_input_source_n": int(cluster_input_stats.get("n_source", len(relabeled_pdf))),
        "cluster_input_after_food_service_n": int(
            cluster_input_stats.get("n_after_food_service_filter", len(cluster_pdf))
        ),
        "cluster_input_after_conf_n": int(cluster_input_stats.get("n_after_conf_filter", len(cluster_pdf))),
        "cluster_input_after_general_n": int(
            cluster_input_stats.get("n_after_general_filter", len(cluster_pdf))
        ),
        "cluster_input_after_review_n": int(cluster_input_stats.get("n_after_review_filter", len(cluster_pdf))),
        "use_hybrid_cluster_features": bool(m07.USE_HYBRID_CLUSTER_FEATURES),
        "hybrid_l2_feature_weight": float(m07.HYBRID_L2_FEATURE_WEIGHT),
        "hybrid_l2_feature_dim": int(hybrid_l2_feature_dim),
        "hybrid_use_l2_top2": bool(m07.HYBRID_USE_L2_TOP2),
        "hybrid_ignore_general_l2": bool(m07.HYBRID_IGNORE_GENERAL_L2),
        "cache_file": str(cache_file),
    }
    pd.DataFrame([run_meta]).to_csv(paths["run_meta_csv"], index=False)

    print(f"[INFO] wrote {paths['cluster_input_csv']}")
    print(f"[INFO] wrote {paths['assignments_csv']}")
    print(f"[INFO] wrote {paths['cluster_keywords_csv']}")
    print(f"[INFO] wrote {paths['cluster_summary_csv']}")
    print(f"[INFO] wrote {paths['cluster_examples_csv']}")
    print(f"[INFO] wrote {paths['run_meta_csv']}")
