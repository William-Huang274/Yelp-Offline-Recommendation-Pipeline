from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


SOURCE_08_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels/full")
BASE_LABEL_RUN_DIR = ""  # optional explicit dir: ..._full_labels
TARGET_REFINE_RUN_DIR = ""  # optional explicit dir: ..._full_target_refine

OUTPUT_ROOT = SOURCE_08_ROOT
RUN_TAG = "profile_merged"

BASE_ASSIGNMENTS_NAME = "biz_cluster_assignments_labeled.csv"
BASE_LABELS_NAME = "cluster_labels.csv"
BASE_META_NAME = "run_meta.csv"

TARGET_ASSIGNMENTS_NAME = "target_refine_assignments.csv"
TARGET_SUMMARY_NAME = "target_refine_summary.csv"
TARGET_META_NAME = "run_meta.csv"


def pick_latest_run(root: Path, suffix: str, required_files: list[str]) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if all((run / f).exists() for f in required_files):
            return run
    raise FileNotFoundError(
        f"No run found under {root} with suffix={suffix} and files={required_files}"
    )


def resolve_base_run() -> Path:
    if BASE_LABEL_RUN_DIR.strip():
        p = Path(BASE_LABEL_RUN_DIR.strip())
        if not p.exists():
            raise FileNotFoundError(f"BASE_LABEL_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(
        SOURCE_08_ROOT, "_full_labels", [BASE_ASSIGNMENTS_NAME, BASE_LABELS_NAME, BASE_META_NAME]
    )


def resolve_target_refine_run() -> Optional[Path]:
    if TARGET_REFINE_RUN_DIR.strip():
        p = Path(TARGET_REFINE_RUN_DIR.strip())
        if not p.exists():
            raise FileNotFoundError(f"TARGET_REFINE_RUN_DIR not found: {p}")
        return p
    try:
        return pick_latest_run(
            SOURCE_08_ROOT,
            "_full_target_refine",
            [TARGET_ASSIGNMENTS_NAME, TARGET_SUMMARY_NAME, TARGET_META_NAME],
        )
    except FileNotFoundError:
        return None


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def titleize_l2(value: str) -> str:
    s = str(value).strip().replace("_", " ")
    if not s:
        return "Refined Segment"
    return " ".join(x.capitalize() for x in s.split())


def main() -> None:
    base_run = resolve_base_run()
    target_run = resolve_target_refine_run()

    base_assign = pd.read_csv(base_run / BASE_ASSIGNMENTS_NAME)
    base_labels = pd.read_csv(base_run / BASE_LABELS_NAME)
    base_meta = pd.read_csv(base_run / BASE_META_NAME)
    source_07_dir = Path(str(base_meta.loc[0, "source_07_dir"]))

    if "business_id" not in base_assign.columns:
        raise RuntimeError("Missing 'business_id' in base assignments.")
    if "cluster" not in base_assign.columns:
        raise RuntimeError("Missing 'cluster' in base assignments.")

    base_assign["base_cluster"] = pd.to_numeric(base_assign["cluster"], errors="coerce").fillna(-1).astype(int)
    if "final_label" not in base_assign.columns:
        label_map = base_labels[["cluster", "final_label"]].copy()
        base_assign = base_assign.merge(label_map, on="cluster", how="left")

    strict_input_csv = source_07_dir / "biz_cluster_input.csv"
    strict_ids: set[str] = set()
    if strict_input_csv.exists():
        strict_df = pd.read_csv(strict_input_csv, usecols=["business_id"])
        strict_ids = set(strict_df["business_id"].astype(str).tolist())

    relabel_csv = source_07_dir / "biz_relabels.csv"
    relabel_cols = [
        "business_id",
        "final_l1_label",
        "final_l2_label_top1",
        "final_l2_label_top2",
        "final_label_confidence",
        "final_label_source",
    ]
    if relabel_csv.exists():
        relabel_df = pd.read_csv(relabel_csv)
        keep_cols = [c for c in relabel_cols if c in relabel_df.columns]
        if keep_cols:
            base_assign = base_assign.merge(
                relabel_df[keep_cols].drop_duplicates("business_id"),
                on="business_id",
                how="left",
                suffixes=("", "_from_relabel"),
            )
            for c in relabel_cols:
                alt = f"{c}_from_relabel"
                if c in base_assign.columns and alt in base_assign.columns:
                    base_assign[c] = base_assign[c].fillna(base_assign[alt])
                    base_assign.drop(columns=[alt], inplace=True)
                elif alt in base_assign.columns:
                    base_assign.rename(columns={alt: c}, inplace=True)

    out = base_assign.copy()
    out["business_id"] = out["business_id"].astype(str)
    out["cluster_level"] = "base"
    out["cluster_for_recsys"] = out["base_cluster"].apply(lambda x: f"base_{x}")
    out["cluster_label_for_recsys"] = out.get("final_label", pd.Series("", index=out.index)).fillna("").astype(str)
    out["cluster_parent"] = out["base_cluster"].astype(str)
    out["in_cluster_strict_input"] = out["business_id"].isin(strict_ids)

    refined_count = 0
    if target_run is not None:
        t_assign = pd.read_csv(target_run / TARGET_ASSIGNMENTS_NAME)
        t_summary = pd.read_csv(target_run / TARGET_SUMMARY_NAME)

        if "business_id" in t_assign.columns and "refine2_cluster_id" in t_assign.columns:
            refine_map = t_assign[
                ["business_id", "refine_group", "parent_cluster", "subcluster_id", "refine2_cluster_id"]
            ].copy()
            refine_map["business_id"] = refine_map["business_id"].astype(str)
            refine_map["refine2_cluster_id"] = refine_map["refine2_cluster_id"].fillna("").astype(str)

            summary_map = {}
            if "refine2_cluster_id" in t_summary.columns:
                for _, r in t_summary.iterrows():
                    cid = str(r.get("refine2_cluster_id", "")).strip()
                    if not cid:
                        continue
                    top_l2 = titleize_l2(str(r.get("top_l2", "")))
                    summary_map[cid] = f"{top_l2} Segment ({cid})"

            out = out.merge(refine_map, on="business_id", how="left")
            has_refined = out["refine2_cluster_id"].fillna("").astype(str) != ""
            refined_count = int(has_refined.sum())

            out.loc[has_refined, "cluster_level"] = "refined"
            out.loc[has_refined, "cluster_for_recsys"] = out.loc[has_refined, "refine2_cluster_id"].astype(str)
            out.loc[has_refined, "cluster_parent"] = (
                out.loc[has_refined, "parent_cluster"].fillna(out.loc[has_refined, "base_cluster"]).astype(str)
            )
            out.loc[has_refined, "cluster_label_for_recsys"] = (
                out.loc[has_refined, "refine2_cluster_id"]
                .astype(str)
                .map(lambda x: summary_map.get(x, f"Refined Segment ({x})"))
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_full_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = out_dir / "biz_profile_merged.csv"
    recsys_csv = out_dir / "biz_profile_recsys.csv"
    meta_csv = out_dir / "run_meta.csv"

    # Keep full merged table for diagnostics.
    out.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    # Lean schema for recommendation side joins.
    recsys_cols = [
        "business_id",
        "name",
        "city",
        "categories",
        "final_l1_label",
        "final_l2_label_top1",
        "final_l2_label_top2",
        "final_label_confidence",
        "base_cluster",
        "cluster_parent",
        "cluster_level",
        "cluster_for_recsys",
        "cluster_label_for_recsys",
        "in_cluster_strict_input",
    ]
    recsys_cols = [c for c in recsys_cols if c in out.columns]
    out[recsys_cols].to_csv(recsys_csv, index=False, encoding="utf-8-sig")

    meta = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "source_08_full_labels_dir": str(base_run),
                "source_07_dir": str(source_07_dir),
                "source_target_refine_dir": str(target_run) if target_run is not None else "",
                "n_businesses": int(len(out)),
                "n_refined_businesses": refined_count,
                "n_cluster_for_recsys": int(out["cluster_for_recsys"].astype(str).nunique()),
                "output_dir": str(out_dir),
            }
        ]
    )
    meta.to_csv(meta_csv, index=False)

    print(f"[INFO] source_08={base_run}")
    print(f"[INFO] source_07={source_07_dir}")
    print(f"[INFO] source_target_refine={target_run if target_run is not None else 'none'}")
    print(f"[INFO] n_businesses={len(out)}, refined_businesses={refined_count}")
    print(f"[INFO] wrote {merged_csv}")
    print(f"[INFO] wrote {recsys_csv}")
    print(f"[INFO] wrote {meta_csv}")


if __name__ == "__main__":
    main()
