from pathlib import Path
import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("Usage: python scripts/stage01_to_stage08/05_freeze_recsys_results.py")
    print("Reads stage03/04 metrics and writes frozen comparison tables / figures.")
    print("Update the metrics path constants or env-backed paths, then run without --help.")
    sys.exit(0)

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_PATH = Path(r"D:/5006 BDA project/data/metrics/la_recsys_valid_test_results.csv")
OUTPUT_DIR = Path(r"D:/5006 BDA project/data/metrics")
OUTPUT_TABLE = OUTPUT_DIR / "recsys_results_table.csv"
OUTPUT_FIG = OUTPUT_DIR / "fig_ndcg_by_bucket.png"

BUCKETS = [2, 5, 10]
MODELS = ["Popular", "Category Popular", "Implicit ALS (best)"]
SPLITS = ["valid", "test"]
PLOT_SPLIT = "test"  # plot NDCG for this split
RUN_ID = ""  # empty -> use latest run_id in the results file


def pick_run_id(df: pd.DataFrame) -> str:
    if RUN_ID:
        return RUN_ID
    run_ids = df["run_id"].dropna().astype(str).unique().tolist()
    if not run_ids:
        return ""
    return sorted(run_ids)[-1]


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"[ERROR] results file not found: {RESULTS_PATH}")
        return

    df = pd.read_csv(RESULTS_PATH)
    if df.empty:
        print(f"[ERROR] results file is empty: {RESULTS_PATH}")
        return

    run_id = pick_run_id(df)
    if not run_id:
        print("[ERROR] no run_id available to select")
        return

    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"] == run_id]
    df = df[df["bucket_min_train_reviews"].isin(BUCKETS)]
    df = df[df["model"].isin(MODELS)]
    df = df[df["split"].isin(SPLITS)]
    df = df.sort_values(["bucket_min_train_reviews", "split", "model"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TABLE, index=False)
    print(f"[INFO] wrote {OUTPUT_TABLE}")

    plot_df = df[df["split"] == PLOT_SPLIT].copy()
    plot_df["ndcg_at_k"] = pd.to_numeric(plot_df["ndcg_at_k"], errors="coerce")
    plot_df = plot_df.dropna(subset=["ndcg_at_k"])
    if plot_df.empty:
        print(f"[WARN] no rows to plot for split={PLOT_SPLIT}")
        return

    pivot = plot_df.pivot_table(
        index="bucket_min_train_reviews",
        columns="model",
        values="ndcg_at_k",
        aggfunc="mean",
    ).reindex(BUCKETS)

    top_k = pd.to_numeric(plot_df["top_k"], errors="coerce").dropna()
    top_k_value = int(top_k.iloc[0]) if not top_k.empty else 10

    ax = pivot.plot(marker="o", figsize=(7, 4))
    ax.set_xlabel("min_train_reviews bucket")
    ax.set_ylabel(f"NDCG@{top_k_value}")
    ax.set_title(f"NDCG@{top_k_value} by bucket ({PLOT_SPLIT})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    print(f"[INFO] wrote {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
