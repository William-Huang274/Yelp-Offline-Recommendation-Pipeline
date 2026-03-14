import json
import os
import shutil
from pathlib import Path
from typing import Any


SOURCE_RUN_DIR = Path(os.getenv("INPUT_11_RUN_DIR", "")).expanduser()
OUTPUT_ROOT = Path(
    os.getenv(
        "OUTPUT_11_REWEIGHT_ROOT_DIR",
        str(SOURCE_RUN_DIR.parent if SOURCE_RUN_DIR else Path.cwd()),
    )
).expanduser()
POS_CLASS_WEIGHT = float(os.getenv("QLORA_POS_CLASS_WEIGHT", "12.0").strip() or 12.0)
COPY_PARQUET = os.getenv("QLORA_REWEIGHT_COPY_PARQUET", "true").strip().lower() in {"1", "true", "yes", "y"}


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def iter_json_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def rewrite_json_dir(src_dir: Path, dst_dir: Path, pos_weight: float) -> dict[str, Any]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    row_count = 0
    pos_count = 0
    neg_count = 0
    weight_min = None
    weight_max = None
    json_files = sorted(src_dir.glob("part-*.json"))
    if not json_files:
        raise FileNotFoundError(f"No json part files found in {src_dir}")
    for src_file in json_files:
        dst_file = dst_dir / src_file.name
        with src_file.open("r", encoding="utf-8") as rf, dst_file.open("w", encoding="utf-8") as wf:
            for raw in rf:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                label = int(row.get("label", 0))
                base_w = float(row.get("sample_weight", 1.0))
                new_w = base_w * pos_weight if label == 1 else base_w
                row["sample_weight"] = float(new_w)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1
                weight_min = new_w if weight_min is None else min(weight_min, new_w)
                weight_max = new_w if weight_max is None else max(weight_max, new_w)
    return {
        "rows": row_count,
        "pos": pos_count,
        "neg": neg_count,
        "sample_weight_min": float(weight_min or 0.0),
        "sample_weight_max": float(weight_max or 0.0),
    }


def main() -> None:
    if not SOURCE_RUN_DIR:
        raise ValueError("INPUT_11_RUN_DIR is required")
    if not SOURCE_RUN_DIR.exists():
        raise FileNotFoundError(f"Missing source run dir: {SOURCE_RUN_DIR}")

    run_name = SOURCE_RUN_DIR.name + f"_posw{str(POS_CLASS_WEIGHT).replace('.', 'p')}"
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_train_src = SOURCE_RUN_DIR / "bucket_10" / "pairwise_pool_train_json"
    pair_eval_src = SOURCE_RUN_DIR / "bucket_10" / "pairwise_pool_eval_json"
    pair_parquet_src = SOURCE_RUN_DIR / "bucket_10" / "pairwise_pool_all_parquet"
    if not pair_train_src.exists() or not pair_eval_src.exists():
        raise FileNotFoundError("pairwise_pool_train_json/eval_json are required")

    # Copy everything first so the derived run remains structurally identical.
    for item in SOURCE_RUN_DIR.iterdir():
        target = output_dir / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)

    pair_train_dst = output_dir / "bucket_10" / "pairwise_pool_train_json"
    pair_eval_dst = output_dir / "bucket_10" / "pairwise_pool_eval_json"
    train_stats = rewrite_json_dir(pair_train_src, pair_train_dst, POS_CLASS_WEIGHT)
    eval_stats = rewrite_json_dir(pair_eval_src, pair_eval_dst, POS_CLASS_WEIGHT)

    if not COPY_PARQUET and pair_parquet_src.exists():
        shutil.rmtree(output_dir / "bucket_10" / "pairwise_pool_all_parquet", ignore_errors=True)

    meta_path = output_dir / "reweight_meta.json"
    meta = {
        "source_run_dir": str(SOURCE_RUN_DIR),
        "derived_run_dir": str(output_dir),
        "pos_class_weight": POS_CLASS_WEIGHT,
        "copy_parquet": COPY_PARQUET,
        "train_stats": train_stats,
        "eval_stats": eval_stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
