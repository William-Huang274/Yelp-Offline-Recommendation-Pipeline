from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_run(input_run_dir: str, input_root: Path, input_suffix: str) -> Path:
    raw = str(input_run_dir or "").strip()
    if raw:
        run_dir = Path(raw)
        if not run_dir.exists():
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        return run_dir
    return pick_latest_run(input_root, input_suffix)


def load_bucket_scores(run_dir: Path, buckets: list[int]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    files: list[str] = []
    for bucket in buckets:
        csv_path = run_dir / f"bucket_{bucket}_scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"bucket score csv missing: {csv_path}")
        pdf = pd.read_csv(csv_path)
        pdf["bucket"] = int(bucket)
        frames.append(pdf)
        files.append(csv_path.as_posix())
    if not frames:
        raise RuntimeError(f"no bucket score csv loaded from {run_dir}")
    return pd.concat(frames, ignore_index=True), files


def descending_rank_pct(values: pd.Series) -> pd.Series:
    arr = pd.Series(values).astype(float)
    if arr.empty:
        return arr.astype(np.float32)
    if len(arr) == 1:
        return pd.Series([1.0], index=arr.index, dtype=np.float32)
    ranks = arr.rank(method="average", ascending=False)
    denom = max(1.0, float(len(arr) - 1))
    pct = 1.0 - ((ranks - 1.0) / denom)
    return pct.astype(np.float32)


def add_rank_features(pdf: pd.DataFrame, raw_score_col: str, calibrated_col: str | None = None) -> pd.DataFrame:
    out = pdf.copy()
    out["pre_norm"] = out.groupby("user_idx", group_keys=False)["pre_score"].transform(descending_rank_pct)
    out["rm_rank_pct"] = out.groupby("user_idx", group_keys=False)[raw_score_col].transform(descending_rank_pct)
    if calibrated_col:
        out["rm_calibrated_rank_pct"] = out.groupby("user_idx", group_keys=False)[calibrated_col].transform(descending_rank_pct)
    out["pre_rank_inv"] = (1.0 / (1.0 + out["pre_rank"].astype(float))).astype(np.float32)
    out["pre_rank_top10"] = (out["pre_rank"].astype(int) <= 10).astype(np.float32)
    out["pre_rank_top25"] = (out["pre_rank"].astype(int) <= 25).astype(np.float32)
    out["pre_rank_11_25"] = ((out["pre_rank"].astype(int) >= 11) & (out["pre_rank"].astype(int) <= 25)).astype(np.float32)
    return out


def fit_isotonic_calibrator(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(scores, labels)
    return {
        "method": "isotonic",
        "x_thresholds": [float(x) for x in ir.X_thresholds_],
        "y_thresholds": [float(y) for y in ir.y_thresholds_],
    }


def fit_platt_calibrator(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(scores.reshape(-1, 1), labels.astype(int))
    return {
        "method": "platt",
        "coef": [float(x) for x in clf.coef_[0].tolist()],
        "intercept": float(clf.intercept_[0]),
    }


def apply_calibrator(scores: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    method = str(payload.get("method", "")).strip().lower()
    xs = np.asarray(scores, dtype=np.float32)
    if method == "isotonic":
        x_thr = np.asarray(payload.get("x_thresholds", []), dtype=np.float32)
        y_thr = np.asarray(payload.get("y_thresholds", []), dtype=np.float32)
        if len(x_thr) == 0 or len(y_thr) == 0:
            raise RuntimeError("invalid isotonic calibrator payload")
        return np.interp(xs, x_thr, y_thr).astype(np.float32)
    if method == "platt":
        coef = float((payload.get("coef", [0.0]) or [0.0])[0])
        intercept = float(payload.get("intercept", 0.0))
        logits = xs * coef + intercept
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    raise RuntimeError(f"unsupported calibrator method: {method}")


def build_stack_feature_frame(pdf: pd.DataFrame, rm_feature_col: str) -> tuple[pd.DataFrame, list[str]]:
    feature_df = pd.DataFrame(
        {
            "pre_norm": pdf["pre_norm"].astype(np.float32),
            "rm_feature": pdf[rm_feature_col].astype(np.float32),
            "rm_rank_pct": pdf["rm_rank_pct"].astype(np.float32),
            "pre_rank_inv": pdf["pre_rank_inv"].astype(np.float32),
            "pre_rank_top10": pdf["pre_rank_top10"].astype(np.float32),
            "pre_rank_top25": pdf["pre_rank_top25"].astype(np.float32),
            "pre_rank_11_25": pdf["pre_rank_11_25"].astype(np.float32),
        }
    )
    feature_df["pre_x_rm"] = (feature_df["pre_norm"] * feature_df["rm_feature"]).astype(np.float32)
    feature_df["rm_minus_pre"] = (feature_df["rm_feature"] - feature_df["pre_norm"]).astype(np.float32)
    feature_names = list(feature_df.columns)
    return feature_df, feature_names


def fit_stack_model(feature_df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(feature_df.values, labels.astype(int).values)
    return {
        "model_type": "logistic_regression",
        "feature_names": list(feature_df.columns),
        "coef": [float(x) for x in clf.coef_[0].tolist()],
        "intercept": float(clf.intercept_[0]),
    }


def apply_stack_model(feature_df: pd.DataFrame, payload: dict[str, Any]) -> np.ndarray:
    feature_names = list(payload.get("feature_names", []))
    coef = np.asarray(payload.get("coef", []), dtype=np.float32)
    intercept = float(payload.get("intercept", 0.0))
    if not feature_names or len(feature_names) != len(coef):
        raise RuntimeError("invalid stack model payload")
    x = feature_df.loc[:, feature_names].to_numpy(dtype=np.float32)
    logits = x @ coef.astype(np.float32) + intercept
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


__all__ = [
    "add_rank_features",
    "apply_calibrator",
    "apply_stack_model",
    "build_stack_feature_frame",
    "fit_isotonic_calibrator",
    "fit_platt_calibrator",
    "fit_stack_model",
    "load_bucket_scores",
    "resolve_run",
    "save_json",
]
