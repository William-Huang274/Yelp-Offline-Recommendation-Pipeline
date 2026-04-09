from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


SPARSE_CONTEXT_COLUMNS = ["business_id", "primary_category", "user_segment"]
WIDE_CROSS_SPEC_DEFAULTS = [
    ("user_segment", "primary_category", "user_segment_x_primary_category"),
    ("user_segment", "business_id", "user_segment_x_business_id"),
    ("user_segment", "has_als", "user_segment_x_has_als"),
    ("user_segment", "has_profile", "user_segment_x_has_profile"),
    ("user_segment", "has_cluster", "user_segment_x_has_cluster"),
    ("user_segment", "has_popular", "user_segment_x_has_popular"),
    ("primary_category", "has_profile", "primary_category_x_has_profile"),
    ("primary_category", "has_als", "primary_category_x_has_als"),
]


def _stable_hash_bucket(value: Any, num_buckets: int) -> int:
    if int(num_buckets) <= 1:
        return 0
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "little") % max(1, int(num_buckets) - 1))


def _series_to_hash_indices(series: pd.Series, num_buckets: int) -> np.ndarray:
    values = [_stable_hash_bucket(v, int(num_buckets)) for v in series.tolist()]
    return np.asarray(values, dtype=np.int64)


def _series_to_direct_item_indices(series: pd.Series, known_max: int) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    out = np.zeros(len(arr), dtype=np.int64)
    if int(known_max) < 0:
        return out
    known_mask = (arr >= 0) & (arr <= int(known_max))
    out[known_mask] = arr[known_mask] + 1
    oov_bucket = int(known_max) + 2
    out[arr > int(known_max)] = oov_bucket
    return out


def _normalize_cross_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    text = str(value).strip()
    return text


def _series_to_cross_indices(
    left: pd.Series,
    right: pd.Series,
    num_buckets: int,
) -> np.ndarray:
    if len(left) != len(right):
        raise ValueError("wide cross series length mismatch")
    values: list[int] = []
    for lv, rv in zip(left.tolist(), right.tolist(), strict=False):
        ltxt = _normalize_cross_value(lv)
        rtxt = _normalize_cross_value(rv)
        if (not ltxt) and (not rtxt):
            values.append(0)
            continue
        values.append(_stable_hash_bucket(f"{ltxt}||{rtxt}", int(num_buckets)))
    return np.asarray(values, dtype=np.int64)


def _build_pairwise_example_index(
    user_idx: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    pre_rank: np.ndarray,
    max_neg_per_pos: int,
    neg_pre_rank_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = max(1, int(max_neg_per_pos))
    df = pd.DataFrame(
        {
            "row_idx": np.arange(len(user_idx), dtype=np.int32),
            "user_idx": pd.to_numeric(pd.Series(user_idx), errors="coerce").fillna(-1).astype(np.int64),
            "label": pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).astype(np.int32),
            "sample_weight": pd.to_numeric(pd.Series(sample_weight), errors="coerce").fillna(1.0).astype(np.float32),
            "pre_rank": pd.to_numeric(pd.Series(pre_rank), errors="coerce").fillna(10**9).astype(np.int64),
        }
    )
    df = df.sort_values(
        ["user_idx", "label", "pre_rank", "sample_weight", "row_idx"],
        ascending=[True, False, True, False, True],
        kind="stable",
    )
    pos_rows: list[int] = []
    neg_rows: list[int] = []
    pair_weight_rows: list[float] = []
    for _, g in df.groupby("user_idx", sort=False):
        pos = g.loc[g["label"] > 0, ["row_idx", "sample_weight"]].to_numpy(dtype=np.float64, copy=False)
        neg_df = g.loc[g["label"] <= 0, ["row_idx", "sample_weight", "pre_rank"]].copy()
        if int(neg_pre_rank_max) > 0:
            neg_head = neg_df.loc[neg_df["pre_rank"] <= int(neg_pre_rank_max)]
            if not neg_head.empty:
                neg_df = neg_head
        neg_df = neg_df.sort_values(["pre_rank", "sample_weight", "row_idx"], ascending=[True, False, True], kind="stable")
        neg = neg_df.to_numpy(dtype=np.float64, copy=False)
        if len(pos) == 0 or len(neg) == 0:
            continue
        neg = neg[:cap]
        for pos_row, pos_w in pos:
            for neg_row, neg_w, _neg_rank in neg:
                pos_rows.append(int(pos_row))
                neg_rows.append(int(neg_row))
                pair_weight_rows.append(float(max(1e-6, 0.5 * (float(pos_w) + float(neg_w)))))
    if not pos_rows:
        empty_i = np.asarray([], dtype=np.int64)
        empty_w = np.asarray([], dtype=np.float32)
        return empty_i, empty_i, empty_w
    return (
        np.asarray(pos_rows, dtype=np.int64),
        np.asarray(neg_rows, dtype=np.int64),
        np.asarray(pair_weight_rows, dtype=np.float32),
    )


def _pairwise_logloss_from_logits(
    logits: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    pair_weight: np.ndarray,
) -> float:
    if len(pos_idx) == 0:
        return float("inf")
    margin = logits[pos_idx] - logits[neg_idx]
    loss_vec = np.logaddexp(0.0, -margin) * pair_weight.astype(np.float64, copy=False)
    denom = float(np.clip(pair_weight.astype(np.float64, copy=False).sum(), 1e-6, None))
    return float(loss_vec.sum() / denom)


def _ranking_metrics_from_logits(
    user_idx: np.ndarray,
    y: np.ndarray,
    pre_rank: np.ndarray,
    logits: np.ndarray,
    top_k: int,
) -> tuple[float, float]:
    if len(user_idx) == 0:
        return 0.0, 0.0
    df = pd.DataFrame(
        {
            "row_idx": np.arange(len(user_idx), dtype=np.int32),
            "user_idx": pd.to_numeric(pd.Series(user_idx), errors="coerce").fillna(-1).astype(np.int64),
            "label": pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).astype(np.int32),
            "pre_rank": pd.to_numeric(pd.Series(pre_rank), errors="coerce").fillna(10**9).astype(np.int64),
            "score": pd.to_numeric(pd.Series(logits), errors="coerce").fillna(-1e9).astype(np.float64),
        }
    )
    df = df.sort_values(
        ["user_idx", "score", "pre_rank", "row_idx"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    df["rank"] = df.groupby("user_idx", sort=False).cumcount() + 1
    pos = (
        df.loc[df["label"] > 0, ["user_idx", "rank"]]
        .groupby("user_idx", as_index=False)["rank"]
        .min()
        .rename(columns={"rank": "hit_rank"})
    )
    users = pd.DataFrame({"user_idx": pd.Series(df["user_idx"].drop_duplicates().tolist(), dtype=np.int64)})
    merged = users.merge(pos, on="user_idx", how="left")
    hit_rank = pd.to_numeric(merged["hit_rank"], errors="coerce").fillna(0).astype(np.int32).to_numpy()
    hit_rank = np.where(hit_rank <= int(top_k), hit_rank, 0).astype(np.int32, copy=False)
    recall = float((hit_rank > 0).mean()) if len(hit_rank) > 0 else 0.0
    if (hit_rank > 0).any():
        ndcg = float((1.0 / np.log2(hit_rank[hit_rank > 0] + 1.0)).sum() / max(1, len(hit_rank)))
    else:
        ndcg = 0.0
    return recall, ndcg


def build_wide_deep_v2_sparse_state(
    train_pdf: pd.DataFrame,
    valid_pdf: pd.DataFrame,
    user_hash_buckets: int,
    business_hash_buckets: int,
    category_hash_buckets: int,
    segment_hash_buckets: int,
    user_embed_dim: int,
    item_embed_dim: int,
    business_embed_dim: int,
    category_embed_dim: int,
    segment_embed_dim: int,
    enable_wide_cross: bool,
    wide_cross_buckets: int,
) -> dict[str, Any]:
    all_columns = set(train_pdf.columns).union(valid_pdf.columns)
    item_series = pd.concat(
        [
            pd.to_numeric(train_pdf.get("item_idx"), errors="coerce"),
            pd.to_numeric(valid_pdf.get("item_idx"), errors="coerce"),
        ],
        ignore_index=True,
    )
    item_series = item_series[item_series.notna()]
    item_known_max = int(item_series.max()) if not item_series.empty else -1
    item_vocab_size = max(2, item_known_max + 3)
    wide_cross_features: list[dict[str, Any]] = []
    if bool(enable_wide_cross):
        for left_source, right_source, name in WIDE_CROSS_SPEC_DEFAULTS:
            if left_source in all_columns and right_source in all_columns:
                wide_cross_features.append(
                    {
                        "name": str(name),
                        "left_source": str(left_source),
                        "right_source": str(right_source),
                        "num_buckets": int(max(2, wide_cross_buckets)),
                    }
                )
    return {
        "features": [
            {
                "name": "user_idx_hash",
                "source": "user_idx",
                "kind": "hash",
                "num_buckets": int(max(2, user_hash_buckets)),
                "embed_dim": int(max(1, user_embed_dim)),
            },
            {
                "name": "item_idx_direct",
                "source": "item_idx",
                "kind": "direct_item",
                "num_buckets": int(item_vocab_size),
                "known_max": int(item_known_max),
                "embed_dim": int(max(1, item_embed_dim)),
            },
            {
                "name": "business_id_hash",
                "source": "business_id",
                "kind": "hash",
                "num_buckets": int(max(2, business_hash_buckets)),
                "embed_dim": int(max(1, business_embed_dim)),
            },
            {
                "name": "primary_category_hash",
                "source": "primary_category",
                "kind": "hash",
                "num_buckets": int(max(2, category_hash_buckets)),
                "embed_dim": int(max(1, category_embed_dim)),
            },
            {
                "name": "user_segment_hash",
                "source": "user_segment",
                "kind": "hash",
                "num_buckets": int(max(2, segment_hash_buckets)),
                "embed_dim": int(max(1, segment_embed_dim)),
            },
        ],
        "wide_cross_features": wide_cross_features,
    }


def build_wide_deep_v2_sparse_matrix(
    pdf: pd.DataFrame,
    sparse_state: dict[str, Any] | None,
) -> np.ndarray:
    features = list((sparse_state or {}).get("features", []))
    if not features:
        return np.zeros((len(pdf), 0), dtype=np.int64)
    cols: list[np.ndarray] = []
    for spec in features:
        source = str(spec.get("source", "")).strip()
        series = pdf[source] if source in pdf.columns else pd.Series([None] * len(pdf))
        kind = str(spec.get("kind", "")).strip().lower()
        if kind == "direct_item":
            vals = _series_to_direct_item_indices(series, known_max=int(spec.get("known_max", -1)))
        else:
            vals = _series_to_hash_indices(series, num_buckets=int(spec.get("num_buckets", 2)))
        cols.append(vals)
    return np.stack(cols, axis=1).astype(np.int64, copy=False)


def build_wide_deep_v2_wide_matrix(
    pdf: pd.DataFrame,
    sparse_state: dict[str, Any] | None,
) -> np.ndarray:
    specs = list((sparse_state or {}).get("wide_cross_features", []))
    if not specs:
        return np.zeros((len(pdf), 0), dtype=np.int64)
    cols: list[np.ndarray] = []
    for spec in specs:
        left_source = str(spec.get("left_source", "")).strip()
        right_source = str(spec.get("right_source", "")).strip()
        left_series = pdf[left_source] if left_source in pdf.columns else pd.Series([None] * len(pdf))
        right_series = pdf[right_source] if right_source in pdf.columns else pd.Series([None] * len(pdf))
        vals = _series_to_cross_indices(
            left=left_series,
            right=right_series,
            num_buckets=int(spec.get("num_buckets", 2)),
        )
        cols.append(vals)
    return np.stack(cols, axis=1).astype(np.int64, copy=False)


class WideDeepV2Net(nn.Module):
    def __init__(
        self,
        dense_input_dim: int,
        sparse_features: list[dict[str, Any]],
        wide_cross_features: list[dict[str, Any]],
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.wide_dense = nn.Linear(int(dense_input_dim), 1)
        self.wide_cross_embeddings = nn.ModuleList(
            [
                nn.Embedding(int(spec["num_buckets"]), 1)
                for spec in wide_cross_features
            ]
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(spec["num_buckets"]), int(spec["embed_dim"]))
                for spec in sparse_features
            ]
        )
        sparse_total_dim = int(sum(int(spec["embed_dim"]) for spec in sparse_features))
        deep_input_dim = int(dense_input_dim) + sparse_total_dim
        deep_layers: list[nn.Module] = []
        prev = deep_input_dim
        for h in hidden_dims:
            deep_layers.append(nn.Linear(prev, int(h)))
            deep_layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                deep_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        deep_layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, dense_x: "torch.Tensor", sparse_x: "torch.Tensor", wide_x: "torch.Tensor") -> "torch.Tensor":
        pieces = [dense_x]
        wide_out = self.wide_dense(dense_x)
        for i, emb in enumerate(self.wide_cross_embeddings):
            idx = wide_x[:, i].long().clamp(min=0, max=emb.num_embeddings - 1)
            wide_out = wide_out + emb(idx)
        for i, emb in enumerate(self.embeddings):
            idx = sparse_x[:, i].long().clamp(min=0, max=emb.num_embeddings - 1)
            pieces.append(emb(idx))
        deep_x = torch.cat(pieces, dim=1) if len(pieces) > 1 else dense_x
        return wide_out + self.deep(deep_x)


def predict_wide_deep_v2_logits(
    model: WideDeepV2Net,
    dense_x: np.ndarray,
    sparse_x: np.ndarray,
    wide_x: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is not installed for wide_deep_v2 backend")
    model.eval()
    outputs: list[np.ndarray] = []
    bs = max(256, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(dense_x), bs):
            end = min(len(dense_x), start + bs)
            xb_dense = torch.from_numpy(dense_x[start:end]).to(device=device, dtype=torch.float32)
            xb_sparse = torch.from_numpy(sparse_x[start:end]).to(device=device, dtype=torch.long)
            xb_wide = torch.from_numpy(wide_x[start:end]).to(device=device, dtype=torch.long)
            logits = model(xb_dense, xb_sparse, xb_wide).squeeze(-1).detach().cpu().numpy().astype(np.float64)
            outputs.append(logits)
    if not outputs:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(outputs, axis=0)


def train_wide_deep_v2_model(
    dense_train: np.ndarray,
    sparse_train: np.ndarray,
    wide_train: np.ndarray,
    user_idx_train: np.ndarray,
    pre_rank_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    dense_valid: np.ndarray,
    sparse_valid: np.ndarray,
    wide_valid: np.ndarray,
    user_idx_valid: np.ndarray,
    pre_rank_valid: np.ndarray,
    y_valid: np.ndarray,
    sparse_state: dict[str, Any],
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    patience: int,
    device: str,
    random_seed: int,
    loss_type: str,
    pairwise_neg_per_pos: int,
    pairwise_neg_pre_rank_max: int,
    early_stop_metric: str,
    early_stop_top_k: int,
) -> tuple[WideDeepV2Net, np.ndarray, dict[str, Any]]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("TRAIN_MODEL_BACKEND=wide_deep_v2 but torch is not installed")
    from sklearn.metrics import log_loss, roc_auc_score

    torch.manual_seed(int(random_seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(random_seed))
    sparse_features = list((sparse_state or {}).get("features", []))
    wide_cross_features = list((sparse_state or {}).get("wide_cross_features", []))
    model = WideDeepV2Net(
        dense_input_dim=int(dense_train.shape[1]),
        sparse_features=sparse_features,
        wide_cross_features=wide_cross_features,
        hidden_dims=[int(x) for x in hidden_dims],
        dropout=float(dropout),
    ).to(device)
    loss_type_norm = str(loss_type or "pointwise").strip().lower() or "pointwise"
    if loss_type_norm not in {"pointwise", "pairwise"}:
        loss_type_norm = "pointwise"
    pair_pos_idx, pair_neg_idx, pair_weight = _build_pairwise_example_index(
        user_idx=user_idx_train,
        y=y_train,
        sample_weight=sample_weight,
        pre_rank=pre_rank_train,
        max_neg_per_pos=int(pairwise_neg_per_pos),
        neg_pre_rank_max=int(pairwise_neg_pre_rank_max),
    )
    valid_pair_pos_idx, valid_pair_neg_idx, valid_pair_weight = _build_pairwise_example_index(
        user_idx=user_idx_valid,
        y=y_valid,
        sample_weight=np.ones(len(y_valid), dtype=np.float32),
        pre_rank=pre_rank_valid,
        max_neg_per_pos=int(pairwise_neg_per_pos),
        neg_pre_rank_max=int(pairwise_neg_pre_rank_max),
    )
    use_pairwise = loss_type_norm == "pairwise" and len(pair_pos_idx) > 0
    early_stop_metric_norm = str(early_stop_metric or "auto").strip().lower() or "auto"
    if early_stop_metric_norm not in {"auto", "logloss", "pair_logloss", "ndcg_at_k"}:
        early_stop_metric_norm = "auto"
    effective_early_stop_metric = (
        "ndcg_at_k" if (early_stop_metric_norm == "auto" and use_pairwise) else
        "logloss" if early_stop_metric_norm == "auto" else
        early_stop_metric_norm
    )
    dense_train_tensor = torch.from_numpy(dense_train.astype(np.float32))
    sparse_train_tensor = torch.from_numpy(sparse_train.astype(np.int64))
    wide_train_tensor = torch.from_numpy(wide_train.astype(np.int64))
    if use_pairwise:
        dataset = TensorDataset(
            torch.from_numpy(pair_pos_idx.astype(np.int64)),
            torch.from_numpy(pair_neg_idx.astype(np.int64)),
            torch.from_numpy(pair_weight.astype(np.float32)),
        )
    else:
        dataset = TensorDataset(
            dense_train_tensor,
            sparse_train_tensor,
            wide_train_tensor,
            torch.from_numpy(y_train.astype(np.float32)),
            torch.from_numpy(sample_weight.astype(np.float32)),
        )
    loader = DataLoader(
        dataset,
        batch_size=max(256, int(batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_state: dict[str, Any] | None = None
    best_valid_loss = float("inf")
    best_epoch = -1
    best_valid_logloss_observed = float("inf")
    best_valid_pair_logloss_observed = float("inf")
    best_valid_ndcg_observed = float("-inf")
    patience_left = int(patience)
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        if use_pairwise:
            for pos_idx_batch, neg_idx_batch, wb in loader:
                pos_idx_batch = pos_idx_batch.long()
                neg_idx_batch = neg_idx_batch.long()
                xb_pos_dense = dense_train_tensor[pos_idx_batch].to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                xb_pos_sparse = sparse_train_tensor[pos_idx_batch].to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                xb_pos_wide = wide_train_tensor[pos_idx_batch].to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                xb_neg_dense = dense_train_tensor[neg_idx_batch].to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                xb_neg_sparse = sparse_train_tensor[neg_idx_batch].to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                xb_neg_wide = wide_train_tensor[neg_idx_batch].to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                wb = wb.to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                pos_logits = model(xb_pos_dense, xb_pos_sparse, xb_pos_wide).squeeze(-1)
                neg_logits = model(xb_neg_dense, xb_neg_sparse, xb_neg_wide).squeeze(-1)
                margin = pos_logits - neg_logits
                target = torch.ones_like(margin)
                loss_vec = criterion(margin, target) * wb
                loss = loss_vec.sum() / torch.clamp(wb.sum(), min=1e-6)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss_vec.sum().detach().cpu())
                weight_sum += float(wb.sum().detach().cpu())
        else:
            for xb_dense, xb_sparse, xb_wide, yb, wb in loader:
                xb_dense = xb_dense.to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                xb_sparse = xb_sparse.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                xb_wide = xb_wide.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
                yb = yb.to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                wb = wb.to(device=device, dtype=torch.float32, non_blocking=(device == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb_dense, xb_sparse, xb_wide).squeeze(-1)
                loss_vec = criterion(logits, yb) * wb
                loss = loss_vec.sum() / torch.clamp(wb.sum(), min=1e-6)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss_vec.sum().detach().cpu())
                weight_sum += float(wb.sum().detach().cpu())
        train_loss = float(loss_sum / max(weight_sum, 1e-6))
        valid_logits = predict_wide_deep_v2_logits(
            model,
            dense_x=dense_valid,
            sparse_x=sparse_valid,
            wide_x=wide_valid,
            device=device,
            batch_size=int(eval_batch_size),
        )
        valid_prob = 1.0 / (1.0 + np.exp(-np.clip(valid_logits, -50.0, 50.0)))
        valid_loss = float(log_loss(y_valid, valid_prob, labels=[0, 1])) if len(y_valid) > 0 else float("inf")
        valid_pair_loss = _pairwise_logloss_from_logits(
            logits=valid_logits,
            pos_idx=valid_pair_pos_idx,
            neg_idx=valid_pair_neg_idx,
            pair_weight=valid_pair_weight,
        ) if use_pairwise else float("inf")
        valid_recall_at_k, valid_ndcg_at_k = _ranking_metrics_from_logits(
            user_idx=user_idx_valid,
            y=y_valid,
            pre_rank=pre_rank_valid,
            logits=valid_logits,
            top_k=int(early_stop_top_k),
        )
        try:
            valid_auc = float(roc_auc_score(y_valid, valid_prob)) if len(np.unique(y_valid)) > 1 else float("nan")
        except Exception:
            valid_auc = float("nan")
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_logloss": float(valid_loss),
                "valid_pair_logloss": None if math.isinf(valid_pair_loss) else float(valid_pair_loss),
                "valid_recall_at_k": float(valid_recall_at_k),
                "valid_ndcg_at_k": float(valid_ndcg_at_k),
                "valid_auc": None if math.isnan(valid_auc) else float(valid_auc),
            }
        )
        if effective_early_stop_metric == "ndcg_at_k":
            optimize_loss = float(valid_ndcg_at_k)
            improved = optimize_loss > (best_valid_ndcg_observed + 1e-6)
        elif effective_early_stop_metric == "pair_logloss":
            optimize_loss = float(valid_pair_loss)
            improved = optimize_loss < (best_valid_loss - 1e-5)
        else:
            optimize_loss = float(valid_loss)
            improved = optimize_loss < (best_valid_loss - 1e-5)
        if improved:
            best_valid_loss = float(optimize_loss)
            best_valid_logloss_observed = float(valid_loss)
            best_valid_pair_logloss_observed = float(valid_pair_loss)
            best_valid_ndcg_observed = float(valid_ndcg_at_k)
            best_epoch = int(epoch)
            patience_left = int(patience)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = int(len(history_rows))
    model.load_state_dict(best_state)
    valid_logits = predict_wide_deep_v2_logits(
        model,
        dense_x=dense_valid,
        sparse_x=sparse_valid,
        wide_x=wide_valid,
        device=device,
        batch_size=int(eval_batch_size),
    )
    valid_prob = 1.0 / (1.0 + np.exp(-np.clip(valid_logits, -50.0, 50.0)))
    meta = {
        "device": str(device),
        "hidden_dims": [int(x) for x in hidden_dims],
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "epochs_requested": int(epochs),
        "epochs_trained": int(len(history_rows)),
        "batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "patience": int(patience),
        "loss_type_requested": str(loss_type_norm),
        "loss_type_effective": "pairwise" if use_pairwise else "pointwise",
        "pairwise_neg_per_pos": int(pairwise_neg_per_pos),
        "pairwise_neg_pre_rank_max": int(pairwise_neg_pre_rank_max),
        "pairwise_train_pairs": int(len(pair_pos_idx)),
        "pairwise_valid_pairs": int(len(valid_pair_pos_idx)),
        "early_stop_metric_requested": str(early_stop_metric_norm),
        "early_stop_metric_effective": str(effective_early_stop_metric),
        "early_stop_top_k": int(early_stop_top_k),
        "best_epoch": int(best_epoch),
        "best_valid_objective": float(best_valid_loss),
        "best_valid_logloss": float(best_valid_logloss_observed),
        "best_valid_pair_logloss": None if math.isinf(best_valid_pair_logloss_observed) else float(best_valid_pair_logloss_observed),
        "best_valid_ndcg_at_k": None if math.isinf(best_valid_ndcg_observed) else float(best_valid_ndcg_observed),
        "sparse_features": sparse_features,
        "wide_cross_features": wide_cross_features,
        "history": history_rows,
    }
    return model, valid_prob.astype(np.float64), meta
