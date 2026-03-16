from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pipeline.project_paths import env_or_project_path, project_path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


RUN_TAG = "stage09_user_profile_vector_recovery"
INPUT_ROOT = env_or_project_path("INPUT_09_USER_PROFILE_ROOT_DIR", "data/output/09_user_profiles")
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_USER_PROFILE_ROOT_DIR", "data/output/09_user_profiles")
CACHE_ROOT = env_or_project_path("CACHE_09_USER_PROFILE_ROOT_DIR", "data/cache/user_profile_embeddings")

INPUT_RUN_DIR = os.getenv("INPUT_USER_PROFILE_RUN_DIR", "").strip()
OUTPUT_RUN_DIR = os.getenv("OUTPUT_USER_PROFILE_RECOVERY_RUN_DIR", "").strip()
OUTPUT_RUN_SUFFIX = os.getenv(
    "USER_PROFILE_RECOVERY_RUN_SUFFIX",
    "_full_stage09_user_profile_vector_recovery",
).strip() or "_full_stage09_user_profile_vector_recovery"

USER_PROFILE_TABLE_FILE = "user_profiles.csv"
USER_PROFILE_TAG_LONG_FILE = "user_profile_tag_profile_long.csv"
USER_PROFILE_SUMMARY_FILE = "user_profiles_summary.csv"
USER_PROFILE_VECTOR_FILE = "user_profile_vectors.npz"
RECOVERY_META_FILE = "vector_recovery_meta.json"

BGE_LOCAL_MODEL_PATH = os.getenv("BGE_LOCAL_MODEL_PATH", project_path("hf_models/BAAI__bge-m3").as_posix()).strip()
PREFERRED_MODEL_NAME = (
    os.getenv("PROFILE_VECTOR_RECOVERY_MODEL_NAME", "bge-m3").strip()
    or "bge-m3"
)
DEVICE = os.getenv("PROFILE_VECTOR_DEVICE", "cpu").strip().lower() or "cpu"
EMBED_BATCH_SIZE = int(os.getenv("PROFILE_VECTOR_RECOVERY_BATCH_SIZE", "32").strip() or 32)
MAX_LENGTH = int(os.getenv("PROFILE_VECTOR_RECOVERY_MAX_LENGTH", "512").strip() or 512)
EMBED_NORMALIZE = os.getenv("PROFILE_VECTOR_RECOVERY_NORMALIZE", "true").strip().lower() == "true"
OVERWRITE_VECTOR = os.getenv("PROFILE_VECTOR_RECOVERY_OVERWRITE", "false").strip().lower() == "true"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def pick_latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and (p / USER_PROFILE_TABLE_FILE).exists()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no user profile run with {USER_PROFILE_TABLE_FILE} found in {root}")
    return runs[0]


def resolve_input_run() -> Path:
    if INPUT_RUN_DIR:
        p = Path(INPUT_RUN_DIR)
        if not (p / USER_PROFILE_TABLE_FILE).exists():
            raise FileNotFoundError(f"input run missing {USER_PROFILE_TABLE_FILE}: {p}")
        return p
    return pick_latest_run(INPUT_ROOT)


def resolve_output_run(input_run: Path) -> Path:
    if OUTPUT_RUN_DIR:
        return Path(OUTPUT_RUN_DIR)
    run_profile = "full" if "_full_" in input_run.name else "sample"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"{run_id}_{run_profile}{OUTPUT_RUN_SUFFIX}"


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _resolve_bge_path(path_text: str) -> str | None:
    p = Path(str(path_text or "").strip())
    if not str(p):
        return None
    if p.exists():
        if (p / "config.json").exists():
            return str(p)
        snapshots = p / "snapshots"
        if snapshots.exists():
            subs = [x for x in snapshots.iterdir() if x.is_dir() and (x / "config.json").exists()]
            if subs:
                subs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(subs[0])
    return None


def resolve_model_name() -> str:
    want = str(PREFERRED_MODEL_NAME or "").strip()
    if want.lower() == "auto":
        local_bge = _resolve_bge_path(BGE_LOCAL_MODEL_PATH)
        return local_bge if local_bge else "BAAI/bge-m3"
    if want.lower() in {"bge", "bge-m3"}:
        local_bge = _resolve_bge_path(BGE_LOCAL_MODEL_PATH)
        return local_bge if local_bge else "BAAI/bge-m3"
    return want


def _cache_file_for_model(model_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_name))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"profile_text_cache_{safe}.npz"


def _load_npz_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype="<U40"), np.zeros((0, 0), dtype=np.float32)
    data = np.load(path, allow_pickle=False)
    return data["text_hashes"].astype(str), data["embeddings"].astype(np.float32)


def _save_npz_cache(path: Path, hashes: np.ndarray, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, text_hashes=hashes, embeddings=vectors.astype(np.float32))


def _sentence_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class HFEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = str(device or "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int, normalize: bool, max_length: int) -> np.ndarray:
        all_parts: list[np.ndarray] = []
        with self._torch.no_grad():
            for s in range(0, len(texts), int(batch_size)):
                batch = texts[s : s + int(batch_size)]
                toks = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks)
                last_hidden = out.last_hidden_state
                attn = toks["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * attn).sum(dim=1)
                denom = attn.sum(dim=1).clamp(min=1e-9)
                emb = (summed / denom).detach().cpu().numpy().astype(np.float32)
                if normalize:
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    norms[norms == 0.0] = 1.0
                    emb = (emb / norms).astype(np.float32)
                all_parts.append(emb)
        if not all_parts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(all_parts, axis=0).astype(np.float32, copy=False)


def load_encoder(model_name: str, device: str) -> tuple[Any, str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)

        class STEncoder:
            def encode(self, texts: list[str], batch_size: int, normalize: bool, max_length: int) -> np.ndarray:
                return model.encode(
                    texts,
                    batch_size=int(batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=bool(normalize),
                    show_progress_bar=True,
                ).astype(np.float32)

        return STEncoder(), "sentence_transformers"
    except Exception:
        return HFEncoder(model_name=model_name, device=device), "transformers_mean_pool"


def load_profile_table(path: Path) -> pd.DataFrame:
    usecols = ["user_id", "profile_text", "n_sentences_selected"]
    pdf = pd.read_csv(path, usecols=lambda c: c in set(usecols), low_memory=False)
    for col in usecols:
        if col not in pdf.columns:
            pdf[col] = ""
    pdf["user_id"] = pdf["user_id"].astype(str)
    pdf["profile_text"] = pdf["profile_text"].map(_normalize_space)
    pdf["n_sentences_selected"] = pd.to_numeric(pdf["n_sentences_selected"], errors="coerce").fillna(0).astype(np.int32)
    pdf = pdf[(pdf["user_id"] != "") & (pdf["profile_text"] != "") & (pdf["n_sentences_selected"] > 0)].copy()
    return pdf


def build_vectors(pdf: pd.DataFrame, encoder: Any, model_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_file = _cache_file_for_model(model_name)
    cache_hashes, cache_vectors = _load_npz_cache(cache_file)
    cache_map = {str(h): i for i, h in enumerate(cache_hashes.tolist())}
    print(
        f"[INFO] building_vectors rows={int(pdf.shape[0])} cache_rows={int(cache_hashes.shape[0])} "
        f"model={model_name} batch_size={int(EMBED_BATCH_SIZE)}"
    )

    text_hashes = pdf["profile_text"].map(_sentence_hash).astype(str).tolist()
    unique_missing_hashes: list[str] = []
    unique_missing_texts: list[str] = []
    seen: set[str] = set()
    for h, txt in zip(text_hashes, pdf["profile_text"].tolist()):
        if h in cache_map or h in seen:
            continue
        seen.add(h)
        unique_missing_hashes.append(h)
        unique_missing_texts.append(str(txt))

    if unique_missing_texts:
        print(f"[INFO] encoding_missing_texts={int(len(unique_missing_texts))}")
        emb = encoder.encode(
            unique_missing_texts,
            batch_size=int(EMBED_BATCH_SIZE),
            normalize=bool(EMBED_NORMALIZE),
            max_length=int(MAX_LENGTH),
        ).astype(np.float32)
        if cache_vectors.shape[0] == 0:
            cache_hashes = np.array(unique_missing_hashes, dtype="<U40")
            cache_vectors = emb
        else:
            cache_hashes = np.concatenate([cache_hashes, np.array(unique_missing_hashes, dtype="<U40")], axis=0)
            cache_vectors = np.concatenate([cache_vectors, emb], axis=0)
        _save_npz_cache(cache_file, cache_hashes, cache_vectors)
        cache_map = {str(h): i for i, h in enumerate(cache_hashes.tolist())}

    user_ids: list[str] = []
    vec_rows: list[np.ndarray] = []
    for uid, h in zip(pdf["user_id"].tolist(), text_hashes):
        idx = cache_map.get(str(h))
        if idx is None:
            continue
        user_ids.append(str(uid))
        vec_rows.append(cache_vectors[int(idx)])
    if not vec_rows:
        raise RuntimeError("no vectors could be materialized from user_profiles.csv")
    mat = np.stack(vec_rows).astype(np.float32)
    meta = {
        "rows_in_profile_table": int(pdf.shape[0]),
        "vectors_written": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "cache_file": str(cache_file),
        "cache_size": int(cache_hashes.shape[0]),
        "missing_encoded": int(len(unique_missing_texts)),
    }
    return np.array(user_ids, dtype="<U64"), mat, meta


def main() -> None:
    input_run = resolve_input_run()
    output_run = resolve_output_run(input_run)
    output_run.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] input_run={input_run}")
    print(f"[INFO] output_run={output_run}")

    input_table = input_run / USER_PROFILE_TABLE_FILE
    input_tag_long = input_run / USER_PROFILE_TAG_LONG_FILE
    input_summary = input_run / USER_PROFILE_SUMMARY_FILE
    input_run_meta = input_run / "run_meta.json"
    output_vector = output_run / USER_PROFILE_VECTOR_FILE

    if output_vector.exists() and (not OVERWRITE_VECTOR):
        raise FileExistsError(f"output vector already exists: {output_vector}")

    copy_if_exists(input_table, output_run / USER_PROFILE_TABLE_FILE)
    copy_if_exists(input_tag_long, output_run / USER_PROFILE_TAG_LONG_FILE)
    copy_if_exists(input_summary, output_run / USER_PROFILE_SUMMARY_FILE)
    copy_if_exists(input_run_meta, output_run / "run_meta.json")

    model_name = resolve_model_name()
    print(f"[INFO] resolved_model={model_name} device={DEVICE}")
    encoder, encoder_backend = load_encoder(model_name=model_name, device=DEVICE)
    pdf = load_profile_table(input_table)
    print(f"[INFO] profile_rows={int(pdf.shape[0])}")
    user_ids, vectors, vec_meta = build_vectors(pdf=pdf, encoder=encoder, model_name=model_name)

    np.savez_compressed(
        output_vector.as_posix(),
        user_ids=user_ids,
        vectors=vectors,
        model_name=np.array([model_name]),
        normalized=np.array([int(EMBED_NORMALIZE)], dtype=np.int32),
    )
    recovery_meta = {
        "run_tag": RUN_TAG,
        "input_run": str(input_run),
        "output_run": str(output_run),
        "model_name": model_name,
        "encoder_backend": encoder_backend,
        "device": DEVICE,
        "batch_size": int(EMBED_BATCH_SIZE),
        "normalize": bool(EMBED_NORMALIZE),
        **vec_meta,
    }
    write_json(output_run / RECOVERY_META_FILE, recovery_meta)
    print(
        f"[INFO] vectors_written={recovery_meta['vectors_written']} "
        f"dim={recovery_meta['dim']} model={model_name} backend={encoder_backend}"
    )


if __name__ == "__main__":
    main()
