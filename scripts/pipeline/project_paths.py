from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _resolve_root_from_env() -> Path | None:
    for env_name in ("BDA_PROJECT_ROOT", "PROJECT_ROOT", "REPO_ROOT"):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        return Path(raw).expanduser().resolve()
    return None


def _discover_project_root() -> Path:
    from_env = _resolve_root_from_env()
    if from_env is not None:
        return from_env
    # scripts/pipeline/project_paths.py -> repo root is 2 parents up
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _discover_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
PARQUET_DIR = DATA_DIR / "parquet"
METRICS_DIR = DATA_DIR / "metrics"
SPARK_TMP_DIR = DATA_DIR / "spark-tmp"
LATEST_RUN_DIR = OUTPUT_DIR / "_latest_runs"
PROD_RUN_DIR = OUTPUT_DIR / "_prod_runs"


def project_path(relative_path: str | Path) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return (PROJECT_ROOT / rel).resolve()


def env_or_project_path(env_var: str, default_relative_path: str | Path) -> Path:
    raw = os.getenv(env_var, "").strip()
    if raw:
        return normalize_legacy_project_path(Path(raw).expanduser())
    return project_path(default_relative_path)


def normalize_legacy_project_path(raw_path: str | Path) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return Path(raw)
    normalized = re.sub(r"/+", "/", raw.replace("\\", "/"))
    for legacy_root in ("D:/5006 BDA project", "D:/5006_BDA_project"):
        if legacy_root in normalized:
            return Path(normalized.replace(legacy_root, PROJECT_ROOT.as_posix()))
    return Path(raw)


def _sanitize_pointer_name(pointer_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(pointer_name or "").strip()).strip("._")
    if not safe:
        raise ValueError("pointer_name is empty")
    return safe


def _pointer_path(pointer_dir: Path, pointer_name: str) -> Path:
    safe = _sanitize_pointer_name(pointer_name)
    return pointer_dir / f"{safe}.json"


def latest_run_pointer_path(pointer_name: str) -> Path:
    return _pointer_path(LATEST_RUN_DIR, pointer_name)


def production_run_pointer_path(pointer_name: str) -> Path:
    return _pointer_path(PROD_RUN_DIR, pointer_name)


def _read_pointer(pointer_path: Path) -> dict[str, Any] | None:
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def read_latest_run_pointer(pointer_name: str) -> dict[str, Any] | None:
    return _read_pointer(latest_run_pointer_path(pointer_name))


def read_production_run_pointer(pointer_name: str) -> dict[str, Any] | None:
    return _read_pointer(production_run_pointer_path(pointer_name))


def _resolve_pointer(pointer_payload: dict[str, Any] | None) -> Path | None:
    if not pointer_payload:
        return None
    raw = str(pointer_payload.get("run_dir", "")).strip()
    if not raw:
        return None
    p = normalize_legacy_project_path(raw)
    if p.exists():
        return p
    return None


def resolve_latest_run_pointer(pointer_name: str) -> Path | None:
    return _resolve_pointer(read_latest_run_pointer(pointer_name))


def resolve_production_run_pointer(pointer_name: str) -> Path | None:
    return _resolve_pointer(read_production_run_pointer(pointer_name))


def _write_pointer(pointer_path: Path, pointer_name: str, run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "pointer_name": str(pointer_name),
        "run_dir": str(Path(run_dir)),
    }
    if extra:
        payload.update(extra)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return pointer_path


def write_latest_run_pointer(pointer_name: str, run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    return _write_pointer(latest_run_pointer_path(pointer_name), pointer_name, run_dir, extra)


def write_production_run_pointer(pointer_name: str, run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    return _write_pointer(production_run_pointer_path(pointer_name), pointer_name, run_dir, extra)
