from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SparkTmpContext:
    script_tag: str
    session_id: str
    base_dir: Path
    spark_local_dir: Path
    py_temp_dir: Path
    py_temp_root: Path
    sessions_root: Path
    scratch_root: Path
    session_isolation: bool
    auto_clean_enabled: bool
    clean_on_exit: bool
    retention_hours: int
    cleanup_summary: dict[str, Any]


def _safe_rm_tree(path: Path) -> bool:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=False)
        return True
    except Exception:
        return False


def _safe_rm_file(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _safe_size(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        return total
    except Exception:
        return 0


def _ts() -> int:
    return int(time.time())


def _sanitize_tag(raw: str) -> str:
    txt = str(raw or "").strip().lower() or "stage"
    out = []
    for ch in txt:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "stage"


def _resolve_existing_executable(raw: str) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    p = Path(txt)
    if p.exists() and p.is_file():
        return str(p)
    return ""


def ensure_pyspark_python_env(default_python: str = "") -> tuple[str, str]:
    fallback = _resolve_existing_executable(default_python)
    if not fallback:
        fallback = _resolve_existing_executable(sys.executable) or sys.executable
    worker_py = _resolve_existing_executable(os.getenv("PYSPARK_PYTHON", "")) or fallback
    driver_py = _resolve_existing_executable(os.getenv("PYSPARK_DRIVER_PYTHON", "")) or worker_py
    os.environ["PYSPARK_PYTHON"] = worker_py
    os.environ["PYSPARK_DRIVER_PYTHON"] = driver_py
    return worker_py, driver_py


def cleanup_stale_tmp(
    *,
    base_dir: Path,
    sessions_root: Path,
    py_temp_root: Path,
    scratch_root: Path,
    retention_hours: int,
    max_entries: int,
) -> dict[str, Any]:
    now = _ts()
    ttl = max(1, int(retention_hours)) * 3600
    max_entries = max(1, int(max_entries))
    deleted_dirs = 0
    deleted_files = 0
    deleted_bytes = 0
    scanned = 0
    errors = 0

    def is_stale(path: Path) -> bool:
        try:
            return (now - int(path.stat().st_mtime)) >= ttl
        except Exception:
            return False

    legacy_prefixes = (
        "spark-",
        "blockmgr-",
        "profile_recall",
        "tower_seq_features",
        "tmp_py",
        "stage10_2_xgb_batches_",
        "stage09-checkpoint",
    )

    # Legacy root-level temp dirs from old scripts (before session isolation).
    if base_dir.exists():
        for child in base_dir.iterdir():
            if scanned >= max_entries:
                break
            if not child.is_dir():
                continue
            name = child.name.strip().lower()
            if name.startswith("_"):
                # Managed roots are handled below.
                continue
            if not any(name.startswith(p) for p in legacy_prefixes):
                continue
            scanned += 1
            if not is_stale(child):
                continue
            sz = _safe_size(child)
            if _safe_rm_tree(child):
                deleted_dirs += 1
                deleted_bytes += sz
            else:
                errors += 1

    for root in (sessions_root, py_temp_root):
        if not root.exists():
            continue
        for child in root.iterdir():
            if scanned >= max_entries:
                break
            scanned += 1
            if not child.is_dir() or not is_stale(child):
                continue
            sz = _safe_size(child)
            if _safe_rm_tree(child):
                deleted_dirs += 1
                deleted_bytes += sz
            else:
                errors += 1

    if scratch_root.exists() and scanned < max_entries:
        for child in scratch_root.rglob("*"):
            if scanned >= max_entries:
                break
            scanned += 1
            if child.is_dir():
                continue
            if not is_stale(child):
                continue
            sz = _safe_size(child)
            if _safe_rm_file(child):
                deleted_files += 1
                deleted_bytes += sz
            else:
                errors += 1

        # Best-effort empty dir cleanup under scratch.
        for child in sorted(scratch_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not child.is_dir():
                continue
            try:
                next(child.iterdir())
            except StopIteration:
                try:
                    child.rmdir()
                except Exception:
                    pass
            except Exception:
                continue

    return {
        "retention_hours": int(retention_hours),
        "max_entries": int(max_entries),
        "scanned_entries": int(scanned),
        "deleted_dirs": int(deleted_dirs),
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "errors": int(errors),
    }


def build_spark_tmp_context(
    *,
    script_tag: str,
    spark_local_dir: str,
    py_temp_root_override: str = "",
    session_isolation: bool = True,
    auto_clean_enabled: bool = True,
    clean_on_exit: bool = True,
    retention_hours: int = 24,
    clean_max_entries: int = 2000,
    set_env_temp: bool = True,
    set_env_pyspark_python: bool = True,
) -> SparkTmpContext:
    tag = _sanitize_tag(script_tag)
    base_dir = Path(str(spark_local_dir or "").strip())
    base_dir.mkdir(parents=True, exist_ok=True)

    sessions_root = base_dir / "_sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    scratch_root = base_dir / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    if py_temp_root_override.strip():
        py_temp_root = Path(py_temp_root_override.strip())
    else:
        py_temp_root = base_dir / "_pytemp"
    py_temp_root.mkdir(parents=True, exist_ok=True)

    cleanup_summary = {
        "retention_hours": int(retention_hours),
        "max_entries": int(clean_max_entries),
        "scanned_entries": 0,
        "deleted_dirs": 0,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "errors": 0,
    }
    if auto_clean_enabled:
        cleanup_summary = cleanup_stale_tmp(
            base_dir=base_dir,
            sessions_root=sessions_root,
            py_temp_root=py_temp_root,
            scratch_root=scratch_root,
            retention_hours=int(retention_hours),
            max_entries=int(clean_max_entries),
        )

    session_id = f"{tag}_{os.getpid()}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    spark_dir = sessions_root / session_id if session_isolation else base_dir
    spark_dir.mkdir(parents=True, exist_ok=True)

    py_temp_dir = py_temp_root / session_id
    py_temp_dir.mkdir(parents=True, exist_ok=True)

    if set_env_temp:
        py_tmp = py_temp_dir.as_posix()
        os.environ["TEMP"] = py_tmp
        os.environ["TMP"] = py_tmp
        os.environ["TMPDIR"] = py_tmp
        tempfile.tempdir = py_tmp
    if set_env_pyspark_python:
        ensure_pyspark_python_env()

    ctx = SparkTmpContext(
        script_tag=tag,
        session_id=session_id,
        base_dir=base_dir,
        spark_local_dir=spark_dir,
        py_temp_dir=py_temp_dir,
        py_temp_root=py_temp_root,
        sessions_root=sessions_root,
        scratch_root=scratch_root,
        session_isolation=bool(session_isolation),
        auto_clean_enabled=bool(auto_clean_enabled),
        clean_on_exit=bool(clean_on_exit),
        retention_hours=int(retention_hours),
        cleanup_summary=cleanup_summary,
    )

    if clean_on_exit:
        atexit.register(cleanup_context_tmp, ctx=ctx)

    return ctx


def cleanup_context_tmp(*, ctx: SparkTmpContext) -> None:
    # Only cleanup session-local dirs to avoid deleting shared caches.
    _safe_rm_tree(ctx.spark_local_dir)
    _safe_rm_tree(ctx.py_temp_dir)


def alloc_scratch_file(
    *,
    ctx: SparkTmpContext,
    subdir: str,
    prefix: str,
    suffix: str,
) -> Path:
    target_dir = ctx.scratch_root / str(subdir).strip().replace("\\", "/")
    target_dir.mkdir(parents=True, exist_ok=True)
    pfx = _sanitize_tag(prefix)
    return target_dir / f"{pfx}_{ctx.session_id}_{uuid.uuid4().hex}{suffix}"
