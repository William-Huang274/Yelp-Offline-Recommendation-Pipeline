#!/usr/bin/env python3
"""Inspect local GPU, RAM, swap, and disk before a heavy Stage11 run."""
from __future__ import annotations

from pathlib import Path

import psutil

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None


def format_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def main() -> int:
    print("=" * 72)
    print("Local Resource Check")
    print("=" * 72)

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"RAM total={format_bytes(mem.total)} available={format_bytes(mem.available)} used={mem.percent:.1f}%")
    print(f"Swap total={format_bytes(swap.total)} free={format_bytes(swap.free)} used={swap.percent:.1f}%")

    disk = psutil.disk_usage(Path.cwd().drive or "/")
    print(f"Disk free={format_bytes(disk.free)} total={format_bytes(disk.total)} used={disk.percent:.1f}%")

    if torch is None or not torch.cuda.is_available():
        print("CUDA not available")
        return 0

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(idx)
        allocated = torch.cuda.memory_allocated(idx)
        free = total - reserved
        print(
            f"GPU {idx} {props.name}: total={format_bytes(total)} "
            f"allocated={format_bytes(allocated)} reserved={format_bytes(reserved)} free={format_bytes(free)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
