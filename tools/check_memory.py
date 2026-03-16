#!/usr/bin/env python3
"""
Memory inspection utility for DPO training.
Use this tool before or during a local run to inspect GPU, RAM, page-file,
and disk capacity.
"""
import psutil
import torch
from pathlib import Path


def format_bytes(bytes_val):
    """Format bytes in a human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def check_cuda():
    """Check CUDA availability."""
    print("=" * 60)
    print("CUDA Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        print("  Please check:")
        print("  1. NVIDIA driver installation")
        print("  2. PyTorch CUDA build availability")
        return False

    print("[OK] CUDA is available")
    print(f"[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] GPU count: {torch.cuda.device_count()}")
    print()
    return True


def check_gpu_memory():
    """Check GPU memory state."""
    if not torch.cuda.is_available():
        return

    print("=" * 60)
    print("GPU Memory Status")
    print("=" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = total_memory - reserved

        print(f"\nGPU {i}: {props.name}")
        print(f"  Total VRAM:  {format_bytes(total_memory)}")
        print(f"  Allocated:   {format_bytes(allocated)} ({allocated / total_memory * 100:.1f}%)")
        print(f"  Reserved:    {format_bytes(reserved)} ({reserved / total_memory * 100:.1f}%)")
        print(f"  Free:        {format_bytes(free)} ({free / total_memory * 100:.1f}%)")

        free_gb = free / (1024 ** 3)
        print("\n  Guidance:")
        if free_gb >= 7:
            print(f"    [OK] Enough free VRAM ({free_gb:.1f}GB). Standard config is viable.")
        elif free_gb >= 5:
            print(f"    [WARN] VRAM is moderate ({free_gb:.1f}GB). Use a low-memory config.")
        else:
            print(f"    [ERROR] VRAM is tight ({free_gb:.1f}GB). Use the ultra-low-memory config.")
            print("    Close other GPU workloads if possible.")


def check_system_memory():
    """Check system RAM and page-file state."""
    print("\n" + "=" * 60)
    print("System Memory Status")
    print("=" * 60)

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print("\nPhysical memory:")
    print(f"  Total:       {format_bytes(mem.total)}")
    print(f"  Available:   {format_bytes(mem.available)} ({mem.percent:.1f}% used)")
    print(f"  Used:        {format_bytes(mem.used)}")

    print("\nPage file / swap:")
    print(f"  Total:       {format_bytes(swap.total)}")
    print(f"  Available:   {format_bytes(swap.free)} ({swap.percent:.1f}% used)")
    print(f"  Used:        {format_bytes(swap.used)}")

    swap_gb = swap.total / (1024 ** 3)
    if swap_gb < 20:
        print(f"\n  [WARN] Page file is small ({swap_gb:.1f}GB)")
        print("  Increase it to 40GB+ to reduce the risk of Windows Error 1455.")
        print("  Path: System Properties -> Advanced -> Performance -> Advanced -> Virtual Memory")
    else:
        print(f"\n  [OK] Page file capacity looks sufficient ({swap_gb:.1f}GB)")


def estimate_memory_usage():
    """Estimate DPO training VRAM demand."""
    print("\n" + "=" * 60)
    print("Estimated DPO VRAM Usage (Qwen3-4B)")
    print("=" * 60)

    configs = [
        {
            "name": "Standard profile",
            "seq_len": 768,
            "lora_r": 16,
            "batch": 1,
            "estimate_gb": 9.5,
        },
        {
            "name": "Low-memory profile",
            "seq_len": 512,
            "lora_r": 8,
            "batch": 1,
            "estimate_gb": 7.5,
        },
        {
            "name": "Ultra-low-memory profile",
            "seq_len": 384,
            "lora_r": 4,
            "batch": 1,
            "estimate_gb": 6.0,
        },
    ]

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print(f"  Sequence length: {cfg['seq_len']}")
        print(f"  LoRA rank:       {cfg['lora_r']}")
        print(f"  Batch size:      {cfg['batch']}")
        print(f"  Estimated VRAM:  ~{cfg['estimate_gb']:.1f} GB")

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if cfg['estimate_gb'] <= total_gb * 0.9:
                print("  Status: [OK] likely to run")
            else:
                print("  Status: [WARN] likely to OOM")


def check_disk_space():
    """Check free disk space."""
    print("\n" + "=" * 60)
    print("Disk Space Check")
    print("=" * 60)

    current_drive = Path.cwd().drive
    if not current_drive:
        current_drive = "/"

    disk = psutil.disk_usage(current_drive)

    print(f"\nDisk {current_drive}")
    print(f"  Total:       {format_bytes(disk.total)}")
    print(f"  Free:        {format_bytes(disk.free)} ({disk.percent:.1f}% used)")
    print(f"  Used:        {format_bytes(disk.used)}")

    free_gb = disk.free / (1024 ** 3)
    if free_gb < 10:
        print(f"\n  [WARN] Low free disk space ({free_gb:.1f}GB)")
        print("  Model outputs and checkpoints need at least 10GB.")
    else:
        print(f"\n  [OK] Disk space looks sufficient ({free_gb:.1f}GB)")


def recommend_config():
    """Recommend a practical config based on current free VRAM."""
    print("\n" + "=" * 60)
    print("Configuration Recommendation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n[ERROR] GPU training is not available")
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_vram = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    ) / (1024 ** 3)

    print(f"\nYour GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"Total VRAM: {total_vram:.1f}GB, currently free: {free_vram:.1f}GB")
    print()

    if free_vram >= 9:
        print("Recommendation: standard profile")
        print("  python scripts\\11_2_dpo_train.py")
    elif free_vram >= 7:
        print("Recommendation: low-memory profile")
        print("  scripts\\run_dpo_low_memory.bat")
        print("  Choose [1] Standard Low Memory")
    elif free_vram >= 5:
        print("Recommendation: ultra-low-memory profile")
        print("  scripts\\run_dpo_low_memory.bat")
        print("  Choose [2] Ultra Low Memory")
    else:
        print("[WARN] VRAM may still be insufficient")
        print("  1. Close other GPU workloads")
        print("  2. Use the ultra-low-memory profile")
        print("  3. Consider a smaller base model such as Qwen3-1.5B")


def main():
    print("\n" + "=" * 60)
    print("DPO Training Environment Check")
    print("=" * 60)
    print()

    check_cuda()
    check_gpu_memory()
    check_system_memory()
    check_disk_space()
    estimate_memory_usage()
    recommend_config()

    print("\n" + "=" * 60)
    print("Check complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
