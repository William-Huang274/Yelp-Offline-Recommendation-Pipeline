#!/usr/bin/env python3
"""
显存监控工具 - 用于 DPO 训练前后的显存检查
Memory Monitor for DPO Training
"""
import torch
import psutil
import os
from pathlib import Path


def format_bytes(bytes_val):
    """格式化字节为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def check_cuda():
    """检查 CUDA 可用性"""
    print("=" * 60)
    print("CUDA 检查")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        print("   请检查:")
        print("   1. NVIDIA 驱动是否安装")
        print("   2. PyTorch 是否安装了 CUDA 版本")
        return False

    print(f"✓ CUDA 可用")
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")
    print()

    return True


def check_gpu_memory():
    """检查 GPU 显存"""
    if not torch.cuda.is_available():
        return

    print("=" * 60)
    print("GPU 显存状态")
    print("=" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = total_memory - reserved

        print(f"\nGPU {i}: {props.name}")
        print(f"  总显存:     {format_bytes(total_memory)}")
        print(f"  已分配:     {format_bytes(allocated)} ({allocated/total_memory*100:.1f}%)")
        print(f"  已保留:     {format_bytes(reserved)} ({reserved/total_memory*100:.1f}%)")
        print(f"  可用:       {format_bytes(free)} ({free/total_memory*100:.1f}%)")

        # 显存建议
        free_gb = free / (1024**3)
        print(f"\n  显存建议:")
        if free_gb >= 7:
            print(f"    ✓ 显存充足 ({free_gb:.1f}GB)，可以使用标准配置")
        elif free_gb >= 5:
            print(f"    ⚠ 显存一般 ({free_gb:.1f}GB)，建议使用低显存配置")
        else:
            print(f"    ❌ 显存不足 ({free_gb:.1f}GB)，必须使用极限配置")
            print(f"       或考虑关闭其他占用显存的程序")


def check_system_memory():
    """检查系统内存"""
    print("\n" + "=" * 60)
    print("系统内存状态")
    print("=" * 60)

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"\n物理内存:")
    print(f"  总量:       {format_bytes(mem.total)}")
    print(f"  可用:       {format_bytes(mem.available)} ({mem.percent:.1f}% 已使用)")
    print(f"  已使用:     {format_bytes(mem.used)}")

    print(f"\n虚拟内存 (页面文件):")
    print(f"  总量:       {format_bytes(swap.total)}")
    print(f"  可用:       {format_bytes(swap.free)} ({swap.percent:.1f}% 已使用)")
    print(f"  已使用:     {format_bytes(swap.used)}")

    # 页面文件建议
    swap_gb = swap.total / (1024**3)
    if swap_gb < 20:
        print(f"\n  ⚠ 警告: 页面文件较小 ({swap_gb:.1f}GB)")
        print(f"     建议增加到 40GB+ 以避免 Windows Error 1455")
        print(f"     设置路径: 系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存")
    else:
        print(f"\n  ✓ 页面文件充足 ({swap_gb:.1f}GB)")


def estimate_memory_usage():
    """估算 DPO 训练显存需求"""
    print("\n" + "=" * 60)
    print("DPO 训练显存估算 (Qwen3-4B)")
    print("=" * 60)

    configs = [
        {
            "name": "标准配置",
            "seq_len": 768,
            "lora_r": 16,
            "batch": 1,
            "estimate_gb": 9.5
        },
        {
            "name": "低显存配置",
            "seq_len": 512,
            "lora_r": 8,
            "batch": 1,
            "estimate_gb": 7.5
        },
        {
            "name": "极限配置",
            "seq_len": 384,
            "lora_r": 4,
            "batch": 1,
            "estimate_gb": 6.0
        }
    ]

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print(f"  序列长度: {cfg['seq_len']}")
        print(f"  LoRA rank: {cfg['lora_r']}")
        print(f"  Batch size: {cfg['batch']}")
        print(f"  预估显存: ~{cfg['estimate_gb']:.1f} GB")

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if cfg['estimate_gb'] <= total_gb * 0.9:
                print(f"  状态: ✓ 可以运行")
            else:
                print(f"  状态: ❌ 可能 OOM")


def check_disk_space():
    """检查磁盘空间"""
    print("\n" + "=" * 60)
    print("磁盘空间检查")
    print("=" * 60)

    # 检查当前目录所在磁盘
    current_drive = Path.cwd().drive
    if not current_drive:
        current_drive = "/"

    disk = psutil.disk_usage(current_drive)

    print(f"\n磁盘 {current_drive}")
    print(f"  总量:       {format_bytes(disk.total)}")
    print(f"  可用:       {format_bytes(disk.free)} ({disk.percent:.1f}% 已使用)")
    print(f"  已使用:     {format_bytes(disk.used)}")

    free_gb = disk.free / (1024**3)
    if free_gb < 10:
        print(f"\n  ⚠ 警告: 磁盘空间不足 ({free_gb:.1f}GB)")
        print(f"     模型和检查点需要至少 10GB 空间")
    else:
        print(f"\n  ✓ 磁盘空间充足 ({free_gb:.1f}GB)")


def recommend_config():
    """推荐配置"""
    print("\n" + "=" * 60)
    print("配置推荐")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n❌ 无法使用 GPU 训练")
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_vram = (torch.cuda.get_device_properties(0).total_memory -
                 torch.cuda.memory_reserved(0)) / (1024**3)

    print(f"\n你的 GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"总显存: {total_vram:.1f}GB, 当前可用: {free_vram:.1f}GB")
    print()

    if free_vram >= 9:
        print("推荐: 标准配置")
        print("  python scripts\\11_2_dpo_train.py")
    elif free_vram >= 7:
        print("推荐: 低显存配置")
        print("  scripts\\run_dpo_low_memory.bat")
        print("  选择 [1] Standard Low Memory")
    elif free_vram >= 5:
        print("推荐: 极限配置")
        print("  scripts\\run_dpo_low_memory.bat")
        print("  选择 [2] Ultra Low Memory")
    else:
        print("⚠ 显存可能不足，建议:")
        print("  1. 关闭其他占用显存的程序")
        print("  2. 使用极限配置")
        print("  3. 考虑使用更小的模型 (如 Qwen3-1.5B)")


def main():
    print("\n" + "=" * 60)
    print("DPO 训练环境检查工具")
    print("=" * 60)
    print()

    # 执行所有检查
    check_cuda()
    check_gpu_memory()
    check_system_memory()
    check_disk_space()
    estimate_memory_usage()
    recommend_config()

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
