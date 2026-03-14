#!/usr/bin/env python3
"""
DPO 训练可视化工具
实时监控训练进度和显存使用
"""
import json
import time
import os
from pathlib import Path
from datetime import datetime


def parse_log_file(log_path: Path):
    """解析训练日志文件"""
    if not log_path.exists():
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    info = {
        "config": {},
        "data": {},
        "training": {},
        "status": "unknown"
    }

    # 解析配置信息
    if "base_model=" in content:
        for line in content.split('\n'):
            if "base_model=" in line:
                info["config"]["base_model"] = line.split("base_model=")[1].split()[0]
            if "DPO β=" in line:
                parts = line.split("DPO β=")[1].split(",")
                info["config"]["dpo_beta"] = parts[0].strip()
            if "LR=" in line:
                parts = line.split("LR=")[1].split(",")
                info["config"]["lr"] = parts[0].strip()

    # 解析数据信息
    if "DPO train_pairs:" in content:
        for line in content.split('\n'):
            if "DPO train_pairs:" in line:
                info["data"]["train_pairs"] = line.split("train_pairs:")[1].split(",")[0].strip()
            if "eval_pairs:" in line:
                info["data"]["eval_pairs"] = line.split("eval_pairs:")[1].strip()

    # 解析训练状态
    if "Starting DPO training" in content:
        info["status"] = "training"
    if "Training completed" in content or "DONE" in content:
        info["status"] = "completed"
    if "ERROR" in content or "Error" in content:
        info["status"] = "error"

    # 解析训练指标
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "'loss':" in line:
            try:
                loss_val = line.split("'loss':")[1].split(",")[0].strip()
                info["training"]["last_loss"] = loss_val
            except:
                pass
        if "'epoch':" in line:
            try:
                epoch_val = line.split("'epoch':")[1].split(",")[0].strip()
                info["training"]["epoch"] = epoch_val
            except:
                pass

    return info


def display_training_status(log_path: Path):
    """显示训练状态"""
    print("\033[2J\033[H")  # 清屏
    print("=" * 70)
    print("DPO 训练监控")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志: {log_path}")
    print()

    info = parse_log_file(log_path)

    if info is None:
        print("⏳ 等待训练开始...")
        print("\n提示: 如果长时间没有响应，请检查:")
        print("  1. 训练脚本是否正在运行")
        print("  2. 日志文件路径是否正确")
        return

    # 显示配置
    if info["config"]:
        print("📋 配置信息:")
        for key, val in info["config"].items():
            print(f"  {key}: {val}")
        print()

    # 显示数据
    if info["data"]:
        print("📊 数据信息:")
        for key, val in info["data"].items():
            print(f"  {key}: {val}")
        print()

    # 显示训练状态
    status_emoji = {
        "training": "🔄",
        "completed": "✅",
        "error": "❌",
        "unknown": "❓"
    }

    print(f"{status_emoji.get(info['status'], '❓')} 状态: {info['status']}")

    if info["training"]:
        print("\n📈 训练进度:")
        for key, val in info["training"].items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("按 Ctrl+C 退出监控")


def monitor_training(log_path: Path, refresh_interval: int = 5):
    """持续监控训练"""
    try:
        while True:
            display_training_status(log_path)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


def main():
    import sys

    # 默认日志路径
    default_log = Path("dpo_train_log.txt")

    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = default_log

    print(f"监控日志文件: {log_path}")
    print("刷新间隔: 5秒")
    print()

    monitor_training(log_path, refresh_interval=5)


if __name__ == "__main__":
    main()
