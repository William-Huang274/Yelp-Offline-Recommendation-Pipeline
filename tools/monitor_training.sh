#!/bin/bash
# DPO 训练监控脚本

echo "=========================================="
echo "DPO 训练进程监控"
echo "=========================================="
echo ""

# 1. 查看训练进程
echo "1. 训练进程状态:"
ps -ef | grep "11_2_dpo_train.py" | grep -v grep
if [ $? -eq 0 ]; then
    echo "   ✓ 训练进程正在运行"
else
    echo "   ✗ 训练进程未运行"
fi
echo ""

# 2. 查看最新日志
echo "2. 最新训练日志 (最后 30 行):"
echo "----------------------------------------"
if [ -f "/d/5006_BDA_project/dpo_train_log.txt" ]; then
    tail -30 /d/5006_BDA_project/dpo_train_log.txt
else
    echo "   日志文件不存在"
fi
echo ""

# 3. 查看训练进度
echo "3. 训练进度:"
echo "----------------------------------------"
if [ -f "/d/5006_BDA_project/dpo_train_log.txt" ]; then
    # 查找 Step 信息
    grep -E "Step|loss|epoch" /d/5006_BDA_project/dpo_train_log.txt | tail -10

    # 查找 TRAIN 或 DONE 标记
    if grep -q "DONE" /d/5006_BDA_project/dpo_train_log.txt; then
        echo ""
        echo "   ✓ 训练已完成！"
    elif grep -q "TRAIN" /d/5006_BDA_project/dpo_train_log.txt; then
        echo ""
        echo "   🔄 训练进行中..."
    else
        echo ""
        echo "   ⏳ 准备阶段..."
    fi
fi
echo ""

# 4. GPU 使用情况
echo "4. GPU 使用情况:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "   无法获取 GPU 信息"
echo ""

echo "=========================================="
echo "监控完成"
echo "=========================================="
