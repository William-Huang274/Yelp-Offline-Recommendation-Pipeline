#!/bin/bash
# DPO training monitor script

echo "=========================================="
echo "DPO training process monitor"
echo "=========================================="
echo ""

# 1. Check process state
echo "1. Training process state:"
ps -ef | grep "11_2_dpo_train.py" | grep -v grep
if [ $? -eq 0 ]; then
    echo "   [OK] Training process is running"
else
    echo "   [INFO] Training process is not running"
fi
echo ""

# 2. Show latest log lines
echo "2. Latest training log (last 30 lines):"
echo "----------------------------------------"
if [ -f "/d/5006_BDA_project/dpo_train_log.txt" ]; then
    tail -30 /d/5006_BDA_project/dpo_train_log.txt
else
    echo "   Log file does not exist"
fi
echo ""

# 3. Show training progress
echo "3. Training progress:"
echo "----------------------------------------"
if [ -f "/d/5006_BDA_project/dpo_train_log.txt" ]; then
    # Search for step information
    grep -E "Step|loss|epoch" /d/5006_BDA_project/dpo_train_log.txt | tail -10

    # Search for TRAIN or DONE markers
    if grep -q "DONE" /d/5006_BDA_project/dpo_train_log.txt; then
        echo ""
        echo "   [DONE] Training completed"
    elif grep -q "TRAIN" /d/5006_BDA_project/dpo_train_log.txt; then
        echo ""
        echo "   [RUN] Training is in progress"
    else
        echo ""
        echo "   [INFO] Preparation stage"
    fi
fi
echo ""

# 4. Show GPU usage
echo "4. GPU usage:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "   Unable to query GPU information"
echo ""

echo "=========================================="
echo "Monitoring complete"
echo "=========================================="
