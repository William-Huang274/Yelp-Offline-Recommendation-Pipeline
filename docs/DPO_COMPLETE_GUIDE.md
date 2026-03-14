# DPO Pairwise 训练完整指南
# 适用于 4060 Laptop 8GB 显存

## 📋 目录

1. [核心概念](#核心概念)
2. [环境准备](#环境准备)
3. [配置选择](#配置选择)
4. [训练流程](#训练流程)
5. [问题排查](#问题排查)
6. [性能优化](#性能优化)

---

## 核心概念

### 什么是 Pairwise DPO？

**DPO (Direct Preference Optimization)** 是一种基于偏好学习的训练方法：

- **Pointwise**: 每个样本独立训练 → `item → YES/NO`
- **Pairwise**: 成对比较训练 → `good_item vs bad_item`

### 你的脚本已经是 Pairwise！

`scripts/11_2_dpo_train.py` 已经实现了 pairwise DPO：
- 使用 `DPOTrainer`
- 构建 chosen/rejected 配对
- 每个用户的正样本 vs 负样本

### Pairwise 的优势

✓ 更好的排序能力
✓ 更符合推荐场景（比较而非分类）
✓ 更稳定的训练过程

### Pairwise 的挑战

✗ 显存需求约 2 倍（同时处理 chosen 和 rejected）
✗ 训练时间更长
✗ 需要每个用户都有正负样本

---

## 环境准备

### 1. 检查硬件

```bash
# 运行显存检查工具
python tools/check_memory.py
```

**最低要求**：
- GPU: 6GB+ 显存
- RAM: 16GB+ 内存
- 磁盘: 10GB+ 可用空间
- 虚拟内存: 40GB+ (避免 Windows Error 1455)

### 2. 检查数据集

```bash
# 检查数据集是否适合 DPO
python tools/check_dpo_dataset.py
```

**数据集要求**：
- 每个用户需要有正样本和负样本
- 至少 30% 用户可生成配对
- 预估配对数 > 500

### 3. 安装依赖

```bash
# 确保安装了 TRL
pip install trl>=0.9

# 其他依赖
pip install transformers datasets peft bitsandbytes
```

---

## 配置选择

### 配置对比

| 配置 | 显存 | 序列长度 | LoRA R | 配对数 | 训练时间 | 适用场景 |
|------|------|---------|--------|--------|---------|---------|
| **标准** | 9.5GB | 768 | 16 | 8 | 30-45分钟 | 12GB+ 显存 |
| **低显存** | 7.5GB | 512 | 8 | 4 | 45-60分钟 | 8GB 显存 ✓ |
| **极限** | 6.0GB | 384 | 4 | 2 | 60-90分钟 | 6GB 显存 |

### 推荐配置：低显存模式

对于你的 **4060 Laptop 8GB 显存**，推荐使用**低显存配置**：

```bash
QLORA_MAX_SEQ_LEN=512
QLORA_LORA_R=8
QLORA_LORA_ALPHA=16
QLORA_BATCH_SIZE=1
QLORA_GRAD_ACC=16
QLORA_DPO_MAX_PAIRS=4
QLORA_DPO_MAX_PROMPT_LENGTH=384
QLORA_DPO_MAX_TARGET_LENGTH=8
```

---

## 训练流程

### 方法 1: 使用启动脚本（推荐）

```bash
# 1. 检查环境
python tools/check_memory.py

# 2. 检查数据集
python tools/check_dpo_dataset.py

# 3. 启动训练
scripts\run_dpo_low_memory.bat
# 选择 [1] Standard Low Memory

# 4. 监控训练
# 查看日志: dpo_train_log.txt
# 查看 GPU: nvidia-smi
```

### 方法 2: 手动设置

```bash
# Windows CMD
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_LORA_ALPHA=16
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=16
set QLORA_DPO_MAX_PAIRS=4
set QLORA_DPO_MAX_PROMPT_LENGTH=384
set QLORA_DPO_MAX_TARGET_LENGTH=8
set QLORA_DPO_BETA=0.1
set QLORA_USE_4BIT=true
set QLORA_GRADIENT_CHECKPOINTING=true
set BUCKETS_OVERRIDE=10

python scripts\11_2_dpo_train.py 2>&1 | tee dpo_train_log.txt
```

### 方法 3: 使用配置文件

```bash
# 加载配置文件
for /f "delims=" %i in (config\dpo_low_memory.env) do set %i

# 运行训练
python scripts\11_2_dpo_train.py
```

---

## 问题排查

### 问题 1: CUDA Out of Memory

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**（按优先级）：

1. **降低序列长度**（影响最大）
   ```bash
   set QLORA_MAX_SEQ_LEN=384  # 从 512 降到 384
   ```

2. **降低 LoRA rank**
   ```bash
   set QLORA_LORA_R=4  # 从 8 降到 4
   ```

3. **减少配对数**
   ```bash
   set QLORA_DPO_MAX_PAIRS=2  # 从 4 降到 2
   ```

4. **只训练 attention 层**
   ```bash
   set QLORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
   ```

5. **使用极限配置**
   ```bash
   scripts\run_dpo_low_memory.bat
   # 选择 [2] Ultra Low Memory
   ```

### 问题 2: Windows Error 1455

**症状**：
```
OSError: [WinError 1455] 页面文件太小，无法完成操作
```

**解决方案**：

1. 增加虚拟内存（页面文件）：
   - 右键"此电脑" → 属性
   - 高级系统设置 → 性能设置
   - 高级 → 虚拟内存 → 更改
   - 取消"自动管理"
   - 设置自定义大小：初始 40960MB，最大 81920MB
   - 确定并重启

2. 或者使用更小的模型：
   ```bash
   set QLORA_BASE_MODEL=Qwen/Qwen3-1.5B
   ```

### 问题 3: 训练太慢

**症状**：
- 每个 step 超过 5 秒
- 预估训练时间超过 2 小时

**解决方案**：

1. **减少配对数**
   ```bash
   set QLORA_DPO_MAX_PAIRS=2
   ```

2. **减少训练轮数**
   ```bash
   set QLORA_EPOCHS=0.5
   ```

3. **只训练单个 bucket**
   ```bash
   set BUCKETS_OVERRIDE=10
   ```

4. **增加 batch size**（如果显存允许）
   ```bash
   set QLORA_BATCH_SIZE=2
   set QLORA_GRAD_ACC=8
   ```

### 问题 4: 无法生成配对

**症状**：
```
RuntimeError: No DPO pairs could be built
```

**原因**：
- 数据集中没有用户同时有正负样本

**解决方案**：

1. 检查数据集：
   ```bash
   python tools/check_dpo_dataset.py
   ```

2. 重新生成数据集，确保负采样正确

3. 检查 `11_1_qlora_build_dataset.py` 的负采样配置

### 问题 5: 训练效果不好

**症状**：
- 评估指标没有提升
- 模型输出不稳定

**解决方案**：

1. **增加 DPO beta**（增强偏好信号）
   ```bash
   set QLORA_DPO_BETA=0.2  # 从 0.1 增加到 0.2
   ```

2. **过滤反转配对**
   ```bash
   set QLORA_DPO_FILTER_INVERTED=true
   ```

3. **增加训练数据**
   ```bash
   set QLORA_DPO_MAX_PAIRS=8
   ```

4. **增加训练轮数**
   ```bash
   set QLORA_EPOCHS=2.0
   ```

5. **调整学习率**
   ```bash
   set QLORA_LR=1e-4  # 尝试更大的学习率
   ```

---

## 性能优化

### 显存优化技巧

1. **序列长度** (影响最大)
   - 标准: 768 → 低显存: 512 → 极限: 384

2. **LoRA rank** (中等影响)
   - 标准: 16 → 低显存: 8 → 极限: 4

3. **Gradient checkpointing** (必须开启)
   ```bash
   set QLORA_GRADIENT_CHECKPOINTING=true
   ```

4. **4-bit 量化** (必须开启)
   ```bash
   set QLORA_USE_4BIT=true
   ```

5. **减少 target modules**
   ```bash
   # 只训练 attention (节省 ~30% 显存)
   set QLORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
   ```

### 训练速度优化

1. **增加 gradient accumulation**
   ```bash
   set QLORA_GRAD_ACC=32  # 补偿小 batch
   ```

2. **减少 logging 频率**
   ```bash
   set QLORA_LOGGING_STEPS=50
   ```

3. **减少 eval 频率**
   ```bash
   set QLORA_EVAL_STEPS=1000
   ```

4. **使用 bf16**（如果 GPU 支持）
   ```bash
   set QLORA_USE_BF16=true
   ```

### 数据优化

1. **优先简单负样本**
   ```bash
   set QLORA_DPO_PREFER_EASY_NEG=true
   ```

2. **过滤反转配对**
   ```bash
   set QLORA_DPO_FILTER_INVERTED=true
   ```

3. **去除排序特征**（如果不需要）
   ```bash
   set QLORA_DPO_STRIP_RANK_FEATURES=true
   ```

---

## 训练监控

### 实时监控

```bash
# 监控 GPU 使用
nvidia-smi -l 1

# 查看训练日志
type dpo_train_log.txt

# 监控显存（Python）
python -c "import torch; print(f'显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

### 关键指标

- **train_loss**: 应该逐渐下降
- **eval_loss**: 应该低于 train_loss
- **显存使用**: 应该稳定在 7-8GB
- **训练速度**: 每 step 约 2-5 秒

### 训练完成后

```bash
# 查看输出目录
dir data\output\11_qlora_models\

# 运行评估
set INPUT_11_2_RUN_DIR=<训练输出目录>
python scripts\11_3_qlora_sidecar_eval.py
```

---

## 总结

### ✓ 你的脚本已经是 Pairwise DPO

不需要修改代码，只需要优化配置以适应 8GB 显存。

### ✓ 推荐配置

对于 4060 Laptop 8GB 显存，使用**低显存配置**：
- 序列长度: 512
- LoRA rank: 8
- 配对数: 4
- Batch size: 1
- Gradient accumulation: 16

### ✓ 快速开始

```bash
# 1. 检查环境
python tools/check_memory.py

# 2. 启动训练
scripts\run_dpo_low_memory.bat

# 3. 选择 [1] Standard Low Memory
```

### ✓ 如果遇到问题

1. OOM → 降低序列长度
2. Error 1455 → 增加虚拟内存
3. 太慢 → 减少配对数
4. 效果差 → 增加 DPO beta

---

## 相关文件

- `config/dpo_low_memory.env` - 低显存配置
- `config/dpo_ultra_low_memory.env` - 极限配置
- `scripts/run_dpo_low_memory.bat` - 启动脚本
- `tools/check_memory.py` - 显存检查
- `tools/check_dpo_dataset.py` - 数据集检查
- `docs/DPO_QUICK_REFERENCE.md` - 快速参考

---

**祝训练顺利！** 🚀
