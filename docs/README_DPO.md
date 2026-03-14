# DPO Pairwise 训练 - README

## 🎯 核心结论

**你的 `scripts/11_2_dpo_train.py` 已经是 pairwise DPO 实现！**

不需要修改代码，只需要针对你的 **4060 Laptop 8GB 显存** 优化配置即可。

---

## 🚀 快速开始（3步）

### 1️⃣ 检查环境

```bash
# 运行完整环境检查
scripts\check_dpo_env.bat

# 或单独检查显存
python tools\check_memory.py
```

### 2️⃣ 启动训练

```bash
# 使用启动脚本（推荐）
scripts\run_dpo_low_memory.bat
# 选择 [1] Standard Low Memory

# 或手动设置
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_MAX_PAIRS=4
python scripts\11_2_dpo_train.py
```

### 3️⃣ 监控训练

```bash
# 实时监控（可选）
python tools\monitor_dpo_training.py

# 查看日志
type dpo_train_log.txt

# 查看 GPU
nvidia-smi
```

---

## 📦 已创建的资源

### 配置文件
- `config/dpo_low_memory.env` - 推荐配置（8GB 显存）
- `config/dpo_ultra_low_memory.env` - 极限配置（6GB 显存）

### 启动脚本
- `scripts/run_dpo_low_memory.bat` - 训练启动器
- `scripts/check_dpo_env.bat` - 环境检查器

### 工具
- `tools/check_memory.py` - 显存和系统检查
- `tools/check_dpo_dataset.py` - 数据集验证
- `tools/monitor_dpo_training.py` - 训练监控

### 文档
- `docs/DPO_SUMMARY.md` - 总结（推荐先看）
- `docs/DPO_QUICK_REFERENCE.md` - 快速参考
- `docs/DPO_COMPLETE_GUIDE.md` - 完整指南
- `docs/DPO_LOW_MEMORY_GUIDE.md` - 低显存指南
- `docs/README_DPO.md` - 本文件

---

## ⚙️ 推荐配置（8GB 显存）

```bash
QLORA_MAX_SEQ_LEN=512          # 序列长度（影响最大）
QLORA_LORA_R=8                 # LoRA rank
QLORA_LORA_ALPHA=16            # LoRA alpha
QLORA_BATCH_SIZE=1             # Batch size
QLORA_GRAD_ACC=16              # 梯度累积
QLORA_DPO_MAX_PAIRS=4          # 每用户配对数
QLORA_DPO_BETA=0.1             # DPO 强度
QLORA_USE_4BIT=true            # 4-bit 量化
QLORA_GRADIENT_CHECKPOINTING=true  # 梯度检查点
```

**预估显存使用**: ~7.5GB
**预估训练时间**: 45-60 分钟/轮

---

## 📊 配置对比

| 配置 | 显存 | 序列长度 | LoRA R | 配对数 | 训练时间 |
|------|------|---------|--------|--------|---------|
| 标准 | 9.5GB | 768 | 16 | 8 | 30-45分钟 |
| **低显存** ✓ | **7.5GB** | **512** | **8** | **4** | **45-60分钟** |
| 极限 | 6.0GB | 384 | 4 | 2 | 60-90分钟 |

---

## ❓ 常见问题

### Q1: 什么是 Pairwise DPO？

**A**: DPO (Direct Preference Optimization) 是基于偏好学习的训练方法：
- **Pointwise**: 每个样本独立 → `item → YES/NO`
- **Pairwise**: 成对比较 → `good_item vs bad_item` ✓

你的脚本使用 pairwise，更适合推荐排序任务。

### Q2: 为什么 Pairwise 需要更多显存？

**A**: Pairwise 需要同时处理 chosen 和 rejected 两个样本，显存需求约为 pointwise 的 2 倍。

### Q3: 如果遇到 CUDA Out of Memory？

**A**: 按优先级尝试：
1. 降低 `QLORA_MAX_SEQ_LEN` (512→384)
2. 降低 `QLORA_LORA_R` (8→4)
3. 减少 `QLORA_DPO_MAX_PAIRS` (4→2)
4. 使用极限配置

### Q4: 如果遇到 Windows Error 1455？

**A**: 增加虚拟内存（页面文件）到 40GB+：
- 系统设置 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存
- 设置自定义大小：初始 40960MB，最大 81920MB
- 重启电脑

### Q5: 训练太慢怎么办？

**A**:
1. 减少 `QLORA_DPO_MAX_PAIRS` (4→2)
2. 减少 `QLORA_EPOCHS` (1.0→0.5)
3. 只训练单个 bucket (`BUCKETS_OVERRIDE=10`)

### Q6: 如何验证是否是 Pairwise？

**A**: 查看脚本第 286-287 行：
```python
"chosen": pos_prompt + " YES",
"rejected": neg_prompt + " YES",
```
这就是 pairwise 的标志：chosen 和 rejected 配对。

---

## 🔧 关键参数说明

### 显存优化参数（按影响排序）

1. **QLORA_MAX_SEQ_LEN** ⭐⭐⭐
   - 影响：最大
   - 推荐：512（标准）→ 384（极限）

2. **QLORA_LORA_R** ⭐⭐
   - 影响：中等
   - 推荐：8（标准）→ 4（极限）

3. **QLORA_DPO_MAX_PAIRS** ⭐
   - 影响：较小
   - 推荐：4（标准）→ 2（极限）

### DPO 核心参数

- **QLORA_DPO_BETA**: DPO 强度（默认 0.1）
  - 越大，偏好信号越强
  - 建议范围：0.05 - 0.3

- **QLORA_DPO_LOSS_TYPE**: 损失函数（默认 sigmoid）
  - 选项：sigmoid, ipo, hinge
  - 推荐：sigmoid

- **QLORA_DPO_PREFER_EASY_NEG**: 优先简单负样本（默认 true）
  - true: 优先使用容易区分的负样本
  - false: 随机选择负样本

- **QLORA_DPO_FILTER_INVERTED**: 过滤反转配对（默认 false）
  - true: 过滤负样本分数高于正样本的配对
  - false: 保留所有配对

---

## 📈 训练流程

```
1. 环境检查
   ├─ 显存检查 (check_memory.py)
   ├─ 数据集检查 (check_dpo_dataset.py)
   └─ 依赖检查 (check_dpo_env.bat)

2. 启动训练
   ├─ 使用启动脚本 (run_dpo_low_memory.bat)
   └─ 或手动设置环境变量

3. 监控训练
   ├─ 实时监控 (monitor_dpo_training.py)
   ├─ 查看日志 (dpo_train_log.txt)
   └─ 查看 GPU (nvidia-smi)

4. 训练完成
   ├─ 检查输出 (data/output/11_qlora_models/)
   └─ 运行评估 (11_3_qlora_sidecar_eval.py)
```

---

## 📝 训练后评估

```bash
# 设置训练输出目录
set INPUT_11_2_RUN_DIR=data\output\11_qlora_models\<你的运行目录>

# 运行评估
python scripts\11_3_qlora_sidecar_eval.py
```

---

## 🔗 相关命令

```bash
# 查看 GPU 状态
nvidia-smi

# 查看训练日志
type dpo_train_log.txt

# 查看输出目录
dir data\output\11_qlora_models\

# 清理显存（如果需要）
taskkill /F /IM python.exe

# 查看环境检查报告
type dpo_env_check.txt
```

---

## 📚 文档索引

- **新手**: 先看 `DPO_SUMMARY.md`
- **快速参考**: 看 `DPO_QUICK_REFERENCE.md`
- **详细指南**: 看 `DPO_COMPLETE_GUIDE.md`
- **问题排查**: 看 `DPO_COMPLETE_GUIDE.md` 的问题排查章节

---

## ✅ 检查清单

训练前确认：
- [ ] 显存 ≥ 6GB（推荐 8GB）
- [ ] 虚拟内存 ≥ 40GB
- [ ] 磁盘空间 ≥ 10GB
- [ ] 已安装 TRL (pip install trl>=0.9)
- [ ] 数据集有正负样本配对
- [ ] 已选择合适的配置

---

## 🎓 Pairwise vs Pointwise 对比

| 维度 | Pointwise | Pairwise (当前) |
|------|-----------|----------------|
| **训练方式** | 独立分类 | 成对比较 ✓ |
| **显存需求** | 低 (~4GB) | 高 (~8GB) |
| **训练时间** | 快 | 慢 |
| **排序能力** | 一般 | 更好 ✓ |
| **适用场景** | 分类任务 | 推荐排序 ✓ |
| **数据要求** | 标注样本 | 正负配对 |

**结论**: 对于推荐系统，pairwise 更合适！

---

## 💡 优化建议

### 如果显存不够
1. 使用极限配置
2. 降低序列长度到 256
3. 只训练 attention 层
4. 考虑更小的模型 (Qwen3-1.5B)

### 如果效果不好
1. 增加 DPO beta (0.1→0.2)
2. 启用 filter_inverted
3. 增加训练数据
4. 增加训练轮数

### 如果训练太慢
1. 减少配对数
2. 减少训练轮数
3. 只训练单个 bucket
4. 使用更大的 batch (如果显存允许)

---

## 🆘 获取帮助

如果遇到问题：
1. 查看 `docs/DPO_COMPLETE_GUIDE.md` 的问题排查章节
2. 运行 `python tools/check_memory.py` 检查环境
3. 查看训练日志 `dpo_train_log.txt`
4. 检查数据集 `python tools/check_dpo_dataset.py`

---

**准备就绪！祝训练顺利！** 🚀
