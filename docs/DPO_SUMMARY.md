# DPO Pairwise 训练总结

## 核心发现

你的 `scripts/11_2_dpo_train.py` **已经是 pairwise DPO 实现**，不需要修改！

## 为你的 4060 Laptop (8GB 显存) 创建的资源

### 📁 配置文件
1. **config/dpo_low_memory.env** - 推荐配置（7.5GB 显存）
2. **config/dpo_ultra_low_memory.env** - 极限配置（6GB 显存）

### 🛠️ 工具脚本
1. **scripts/run_dpo_low_memory.bat** - 一键启动训练
2. **tools/check_memory.py** - 显存和系统检查
3. **tools/check_dpo_dataset.py** - 数据集验证

### 📚 文档
1. **docs/DPO_COMPLETE_GUIDE.md** - 完整指南（问题排查、优化技巧）
2. **docs/DPO_LOW_MEMORY_GUIDE.md** - 低显存优化指南
3. **docs/DPO_QUICK_REFERENCE.md** - 快速参考卡片

## 快速开始（3步）

```bash
# 1. 检查环境
python tools/check_memory.py

# 2. 检查数据集（可选）
python tools/check_dpo_dataset.py

# 3. 启动训练
scripts\run_dpo_low_memory.bat
# 选择 [1] Standard Low Memory
```

## 推荐配置（8GB 显存）

```bash
QLORA_MAX_SEQ_LEN=512          # 序列长度
QLORA_LORA_R=8                 # LoRA rank
QLORA_BATCH_SIZE=1             # Batch size
QLORA_GRAD_ACC=16              # 梯度累积
QLORA_DPO_MAX_PAIRS=4          # 每用户配对数
QLORA_DPO_BETA=0.1             # DPO 强度
```

## 关键优化参数（按影响排序）

1. **QLORA_MAX_SEQ_LEN** ⭐⭐⭐ (最大影响)
   - 768 → 512 → 384

2. **QLORA_LORA_R** ⭐⭐ (中等影响)
   - 16 → 8 → 4

3. **QLORA_DPO_MAX_PAIRS** ⭐ (小影响)
   - 8 → 4 → 2

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | 降低 MAX_SEQ_LEN (512→384) |
| Windows Error 1455 | 增加虚拟内存到 40GB+ |
| 训练太慢 | 减少 DPO_MAX_PAIRS (4→2) |
| 无法生成配对 | 检查数据集（每用户需正负样本） |
| 效果不好 | 增加 DPO_BETA (0.1→0.2) |

## Pairwise vs Pointwise

| 特性 | Pointwise | Pairwise (当前) |
|------|-----------|----------------|
| 训练方式 | 独立分类 | 成对比较 ✓ |
| 显存需求 | 低 | 高 (~2倍) |
| 排序能力 | 一般 | 更好 ✓ |
| 适用场景 | 分类任务 | 推荐排序 ✓ |

## 显存估算

- **基础模型** (Qwen3-4B 4-bit): ~2.5GB
- **LoRA 参数** (r=8): ~50MB
- **激活值** (seq=512, batch=1): ~3-4GB
- **梯度和优化器**: ~1-2GB
- **总计**: ~7-8GB ✓ 适合你的 8GB 显存

## 训练时间估算

- **数据量**: 1000-2000 配对
- **配置**: 低显存模式
- **预估时间**: 45-60 分钟/轮

## 下一步

训练完成后运行评估：

```bash
set INPUT_11_2_RUN_DIR=<你的训练输出目录>
python scripts\11_3_qlora_sidecar_eval.py
```

## 文件清单

```
config/
  ├── dpo_low_memory.env          ✓ 低显存配置
  └── dpo_ultra_low_memory.env    ✓ 极限配置

scripts/
  ├── 11_2_dpo_train.py           ✓ DPO训练脚本（已是pairwise）
  └── run_dpo_low_memory.bat      ✓ 启动脚本

tools/
  ├── check_memory.py             ✓ 显存检查
  └── check_dpo_dataset.py        ✓ 数据集检查

docs/
  ├── DPO_COMPLETE_GUIDE.md       ✓ 完整指南
  ├── DPO_LOW_MEMORY_GUIDE.md     ✓ 低显存指南
  ├── DPO_QUICK_REFERENCE.md      ✓ 快速参考
  └── DPO_SUMMARY.md              ✓ 本文件
```

---

**准备就绪！现在可以开始训练了。** 🚀
