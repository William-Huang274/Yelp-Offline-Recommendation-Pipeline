# DPO 训练最终推荐方案

## 🎯 核心发现总结

1. ✅ **你的脚本已经是 pairwise DPO**
2. ✅ **Prompt 包含完整信息**（含 review）
3. ✅ **有 semantic 模式开关**可以移除排序特征
4. ✅ **实际数据分析**：平均 432 tokens, P95=477

## 🚀 最终推荐方案

### 方案 A: 快速开始（推荐）

**使用现有数据集 + 动态移除排序特征**

```bash
# 1. 设置环境变量
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_MAX_PAIRS=4
set QLORA_DPO_STRIP_RANK_FEATURES=true
set QLORA_USE_4BIT=true
set QLORA_GRADIENT_CHECKPOINTING=true

# 2. 启动训练
python scripts\11_2_dpo_train.py
```

**优点**：
- ✓ 立即可用，不需要重新生成数据集
- ✓ 移除排序特征，避免 DPO 矛盾
- ✓ 512 覆盖 98.7%，显存安全 (~7.5GB)

**适合**：
- 想快速开始训练
- 不想等待数据集重新生成

---

### 方案 B: 最优方案（长期）

**重新生成 semantic 模式数据集**

```bash
# 1. 重新生成数据集
set QLORA_PROMPT_MODE=semantic
python scripts\11_1_qlora_build_dataset.py

# 2. 分析新数据集长度
python tools\analyze_prompt_length.py

# 3. 训练（预计可以用更大的 MAX_SEQ_LEN）
set QLORA_MAX_SEQ_LEN=640
set QLORA_LORA_R=8
python scripts\11_2_dpo_train.py
```

**优点**：
- ✓ 数据集原生 semantic 模式，更干净
- ✓ Prompt 更短（预计 ~380-400 tokens）
- ✓ 可以用 640 覆盖 99.9%+

**适合**：
- 有时间重新生成数据集
- 追求最优效果

---

## 📊 配置对比

| 方案 | 数据集 | MAX_SEQ_LEN | 覆盖率 | 显存 | 时间成本 |
|------|--------|-------------|--------|------|---------|
| **A (推荐)** | 现有 | 512 | 98.7% | 7.5GB | 0 |
| **B (最优)** | 重新生成 | 640 | 99.9%+ | 8.5GB | 需重新生成 |

---

## 🎯 我的建议

### 第一步：使用方案 A

```bash
# 快速配置
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_STRIP_RANK_FEATURES=true
set QLORA_DPO_MAX_PAIRS=4
set QLORA_USE_4BIT=true
set QLORA_GRADIENT_CHECKPOINTING=true
set QLORA_DPO_BETA=0.1

# 启动训练
python scripts\11_2_dpo_train.py 2>&1 | tee dpo_train_log.txt
```

**理由**：
1. 不需要等待数据集重新生成
2. 98.7% 覆盖率已经很好
3. 动态移除排序特征效果应该不错
4. 显存安全

### 第二步：评估效果

```bash
set INPUT_11_2_RUN_DIR=<训练输出目录>
python scripts\11_3_qlora_sidecar_eval.py
```

### 第三步：如果效果好，考虑方案 B

如果方案 A 效果满意，可以考虑：
- 重新生成 semantic 模式数据集
- 用 640 训练，获得更好的覆盖率

---

## 📁 相关文件

```
config/
  ├── dpo_safe_512.env           # 保守配置
  ├── dpo_optimized_640.env      # 激进配置
  └── dpo_semantic_mode.env      # Semantic 模式配置

docs/
  ├── DPO_FINAL_RECOMMENDATION.md      # 之前的推荐
  ├── DPO_SEMANTIC_MODE_GUIDE.md       # Semantic 模式指南
  └── DPO_ULTIMATE_RECOMMENDATION.md   # 本文件（最终推荐）

tools/
  └── analyze_prompt_length.py   # 长度分析工具
```

---

## ✅ 检查清单

训练前确认：
- [ ] 显存 ≥ 6GB（推荐 8GB）
- [ ] 虚拟内存 ≥ 40GB
- [ ] 已安装 TRL (pip install trl>=0.9)
- [ ] 数据集存在且有正负样本配对
- [ ] 决定使用方案 A 还是方案 B

---

## 🎉 准备就绪

**推荐立即开始**：

```bash
# 方案 A - 快速开始
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_DPO_STRIP_RANK_FEATURES=true
set QLORA_DPO_MAX_PAIRS=4
set QLORA_USE_4BIT=true
set QLORA_GRADIENT_CHECKPOINTING=true

python scripts\11_2_dpo_train.py 2>&1 | tee dpo_train_log.txt
```

**预计训练时间**：45-60 分钟/轮

祝训练顺利！🚀
