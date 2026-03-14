# DPO Semantic Mode 使用指南

## 🎯 重要发现

你的数据集构建脚本有一个 **`QLORA_PROMPT_MODE`** 开关！

### 两种模式

**1. `full` 模式 (默认)**
```
包含: 用户信息 + 物品信息 + 排序特征
排序特征: als_rank, cluster_rank, profile_rank, popular_rank,
         candidate_sources, user_segment
```

**2. `semantic` 模式 (推荐用于 DPO)**
```
包含: 用户信息 + 物品信息 (仅语义特征)
移除: 所有排序特征
```

## 🤔 为什么要用 semantic 模式？

### 问题：排序特征在 DPO 中会产生矛盾

**场景**：
- Positive item: als_rank=50 (排名较低)
- Negative item: als_rank=10 (排名较高，hard negative)

**矛盾**：
- DPO 想让模型学习：positive > negative
- 但 prompt 中显示：negative 的 rank 更高
- 模型会困惑：到底该相信 rank 还是相信标签？

### 解决：semantic 模式

移除排序特征后：
- ✓ 只保留语义/内容特征
- ✓ 避免矛盾信号
- ✓ DPO 专注于学习语义匹配
- ✓ Prompt 更短，节省显存

## 📊 Prompt 长度对比

### 当前分析结果 (full 模式)
```
平均长度: 432 tokens
P95: 477 tokens
P99: 525 tokens
```

### 预估 semantic 模式
```
预估平均: ~380-400 tokens (减少 30-50 tokens)
预估 P95: ~420-450 tokens
预估 P99: ~470-500 tokens
```

**好处**：
- 512 的覆盖率会从 98.7% 提升到 99%+
- 640 几乎能覆盖 100%
- 显存压力更小

## 🚀 如何使用

### 方案 1: 重新生成数据集 (推荐)

```bash
# 设置 semantic 模式
set QLORA_PROMPT_MODE=semantic

# 重新运行数据集构建
python scripts\11_1_qlora_build_dataset.py
```

**优点**：
- 数据集本身就是 semantic 模式
- Prompt 更短，更适合 DPO
- 避免排序特征矛盾

**缺点**：
- 需要重新生成数据集（需要时间）

---

### 方案 2: 训练时动态移除 (已支持)

DPO 训练脚本已经支持动态移除排序特征：

```bash
# 在 DPO 训练时设置
set QLORA_DPO_STRIP_RANK_FEATURES=true

# 使用现有数据集训练
python scripts\11_2_dpo_train.py
```

**优点**：
- 不需要重新生成数据集
- 立即可用

**缺点**：
- 使用正则表达式移除，可能不如原生 semantic 模式干净
- Prompt 长度统计仍然基于 full 模式

---

## 💡 最终建议

### 推荐流程

**第一步：先用现有数据集 + 动态移除**

```bash
# 使用配置文件
for /f "delims=" %i in (config\dpo_semantic_mode.env) do set %i

# 或手动设置
set QLORA_DPO_STRIP_RANK_FEATURES=true
set QLORA_MAX_SEQ_LEN=512

# 训练
python scripts\11_2_dpo_train.py
```

**第二步：如果效果好，考虑重新生成数据集**

```bash
# 重新生成 semantic 模式数据集
set QLORA_PROMPT_MODE=semantic
python scripts\11_1_qlora_build_dataset.py

# 然后用新数据集训练
python scripts\11_2_dpo_train.py
```

---

## 📋 配置文件

已创建 `config/dpo_semantic_mode.env`：
- 设置 `QLORA_DPO_STRIP_RANK_FEATURES=true`
- MAX_SEQ_LEN=640 (因为 prompt 会更短)
- 其他参数优化

---

## 🔍 验证效果

### 检查 prompt 是否移除了排序特征

训练开始后，查看日志中的 prompt 样本，确认是否包含：
- ❌ `als_rank:`
- ❌ `cluster_rank:`
- ❌ `profile_rank:`
- ❌ `popular_rank:`
- ❌ `candidate_sources:`
- ❌ `user_segment:`

如果这些都不存在，说明成功移除了排序特征。

---

## 📈 预期改进

使用 semantic 模式后：

1. **Prompt 更短**
   - 减少 30-50 tokens
   - 512 覆盖率 > 99%

2. **DPO 效果更好**
   - 避免排序特征矛盾
   - 专注语义匹配

3. **显存更充裕**
   - 可以用 640 而不担心 OOM
   - 或者用 512 + 更大的 LoRA rank

---

## 🎯 总结

**立即可用**：
```bash
# 使用现有数据集 + 动态移除排序特征
set QLORA_DPO_STRIP_RANK_FEATURES=true
set QLORA_MAX_SEQ_LEN=512
python scripts\11_2_dpo_train.py
```

**长期优化**：
```bash
# 重新生成 semantic 模式数据集
set QLORA_PROMPT_MODE=semantic
python scripts\11_1_qlora_build_dataset.py
```

两种方式都能达到移除排序特征的目的，推荐先用第一种快速验证效果！
