# DPO Pairwise 训练指南 - 低显存优化
# 适用于 4060 Laptop 8GB 显存

## 当前状态确认

你的 `11_2_dpo_train.py` 脚本**已经是 pairwise DPO**：
- 使用 `DPOTrainer` 进行偏好对训练
- 构建 chosen/rejected 配对数据
- 每个用户的正样本（好物品）vs 负样本（差物品）

## 显存优化策略

### 方案1：标准低显存配置 (推荐先试这个)

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
set QLORA_USE_4BIT=true
set QLORA_GRADIENT_CHECKPOINTING=true
set BUCKETS_OVERRIDE=10

python scripts/11_2_dpo_train.py
```

或者直接加载配置文件：
```bash
# 加载环境变量
for /f "delims=" %i in (config\dpo_low_memory.env) do set %i
python scripts/11_2_dpo_train.py
```

### 方案2：极限低显存配置 (如果方案1仍然OOM)

```bash
# 使用更激进的优化
for /f "delims=" %i in (config\dpo_ultra_low_memory.env) do set %i
python scripts/11_2_dpo_train.py
```

## 关键优化参数说明

| 参数 | 标准值 | 低显存值 | 极限值 | 影响 |
|------|--------|----------|--------|------|
| `QLORA_MAX_SEQ_LEN` | 768 | 512 | 384 | 序列长度，影响最大 |
| `QLORA_LORA_R` | 16 | 8 | 4 | LoRA rank，降低参数量 |
| `QLORA_BATCH_SIZE` | 1 | 1 | 1 | 已经最小 |
| `QLORA_GRAD_ACC` | 8 | 16 | 32 | 补偿小batch |
| `QLORA_DPO_MAX_PAIRS` | 8 | 4 | 2 | 每用户配对数 |
| `QLORA_DPO_MAX_PROMPT_LENGTH` | 512 | 384 | 256 | Prompt最大长度 |
| `QLORA_TARGET_MODULES` | 全部 | 全部 | 仅attention | 训练的层数 |

## Pairwise vs Pointwise 对比

### Pointwise (原11_2_qlora_train.py)
- 每个样本独立：item → YES/NO
- 显存需求：较低
- 训练目标：二分类交叉熵

### Pairwise (当前11_2_dpo_train.py) ✓
- 成对比较：good_item vs bad_item
- 显存需求：**约2倍** (同时处理chosen和rejected)
- 训练目标：DPO偏好优化
- 优势：更好的排序能力

## 显存估算

对于 Qwen3-4B + 4-bit 量化：
- 基础模型：~2.5GB
- LoRA参数 (r=8)：~50MB
- 激活值 (seq_len=512, batch=1)：~3-4GB
- 梯度和优化器状态：~1-2GB
- **总计：约7-8GB** (刚好适合8GB显存)

## 监控和调试

### 1. 监控显存使用
```python
# 在训练开始后查看显存
import torch
print(f"显存已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"显存缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

### 2. 如果遇到 OOM (Out of Memory)

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决步骤**：
1. 降低 `QLORA_MAX_SEQ_LEN` (512 → 384 → 256)
2. 降低 `QLORA_LORA_R` (8 → 4)
3. 降低 `QLORA_DPO_MAX_PAIRS` (4 → 2)
4. 减少 target_modules (只训练 attention)
5. 考虑使用更小的模型 (如 Qwen3-1.5B)

### 3. 如果遇到 Windows Error 1455

**症状**：
```
OSError: [WinError 1455] 页面文件太小，无法完成操作
```

**解决**：
1. 增加虚拟内存（页面文件）到 40GB+
2. 系统设置 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存
3. 重启电脑

## 训练时间估算

对于 8GB 显存配置：
- 数据对数：~1000-2000 pairs
- Batch size: 1, Grad acc: 16
- 有效 batch: 16
- 1 epoch: 约 30-60 分钟 (取决于数据量)

## 验证训练效果

训练完成后：
```bash
# 使用现有的评估脚本
set INPUT_11_2_RUN_DIR=<你的训练输出目录>
python scripts/11_3_qlora_sidecar_eval.py
```

## 进一步优化建议

如果训练速度太慢：
1. 减少 `QLORA_DPO_MAX_PAIRS` (减少训练数据量)
2. 减少 `QLORA_EPOCHS` (0.5 epoch 也可能有效)
3. 只训练单个 bucket (BUCKETS_OVERRIDE=10)

如果效果不好：
1. 增加 `QLORA_DPO_BETA` (0.1 → 0.2) 增强偏好信号
2. 设置 `QLORA_DPO_FILTER_INVERTED=true` 过滤反转对
3. 增加训练数据 (QLORA_DPO_MAX_PAIRS=8)
