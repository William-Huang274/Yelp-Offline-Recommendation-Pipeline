# DPO Pairwise 训练快速参考

## 🎯 核心要点

你的脚本 **已经是 pairwise DPO**，不需要修改！

## 🚀 快速开始

### 方法1: 使用启动脚本（推荐）
```bash
# 运行显存检查
python tools\check_memory.py

# 启动训练（会提示选择配置）
scripts\run_dpo_low_memory.bat
```

### 方法2: 手动设置环境变量
```bash
# 低显存配置
set QLORA_MAX_SEQ_LEN=512
set QLORA_LORA_R=8
set QLORA_BATCH_SIZE=1
set QLORA_GRAD_ACC=16
set QLORA_DPO_MAX_PAIRS=4

python scripts\11_2_dpo_train.py
```

## 📊 配置对比表

| 配置 | 显存需求 | 序列长度 | LoRA Rank | 训练速度 | 适用场景 |
|------|---------|---------|-----------|---------|---------|
| 标准 | ~9.5GB | 768 | 16 | 快 | 12GB+ 显存 |
| 低显存 | ~7.5GB | 512 | 8 | 中等 | 8GB 显存 ✓ |
| 极限 | ~6.0GB | 384 | 4 | 慢 | 6GB 显存 |

## 🔧 关键参数速查

```bash
# 显存优化（按影响程度排序）
QLORA_MAX_SEQ_LEN=512          # ⭐⭐⭐ 最大影响
QLORA_LORA_R=8                 # ⭐⭐ 中等影响
QLORA_DPO_MAX_PAIRS=4          # ⭐ 小影响
QLORA_DPO_MAX_PROMPT_LENGTH=384

# DPO 核心参数
QLORA_DPO_BETA=0.1             # DPO 强度
QLORA_DPO_LOSS_TYPE=sigmoid    # 损失函数
QLORA_DPO_PREFER_EASY_NEG=true # 优先简单负样本

# 训练参数
QLORA_BATCH_SIZE=1             # 保持为1
QLORA_GRAD_ACC=16              # 补偿小batch
QLORA_EPOCHS=1.0
QLORA_LR=5e-5
```

## ⚠️ 常见问题

### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**解决**：
- 降低 `QLORA_MAX_SEQ_LEN` (512→384→256)
- 降低 `QLORA_LORA_R` (8→4)
- 使用极限配置

### 2. Windows Error 1455
```
OSError: [WinError 1455] 页面文件太小
```
**解决**：
- 增加虚拟内存到 40GB+
- 系统设置 → 高级 → 虚拟内存
- 重启电脑

### 3. 训练太慢
**优化**：
- 减少 `QLORA_DPO_MAX_PAIRS` (4→2)
- 减少 `QLORA_EPOCHS` (1.0→0.5)
- 只训练单个 bucket

## 📁 文件位置

```
config/
  ├── dpo_low_memory.env          # 低显存配置
  └── dpo_ultra_low_memory.env    # 极限配置

scripts/
  ├── 11_2_dpo_train.py           # DPO训练脚本（pairwise）
  └── run_dpo_low_memory.bat      # 启动脚本

tools/
  └── check_memory.py             # 显存检查工具

docs/
  └── DPO_LOW_MEMORY_GUIDE.md     # 详细指南
```

## 🎓 Pairwise vs Pointwise

| 特性 | Pointwise | Pairwise (当前) |
|------|-----------|----------------|
| 训练方式 | 独立分类 | 成对比较 |
| 显存需求 | 低 | 高 (~2倍) |
| 排序能力 | 一般 | 更好 ✓ |
| 训练目标 | 交叉熵 | DPO偏好 |

## 📈 训练流程

1. **检查环境**
   ```bash
   python tools\check_memory.py
   ```

2. **启动训练**
   ```bash
   scripts\run_dpo_low_memory.bat
   ```

3. **监控训练**
   - 查看日志: `dpo_train_log.txt`
   - 输出目录: `data\output\11_qlora_models\`

4. **评估模型**
   ```bash
   set INPUT_11_2_RUN_DIR=<训练输出目录>
   python scripts\11_3_qlora_sidecar_eval.py
   ```

## 💡 优化建议

### 如果显存不够
1. 使用极限配置
2. 减少序列长度
3. 只训练 attention 层
4. 考虑更小的模型

### 如果效果不好
1. 增加 `QLORA_DPO_BETA` (0.1→0.2)
2. 启用 `QLORA_DPO_FILTER_INVERTED=true`
3. 增加训练数据量
4. 增加训练轮数

### 如果训练太慢
1. 减少每用户配对数
2. 减少训练轮数
3. 只训练单个 bucket
4. 使用更大的 batch (如果显存允许)

## 🔗 相关命令

```bash
# 查看 GPU 状态
nvidia-smi

# 查看训练进度
type dpo_train_log.txt

# 清理显存
taskkill /F /IM python.exe

# 查看输出目录
dir data\output\11_qlora_models\
```
