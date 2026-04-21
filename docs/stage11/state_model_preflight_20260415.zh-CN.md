# 用户状态/商家状态前置准备（2026-04-15）

这份记录只回答两个问题：

1. 为什么当前 `35B raw evidence -> full profile` 的 SFT 不应继续作为唯一主线。
2. 在不停止云端当前训练的前提下，本地先把哪些脚本、审计和样本准备好。

## 1. 当前判断

基于远端 `Qwen3.5-35B-A3B` 这轮 `11_2` 的 `step225` 与 `step300`：

- `step225` 是当前最均衡的 checkpoint
- `step300` 的 `loss` 更低，但 `cuisine / meal / scene exact` 没继续提升
- 高频 cuisine / scene 模板偏置仍明显
- `open_discoveries` 开始出现，但还没有转化成稳定的字段级质量增益

所以当前结论不是“大模型没用”，而是：

- 更大模型对格式和部分字段有帮助
- 但把“用户状态学习”和“结构化解释”压在一条 SFT 上，已经开始碰到上限

## 2. 这轮本地先准备的产物

### 2.1 商家状态审计与 prompt 样本

脚本：

- [09_merchant_state_v2_audit_sample.py](D:\5006_BDA_project\scripts\09_merchant_state_v2_audit_sample.py)

作用：

- 审计当前 `merchant_persona_schema_v1` 是否足够支撑下一步 `merchant_state_v2`
- 接入 `photo summary` 做稀疏商家补强覆盖检查
- 输出 prompt-only 审计样本，而不是直接开 merchant SFT

当前本地输出：

- `data/output/09_merchant_state_v2_audit/.../merchant_state_v2_audit_summary.json`
- `data/output/09_merchant_state_v2_audit/.../merchant_state_v2_sample.json`
- `data/output/09_merchant_state_v2_audit/.../merchant_state_v2_prompt_audit.jsonl`

### 2.2 用户状态训练资产

脚本：

- [09_user_state_training_assets_v1_build.py](D:\5006_BDA_project\scripts\09_user_state_training_assets_v1_build.py)

作用：

- 构造 `user_state_model_v1` 的第一版训练资产
- 当前先做：
  - `pairwise preference` 训练对
  - 用户状态标签审计
- 不直接做新一轮 profile SFT

当前本地输出：

- `data/output/09_user_state_training_assets_v1/.../user_state_pairwise_train_v1.parquet`
- `data/output/09_user_state_training_assets_v1/.../user_state_labels_v1.parquet`
- `data/output/09_user_state_training_assets_v1/.../user_state_training_assets_v1_summary.json`
- `data/output/09_user_state_training_assets_v1/.../user_state_training_assets_v1_sample.json`

## 3. 下一步门槛

下一步不是继续盲目拉长 `11_2`，而是先过两个门：

### 门 1：商家状态门

- 稀疏商家也要有可用状态
- `photo summary` 对 sparse merchant 的补强要能量化
- schema 边界通过小样本 prompt 审计

### 门 2：用户状态门

- `pairwise preference` 资产要够覆盖
- 低/中/高密度用户都要有训练对
- `recent_shift / service_risk / novelty_style` 标签要不是全 `unknown`

这两个门不过，不开下一轮 merchant SFT，也不开下一轮 profile SFT。

## 4. 当前最实际的方向

当前更合理的路线：

1. 继续保留远端 `35B` 训练作为 profile generator baseline
2. 本地先把 `merchant_state_v2` 审计和 `user_state_model_v1` 资产准备好
3. 等商家状态门和用户状态门通过后，再决定：
   - merchant-side SFT
   - state-first 的 profile generator SFT vNext
