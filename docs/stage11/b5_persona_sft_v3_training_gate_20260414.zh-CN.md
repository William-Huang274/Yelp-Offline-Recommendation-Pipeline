# b5 Persona SFT v3 训练前门控

这份文档收口 `b5 persona SFT v3` 进入训练前需要通过的三件事：

1. `v3` 输入是否已经足够丰富，且 teacher overlap 足够低
2. 强监督与弱监督信号是否已经可用
3. 训练时的 packing / cap 方案是否已经能控制高密度长尾而不明显损失语义

当前结论：

- `v3` 数据输入：通过
- `core + open_discoveries` 监督：通过
- 分密度 packing / cap 审计：通过
- 当前可以进入下一步训练准备

## 1. 当前使用的数据口径

云端最新 run：

- `target schema`: `/root/autodl-tmp/5006_BDA_project/data/output/09_user_preference_target_schema_v1/20260414_202228_full_stage09_user_preference_target_schema_v1_build`
- `v3 dataset`: `/root/autodl-tmp/5006_BDA_project/data/output/11_persona_sft_data/20260414_202332_full_stage11_1_persona_sft_v3_full_build_dataset_refined`
- `packing audit`: `/root/autodl-tmp/5006_BDA_project/data/output/11_persona_sft_packing_audit/20260414_202905_full_stage11_persona_sft_v3_packing_audit_refined`

本地脚本：

- [09_user_preference_target_schema_v1_build.py](D:\5006_BDA_project\scripts\09_user_preference_target_schema_v1_build.py)
- [11_1_persona_sft_build_dataset.py](D:\5006_BDA_project\scripts\11_1_persona_sft_build_dataset.py)
- [audit_persona_sft_v3_packing.py](D:\5006_BDA_project\scripts\audit_persona_sft_v3_packing.py)

## 2. 强监督与弱监督

### 2.1 强监督

强监督只用于 `core_profile`，目标仍然是稳定、可评估、可回灌排序链路的结构化画像。

当前强监督字段：

- `stable_preferences`
  - `preferred_cuisines`
  - `preferred_meals`
  - `preferred_scenes`
  - `preferred_properties`
  - `top_city`
  - `geo_style`
- `avoid_signals`
  - `avoided_cuisines`
  - `avoided_scenes`
  - `service_risk_sensitivity`
- `recent_preferences`
  - `recent_focus_cuisines`
  - `recent_focus_meals`
  - `recent_focus_scenes`
  - `recent_focus_properties`
  - `recent_shift`
- `behavior_summary`
  - `social_mode`
  - `time_mode`
  - `novelty_style`
- `evidence_refs`
- `confidence`
- `ambivalent_signals`

### 2.2 弱监督

`open_discoveries` 不再是空数组，而是保守的弱标签。  
目标不是把模型锁死，而是给它一个“可以探索但不能乱写”的起点。

当前可用类型：

- `preference_conflict`
- `tolerance_hypothesis`
- `context_dependent_preference`
- `scenario_specific_preference`
- `uncertainty_note`

当前弱监督审计结果：

- `rows = 9235`
- `mean_open_discoveries_count = 1.0918`
- `p50_open_discoveries_count = 1`
- `p90_open_discoveries_count = 2`
- `discovery_with_refs_ratio = 1.0`

类型分布：

- `preference_conflict = 4480`
- `tolerance_hypothesis = 2283`
- `uncertainty_note = 2120`
- `context_dependent_preference = 803`
- `scenario_specific_preference = 397`

这说明：

- 弱监督已经不是空白
- 但也没有膨胀成几乎全量三条 discovery 的高干预 teacher
- 每条 discovery 都带证据引用，可以接受

## 3. Prompt 模板

当前 `v3` prompt 结构已经固定：

1. 任务和输出规则
2. 目标 JSON schema
3. `User evidence stream`
4. `Behavior stats`
5. `Supplemental cleaned archive digest`  
   只在证据偏薄时补，当前覆盖率 `23.6%`
6. `Recent event sequence`
7. `Positive anchor events`
8. `Negative anchor events`

当前不再进入主输入的内容：

- `long_term preference narrative`
- `recent intent narrative`
- `negative avoid narrative`

这意味着 teacher-like narrative 已经退出主输入，输入主体是清洗后的原始证据流。

## 4. Packing / Cap 方案

目标不是为了把 prompt 压得很短，而是：

- 不让高密度极端长尾拖垮训练
- 同时尽量保留语义丰富度
- 只在 evidence stream 层做低干预 packing

### 4.1 分密度配置

按训练期可见交互数分层：

- 低密度：`5-7`
- 中密度：`8-17`
- 高密度：`18+`

当前建议配置：

- 低密度：
  - `total_cap = 18`
  - `tip_cap = 4`
  - `neg_floor = 2`
- 中密度：
  - `total_cap = 24`
  - `tip_cap = 6`
  - `neg_floor = 3`
- 高密度：
  - `total_cap = 36`
  - `tip_cap = 12`
  - `neg_floor = 4`

packing 规则很克制：

1. 优先保留少量负向 review，防止全被正向高频文本淹没
2. 保留 tip，但不让高密度用户的海量 tip 抢占全部上下文
3. 其余部分仍按原 evidence stream 顺序保留

这不是在重新造 teacher，只是在做训练期上下文裁剪。

### 4.2 Packing 审计结果

当前 refined `v3` packing 审计：

- `orig_full_tokens_mean = 3101.76`
- `packed_full_tokens_mean = 2994.67`
- `orig_full_tokens_p95 = 4283`
- `packed_full_tokens_p95 = 4097.2`
- `orig_full_tokens_max = 32259`
- `packed_full_tokens_max = 4778`

预算下溢出率：

- `3072`
  - `orig_overflow = 36.32%`
  - `packed_overflow = 35.46%`
- `4096`
  - `orig_overflow = 7.10%`
  - `packed_overflow = 5.01%`
- `5120`
  - `orig_overflow = 2.22%`
  - `packed_overflow = 0.0%`
- `6144`
  - `orig_overflow = 1.04%`
  - `packed_overflow = 0.0%`

按密度看：

- 低密度 packed `p50 = 9`，`p90 = 16`
- 中密度 packed `p50 = 13`，`p90 = 24`
- 高密度 packed `p50 = 28`，`p90 = 36`

门控结果：

- `min_packed_items_lower = 9`
- `min_packed_items_mid = 13`
- `min_packed_items_high = 28`
- `audit_pass = true`

这说明当前 packing 方案没有把高密度用户压扁，也没有让低密度用户只剩下几个标签。

## 5. 当前训练建议

如果下一步进入训练，建议口径是：

- 使用 refined `v3` dataset
- 开启 `density-aware packing`
- 如果优先考虑训练质量而不是吞吐，优先考虑 `max_seq_len >= 5120`

原因很直接：

- `4096` 下 packed 仍有 `5.01%` overflow
- `5120` 下 packed overflow 已经为 `0`
- 当前输入重点是丰富证据，不应因为追求更小长度重新回到压缩 teacher narrative 的老路

## 6. 最终结论

当前 `SFT v3` 的三道门控都已经过了：

1. 输入：
   - evidence-first
   - derived narrative 已移除
   - 低 / 中 / 高密度用户语义都足够厚
2. 监督：
   - `core_profile` 强监督可用
   - `open_discoveries` 弱监督可用，且不过度泛滥
3. packing：
   - 高密度长尾已可控
   - 语义丰富度没有明显牺牲

因此，下一步不需要再补数据定义，直接进入训练配置和训练执行即可。
