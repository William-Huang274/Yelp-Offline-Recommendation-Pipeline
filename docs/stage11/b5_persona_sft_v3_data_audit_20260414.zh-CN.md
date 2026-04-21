# b5 Persona SFT v3 数据审计

这轮工作只做了 `v3` 数据构建和审计，没有启动训练。

目标：

- 让低、中、高密度 `b5` 用户都拥有更丰富的语义证据
- 减少 teacher-like narrative 输入
- 用更原始、清洗后的用户证据替代过度人工提纯的素材
- 先看数据质量，不看 token 上限

## 1. 本轮 v3 数据构建

云端产物：

- `/root/autodl-tmp/5006_BDA_project/data/output/11_persona_sft_data/20260414_195947_full_stage11_1_persona_sft_v3_build_dataset`

当前输入来源：

- `09_user_preference_schema_v1`
- `09_user_profiles`
- `09_tip_signals`
- `09_sequence_view_schema_v1`
- `09_user_preference_target_schema_v1`

v3 的主要变化：

- 不再使用 `long_term / recent_intent / negative_avoid narrative` 作为主输入
- 改成 `User evidence stream`
- 保留 `Recent event sequence`
- 保留 `Positive / Negative anchor events`
- teacher target 包装成 `core_profile + open_discoveries` schema

## 2. 全量结果

总体规模：

- `rows_total = 9197`
- `rows_train = 7389`
- `rows_eval = 1808`

总体输入厚度：

- `mean_prompt_chars_train = 8835.85`
- `p95_prompt_chars_train = 12647.2`
- `mean_review_evidence_train = 13.61`
- `mean_tip_evidence_train = 2.996`
- `mean_unified_evidence_train = 16.60`

结构审计：

- `has_user_evidence_stream_rate = 1.0`
- `has_recent_sequence_rate = 1.0`
- `has_positive_anchor_events_rate = 1.0`
- `has_negative_anchor_events_rate = 1.0`
- `has_derived_narrative_rate = 0.0`
- `audit_pass_v3 = true`

这说明：

- `v3` 已经按设计切到了 evidence-first 输入
- teacher-like narrative 已经从主输入中移除
- 数据结构完整，可进入下一阶段训练前审计

## 3. 按 b5 内部密度分层

这里按 split 后训练期可见交互数分三档：

- 低密度：`5-7`
- 中密度：`8-17`
- 高密度：`18+`

### 3.1 低密度 `5-7`

- `rows = 2526`
- `mean_prompt_chars = 7554.07`
- `p95_prompt_chars = 9030.75`
- `mean_review_evidence_count = 8.69`
- `mean_tip_evidence_count = 0.71`
- `mean_unified_evidence_count = 9.40`
- `mean_sequence_event_count = 7.80`

分布：

- unified evidence `q50 = 9`
- unified evidence `q90 = 16`

判断：

- 低密度用户已经不再只是几个聚合标签
- 当前能看到接近 10 条统一证据，外加最近 sequence 和 anchor
- 对低密度用户来说，这一轮的语义空间已经明显更充足

### 3.2 中密度 `8-17`

- `rows = 4372`
- `mean_prompt_chars = 8400.58`
- `p95_prompt_chars = 11109.0`
- `mean_review_evidence_count = 12.81`
- `mean_tip_evidence_count = 1.60`
- `mean_unified_evidence_count = 14.41`
- `mean_sequence_event_count = 8.0`

分布：

- unified evidence `q50 = 13`
- unified evidence `q90 = 26`

判断：

- 中密度用户是当前最均衡的一档
- review evidence 已经足够厚，tip 也开始补进来
- 这是最适合 first-pass `SFT v3` 的主战场

### 3.3 高密度 `18+`

- `rows = 2299`
- `mean_prompt_chars = 11060.85`
- `p95_prompt_chars = 16602.9`
- `mean_review_evidence_count = 20.50`
- `mean_tip_evidence_count = 8.11`
- `mean_unified_evidence_count = 28.61`
- `mean_sequence_event_count = 8.0`

分布：

- unified evidence `q50 = 29`
- unified evidence `q90 = 44`

判断：

- 高密度用户的语义信息最丰富
- review / tip 都明显更厚
- 当前 v3 已经给了模型足够大的语义空间去学习上下文依赖偏好
- 这档用户的主要问题已经不是信息不够，而是后续训练时如何做更好的长度管理和证据 packing

## 4. 代表性样本

### 低密度样本

- `user_id = rxiYKI40S7UunZMP1P0ltQ`
- `n_train_max = 5`
- `review_evidence_count = 9`
- `tip_evidence_count = 0`
- `sequence_event_count = 7`

当前 evidence stream 里已经能看到：

- atmosphere
- price/value
- wait_long
- cajun_creole
- deli_sandwich
- service
- pizza

这说明低密度用户虽然历史少，但已经不是“只有一个 top cuisine”。

### 中密度样本

- `user_id = G85ZNADVe4c4BOjGD56ZTA`
- `n_train_max = 12`
- `review_evidence_count = 11`
- `tip_evidence_count = 2`
- `sequence_event_count = 8`

当前 evidence stream 里已经能看到：

- cajun_creole
- pizza
- cocktail_bar
- steakhouse
- wait_long
- service

sequence 里还能看到不同 cuisine / city / meal / scene 的上下文切换。

### 高密度样本

- `user_id = r8FVQHXxEQpu_Enf2Ux5Rg`
- `n_train_max = 145`
- `review_evidence_count = 28`
- `tip_evidence_count = 1`
- `sequence_event_count = 8`

当前 evidence stream 里已经能看到：

- pizza
- seafood
- breakfast_brunch
- price_value
- deli_sandwich
- service
- coffee_tea

而 sequence block 继续补了 meal / scene / reliability / merchant text 的上下文。

## 5. 极端长样本

存在极端高活跃用户：

- `user_id = oHjUPJHEOGJsrPqu3B_MnA`
- `n_train_max = 178`
- `review_evidence_count = 30`
- `tip_evidence_count = 310`
- `unified_evidence_count = 340`
- `prompt_chars = 103366`

这不是脏数据，而是**极高活跃用户的自然长尾**。  
当前阶段先保留它，原因是本轮目标是先确认语义信息能否尽量丰富。

后续进入训练前，再决定是否：

- 对极端 tip-heavy 用户单独做 packing
- 还是设置更高 token 预算
- 或对极端长样本单独分桶

## 6. 结论

这轮 `v3` 数据已经达到预期：

- 低、中、高密度 `b5` 用户都获得了比 `v2` 更丰富的语义证据
- review / tip 尽量宽进，只过滤明显无效文本
- teacher-like narrative 不再作为主输入
- 模型现在看到的是更接近真实世界的清洗后证据流

当前可以下的结论是：

- **语义信息层面，v3 已经过关**
- **接下来真正需要控制的是训练时的长度预算和 evidence packing，而不是继续补更多同类信息**
