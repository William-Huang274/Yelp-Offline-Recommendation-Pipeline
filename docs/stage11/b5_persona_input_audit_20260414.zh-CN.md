# b5 Persona 输入面全面审计（2026-04-14）

## 1. 审计范围

本轮只做输入资产与首版 `SFT` 训练效果审计，不启动新训练，不启动新构建任务。

审计目标：

1. 判断首版 `SFT` 是否已经证明 schema 可学。
2. 判断用户侧输入是否过于贴近 teacher，是否需要去掉派生 narrative。
3. 判断商家侧是否应该继续以 review / tip 为主，并评估 `photo summary` 的补强价值。
4. 判断用户侧与商家侧是否应拆成两个任务 / 两个 adapter，而不是先混成单模型。

## 2. 首版 SFT 训练效果

云端首版训练产物：

- `/root/autodl-tmp/5006_BDA_project/data/output/11_persona_sft_models/20260414_170209_full_stage11_2_persona_sft_train`

关键结果：

- `train_rows = 2048`
- `eval_rows = 128`
- 训练 loss 明显下降
- 但严格 generation audit：
  - `json_valid_rate = 0.0`
  - `preferred_cuisines_exact_rate = 0.0`
  - `preferred_meals_exact_rate = 0.0`
  - `preferred_scenes_exact_rate = 0.0`

结合生成样本判断，这轮失败模式不是“模型没学到内容”，而是：

1. 输出边界控制不好，模型会把 prompt 尾部一起续写进结果。
2. 模型会在一个基本正确的 JSON 后继续重复输出第二份 JSON。
3. 预测结构已经明显接近 target schema，但由于拼接、重复和额外 token，导致严格 JSON 校验全部失败。

结论：

- **schema 可学性：通过**
- **当前 prompt + decoding 直接产出稳定 JSON：未通过**

这说明下一轮的重点不是“换任务”，而是：

1. 进一步减少 teacher-like narrative 输入
2. 加强 stop / boundary / postprocess 设计
3. 先把 extractor 做成稳定结构化输出器

### 2.1 generation sample 结构错误分析

对首版 `generation_samples.json` 做结构错误扫描后，得到：

- `contains_json_brace_anywhere_rate = 0.8125`
- `starts_with_brace_rate = 0.4375`
- `contains_double_json_pattern_rate = 0.4375`
- `contains_assistant_token_rate = 0.4375`
- `contains_prompt_leak_markers_rate = 0.5000`

这组结果说明：

1. 大多数样本已经开始生成目标 JSON 结构，而不是完全答非所问。
2. 主要问题集中在：
   - 输出前部残留 prompt 尾巴
   - 输出完第一份 JSON 后继续重复第二份
   - 输出中混入 `<|assistant|>` 之类的边界 token
3. 当前失败更像 **format / decoding failure**，而不是 **semantic learning failure**。

因此，下一轮不能只看 strict JSON valid rate，也要单独看：

- `json_start_rate`
- `single_json_rate`
- `prompt_leak_rate`
- `assistant_token_rate`
- 字段级弱匹配率

## 3. 用户侧输入面审计

### 3.1 当前可用覆盖

当前用户侧资产：

- `user_preference_schema_v1`
- `user_preference_target_schema_v1`
- `sequence_view_schema_v1`

用户总量：

- `9235`

关键覆盖率：

- `sft_ready_ratio = 0.9959`
- `has_profile_text_short = 0.9999`
- `has_long_pref_text = 1.0000`
- `has_recent_intent_text = 0.9951`
- `has_negative_avoid_text = 0.9997`
- `has_pos_evidence_sentences = 0.9857`
- `has_neg_evidence_sentences = 0.5811`
- `has_recent_sequence_json = 1.0000`
- `has_positive_anchor_json = 1.0000`
- `has_negative_anchor_json = 1.0000`
- `mean_sequence_event_count = 7.9448`

结论：

- 当前用户侧输入**不算过度压缩**
- 证据厚度足够进入 `SFT`
- sequence block 和 anchor block 已经具备可学性

### 3.2 当前最主要的问题

当前 prompt 里，以下派生 narrative 全部高覆盖出现：

- `Long-term preference narrative`
- `Recent intent narrative`
- `Negative or avoid narrative`

而且在 sample 中这三块和 teacher target 的字段语义过近。

这不是数据泄露问题，而是 **teacher overlap 过高**：

- 模型更容易学会“复述现有规则画像”
- 不够像“从证据中抽偏好”

当前审计结果：

- `derived_without_pos_evidence_ratio = 0.0143`
- `derived_without_sequence_ratio = 0.0`

这说明：

- 这些 narrative 并不是完全脱离证据凭空出现
- 但它们仍然过强，不适合作为下一轮主输入
- 同时，底层已经存在的 `positive_anchor_sequence_json / negative_anchor_sequence_json` 虽然覆盖率是 `100%`，但第一版 prompt 并没有把这两块作为主输入稳定用起来，导致模型更多是在跟随聚合叙述，而不是学习高价值行为锚点

### 3.3 用户侧输入的建议调整

下一轮用户 `SFT` 输入，建议按“清洗而非提纯”重构：

保留：

- `positive_evidence_sentences`
- `negative_evidence_sentences`
- `recent_event_sequence_json`
- `positive_anchor_sequence_json`
- `negative_anchor_sequence_json`
- 少量行为统计
- 少量 merchant text snippet

降权或移除：

- `user_long_pref_text`
- `user_recent_intent_text`
- `user_negative_avoid_text`

原则：

- 只剔除明显噪声、废话、重复文本
- 不做强人工裁剪
- 尽量保留训练期内真实 review / tip 差异

## 4. 商家侧输入面审计

### 4.1 当前文本资产已经具备的能力

当前商家文本资产包括：

- `merchant_semantic_card_v2`
- `merchant_text_views_v3`
- `merchant_structured_text_views_v2`

从现有样本看，商家侧已经能提供：

- 核心类别和菜系
- meal / scene / property
- 正向服务特征
- 负向投诉特征
- 语义摘要文本
- 上下文文本

这意味着商家侧第一版 `persona SFT` 并不缺主干文本。

### 4.2 review / tip 仍应作为主干

当前 review / tip 资产已经具备权重基础：

review 侧已有：

- `useful / funny / cool`
- `vote_weight`
- `text_weight`
- `recency_weight`
- `evidence_weight_v1`

tip 侧已有：

- `compliment_count`
- `text_weight`
- `recency_weight`
- `tip_weight_v1`

用户原生表上还能进一步得到：

- `review_count`
- `useful`
- `cool`
- `fans`
- `elite`

本地按简单用户信号分数审计后：

- `user_signal_q80 = 112.3`
- 高信号用户比例约 `20.0%`

结论：

- 商家侧可以优先使用高权重 review / tip
- 也可以引入“高信号用户文本优先”策略
- 但**不应硬过滤掉中低权重但语义有效的文本**

推荐策略：

1. review / tip 多的商家：
   - 高权重文本优先
   - 但中低权重有效文本保留
2. review / tip 少的商家：
   - 去噪后尽量全喂
3. 只剔除：
   - 极短废话
   - 重复文本
   - 乱码
   - 与餐饮属性完全无关的模板话

### 4.3 photo summary 的补强价值

本轮已将以下资产上传到云端：

- `business_photo_summary.parquet`
- `build_stats.json`

photo 资产覆盖情况：

- `photo_rows = 36680`
- 在目标商家集（`merchant_semantic_card_v2` 的 `4518` 家商户）上的覆盖：
  - `photo_cover_on_target_merchants = 0.5795`
  - `photo_nonempty_caption_cover_on_target_merchants = 0.4327`
  - `food_photo_cover_on_target_merchants = 0.4378`
  - `inside_photo_cover_on_target_merchants = 0.3406`
  - `menu_photo_cover_on_target_merchants = 0.0292`

对 sparse merchant 的补强价值：

- `sparse_merchant_ratio = 0.3220`
- `photo_cover_on_sparse_merchants = 0.3086`
- `photo_caption_cover_on_sparse_merchants = 0.1333`

对 very sparse merchant：

- `very_sparse_merchant_ratio = 0.0706`
- `photo_cover_on_very_sparse_merchants = 0.2257`
- `photo_caption_cover_on_very_sparse_merchants = 0.0972`

结论：

- `photo summary` 对商家侧是**有效补强**
- 但它不是主干，只是补强层
- 对稀疏商家最有价值的不是“所有商家都依赖 photo”，而是：
  - 文本弱时由 photo 补一层 food / inside / vibe / menu 证据

### 4.4 原始图片是否现在就需要上云

本地 `Yelp-Photos.zip` 中已确认：

- `yelp_photos.tar` 内含 raw jpg
- 同时含 `photos.json`
- `photos.json` 里有：
  - `photo_id`
  - `business_id`
  - `caption`
  - `label`

所以原始图片确实可用。

但当前阶段不建议优先上传 raw image 包，原因：

1. 现有 `business_photo_summary.parquet` 已足够支撑 merchant-side v1 审计和样本设计
2. 现在还没有进入视觉模型 / VLM / OCR 阶段
3. raw image 包体积太大，当前 ROI 不高

结论：

- **photo summary：现在就有用**
- **raw photos：二阶段增强再考虑**

## 5. 模型结构建议

当前建议仍保持：

- 同一个 `Qwen3.5-9B` 主干
- 用户侧一个 adapter
- 商家侧一个 adapter
- 不先做单模型混训

原因：

1. 用户侧输入单位是 `user_id`
2. 商家侧输入单位是 `business_id`
3. 两边输入结构不同
4. 两边输出 schema 不同
5. 两边噪声模式不同

在 schema 和输入边界都还未完全稳定前，先混训只会让定位问题更难。

## 6. 综合审计结论

### 6.1 用户侧

- 继续做 `SFT`
- 输入改成“原始证据优先”
- 少用派生 narrative
- 只做轻度去噪，不做强人工裁剪

### 6.2 商家侧

- review / tip 继续是主干
- 高权重文本优先，但不硬过滤中低权重有效文本
- sparse merchant 用 `photo summary` 补强
- 先做 merchant persona 样本设计，再考虑训练

### 6.3 当前是否进入下一阶段

本轮结论：

- **可以进入用户侧 `SFT v2` 输入重构**
- **可以进入 merchant-side 样本设计**
- **暂不进入 merchant-side 训练**
- **暂不进入用户 / 商家混训**

## 7. 当前不缺的云端资产

现在云端已具备：

- `yelp_academic_dataset_user`
- `business_photo_summary.parquet`
- `build_stats.json`

因此本轮审计后，**没有必须立即再上传的新资产**。

如果下一阶段要做 merchant 视觉增强，再考虑上传：

- `Yelp-Photos.zip`
- 或其中的 `yelp_photos.tar`
