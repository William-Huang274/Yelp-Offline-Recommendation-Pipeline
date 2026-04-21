# b5 用户偏好识别 / Persona 建模审计与 SFT-DPO 方案（2026-04-14）

## 目标
围绕 `b5`（中高交互用户集）设计一条新的用户偏好识别与用户-商户匹配方案，用于后续：

- `SFT`：学习稳定、结构化、可控的用户偏好输出
- `DPO`：学习在多个“看起来都合理”的画像或匹配结论中，优先选择更有证据、对排序更有用的版本
- 下游回灌：先服务 `Stage10` 特征，再考虑接入 `Stage11`

本轮先做**数据审计和训练方案设计**，不做云端训练。

## 约束
这条线必须满足：

1. 训练与推理输入不能过度压缩，不能只给很少几列 JSON 特征。
2. 严格无数据泄露。所有用户画像、序列视图、用户-商户匹配都只能使用训练期可见信息。
3. 优先考虑可学性。先用覆盖高、证据明确、可以结构化监督的字段，不先做抽象自由文本人格。
4. 新方案先作为独立资产接入，不影响当前 `b5` 主线排序设计。

## 当前资产现状
当前 `b5` 画像和 `Stage11` 语义资产已经有较强基础，但主抽取逻辑仍以规则/关键词/聚合为主：

- 用户画像聚合：场景、餐时、属性、长期/近期/最新偏好  
  见 [09_user_intent_profile_v2_build.py:47](D:\5006_BDA_project\scripts\09_user_intent_profile_v2_build.py:47)、[09_user_intent_profile_v2_build.py:68](D:\5006_BDA_project\scripts\09_user_intent_profile_v2_build.py:68)、[09_user_intent_profile_v2_build.py:91](D:\5006_BDA_project\scripts\09_user_intent_profile_v2_build.py:91)
- `Stage11` 源语义材料：正负 cue、semantic hints、故事污染清洗  
  见 [09_stage11_source_semantic_materials_v1_build.py:61](D:\5006_BDA_project\scripts\09_stage11_source_semantic_materials_v1_build.py:61)、[09_stage11_source_semantic_materials_v1_build.py:76](D:\5006_BDA_project\scripts\09_stage11_source_semantic_materials_v1_build.py:76)、[09_stage11_source_semantic_materials_v1_build.py:107](D:\5006_BDA_project\scripts\09_stage11_source_semantic_materials_v1_build.py:107)、[09_stage11_source_semantic_materials_v1_build.py:178](D:\5006_BDA_project\scripts\09_stage11_source_semantic_materials_v1_build.py:178)
- 语义文本资产：负向 cue、semantic hints、story pattern  
  见 [09_stage11_semantic_text_assets_v1_build.py:128](D:\5006_BDA_project\scripts\09_stage11_semantic_text_assets_v1_build.py:128)、[09_stage11_semantic_text_assets_v1_build.py:137](D:\5006_BDA_project\scripts\09_stage11_semantic_text_assets_v1_build.py:137)、[09_stage11_semantic_text_assets_v1_build.py:186](D:\5006_BDA_project\scripts\09_stage11_semantic_text_assets_v1_build.py:186)
- 用户-商户匹配已具备较完整的结构化骨架  
  见 [09_user_business_match_features_v2_build.py:171](D:\5006_BDA_project\scripts\09_user_business_match_features_v2_build.py:171)、[09_user_business_match_features_v2_build.py:175](D:\5006_BDA_project\scripts\09_user_business_match_features_v2_build.py:175)、[09_user_business_match_channels_v2_build.py:65](D:\5006_BDA_project\scripts\09_user_business_match_channels_v2_build.py:65)、[09_user_business_match_channels_v2_build.py:71](D:\5006_BDA_project\scripts\09_user_business_match_channels_v2_build.py:71)

当前问题不是“没有资产”，而是：

- 用户侧信息已经够做更细的结构化偏好抽取
- 商户侧信息也够做更细的语义 schema
- 但现在仍然停在规则聚合层，尚未进入 `SFT / DPO` 的可监督用户偏好建模

## b5 审计结果

### 1. 基本规模
- 用户数：`9765`
- 商户数：`1798`
- 训练期交互数：`120045`
- 用户平均训练交互：`12.89`
- 用户训练交互中位数：`7`
- 用户训练交互 P90：`25`

这说明：

- `b5` 并不是极稀疏集
- 足够支持“训练期行为序列 + review/tip 证据”输入
- 适合做结构化偏好抽取和 DPO 二选一偏好对齐

### 2. 用户侧特征覆盖

#### 当前可直接复用的结构化偏好
- `long_term_top_cuisine` 覆盖：`94.15%`
- `negative_top_cuisine` 覆盖：`51.68%`
- `recent_top_cuisine` 覆盖：`12.86%`
- `top_city` 覆盖：`94.57%`
- `profile_confidence_v2` 覆盖：`99.19%`
- `profile_text_short` 覆盖：`99.18%`
- `user_long_pref_text` 覆盖：`94.57%`
- `user_recent_intent_text` 覆盖：`12.95%`
- `user_negative_avoid_text` 覆盖：`94.54%`
- `user_context_text` 覆盖：`94.57%`

#### 训练期行为与文本证据
- `review_evidence` 覆盖：`100%`
- 至少 `2` 个正向历史事件：`91.45%`
- 至少 `1` 个负向历史事件：`52.30%`
- 最近 `180` 天内至少 `1` 个事件：`54.33%`
- 最近 `365` 天内至少 `1` 个事件：`67.72%`
- 拥有最近 `5` 个训练期行为序列：`86.65%`

#### 最近 5 条行为能否接到商户语义
- 最近 `5` 条序列全部有 merchant text：`95.38%`
- 最近 `5` 条序列全部有 merchant card：`95.38%`
- 最近 `5` 条序列里至少 `1` 条有 refined schema：`28.92%`

结论：

- **长期偏好、负向偏好、上下文偏好覆盖非常好**
- **近期意图字段覆盖不够，不适合直接做主监督任务**
- **行为序列是可用的，而且最近 5 条行为大多数都能接到商户语义文本**
- refined merchant schema 覆盖较低，适合做高精度补充，不适合做唯一依赖

### 3. 商户侧特征覆盖
- merchant text 覆盖：`100%`
- merchant semantic card 覆盖：`100%`
- refined merchant schema 覆盖：`8.73%`
- review evidence 覆盖：`100%`
- 最近 2 年 review 覆盖：`99.83%`

结论：

- 商户侧完全可以先依赖 `merchant_text_views_v1 + merchant_semantic_card_v2`
- refined schema 现在更适合做 teacher/anchor，不适合做全面训练输入
- 如果要做更细 persona，优先补一版 **LLM merchant persona schema v1**，覆盖全 `1798` 个商户

### 4. 用户-商户匹配信号覆盖
- user-business pairs：`119175`
- 覆盖用户数：`9235`
- `mean_match_total_v1`：`0.2730`
- `mean_match_positive_evidence > 0` 比例：`95.05%`
- `mean_match_negative_conflict > 0` 比例：`71.43%`

#### 匹配通道
- `mean_channel_preference_core_v1`：`0.2409`
- `mean_channel_recent_intent_v1`：`0.0226`
- `mean_channel_context_geo_v1`：`0.7047`
- `mean_channel_recent_intent_v1 > 0`：`12.44%`
- `mean_channel_conflict_v1 > 0`：`71.92%`

结论：

- **用户-商户匹配已经是当前最成熟的下游接口**
- `recent_intent` 通道过弱，不适合作为新项目主轴
- `preference_core / context_geo / conflict` 是现阶段最适合保留的监督与下游特征

### 5. 可学性判断
按当前本地资产口径，粗略定义三类准备度：

- `sft_structured_ready`：能构造出结构化用户偏好输入  
  条件：长期偏好、城市、long_pref_text、context_text、训练事件数齐全  
  覆盖：`55.47%`
- `sft_rich_ready`：适合做 richer SFT  
  条件：上面基础上，再要求 review/sentence 证据和最近 5 条序列文本覆盖够高  
  覆盖：`48.39%`
- `dpo_pair_ready`：适合做 DPO 偏好对比  
  条件：结构化输入齐全，并且至少有正负历史事件、最近 5 条序列可接商户卡  
  覆盖：`35.996%`

结论：

- **`SFT` 可以直接做，而且用户池够大**
- **`DPO` 也能做，但可学池会更小**
- 当前最稳的顺序是：`先 SFT，再 DPO`

## 真实样本说明

### 用户侧样本
样本来自 [user_intent_profile_v2_sample.json](D:\5006_BDA_project\data\output\09_user_intent_profile_v2\20260322_200648_full_stage09_user_intent_profile_split_aware_v2_build\user_intent_profile_v2_sample.json)：

一个 `family_users` 用户可见字段包括：
- `long_term_top_cuisine = japanese_sushi`
- `top_city = Metairie`
- `dinner_pref = 1.0`
- `late_night_pref = 1.0`
- `family_scene_pref = 1.0`
- `nightlife_scene_pref = 0.8`
- `geo_concentration_ratio = 0.3216`

当前系统能表达：
- 喜欢寿司
- 偏晚餐和夜间
- 家庭/夜生活都可接受

但还不能稳定表达：
- 更偏“家庭局”还是“夜间社交探索”
- 是否接受较吵闹场景
- 偏计划型还是即时型
- 面对差服务、长等待的容忍度

这类更高层的行为偏好，就是下一版 `SFT / DPO` 应该补的内容。

### 商户侧样本
样本来自 [merchant_schema_v1_refined_sample.json](D:\5006_BDA_project\data\output\09_merchant_schema_v1_refined\20260323_002416_full_stage09_merchant_schema_v1_refined_build\merchant_schema_v1_refined_sample.json)：

某商户已经能抽出：
- `discovered_dishes`: oyster, crawfish, burger, taco, sandwich
- `discovered_scenes`: nightlife, outdoor_dining, date_night, group_dining, quick_bite
- `discovered_service_strengths`: good_value, attentive_service, friendly_service, clean_space
- `discovered_complaints`: long_wait, rude_service
- `discovered_time_cues`: happy_hour, weekend, dinner, lunch, late_night
- `discovered_properties`: beer_and_wine, table_service, outdoor_seating, takeout, parking_available

这已经接近一个“商户 persona”了，但覆盖只到 `8.73%`。  
因此下一版需要把这种 schema 扩展到全量商户。

## 训练方案建议

## 一、先做 SFT
目标：

- 让模型学会稳定输出结构化用户偏好
- 控制输出格式
- 学会区分长期偏好、近期偏好、回避信号和行为风格
- 不让模型写泛化的 persona 散文

### 推荐的输入
不是只给压缩 JSON，也不是直接把全量长序列硬塞进去。  
建议输入由四部分组成：

1. **训练期行为序列视图**
   - 最近 `5` 次行为的商户语义摘要
   - 每次行为的：
     - 商户主菜系
     - 场景
     - 时间 cue
     - 正负反馈
     - 简短证据摘要
2. **长期偏好视图**
   - long-term cuisine / meal / scene / property
3. **负向/回避视图**
   - negative cuisine
   - negative pressure
   - 负向高权重 evidence 句子
4. **地理与上下文视图**
   - top_city
   - geo concentration
   - context text

### 为什么不是原始长序列
因为原始长序列：
- 成本高
- 噪声大
- 监督难
- 容易让模型风格漂移

### 为什么也不能只给非常少的 JSON
因为过度压缩会丢掉：
- 近期行为变化
- 正负证据
- 场景切换
- 容忍度线索

所以正确输入是：
**结构化字段 + 最近几条行为的可见证据摘要**

## 二、再做 DPO
目标：

- 在多个“都像样”的用户画像里，更偏向证据更强、对排序更有用的版本
- 减少幻觉
- 减少过度文学化总结

### DPO 推荐任务
不要做成“改写风格”。  
要做成：

- 同一用户，给两个候选画像 A / B
- A 更贴近训练期证据
- B 结构对，但更空泛或更容易幻觉
- 让模型学会偏好 A

### DPO 何时开始
先等 `SFT` 输出稳定再做。  
否则 DPO 会在格式和语义上同时纠偏，成本太高。

## Schema 设计建议

## 1. user_preference_schema_v1
```json
{
  "stable_preferences": {
    "preferred_cuisines": [],
    "preferred_meals": [],
    "preferred_scenes": [],
    "preferred_properties": []
  },
  "avoid_signals": {
    "avoided_cuisines": [],
    "service_risk_sensitivity": "",
    "noise_tolerance": "",
    "wait_tolerance": ""
  },
  "behavior_style": {
    "social_mode": "",
    "geo_style": "",
    "planning_style": "",
    "novelty_style": ""
  },
  "recent_shift": {
    "active_recent": false,
    "recent_focus": [],
    "shift_vs_long_term": ""
  },
  "evidence_spans": []
}
```

### 第一版强烈建议保留的字段
- preferred cuisines
- preferred meals
- preferred scenes
- avoided cuisines
- geo style
- social mode
- service/noise/wait tolerance

### 第一版不建议强上
- 太抽象的人格标签
- 纯自由文本“persona story”
- 低覆盖的 recent-only 细粒度任务

## 2. sequence_view_schema_v1
```json
{
  "recent_interactions": [
    {
      "order": 1,
      "feedback": "positive|neutral|negative",
      "merchant_summary": "",
      "cuisine": "",
      "scenes": [],
      "time_cues": [],
      "service_risk": [],
      "evidence": []
    }
  ],
  "sequence_summary": {
    "recent_positive_streak": 0,
    "recent_negative_streak": 0,
    "cuisine_switch": "",
    "scene_switch": "",
    "exploration_vs_repeat": ""
  }
}
```

### 设计原则
- 用最近 `5` 条即可，不追求长序列
- 每条保留“可见证据 + 商户语义卡”
- 不要直接塞原始全文 review

## 3. merchant_persona_schema_v1
```json
{
  "cuisine_identity": [],
  "occasion_fit": [],
  "vibe_profile": [],
  "service_strengths": [],
  "service_risks": [],
  "time_fit": [],
  "convenience_profile": [],
  "evidence_spans": []
}
```

### 这一步为什么重要
用户偏好识别如果没有更细商户 schema，只能学到粗粒度偏好。  
先把商户 persona 做起来，后续用户画像和用户-商户匹配都会更稳定。

## 推荐的推进顺序

### Phase 0：数据审计
本轮已完成，结论是：
- `SFT` 可以启动
- `DPO` 可以做，但要比 `SFT` 晚
- recent-only 不适合作为主监督
- refined merchant schema 覆盖低，需要补 merchant persona v1

### Phase 1：merchant persona v1
先对全量 `1798` 个 `b5` 商户做一版 LLM 结构化抽取。

### Phase 2：user preference SFT
用训练期历史 + sequence view + merchant persona，训练结构化用户画像抽取。

### Phase 3：DPO
围绕同一用户的两个候选画像做偏好对齐。

### Phase 4：下游验证
先接 `Stage10`，再看是否值得进入 `Stage11`。

## 当前不建议直接做的事情
- 直接上自由文本 persona
- 直接全量接入 `Stage11`
- 直接把 recent_intent 作为主监督任务
- 直接喂原始长行为序列和长 review 正文

## 结论
`b5` 现有数据已经足够支持：

- 更细的用户偏好识别
- 更细的用户-商户匹配
- `SFT -> DPO` 逐步推进

但最稳的落地方式不是“让大模型直接写人格故事”，而是：

**先把用户偏好、行为序列和商户语义做成更细、但仍然结构化且无泄露的训练资产，再进入 `SFT / DPO`。**
