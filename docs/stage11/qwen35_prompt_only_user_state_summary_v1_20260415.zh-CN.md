# Qwen3.5-35B Prompt-Only User State Summary V1

## 目的
这一版先不训练，只验证 `Qwen3.5-35B-A3B` 在 `prompt-only` 条件下，能不能从清洗后的用户证据中：

1. 自己发现有价值的用户状态信息
2. 输出带引用证据的结构化总结
3. 在不被低熵 teacher target 限死的前提下，给出比当前 head-heavy schema 更细的区分

这一版不是最终资产生成方案，也不是下游排序特征方案。它的角色是：

- 先观察大模型自然会发现什么
- 先验证模型能不能稳定做 evidence-grounded state summary
- 先判断当前输入面里还有哪些信息质量问题

## 为什么先做 Prompt-Only
当前 `35B raw evidence -> full profile SFT` 已经证明两件事：

1. 更大模型有价值，至少比 `9B` 更能学会格式和部分核心字段
2. 继续在当前 teacher target 上深训，不会自然长成成熟的用户理解模型

主要原因不是单纯模型不够大，而是：

- current teacher target 低熵、头部偏重
- `preferred_cuisines` 和 `recent_focus_*` 信息密度很低
- direct SFT 把“状态学习”和“结构化解释”压在同一条任务里

所以这一步先不用 teacher 去限制模型要说什么，而是先看：

- 模型会自然发现哪些 `grounded facts`
- 模型会不会提出有用的 `hypotheses`
- 模型能不能在不乱编的前提下输出证据引用

## 任务定义
模型输入是一位用户在训练期内的多源证据：

- 用户文本证据流
- recent sequence
- 事件级 merchant state snippets
- 地理、时间、重复访问等上下文

模型输出不是最终画像标签，而是 `user_state_summary_v1`：

1. `grounded_facts`
2. `state_hypotheses`
3. `discriminative_signals`
4. `unknowns`
5. per-claim `evidence_refs`
6. `confidence`

## 核心原则
### 1. 不要被规则模板绑死
如果证据支持更细的发现，模型可以输出；不要因为当前 schema 没有就不敢说。

### 2. 不允许无证据输出
所有 claim 都必须带 `evidence_refs`。没有证据就写到 `unknowns` 或不写。

### 3. 区分“用户偏好”与“商家本身热门”
不能把热门店、近距离、方便、旅游场景自动当成稳定偏好。

### 4. 允许不确定和冲突
如果 evidence 冲突或不足，必须写 `unknown`、`ambiguous`、`mixed evidence`，不能硬填。

### 5. 要有区分度
优先输出能区分这个用户和“平均用户”的信息，而不是只输出泛化的 head categories。

## 输入块设计
### A. User Meta
只放背景，不放聚合结论：

- `user_id`
- `density_band`
- `n_train_events`
- `n_review_evidence`
- `n_tip_evidence`
- `sequence_span_days`
- `repeat_business_ratio`
- `geo_span`

### B. User Evidence Stream
按证据来源分 4 类，但不把结论写死：

1. `explicit_positive_text`
2. `explicit_negative_text`
3. `ambiguous_or_neutral_text`
4. `behavior_supported_weak_signals`

每条 evidence 至少要带：

- `evidence_id`
- `source_type`
- `merchant_id`
- `merchant_name`
- `time_bucket`
- `merchant_state_snippet`
- `text`

### C. Recent Event Sequence
放最近 `K` 条行为事件，每条带：

- `event_id`
- `event_time_bucket`
- `merchant_id`
- `merchant_name`
- `merchant_state_snippet`
- `repeat_flag`
- `distance_band`

### D. Anchor Events
放高置信的关键事件：

- `anchor_positive_events`
- `anchor_negative_events`
- `anchor_conflict_events`

### E. Context Notes
只放事实，不放画像结论：

- 常去区域
- 时间模式
- 工作日/周末分布
- 明显的 sequence shift 线索

## 输出 Schema
```json
{
  "grounded_facts": {
    "stable_preferences": [
      {
        "claim": "",
        "confidence": "high|medium|low",
        "evidence_refs": []
      }
    ],
    "avoid_signals": [
      {
        "claim": "",
        "confidence": "high|medium|low",
        "evidence_refs": []
      }
    ],
    "recent_signals": [
      {
        "claim": "",
        "confidence": "high|medium|low",
        "evidence_refs": []
      }
    ],
    "context_rules": [
      {
        "claim": "",
        "confidence": "high|medium|low",
        "evidence_refs": []
      }
    ]
  },
  "state_hypotheses": [
    {
      "type": "conditional_preference|tolerance|shift|conflict|latent_preference|other",
      "claim": "",
      "confidence": "high|medium|low",
      "evidence_refs": []
    }
  ],
  "discriminative_signals": [
    {
      "claim": "",
      "why_not_generic": "",
      "confidence": "high|medium|low",
      "evidence_refs": []
    }
  ],
  "unknowns": [
    {
      "field": "",
      "reason": "",
      "evidence_refs": []
    }
  ],
  "confidence": {
    "overall": "high|medium|low",
    "coverage": "high|medium|low"
  }
}
```

## 字段解释
### grounded_facts
只放模型认为“证据已经足够支撑”的内容。这里应尽量保守。

### state_hypotheses
允许模型提出更细的推测，例如：

- 社交场景下偏热闹，日常用餐更保守
- 对 wait/noise 容忍，但对服务波动敏感
- 近期从 family-centered 转向 nightlife/group

这些都不是硬真值，所以必须带 `confidence` 和 `evidence_refs`。

### discriminative_signals
这一层专门压制模板化输出。只写“能区分这个用户”的信号。

错误示例：

- likes dinner
- likes group dining

正确示例：

- 用户更像“夜间社交型海鲜/酒吧附近餐饮偏好”，而不是一般性的家庭式晚餐偏好

### unknowns
必须单独存在。没有这层，模型容易为了显得聪明而乱填。

## 生成规则
### 允许输出
- 证据支持的细粒度偏好
- 条件偏好
- 容忍度判断
- 冲突与不确定

### 不允许输出
- 无 evidence refs 的 claim
- 仅由热门商家/近距离推出来的稳定偏好
- 空泛的 safe template
- 把 behavior-only 事件直接当成硬正例

## 推荐推理设置
这一版还不跑推理，但建议预设两种模式：

### 模式 A：Audit / Stable
- `temperature = 0.1 ~ 0.2`
- `top_p = 0.9`
- `do_sample = false` 或极弱采样

目的：
- 看模型是否能稳定输出 schema
- 看 evidence refs 是否可靠

### 模式 B：Discovery / Exploratory
- `temperature = 0.4 ~ 0.6`
- `top_p = 0.9 ~ 0.95`

目的：
- 看模型会不会提出更多 hypotheses

注意：
- 模式 B 不能直接产最终资产
- 只能做 discovery 候选审计

## 审计重点
这一版 prompt-only 需要回答 5 个问题：

1. 模型能否稳定输出有效 JSON
2. 模型能否输出带 `evidence_refs` 的 `grounded_facts`
3. 模型会不会自动生成有价值的 `state_hypotheses`
4. 模型会不会大量复读 head categories
5. 模型能否正确使用 `unknowns`

## 下一步门槛
只有当 prompt-only 审计通过下面这些门，才值得进入下一步训练：

- `json_valid_rate` 稳定
- `evidence grounding rate` 足够高
- `unknowns` 使用合理
- `discriminative_signals` 不全是模板
- `state_hypotheses` 有一定信息量且不过度幻觉

如果这些都过不了，再训 SFT 只会把问题蒸馏进去。
