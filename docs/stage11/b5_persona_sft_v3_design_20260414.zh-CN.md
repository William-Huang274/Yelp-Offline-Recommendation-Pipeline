# b5 Persona SFT v3 设计稿

这份设计稿用于下一轮 `b5` 用户偏好建模 `SFT`。目标不是继续训练一个“会复述 teacher 标签”的模型，而是训练一个**基于训练期证据进行结构化归纳**的用户画像抽取器。

当前设计原则：

- 输入优先使用清洗后的原始证据，不把派生 narrative 当主输入
- 输出既要保留稳定、可监督的核心字段，也要允许模型发现上下文依赖偏好和潜在偏好
- 所有开放发现都必须带证据引用和置信度
- 允许输出 `unknown`、`ambivalent`、`possible`，不强迫模型编造完整画像

## 1. 任务定义

`SFT v3` 的任务定义是：

- 输入：用户训练期行为、评论、tips、商家语义和序列证据
- 输出：严格 JSON
- 输出分两层：
  - `core_profile`：可监督、可评估、可直接接入排序链路
  - `open_discoveries`：允许模型结合上下文发现更细的偏好模式，但必须证据可回指

这轮任务不做：

- 自由文本人格小作文
- 直接输出 rerank 结果
- 直接学习 teacher narrative 表达风格

## 2. 输入设计

### 2.1 输入组成

下一轮 prompt 主体建议由这 5 块组成：

1. `User evidence stream`
2. `Recent event sequence`
3. `Positive anchor events`
4. `Negative anchor events`
5. `Behavior stats`

### 2.2 User evidence stream

用户文本不再切成大块 `positive evidence` / `negative evidence`，而是统一成证据流。每条证据保留最少必要元信息：

```json
{
  "evidence_id": "ev_12",
  "source": "review",
  "sentiment": "positive",
  "time_bucket": "older",
  "weight": 0.87,
  "text": "The oyster platter was fresh and the service was friendly even late at night."
}
```

要求：

- 只剔除无效文本、乱码、极短废话、重复文本
- 不提前做人设总结
- 可以保留轻量 `sentiment` 和 `weight`
- 不把 `long_term_top_cuisine`、`recent_intent_text` 这类 teacher-like 总结直接塞回输入

### 2.3 Recent event sequence

保留最近 `K` 条行为事件，当前默认还是 `K=8`。每条事件包含：

- 事件类型
- 商家主/副菜系
- city
- meal tags
- scene tags
- quality / reliability band
- merchant core text short

这部分用于让模型看到：

- 偏好是否稳定
- 偏好是否切换
- 行为模式是否随上下文变化

### 2.4 Positive / Negative anchor events

anchor 继续保留，因为它是高信号事件，不是 teacher 标签。  
但在输入里不再把它们解释成“你应该学到什么”，只作为高权重上下文提供。

### 2.5 Behavior stats

只保留少量统计量：

- `training_interactions_max`
- `training_events_total`
- `distinct_businesses`
- `recent_sequence_count`
- `repeat_business_ratio`
- `sequence_span_days`

这些字段辅助模型理解用户行为密度，不应成为主监督来源。

## 3. 输出 Schema

`SFT v3` 输出统一为严格 JSON，结构如下：

```json
{
  "core_profile": {
    "stable_preferences": {
      "preferred_cuisines": [],
      "preferred_meals": [],
      "preferred_scenes": [],
      "preferred_properties": [],
      "top_city": "",
      "geo_style": "hyperlocal|metro|mixed|unknown"
    },
    "avoid_signals": {
      "avoided_cuisines": [],
      "avoided_scenes": [],
      "service_risk_sensitivity": "low|medium|high|unknown"
    },
    "recent_preferences": {
      "recent_focus_cuisines": [],
      "recent_focus_meals": [],
      "recent_focus_scenes": [],
      "recent_focus_properties": [],
      "recent_shift": "stable|switching|broadening|unknown"
    },
    "behavior_summary": {
      "social_mode": [],
      "time_mode": [],
      "novelty_style": "low|medium|high|unknown"
    }
  },
  "open_discoveries": [],
  "evidence_refs": {
    "positive_event_refs": [],
    "negative_event_refs": [],
    "positive_evidence_refs": [],
    "negative_evidence_refs": []
  },
  "confidence": {
    "overall": "low|medium|high"
  },
  "ambivalent_signals": {
    "cuisines": [],
    "scenes": []
  }
}
```

## 4. Open Discoveries 字段定义

`open_discoveries` 是这轮最关键的新部分。它允许模型在不脱离证据的前提下，发现 teacher 没显式编码的模式。

每条 discovery 统一结构：

```json
{
  "type": "context_dependent_preference",
  "claim": "user may prefer lively late-night venues in group settings but not for routine meals",
  "confidence": "medium",
  "evidence_refs": ["event_2", "event_5", "ev_7"],
  "reason_short": "recent late-night group visits are positive, but solo negative evidence is more cautious"
}
```

### 4.1 允许的 discovery type

第一版建议限制为以下 6 类，避免模型乱造：

1. `context_dependent_preference`
2. `tolerance_hypothesis`
3. `scenario_specific_preference`
4. `latent_preference`
5. `preference_conflict`
6. `uncertainty_note`

### 4.2 各类定义

`context_dependent_preference`

- 某个偏好只有在特定场景、时间或社交上下文下成立

`tolerance_hypothesis`

- 用户对等待、噪音、服务不稳定、价格波动等的容忍度假设

`scenario_specific_preference`

- 用户在 group / family / date / quick bite 等不同场景下偏好不同

`latent_preference`

- teacher 未明确编码，但模型从多条证据中归纳出的潜在倾向

`preference_conflict`

- 用户存在冲突偏好，或近期与长期偏好冲突

`uncertainty_note`

- 证据不足时显式说明不确定点，避免模型硬填

## 5. 监督策略

### 5.1 core_profile

`core_profile` 继续用清洗后的 teacher schema 监督：

- 这是输出范式的主监督
- 也是下游排序可直接复用的结构化字段

### 5.2 open_discoveries

`open_discoveries` 不建议做死监督。  
第一版更稳的方式：

- 只用少量弱监督模板
- 或让 teacher 只给 very coarse candidate
- 模型在 constrained type set 下自行归纳

也就是说：

- `core_profile` 强监督
- `open_discoveries` 弱监督

这样才能避免模型继续只学 teacher 的表达方式。

## 6. Prompt 模板

下面是一版可直接进入下一轮实现的 prompt 模板。

```text
Task: infer a structured user preference profile from training-period evidence only.

Output rules:
- return exactly one valid JSON object
- start with '{' and end with '}'
- do not repeat the JSON
- do not echo the prompt
- if evidence is weak, use empty arrays or "unknown"
- do not invent unsupported cuisines, scenes, properties, or behavior styles
- every open discovery must include evidence_refs and confidence
- use open_discoveries only when there is enough evidence

Required output schema:
{...JSON schema here...}

User evidence stream:
- ev_1: source=review; sentiment=positive; weight=0.92; time_bucket=older; text=...
- ev_2: source=tip; sentiment=neutral; weight=0.66; time_bucket=recent; text=...
- ev_3: source=review; sentiment=negative; weight=0.88; time_bucket=recent; text=...

Behavior stats:
- training_interactions_max: ...
- training_events_total: ...
- distinct_businesses: ...
- recent_sequence_count: ...
- repeat_business_ratio: ...
- sequence_span_days: ...

Recent event sequence:
- event_1: ...
- event_2: ...

Positive anchor events:
- event_3: ...
- event_6: ...

Negative anchor events:
- event_4: ...

Return the JSON object now.
```

## 7. 评估建议

下一轮不只看 JSON 格式，还要看下面 5 类指标：

1. `JSON valid rate`
2. `core_profile field F1`
3. `evidence grounding rate`
4. `hallucination rate`
5. `open_discovery usefulness`

其中 `open_discovery usefulness` 至少要做两层检查：

- discovery 是否有证据支撑
- discovery 是否能为后续 `Stage10 / Stage11` 提供增量信息

## 8. 当前最推荐的落地顺序

1. 先把 `v3` 的 `user evidence stream` packing 做出来
2. 重新做 full token audit
3. 按 `3072 / 3584 / 4096` 三档比较覆盖率
4. 通过后再启动 full `SFT v3`

当前不建议直接开训。  
先把输入组织做对，比盲目继续加数据更重要。
