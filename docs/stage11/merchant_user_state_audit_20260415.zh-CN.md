# 商家状态小样本审计与 b5 用户分层样本分析（2026-04-15）

这份记录回答两个问题：

1. 当前 `merchant_state_v2` 小样本 prompt 审计看到了什么问题。
2. 为什么当前直接让大模型做 `raw evidence -> full profile` 的 SFT，学不出细腻的语义区分度。

## 1. 商家状态小样本 prompt 审计

脚本：

- [09_merchant_state_v2_audit_sample.py](D:\5006_BDA_project\scripts\09_merchant_state_v2_audit_sample.py)

当前本地结果：

- [merchant_state_v2_audit_summary.json](D:\5006_BDA_project\data\output\09_merchant_state_v2_audit\20260415_191030_full_stage09_merchant_state_v2_audit_sample\merchant_state_v2_audit_summary.json)
- [merchant_state_v2_sample.json](D:\5006_BDA_project\data\output\09_merchant_state_v2_audit\20260415_191030_full_stage09_merchant_state_v2_audit_sample\merchant_state_v2_sample.json)
- [merchant_state_v2_prompt_audit.jsonl](D:\5006_BDA_project\data\output\09_merchant_state_v2_audit\20260415_191030_full_stage09_merchant_state_v2_audit_sample\merchant_state_v2_prompt_audit.jsonl)

### 1.1 覆盖层面

- 商家总数：`4518`
- `photo_cover_rate = 0.5795`
- `photo_caption_cover_rate = 0.4327`
- `sparse_merchant_ratio = 0.2594`
- `photo_cover_on_sparse = 0.2841`
- `avg_state_signal_count_v1 = 12.25`

这说明：

- 现有 merchant 资产已经足够做一轮 prompt-only schema 审计
- `photo summary` 对 sparse merchant 是有效补强，但不是主干
- review/tip 仍然是 merchant 语义的主信号来源

### 1.2 直接看到的语义噪声

当前 `merchant_persona_schema_v1` 里已经有一批明显会误导后续用户模型的样本：

- `CVS Pharmacy`、`Target`、`Shopping Center` 这类非典型餐饮主体，被抽出了 `nightlife / quick_bite / late_night`
- `Popeyes`、`MOOYAH`、`BEN'S Burgers` 这类快餐/汉堡店，主菜系会漂到 `dessert_bakery`
- `Whole Foods Market`、`Auction House Market` 这类复合主体，会混出 `coffee_tea / sushi / dessert / seafood` 等杂糅语义

这些不是个别样本，而是说明：

**当前 merchant 侧的规则资产在“边界主体”“复合主体”“稀疏主体”上，存在真实的 schema 噪声。**

如果不先把这层审计和 canonicalization 做掉，后面用户状态模型学到的不是细粒度偏好，而是 merchant 语义噪声。

## 2. 当前 direct SFT 为什么学不出细腻区分

这里的“学不细”不是一句话原因，而是 4 个原因叠加。

### 2.1 teacher target 熵太低，而且天然模板化

对当前本地 `user_preference_target_schema_v1` 的统计：

- `preferred_cuisines` 为空的用户占比：`33.64%`
- `open_discoveries` 为空的用户占比：`100%`
- 头部字段分布极度集中：
  - `dinner = 9055`
  - `late_night = 8962`
  - `family_friendly = 8458`
  - `nightlife = 8296`
  - `group_dining = 8080`
  - `takeout = 8879`
  - `weekend = 8035`

这意味着：

- teacher 本身就在鼓励模型学“头部通用画像”
- 很多用户 target 里根本没有足够强的细分主菜系
- `discoveries` 在 teacher 里又是空的  

所以模型最容易学会的是：
- 把 JSON 填稳
- 把高频 meals/scenes/properties 填上

而不是：
- 做真正细粒度区分

### 2.2 当前用户输入虽然更原始，但仍缺“完整上下文”

`b5` 三档用户中，当前输入的主要构成还是：

- 清洗后的 review/tip 证据流
- 固定最近 `8` 条行为
- 正负 anchor 事件

问题在于：

- 低密度用户本来就只有少量 review/tip
- 高密度用户虽然证据厚，但 recent sequence 仍然只看最近 `8` 条
- review 证据是按 `tag/weight` 聚合过的片段，不是完整评论上下文

所以模型能学到：
- 稳定高频模式

但很难学到：
- 条件偏好
- 冲突偏好
- 容忍度
- 上下文依赖偏好

### 2.3 merchant 语义噪声直接污染用户状态学习

用户最近事件和 anchor 现在依赖的 merchant 语义，本身已经存在：

- 非餐饮主体被当成餐饮场景
- 快餐汉堡店被抽成 `dessert_bakery`
- 复合主体被抽成多头杂糅 cuisine

这会导致用户模型面对同样的用户证据时：

- 学到“这个用户偏 nightlife/quick_bite”
- 但这不一定来自真实餐饮偏好，可能来自 merchant schema 噪声

一句话：

**用户 SFT 学不细，不只是用户侧问题，商家侧状态不稳是直接上游原因。**

### 2.4 当前任务定义本身就把“状态学习”和“解释输出”混在了一起

当前 `11_2` 要一次性学会：

- 用户长期偏好
- 用户近期偏好
- 冲突解决
- 容忍度
- JSON 格式
- evidence refs
- open discoveries

对 SFT 来说，这个任务太重了。

它更容易先学会：

- 高概率 schema 模板
- 头部标签模式

而不是：

- 更细的用户差异

## 3. b5 分层样本直观看到的现象

结合当前 `35B step225` 的样本：

- 低密度用户：
  - 模型经常给出泛化的 `cajun_creole / family_friendly / nightlife / group_dining`
  - 即使 target 更细，也容易被吸回头部模式

- 中密度用户：
  - 模型已经能做一部分 cuisine/meal 区分
  - 但 scene/property 仍容易套模板

- 高密度用户：
  - 模型开始有更好的 cuisine/meal 表现
  - 但 scene、property、discoveries 还是容易写得泛而全

这说明当前 SFT 的真实角色更像：

**一个高质量结构化解释器**

而不是：

**真正的用户状态学习器**

## 4. 现在最该做的事

### 4.1 先做 merchant_state_v2 的 prompt-only 审计，不急着 merchant SFT

原因：

- 先定 schema 边界
- 先抓出 noisy merchants
- 先决定 canonical label 集合

### 4.2 先做 user_state_model_v1 的判别任务

当前已经准备好第一版训练资产：

- [09_user_state_training_assets_v1_build.py](D:\5006_BDA_project\scripts\09_user_state_training_assets_v1_build.py)
- [user_state_training_assets_v1_summary.json](D:\5006_BDA_project\data\output\09_user_state_training_assets_v1\20260415_191252_full_stage09_user_state_training_assets_v1_build\user_state_training_assets_v1_summary.json)

当前资产情况：

- 用户总数：`9197`
- 有 pairwise 样本的用户：`4839`
- pairwise 样本总数：`13818`
- 三档密度都有覆盖
- `recent_shift / service_risk / novelty_style` 不再是全 `unknown`

### 4.3 把下一轮 SFT 改成“解释层”，不要再让它直接学用户状态本体

下一轮 SFT 应该输入：

- `user_state`
- 少量 anchor 和 evidence slice
- merchant state refs

而不是继续：

- `raw evidence -> full profile`

## 5. 直接结论

当前 direct SFT 学不出细腻区分，不是因为模型不够大，而是：

1. teacher target 头部模式太强、discoveries 太弱
2. 用户输入缺少真正完整的上下文
3. merchant 侧状态噪声会直接污染用户建模
4. 当前 SFT 被要求同时学状态和解释，任务定义过重

所以现在正确方向不是继续硬拉一条 full-profile SFT，而是：

1. 先把 merchant_state_v2 审计和 canonicalization 做稳
2. 先训 user_state_model_v1
3. 再让 35B 做解释层 SFT
