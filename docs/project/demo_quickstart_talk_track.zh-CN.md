# Demo 快速启动与讲解口径

这份文档不是完整工程 runbook，而是面向面试、老师演示和项目展示的压缩版讲稿。

目标只有两个：

1. 你知道 demo 从哪里开始。
2. 你知道每条命令跑出来之后该怎么说。

如果只想看完整工程材料，继续参考 [demo_runbook.md](./demo_runbook.md) 和
[cloud_and_local_demo_runbook.zh-CN.md](./cloud_and_local_demo_runbook.zh-CN.md)。

## 1. 先讲什么

先用一句话把项目定住，不要一上来就念命令：

`这是一个 Yelp 餐饮发现的离线排序与发布模拟项目，主线是 Stage09 召回路由、Stage10 XGBoost 结构化精排，以及 Stage11 bounded reward-model rescue rerank。`

然后补一句边界：

`它不是线上生产系统，没有真实流量和 A/B，但我保留了冻结结果、mock serving、fallback、load test 和全链路 smoke，用来证明训练-评估-发布闭环意识。`

这两句比先讲模型名更重要。

## 2. How To Start

### 最小启动顺序

在仓库根目录运行：

```powershell
python tools/run_release_checks.py --skip-pytest
python tools/demo_recommend.py summary
python tools/demo_recommend.py show-case --case boundary_11_30
python tools/batch_infer_demo.py --strategy reward_rerank
```

这 4 条已经够支撑大部分展示。

### 如果只想做 2 分钟快演示

```powershell
python tools/demo_recommend.py summary
python tools/demo_recommend.py show-case --case boundary_11_30
```

### 如果对方追问“有没有服务意识”

```powershell
python tools/mock_serving_api.py --self-test
python tools/load_test_mock_serving.py --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

### 如果对方追问“是不是能全链路跑通”

```powershell
python tools/run_stage01_11_minidemo.py
```

## 3. 推荐展示顺序

### 第一步：先展示冻结主线结果

命令：

```powershell
python tools/demo_recommend.py summary
```

你可以这样说：

- `我先不现场重训，而是先展示当前冻结发布线，因为现场 demo 的目标是讲清系统设计和结果，而不是赌一次临时训练。`
- `Stage09 的重点是尽量把真值保留在候选池里，bucket5 当前 truth_in_pretrim150 是 0.7451，hard_miss 是 0.1190。`
- `Stage10 是主排序 backbone，在 bucket2 / bucket5 / bucket10 都有正向增益，其中 bucket5 从 0.0935 / 0.0440 提升到 0.1261 / 0.0581。`
- `Stage11 不是全量重排，而是 bounded rescue。当前对外冻结线是 tri-band v124，指标是 0.1857 / 0.0838。`

这一步的重点不是把所有数字念完，而是把三层职责讲清楚：

- `Stage09` 负责保真
- `Stage10` 负责主排序
- `Stage11` 负责局部救援

### 第二步：讲一个 Stage11 具体案例

命令：

```powershell
python tools/demo_recommend.py show-case --case boundary_11_30
```

你可以这样说：

- `这个 case 的意义不是证明大模型无敌，而是说明为什么我把 LLM 限制在边界窗口。`
- `这里真值在 Stage10 下已经接近前排，但还没进最终理想位置；Stage11 只对边界候选做局部比较，把它从 learned_rank 17 推到了 final_rank 8。`
- `这说明 reward-model 更适合处理近邻候选之间的细粒度比较，而不是替代整个主排序。`

如果对方继续问“为什么不 full-list rerank”，就接：

- `因为 full-list rerank 成本高、延迟不稳、回滚难，我这里故意把它限制在 11-30 / 31-60 / 61-100 这样的 bounded window。`

### 第三步：展示 mock batch inference

命令：

```powershell
python tools/batch_infer_demo.py --strategy reward_rerank
```

你可以这样说：

- `这一步不是线上 API，只是一个受控 mock batch inference，用来把 release surface、候选 trace 和最终 top-k 串起来。`
- `你可以看到这里有一个候选从 stage10_rank=12 被救到了 final_rank=5，这正好对应 Stage11 的边界救援设计。`
- `输出里同时保留了 strategy、latency、fallback_count 和 release reference，这样演示时能把模型结果和发布口径连起来。`

### 第四步：如果需要，再补服务和 fallback

命令：

```powershell
python tools/mock_serving_api.py --self-test
python tools/load_test_mock_serving.py --request-sample-size 5 --warmup-requests 5 --requests 20 --concurrency 2 --strategy reward_rerank --stage09-mode lookup_live --stage10-mode xgb_live --stage11-mode replay --traffic-profile mixed --cache-miss-rate 0.2 --strategy-failure-rate 0.1 --xgboost-rate 0.1 --output data/output/serving_validation/latest_summary.json
python tools/export_serving_validation_report.py --input data/output/serving_validation/latest_summary.json --output docs/serving_validation_report.md --strict
```

你可以这样说：

- `这不是生产服务，而是本地 mock serving，用来证明我有服务暴露、策略切换和 fallback 的意识。`
- `load test 不是线上 A/B，只是演示在本地 replay 条件下也能看到多 request_id、混合流量、Stage09/10/11 分段延迟、cache miss 和 fallback。`
- `Markdown 报告把压测 JSON 固化下来，方便面试时解释哪些是 serving path，哪些只是 offline audit。`
- `所以这部分我不会夸成线上经验，但它能证明我不是只会离线打分。`

### 第五步：如果被问“你是不是能从头跑到尾”

命令：

```powershell
python tools/run_stage01_11_minidemo.py
```

你可以这样说：

- `这个 mini demo 不是完整 Spark/GPU 重训，而是 contract-level 的 Stage01->Stage11 轻量样本。`
- `它的作用是证明整个链路入口是连起来的，而不是临时拼接几个单点脚本。`
- `真正的 Stage09/10/11 训练仍然依赖完整数据和更重的运行环境，所以现场不做 live retrain。`

## 4. 一套 5 分钟讲法

如果面试官只给你 5 分钟，可以按下面讲：

1. `这是一个 Yelp 餐饮发现的离线推荐排序项目，主线是召回、精排和 bounded rerank。`
2. `Stage09 解决候选保真，Stage10 做主排序，Stage11 只在边界窗口做 reward-model rescue。`
3. `当前冻结 bucket5 主线里，Stage10 把 Recall@10 / NDCG@10 从 0.0935 / 0.0440 提升到 0.1261 / 0.0581。`
4. `Stage11 不做全量重排，而是对需要救援的候选做局部比较；我保留了案例、mock serving 和 fallback，用来表达上线前工程意识。`
5. `这个项目没有真实线上流量，但它不是 notebook 实验，而是一条有 release surface、验证脚本和 demo 入口的离线排序系统。`

## 5. 一套 30 秒讲法

如果对方只想先听一句项目定位，可以这么说：

`我做的是一个 Yelp 离线推荐排序项目，主线是 Stage09 召回路由、Stage10 XGBoost 精排和 Stage11 bounded reward-model rescue rerank。它没有真实线上流量，但我把冻结指标、mock serving、fallback 和 smoke checks 都保留下来了，用来证明训练-评估-发布闭环。`

## 6. 现场容易被问到的问题

### 1. 为什么不现场训练

回答：

- `现场训练不稳定，而且会把演示重点从系统设计和结果解释带偏。`
- `我现在展示的是冻结发布线，它更适合回答“系统怎么设计、结果怎么验证、为什么这样取舍”。`

### 2. 这个是不是线上系统

回答：

- `不是线上生产系统。`
- `它是离线排序与发布模拟项目，服务层是 mock serving，不是真实流量服务。`
- `但我故意保留了 release pointer、fallback、rollback 和 load test，这部分是为了证明上线意识。`

### 3. bucket2 / bucket5 / bucket10 是什么

回答：

- `这是按用户交互密度切的评测集。`
- `bucket2 更偏冷启动和轻量用户，bucket5 是当前主线的中高交互用户集，bucket10 是高交互用户集。`

### 4. Stage11 的 0.1857 / 0.0838 怎么理解

回答：

- `这是当前冻结 rescue 评估线，不是把全量 bucket5 用户都重排后的 headline。`
- `它对应的是 Stage11 在定义好的救援评估子集上的最终融合结果。`

### 5. 为什么你坚持 bounded rerank

回答：

- `因为我想要的是可控增益，而不是让 LLM 接管整个排序。`
- `bounded rerank 更容易控制成本、延迟、回滚风险，也更符合当前项目的资源边界。`

## 7. 不要这样讲

- 不要把 `Stage11` 讲成整个系统的主角。
- 不要把 `mock serving` 讲成真实线上服务。
- 不要把 `best-known line` 讲成当前冻结发布线。
- 不要一上来就念一串脚本名和目录名。
- 不要把整个项目讲成“我用了很多模型”，要讲成“我解决了哪一层问题”。

## 8. 推荐你实际演示时只开这几个窗口

1. 一个终端
2. 一个 [README.md](../../README.md)
3. 一个 [docs/recruiter_pitch.zh-CN.md](../recruiter_pitch.zh-CN.md)
4. 一个 [docs/stage11/stage11_case_notes_20260409.md](../stage11/stage11_case_notes_20260409.md)

这样够了，不要现场切太多目录。

## 9. 最后一句怎么收

可以这样收尾：

`这个项目最想证明的不是我能不能把大模型塞进推荐，而是我能不能把召回、精排、局部重排、评估和发布表面整理成一条可解释、可复核、可演示的排序链路。`
