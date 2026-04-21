# 面向搜广推 / 机器学习岗位的项目说明

这不是一个只给离线分数的课程仓库，而是一个 Yelp 餐厅发现推荐系统的离线排序与发布模拟项目。

核心链路：

`用户 / 商户信号 -> Stage09 召回路由 -> 候选裁剪 -> Stage10 XGBoost 结构化精排 -> Stage11 Qwen3.5-9B reward-model 有边界救援重排 -> Top-K`

## 一句话定位

项目重点是推荐 / 搜索排序中的召回、精排、冷启动迁移和 bounded rerank。当前没有真实线上流量，但提供了 mock serving、release pointer、fallback / rollback、压测和全链路 smoke，用来证明训练-评估-发布闭环意识。

## 和搜广推岗位的对应关系

| 岗位关注点 | 仓库里的证据 |
| --- | --- |
| 召回漏斗 | Stage09 route-aware candidate routing，关注 truth retention 和 hard miss |
| 精排模型 | Stage10 `LearnedBlendXGBCls@10`，跨 `bucket2 / bucket5 / bucket10` 对比 |
| 重排 / rescue | Stage11 bounded reward-model rerank，只在 shortlist 上做有边界救援 |
| 冷启动 | `bucket2` 覆盖冷启动与轻量用户，脚本支持 `0-3` / `4-6` cohort |
| 发布意识 | `data/output/_prod_runs` 记录 champion / fallback / baseline pointer |
| 服务意识 | `tools/mock_serving_api.py` 提供 `/health` 和 `/rank` |
| 稳定性意识 | `config/serving.yaml` 管理策略、版本、fallback 和 latency budget |
| 可复现性 | `tools/run_full_chain_smoke.py` 和 `tools/run_stage01_11_minidemo.py` |

## 当前不是线上经验

需要明确：这个仓库没有真实用户流量、线上 A/B 实验、线上监控报警或生产事故处理记录。

它能证明的是：

- 能把离线排序链路拆成召回、精排、重排和评估层
- 能把实验结果整理成 frozen release surface
- 能用配置、fallback、rollback 和 mock serving 表达上线前工程思维
- 能解释为什么不做 full-list LLM reranking，而选择 bounded rerank

## 面试演示命令

```bash
python tools/run_stage01_11_minidemo.py
python tools/batch_infer_demo.py --strategy reward_rerank
python tools/mock_serving_api.py --self-test
python tools/load_test_mock_serving.py --requests 20 --concurrency 4 --simulate-fallback-every 5
python tools/run_release_checks.py --skip-pytest
```


