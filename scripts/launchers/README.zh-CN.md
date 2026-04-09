# Launcher 入口说明

[English](./README.md) | [中文](./README.zh-CN.md)

这里是当前版本统一的对外 launcher 入口。

根目录下的长名 `run_stage*.sh` 继续保留，用于兼容旧调用；文档、演示和 release 说明统一引用这里的 wrapper。

## 命名方式

- `stage09_*`：Stage09 主线与 Stage11 前置资产
- `stage10_*`：Stage10 训练与评估主线
- `stage11_*`：Stage11 数据、导出、训练、评估与审计入口

## 当前 wrapper

### Stage09

- `stage09_bucket5_mainline.sh`
- `stage09_bucket5_typed_intent_assets.sh`
- `stage09_bucket5_stage11_assets.sh`

### Stage10

- `stage10_bucket5_mainline.sh`
- `stage10_bucket2_mainline.sh`
- `stage10_bucket10_mainline.sh`

### Stage11

- `stage11_bucket5_11_1.sh`
- `stage11_bucket5_export_only.sh`
- `stage11_bucket5_train.sh`
- `stage11_bucket5_eval.sh`
- `stage11_bucket5_watch.sh`
- `stage11_bucket5_constructability_audit.sh`
- `stage11_bucket5_pool_audit.sh`

## 变量命名口径

下面三个名字有明确分工，不能混用：

| 名称 | 含义 |
| --- | --- |
| `TOP_K` | 最终指标截断位，例如 Recall@10、NDCG@10 |
| `RERANK_TOPN` | 模型在最终排序前实际重打分的候选窗口 |
| `PAIRWISE_POOL_TOPN` | 只用于 Stage11 pairwise pool 挖样的 learned-rank 截断位 |

完整定义和各阶段映射见：

- [../../docs/contracts/launcher_env_conventions.zh-CN.md](../../docs/contracts/launcher_env_conventions.zh-CN.md)

## 共享路径约定

当前 active 兼容 launcher 统一使用：

- `scripts/launchers/_path_contract.sh`

这个文件集中维护远端仓库根目录、输出根目录、模型缓存目录、临时目录和
Python 回退路径，避免每个 launcher 各自写死绝对路径，同时保持默认展开后
的实际运行位置不变。

## Wrapper 规则

wrapper 保持薄封装：

- 转发到当前 active root launcher
- 原样透传环境变量
- 不承载业务逻辑
