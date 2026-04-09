# 脚本目录说明

[English](./README.md) | [中文](./README.zh-CN.md)

这里保留当前版本需要直接使用的脚本表面。

历史脚本继续保留在本地，但不进入当前对外公开的脚本表面。

## 当前有效分区

### Stage01 到 Stage08

- [stage01_to_stage08](./stage01_to_stage08)

这部分仍属于仓库主线历史，继续保留可见；但它不属于当前可复现的 `stage09 -> stage11` 冻结链路。

### Stage09

当前有效 Stage09 表面包括：

- `09_candidate_fusion.py`
- `09_1_recall_audit.py`
- typed-intent 与匹配特征构建脚本
- Stage10 特征构建脚本
- Stage11 源材料与语义资产构建脚本

### Stage10

当前有效 Stage10 表面包括：

- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`
- `10_4_bucket5_focus_slice_eval.py`
- `bucket5`、`bucket2`、`bucket10` 的 launcher

### Stage11

当前有效 Stage11 表面包括：

- `11_1_qlora_build_dataset.py`
- `11_2_dpo_export_pairs.py`
- `11_2_rm_train.py`
- `11_3_qlora_sidecar_eval.py`
- compact export / train / eval / watcher launcher
- 当前 Stage11 审计脚本

### 共享支撑

- `pipeline/`
- [launchers/](./launchers)：当前对外稳定入口

## Launcher 规则

根目录下的 `run_stage*.sh` 仍然保留，用于兼容旧调用。

对外展示、文档引用和演示入口统一指向：

- [launchers/](./launchers)

launcher 变量命名口径见：

- [../docs/contracts/launcher_env_conventions.zh-CN.md](../docs/contracts/launcher_env_conventions.zh-CN.md)

## 清理规则

脚本保留在 active surface，至少要满足以下一项：

1. 被当前 launcher 直接调用
2. 属于当前 release 范围
3. 被当前主链 helper 依赖

不满足以上条件的脚本，统一进入本地归档区，不再放在当前对外脚本表面。
