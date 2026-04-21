# 云端 Stage11 与本地 Stage01-10 调试说明

这份说明用于当前演示分工：

- `Stage01-10`：本地可调试，默认使用保守 Spark 配置。
- `Stage11`：训练和大模型推理继续放在云端，demo 时可直接连云端查看。

仓库不会保存云端密码。运行云端工具时，可以临时设置环境变量，或让脚本交互式输入密码。

## 本地 demo 先验检查

先在仓库根目录运行：

```powershell
python tools/demo_recommend.py
python tools/cloud_stage11.py local-check
```

`demo_recommend.py` 不带参数时默认打印 `summary`，适合 VS Code 直接点击运行。

## 当前本地缺口

本地已经有 current release 摘要结果，足够支撑 PPT 和轻量 demo：

- `data/output/current_release/stage09`
- `data/output/current_release/stage10`
- `data/output/current_release/stage11`
- `data/metrics/current_release`

本地 Stage10 bucket5 完整重跑还缺两个 source-parity 前置资产。如果不在本地重新生成，需要从云端拉：

- `stage10_profile_sourceparity`
- `stage10_match_channels_sourceparity`

对应命令：

```powershell
python tools/cloud_stage11.py pull --item stage10_profile_sourceparity
python tools/cloud_stage11.py pull --item stage10_match_channels_sourceparity
```

这两个资产在云端约为 `32M` 和 `132M`，适合拉到本地用于 Stage10 调试。

如果要把 `bucket2` 冷启动线也在本地重放，当前还缺一份 `Stage09 bucket2 source-parity` 候选根目录：

- `stage09_bucket2_sourceparity`

对应命令：

```powershell
python tools/cloud_stage11.py pull --item stage09_bucket2_sourceparity --allow-large
```

这份目录大约 `2.8G`，默认不建议随手拉，但如果你要在本地调试 `bucket2`
或更细的 `0-3 / 4-6` 交互冷启动 cohort，这份资产是最直接的入口。

## 云端 Stage11 inventory

建议先只做只读 inventory：

```powershell
$env:BDA_CLOUD_HOST="connect.westb.seetacloud.com"
$env:BDA_CLOUD_PORT="20804"
$env:BDA_CLOUD_USER="root"
python tools/cloud_stage11.py inventory
```

如果不设置 `$env:BDA_CLOUD_PASSWORD`，脚本会交互式要求输入密码。不要把密码写入仓库文件。
`paramiko` 已经包含在根目录 [../../requirements.txt](../../requirements.txt) 中，安装
`requirements.txt` 后即可使用这个云端 helper。

只想打印 SSH 命令和已知路径：

```powershell
python tools/cloud_stage11.py print-ssh
```

## Stage11 哪些需要拉，哪些不建议拉

轻量审阅建议拉：

```powershell
python tools/cloud_stage11.py pull --item stage11_freeze_pack
```

如果要本地看完整 score dump，再拉：

```powershell
python tools/cloud_stage11.py pull --item stage11_v120_eval_full
python tools/cloud_stage11.py pull --item stage11_v124_eval_full
```

不建议默认拉 adapter 模型：

- `stage11_v101_11_30_adapter`
- `stage11_v117_31_60_adapter`
- `stage11_v122_61_100_adapter`

这些每个接近 `700M-900M`，demo 时直接留在云端更合理。只有需要离线证明模型文件存在时，才加 `--allow-large` 显式拉取。

## 本地 Stage09 调试入口

Windows PowerShell：

```powershell
.\tools\run_stage09_local.ps1
```

只检查本地路径和环境，不启动 Spark：

```powershell
.\tools\run_stage09_local.ps1 -CheckOnly
```

默认本地资源配置：

- `SPARK_MASTER=local[4]`
- `SPARK_DRIVER_MEMORY=6g`
- `SPARK_EXECUTOR_MEMORY=6g`
- `SPARK_SQL_SHUFFLE_PARTITIONS=24`
- `SPARK_LOCAL_DIR=data/spark-tmp`

这些配置只影响本地 wrapper，不改变 Stage09 的 label、candidate boundary、split 或 metric 定义。

## 本地 Stage10 bucket5 调试入口

先确保 source-parity 资产存在：

```powershell
python tools/cloud_stage11.py local-check
```

如果缺前置资产，先拉：

```powershell
python tools/cloud_stage11.py pull --item stage10_profile_sourceparity
python tools/cloud_stage11.py pull --item stage10_match_channels_sourceparity
```

然后运行：

```powershell
.\tools\run_stage10_bucket5_local.ps1
```

只检查 Stage10 bucket5 前置资产，不启动训练：

```powershell
.\tools\run_stage10_bucket5_local.ps1 -CheckOnly
```

这个 wrapper 只封装当前 Stage10 bucket5 主线，并使用保守本地 Spark 配置。它不会改动训练标签、候选边界、split 或 metric 定义。

## 本地 Stage10 bucket2 / 冷启动调试入口

先确认 `bucket2` 的 Stage09 source-parity 资产已经在本地：

```powershell
python tools/cloud_stage11.py local-check
python tools/cloud_stage11.py pull --item stage09_bucket2_sourceparity --allow-large
```

然后运行：

```powershell
.\tools\run_stage10_bucket2_local.ps1
```

只检查 `bucket2` 前置资产，不启动训练：

```powershell
.\tools\run_stage10_bucket2_local.ps1 -CheckOnly
```

如果你要单独重放更细的冷启动 cohort，例如 `0-3` 或 `4-6` 交互：

- Stage09 侧用 `CANDIDATE_FUSION_USER_COHORT_PATH` 指向目标 cohort CSV
- Stage10 侧用 `RANK_EVAL_USER_COHORT_PATH` 指向同一批用户

当前仓库的 headline release 表仍然只汇总公开 `bucket2` 口径；更细 cohort
replay 已经有脚本入口，但还没有冻结进 `current_release` 摘要表。

## 演示建议

现场 demo 推荐顺序：

1. 本地运行 `python tools/demo_recommend.py`，展示冻结结果总览。
2. 本地运行 `python tools/demo_recommend.py show-case --case boundary_11_30`，展示 Stage11 案例。
3. 云端运行 `python tools/cloud_stage11.py print-ssh` 或直接 SSH，展示 Stage11 产物路径和日志。
4. 如果老师问本地可复现性，运行 `python tools/cloud_stage11.py local-check`，说明 Stage01-10 本地调试入口和当前缺口。
