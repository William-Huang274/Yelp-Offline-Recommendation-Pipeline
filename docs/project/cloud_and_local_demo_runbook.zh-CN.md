# 云端与本地运行说明

本文档记录本地 review/demo 路径和可选云端 Stage11 资产操作。仓库不保存云端密码、私钥或临时凭证。

## 本地验证入口

在仓库根目录运行：

```powershell
python tools/demo_recommend.py summary
python tools/cloud_stage11.py local-check
python tools/run_stage11_model_prompt_smoke.py
python tools/run_full_chain_smoke.py
```

这些命令读取当前 release surface、检查本地资产和运行轻量 smoke；不启动完整 Spark/GPU 训练。

## 本地 release surface

当前本地 release 摘要依赖：

- `data/output/current_release/stage09`
- `data/output/current_release/stage10`
- `data/output/current_release/stage11`
- `data/metrics/current_release`

## 云端连接配置

云端 helper 通过环境变量读取连接信息：

```powershell
$env:BDA_CLOUD_HOST="<cloud-host>"
$env:BDA_CLOUD_PORT="<ssh-port>"
$env:BDA_CLOUD_USER="<ssh-user>"
python tools/cloud_stage11.py inventory
```

如果不设置 `$env:BDA_CLOUD_PASSWORD`，脚本会交互式要求输入密码。不要把密码、token 或私钥写入仓库文件。

只打印当前 helper 识别的 SSH 配置和路径：

```powershell
python tools/cloud_stage11.py print-ssh
```

## 可选 Stage11 资产同步

轻量 freeze pack：

```powershell
python tools/cloud_stage11.py pull --item stage11_freeze_pack
```

完整 eval dump：

```powershell
python tools/cloud_stage11.py pull --item stage11_v120_eval_full
python tools/cloud_stage11.py pull --item stage11_v124_eval_full
```

Adapter 模型通常较大，默认不拉取；需要本地文件级验证时再显式使用 `--allow-large`。

## 本地 Stage09 调试入口

Windows PowerShell：

```powershell
.\tools\run_stage09_local.ps1
```

只检查前置路径和环境：

```powershell
.\tools\run_stage09_local.ps1 -CheckOnly
```

默认本地资源配置：

- `SPARK_MASTER=local[4]`
- `SPARK_DRIVER_MEMORY=6g`
- `SPARK_EXECUTOR_MEMORY=6g`
- `SPARK_SQL_SHUFFLE_PARTITIONS=24`
- `SPARK_LOCAL_DIR=data/spark-tmp`

这些 wrapper 配置不改变 label、candidate boundary、split 或 metric 定义。

## 本地 Stage10 Bucket5 调试入口

先检查本地资产：

```powershell
python tools/cloud_stage11.py local-check
```

如果缺少 source-parity 前置资产：

```powershell
python tools/cloud_stage11.py pull --item stage10_profile_sourceparity
python tools/cloud_stage11.py pull --item stage10_match_channels_sourceparity
```

运行 bucket5 wrapper：

```powershell
.\tools\run_stage10_bucket5_local.ps1
```

只检查前置资产：

```powershell
.\tools\run_stage10_bucket5_local.ps1 -CheckOnly
```

## 本地 Stage10 Bucket2 调试入口

确认 bucket2 Stage09 source-parity 资产：

```powershell
python tools/cloud_stage11.py local-check
python tools/cloud_stage11.py pull --item stage09_bucket2_sourceparity --allow-large
```

运行 bucket2 wrapper：

```powershell
.\tools\run_stage10_bucket2_local.ps1
```

只检查前置资产：

```powershell
.\tools\run_stage10_bucket2_local.ps1 -CheckOnly
```

更细的冷启动 cohort 可通过以下变量限定输入范围：

- `CANDIDATE_FUSION_USER_COHORT_PATH`
- `RANK_EVAL_USER_COHORT_PATH`

当前公开 headline release 表仍使用 aggregate `bucket2` 口径；更细 cohort replay 属于诊断入口。
