# Environment Setup

## Purpose

This note separates two environments:

1. a lightweight review environment for frozen assets, documentation, validation,
   and the demo CLI
2. a heavier training environment for full `Stage09 -> Stage11` reproduction

## 1. Review-Only Environment

This is the recommended environment for lecturers, reviewers, and project-demo
dry runs.

### Python

- Python `3.10+`
- Install CPU-friendly repository dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

These dependencies are sufficient for:

- reading the release surface
- running the validators
- using the demo CLI
- using the Stage11 cloud inventory / pull helper
- inspecting current documentation

### Recommended Commands

```bash
python tools/release/run_release_checks.py
python tools/demo/demo_recommend.py summary
python tools/stage/cloud_stage11.py local-check
```

For Windows PowerShell:

```powershell
.\tools\release\run_release_checks.ps1
```

## 2. Full Pipeline Environment

Use this only if you intend to rerun stage scripts or rebuild artifacts beyond
the checked-in frozen release surface.

### Required Components

- Python `3.10+`
- Java compatible with the local Spark distribution
- Apache Spark `3.5+`
- Local Hadoop Windows binaries when running Spark on Windows
- Enough local disk for parquet intermediates and Spark temp files

### Base Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### Optional GPU / Stage11 Dependencies

`Stage11` reward-model training is GPU-oriented and uses the extra dependencies
listed in [../../requirements-stage11-qlora.txt](../../requirements-stage11-qlora.txt):

```bash
python -m pip install -r requirements-stage11-qlora.txt
```

These extras are not required for review-only workflows.

Current Stage11 model surfaces:

- frozen reward-model mainline: `Qwen3.5-9B`
- current public checks only cover the frozen reward-model line

The quickest public verification for that split is:

```bash
python tools/stage/run_stage11_model_prompt_smoke.py
```

## 3. Windows-Specific Spark Notes

The older ingest and local Spark scripts assume:

- `winutils.exe` and `hadoop.dll` exist under `tools/hadoop/bin` or
  `%HADOOP_HOME%\bin`
- Spark temp folders point to a writable local disk

The parquet ingest helper checks these paths directly:

- [../../scripts/stage01_to_stage08/01_data prep.py](../../scripts/stage01_to_stage08/01_data%20prep.py)

Recommended local environment variables for Windows review and Spark work:

```powershell
$env:SPARK_LOCAL_MASTER = "local[2]"
$env:SPARK_DRIVER_MEMORY = "6g"
$env:SPARK_EXECUTOR_MEMORY = "6g"
$env:TEMP = "D:\\tmp"
$env:TMP = "D:\\tmp"
$env:TMPDIR = "D:\\tmp"
```

## 4. Raw Data Placement

The repository does not version the original Yelp source dumps. For a full local
rebuild, place the raw files in the expected local locations or override with
environment variables.

Typical raw assets are:

- `business.json`
- `checkin.json`
- `tip.json`
- `user.json`
- `review.json`
- optional `Yelp-Photos.zip`

See [data_lineage_and_storage.md](./data_lineage_and_storage.md) for the
storage-path contract.

## 5. Launchers vs Review Tools

Use these when you want the canonical pipeline entry points:

- [../../scripts/launchers/README.md](../../scripts/launchers/README.md)

Use these when you want review-time validation only:

- [../../tools/release/validate_public_surface.py](../../tools/release/validate_public_surface.py)
- [../../tools/release/validate_current_release.py](../../tools/release/validate_current_release.py)
- [../../tools/release/run_release_checks.py](../../tools/release/run_release_checks.py)
- [../../tools/release/run_full_chain_smoke.py](../../tools/release/run_full_chain_smoke.py)
- [../../tools/stage/run_stage11_model_prompt_smoke.py](../../tools/stage/run_stage11_model_prompt_smoke.py)
- [../../tools/stage/cloud_stage11.py](../../tools/stage/cloud_stage11.py)

Cloud helper notes:

- `paramiko` is included in [../../requirements.txt](../../requirements.txt).
- `BDA_CLOUD_HOST`, `BDA_CLOUD_PORT`, and `BDA_CLOUD_USER` can override the
  default cloud endpoint.
- `BDA_CLOUD_PASSWORD` can be set temporarily in the shell, or omitted so the
  helper prompts interactively. Do not write passwords into repository files.

## 6. Installation Evidence For Final Report

The briefing asks the final report to include installation instructions for
additional tools. The minimal list to report is:

- Python version used
- `requirements.txt`
- whether `requirements-stage11-qlora.txt` was needed
- Java / Spark version
- whether Windows Hadoop binaries were needed
- whether Git Bash / WSL / Linux shell was used for `.sh` launchers
