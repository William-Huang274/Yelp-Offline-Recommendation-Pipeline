param(
    [string]$Python = "python",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$env:BDA_PROJECT_ROOT = $RepoRoot
$env:PARQUET_BASE_DIR = if ($env:PARQUET_BASE_DIR) { $env:PARQUET_BASE_DIR } else { Join-Path $RepoRoot "data\parquet" }
$env:OUTPUT_ROOT_DIR = if ($env:OUTPUT_ROOT_DIR) { $env:OUTPUT_ROOT_DIR } else { Join-Path $RepoRoot "data\output\09_candidate_fusion_structural_v5_sourceparity" }
$env:OUTPUT_09_AUDIT_ROOT_DIR = if ($env:OUTPUT_09_AUDIT_ROOT_DIR) { $env:OUTPUT_09_AUDIT_ROOT_DIR } else { Join-Path $RepoRoot "data\output\09_recall_audit_sourceparity_v5" }
$env:AUDIT_USER_COHORT_CSV = if ($env:AUDIT_USER_COHORT_CSV) { $env:AUDIT_USER_COHORT_CSV } else { Join-Path $RepoRoot "data\output\09_bucket_roster_diff\20260324_accepted_vs_v5_bucket5\bucket_5_accepted_truth_userid_roster.csv" }

$SparkTmp = Join-Path $RepoRoot "data\spark-tmp"
New-Item -ItemType Directory -Force -Path $env:OUTPUT_ROOT_DIR, $env:OUTPUT_09_AUDIT_ROOT_DIR, $SparkTmp | Out-Null

if (-not (Test-Path $env:PARQUET_BASE_DIR)) {
    throw "PARQUET_BASE_DIR missing: $env:PARQUET_BASE_DIR"
}
if (-not (Test-Path $env:AUDIT_USER_COHORT_CSV)) {
    throw "AUDIT_USER_COHORT_CSV missing: $env:AUDIT_USER_COHORT_CSV"
}

$env:SPARK_MASTER = if ($env:SPARK_MASTER) { $env:SPARK_MASTER } else { "local[4]" }
$env:SPARK_DRIVER_MEMORY = if ($env:SPARK_DRIVER_MEMORY) { $env:SPARK_DRIVER_MEMORY } else { "6g" }
$env:SPARK_EXECUTOR_MEMORY = if ($env:SPARK_EXECUTOR_MEMORY) { $env:SPARK_EXECUTOR_MEMORY } else { "6g" }
$env:SPARK_SQL_SHUFFLE_PARTITIONS = if ($env:SPARK_SQL_SHUFFLE_PARTITIONS) { $env:SPARK_SQL_SHUFFLE_PARTITIONS } else { "24" }
$env:SPARK_DEFAULT_PARALLELISM = if ($env:SPARK_DEFAULT_PARALLELISM) { $env:SPARK_DEFAULT_PARALLELISM } else { "24" }
$env:SPARK_LOCAL_DIR = if ($env:SPARK_LOCAL_DIR) { $env:SPARK_LOCAL_DIR } else { $SparkTmp }
$env:SPARK_NETWORK_TIMEOUT = if ($env:SPARK_NETWORK_TIMEOUT) { $env:SPARK_NETWORK_TIMEOUT } else { "600s" }
$env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL = if ($env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL) { $env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL } else { "60s" }
$env:PYTHONUNBUFFERED = "1"

Write-Host "[INFO] Stage09 local run"
Write-Host "[INFO] repo=$RepoRoot"
Write-Host "[INFO] parquet=$env:PARQUET_BASE_DIR"
Write-Host "[INFO] output=$env:OUTPUT_ROOT_DIR"
Write-Host "[INFO] spark=$env:SPARK_MASTER driver=$env:SPARK_DRIVER_MEMORY executor=$env:SPARK_EXECUTOR_MEMORY"

if ($CheckOnly) {
    Write-Host "[PASS] Stage09 local prerequisites are present."
    exit 0
}

& $Python (Join-Path $RepoRoot "scripts\09_candidate_fusion.py")

$Latest = Get-ChildItem -Directory $env:OUTPUT_ROOT_DIR |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $Latest) {
    throw "No Stage09 run directory found under $env:OUTPUT_ROOT_DIR"
}

$env:INPUT_09_RUN_DIR = $Latest.FullName
Write-Host "[INFO] Stage09 latest run=$env:INPUT_09_RUN_DIR"
& $Python (Join-Path $RepoRoot "scripts\09_1_recall_audit.py")
