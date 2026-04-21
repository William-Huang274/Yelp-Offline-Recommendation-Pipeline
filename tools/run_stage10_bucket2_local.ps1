param(
    [string]$Python = "python",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

function Latest-RunDir([string]$Root) {
    if (-not $Root -or -not (Test-Path $Root)) {
        return $null
    }
    return Get-ChildItem -Directory $Root | Sort-Object Name | Select-Object -Last 1
}

function Require-Path([string]$Label, [string]$PathValue, [string]$Hint) {
    if (-not (Test-Path $PathValue)) {
        Write-Host "[FAIL] $Label missing: $PathValue"
        Write-Host "[HINT] $Hint"
        exit 1
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:BDA_PROJECT_ROOT = $RepoRoot

$Stage09Root = if ($env:STAGE09_ROOT) { $env:STAGE09_ROOT } else { Join-Path $RepoRoot "data\output\09_candidate_fusion_bucket2_baseline_fixed_sourceparity" }
$Stage09RunDir = if ($env:STAGE09_RUN_DIR) { $env:STAGE09_RUN_DIR } else { (Latest-RunDir $Stage09Root).FullName }
$TextMatchRunDir = if ($env:TEXT_MATCH_RUN_DIR) { $env:TEXT_MATCH_RUN_DIR } else { Join-Path $RepoRoot "data\output\09_candidate_wise_text_match_features_v1\20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build" }
$GroupGapRunDir = if ($env:GROUP_GAP_RUN_DIR) { $env:GROUP_GAP_RUN_DIR } else { Join-Path $RepoRoot "data\output\09_stage10_group_gap_features_v1\20260323_174757_full_stage09_stage10_group_gap_features_v1_build" }
$FixedEvalCohortPath = if ($env:FIXED_EVAL_COHORT_PATH) { $env:FIXED_EVAL_COHORT_PATH } else { Join-Path $RepoRoot "data\output\fixed_eval_cohorts\bucket2_gate_eval_users_5344_useridx.csv" }

Require-Path "Stage09 bucket2 source-parity root" $Stage09Root "Run python tools/cloud_stage11.py pull --item stage09_bucket2_sourceparity, or set STAGE09_RUN_DIR to an existing bucket2 Stage09 run."
Require-Path "Stage09 bucket2 run dir" $Stage09RunDir "Use STAGE09_RUN_DIR to point to the exact bucket2 Stage09 run you want to replay."
Require-Path "Text-match run dir" $TextMatchRunDir "This is required by the frozen bucket2 Stage10 feature contract."
Require-Path "Group-gap run dir" $GroupGapRunDir "This is required by the frozen bucket2 Stage10 feature contract."
Require-Path "Stage10 bucket2 fixed eval cohort" $FixedEvalCohortPath "This fixed cohort defines the checked-in bucket2 evaluation boundary."

$TrainOutputRoot = if ($env:TRAIN_OUTPUT_ROOT) { $env:TRAIN_OUTPUT_ROOT } else { Join-Path $RepoRoot "data\output\10_rank_models_bucket2_fixedcohort_v3_1_local" }
$InferOutputRoot = if ($env:INFER_OUTPUT_ROOT) { $env:INFER_OUTPUT_ROOT } else { Join-Path $RepoRoot "data\output\10_2_rank_infer_eval_bucket2_fixedcohort_v3_1_local" }
$MetricsPath = if ($env:METRICS_PATH) { $env:METRICS_PATH } else { Join-Path $RepoRoot "data\metrics\recsys_stage10_results_bucket2_fixedcohort_v3_1_local.csv" }
$SparkTmp = Join-Path $RepoRoot "data\spark-tmp"

New-Item -ItemType Directory -Force -Path $TrainOutputRoot, $InferOutputRoot, (Split-Path $MetricsPath), $SparkTmp | Out-Null

$env:INPUT_09_RUN_DIR = $Stage09RunDir
$env:INPUT_09_TEXT_MATCH_RUN_DIR = $TextMatchRunDir
$env:INPUT_09_GROUP_GAP_RUN_DIR = $GroupGapRunDir
$env:OUTPUT_10_1_ROOT_DIR = $TrainOutputRoot
$env:OUTPUT_10_2_ROOT_DIR = $InferOutputRoot
$env:STAGE10_RESULTS_METRICS_PATH = $MetricsPath
$env:RANK_EVAL_USER_COHORT_PATH = $FixedEvalCohortPath
$env:TRAIN_BUCKETS_OVERRIDE = if ($env:TRAIN_BUCKETS_OVERRIDE) { $env:TRAIN_BUCKETS_OVERRIDE } else { "2" }
$env:RANK_BUCKETS_OVERRIDE = if ($env:RANK_BUCKETS_OVERRIDE) { $env:RANK_BUCKETS_OVERRIDE } else { "2" }
$env:TRAIN_MODEL_BACKEND = if ($env:TRAIN_MODEL_BACKEND) { $env:TRAIN_MODEL_BACKEND } else { "xgboost_cls" }
$env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS = if ($env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS) { $env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS } else { "false" }
$env:RANK_DIAGNOSTICS_ENABLE = if ($env:RANK_DIAGNOSTICS_ENABLE) { $env:RANK_DIAGNOSTICS_ENABLE } else { "false" }
$env:ENABLE_STAGE10_V2_TEXT_MATCH = if ($env:ENABLE_STAGE10_V2_TEXT_MATCH) { $env:ENABLE_STAGE10_V2_TEXT_MATCH } else { "true" }
$env:ENABLE_STAGE10_V2_GROUP_GAP = if ($env:ENABLE_STAGE10_V2_GROUP_GAP) { $env:ENABLE_STAGE10_V2_GROUP_GAP } else { "true" }
$env:STAGE10_V2_TEXT_MATCH_BUCKETS = if ($env:STAGE10_V2_TEXT_MATCH_BUCKETS) { $env:STAGE10_V2_TEXT_MATCH_BUCKETS } else { "2" }
$env:STAGE10_V2_GROUP_GAP_BUCKETS = if ($env:STAGE10_V2_GROUP_GAP_BUCKETS) { $env:STAGE10_V2_GROUP_GAP_BUCKETS } else { "2" }
$env:SPARK_MASTER = if ($env:SPARK_MASTER) { $env:SPARK_MASTER } else { "local[4]" }
$env:SPARK_DRIVER_MEMORY = if ($env:SPARK_DRIVER_MEMORY) { $env:SPARK_DRIVER_MEMORY } else { "6g" }
$env:SPARK_EXECUTOR_MEMORY = if ($env:SPARK_EXECUTOR_MEMORY) { $env:SPARK_EXECUTOR_MEMORY } else { "6g" }
$env:SPARK_LOCAL_DIR = if ($env:SPARK_LOCAL_DIR) { $env:SPARK_LOCAL_DIR } else { $SparkTmp }
$env:SPARK_SQL_SHUFFLE_PARTITIONS = if ($env:SPARK_SQL_SHUFFLE_PARTITIONS) { $env:SPARK_SQL_SHUFFLE_PARTITIONS } else { "24" }
$env:SPARK_DEFAULT_PARALLELISM = if ($env:SPARK_DEFAULT_PARALLELISM) { $env:SPARK_DEFAULT_PARALLELISM } else { "24" }
$env:SPARK_NETWORK_TIMEOUT = if ($env:SPARK_NETWORK_TIMEOUT) { $env:SPARK_NETWORK_TIMEOUT } else { "600s" }
$env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL = if ($env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL) { $env:SPARK_EXECUTOR_HEARTBEAT_INTERVAL } else { "60s" }
$env:TRAIN_CACHE_MODE = if ($env:TRAIN_CACHE_MODE) { $env:TRAIN_CACHE_MODE } else { "disk" }
$env:PYTHONUNBUFFERED = "1"

Write-Host "[INFO] Stage10 bucket2 local train+infer"
Write-Host "[INFO] repo=$RepoRoot"
Write-Host "[INFO] stage09=$Stage09RunDir"
Write-Host "[INFO] text=$TextMatchRunDir"
Write-Host "[INFO] group_gap=$GroupGapRunDir"
Write-Host "[INFO] eval_cohort=$FixedEvalCohortPath"
Write-Host "[INFO] spark=$env:SPARK_MASTER driver=$env:SPARK_DRIVER_MEMORY executor=$env:SPARK_EXECUTOR_MEMORY"
Write-Host "[INFO] finer cold-start slices (for example 0-3 or 4-6 interactions) can be replayed by overriding CANDIDATE_FUSION_USER_COHORT_PATH upstream and RANK_EVAL_USER_COHORT_PATH here."

if ($CheckOnly) {
    Write-Host "[PASS] Stage10 bucket2 local prerequisites are present."
    exit 0
}

& $Python (Join-Path $RepoRoot "scripts\10_1_rank_train.py")

$LatestTrain = Latest-RunDir $TrainOutputRoot
if (-not $LatestTrain) {
    throw "No Stage10 bucket2 train output found under $TrainOutputRoot"
}
$env:RANK_MODEL_JSON = Join-Path $LatestTrain.FullName "rank_model.json"
Require-Path "Rank model" $env:RANK_MODEL_JSON "Training did not produce rank_model.json."

& $Python (Join-Path $RepoRoot "scripts\10_2_rank_infer_eval.py")
