param(
    [string]$Python = "python",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

function Latest-RunDir([string]$Root) {
    if (-not (Test-Path $Root)) {
        return $null
    }
    return Get-ChildItem -Directory $Root | Sort-Object Name | Select-Object -Last 1
}

function Require-Path([string]$Label, [string]$PathValue, [string]$Hint) {
    if (-not (Test-Path $PathValue)) {
        throw "$Label missing: $PathValue`n$Hint"
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$env:BDA_PROJECT_ROOT = $RepoRoot

$Stage09RunDir = if ($env:STAGE09_RUN_DIR) { $env:STAGE09_RUN_DIR } else { Join-Path $RepoRoot "data\output\09_candidate_fusion_structural_v5_sourceparity\20260324_030511_full_stage09_candidate_fusion" }
$TextMatchRunDir = if ($env:TEXT_MATCH_RUN_DIR) { $env:TEXT_MATCH_RUN_DIR } else { Join-Path $RepoRoot "data\output\09_candidate_wise_text_match_features_v1\20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build" }
$GroupGapRunDir = if ($env:GROUP_GAP_RUN_DIR) { $env:GROUP_GAP_RUN_DIR } else { Join-Path $RepoRoot "data\output\09_stage10_group_gap_features_v1\20260323_174757_full_stage09_stage10_group_gap_features_v1_build" }

$UserIntentRoot = if ($env:USER_INTENT_ROOT) { $env:USER_INTENT_ROOT } else { Join-Path $RepoRoot "data\output\09_user_intent_profile_v2_bucket5_sourceparity" }
$MatchChannelRoot = if ($env:MATCH_CHANNEL_ROOT) { $env:MATCH_CHANNEL_ROOT } else { Join-Path $RepoRoot "data\output\09_user_business_match_channels_v2_bucket5_sourceparity" }
$ProfileRunDir = if ($env:PROFILE_RUN_DIR) { $env:PROFILE_RUN_DIR } else { (Latest-RunDir $UserIntentRoot).FullName }
$MatchChannelRunDir = if ($env:MATCH_CHANNEL_RUN_DIR) { $env:MATCH_CHANNEL_RUN_DIR } else { (Latest-RunDir $MatchChannelRoot).FullName }

Require-Path "Stage09 run dir" $Stage09RunDir "Run .\tools\stage\run_stage09_local.ps1 or pull the canonical run from cloud if needed."
Require-Path "Text-match run dir" $TextMatchRunDir "This is required by the frozen Stage10 feature contract."
Require-Path "Group-gap run dir" $GroupGapRunDir "This is required by the frozen Stage10 feature contract."
Require-Path "Source-parity profile root" $UserIntentRoot "Run python tools/stage/cloud_stage11.py pull --item stage10_profile_sourceparity, or rebuild the profile asset locally."
Require-Path "Source-parity match-channel root" $MatchChannelRoot "Run python tools/stage/cloud_stage11.py pull --item stage10_match_channels_sourceparity, or rebuild the match-channel asset locally."
Require-Path "Profile run parquet" (Join-Path $ProfileRunDir "user_intent_profile_v2.parquet") "Use PROFILE_RUN_DIR to point to a valid profile run."
Require-Path "Match channel parquet" (Join-Path $MatchChannelRunDir "user_business_match_channels_v2_user_item.parquet") "Use MATCH_CHANNEL_RUN_DIR to point to a valid match-channel run."

$TrainOutputRoot = if ($env:TRAIN_OUTPUT_ROOT) { $env:TRAIN_OUTPUT_ROOT } else { Join-Path $RepoRoot "data\output\10_rank_models_joint_min_cls_v5_typed_intent_phase3_slicefix" }
$InferOutputRoot = if ($env:INFER_OUTPUT_ROOT) { $env:INFER_OUTPUT_ROOT } else { Join-Path $RepoRoot "data\output\10_2_rank_infer_eval_joint_min_cls_v5_typed_intent_phase3_slicefix" }
$FocusEvalOutputRoot = if ($env:FOCUS_EVAL_OUTPUT_ROOT) { $env:FOCUS_EVAL_OUTPUT_ROOT } else { Join-Path $RepoRoot "data\output\10_4_bucket5_focus_slice_eval_typed_phase3_slicefix" }
$MetricsPath = if ($env:METRICS_PATH) { $env:METRICS_PATH } else { Join-Path $RepoRoot "data\metrics\recsys_stage10_results_joint_min_cls_v5_typed_intent_phase3_slicefix.csv" }
$SparkTmp = Join-Path $RepoRoot "data\spark-tmp"

New-Item -ItemType Directory -Force -Path $TrainOutputRoot, $InferOutputRoot, $FocusEvalOutputRoot, (Split-Path $MetricsPath), $SparkTmp | Out-Null

$env:INPUT_09_RUN_DIR = $Stage09RunDir
$env:INPUT_09_TEXT_MATCH_RUN_DIR = $TextMatchRunDir
$env:INPUT_09_GROUP_GAP_RUN_DIR = $GroupGapRunDir
$env:INPUT_09_MATCH_CHANNELS_RUN_DIR = $MatchChannelRunDir
$env:OUTPUT_10_1_ROOT_DIR = $TrainOutputRoot
$env:OUTPUT_10_2_ROOT_DIR = $InferOutputRoot
$env:STAGE10_RESULTS_METRICS_PATH = $MetricsPath
$env:RANK_EVAL_USER_COHORT_PATH = if ($env:RANK_EVAL_USER_COHORT_PATH) { $env:RANK_EVAL_USER_COHORT_PATH } else { Join-Path $RepoRoot "data\output\fixed_eval_cohorts\bucket5_accepted_test_users_1935_userid.csv" }
$env:RANK_BLEND_ALPHA = if ($env:RANK_BLEND_ALPHA) { $env:RANK_BLEND_ALPHA } else { "0.15" }
$env:TRAIN_BUCKETS_OVERRIDE = if ($env:TRAIN_BUCKETS_OVERRIDE) { $env:TRAIN_BUCKETS_OVERRIDE } else { "5" }
$env:RANK_BUCKETS_OVERRIDE = if ($env:RANK_BUCKETS_OVERRIDE) { $env:RANK_BUCKETS_OVERRIDE } else { "5" }
$env:TRAIN_MODEL_BACKEND = if ($env:TRAIN_MODEL_BACKEND) { $env:TRAIN_MODEL_BACKEND } else { "xgboost_cls" }
$env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS = if ($env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS) { $env:TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS } else { "false" }
$env:RANK_DIAGNOSTICS_ENABLE = if ($env:RANK_DIAGNOSTICS_ENABLE) { $env:RANK_DIAGNOSTICS_ENABLE } else { "true" }
$env:ENABLE_STAGE10_V2_MATCH_CHANNELS = if ($env:ENABLE_STAGE10_V2_MATCH_CHANNELS) { $env:ENABLE_STAGE10_V2_MATCH_CHANNELS } else { "true" }
$env:ENABLE_STAGE10_V2_TEXT_MATCH = if ($env:ENABLE_STAGE10_V2_TEXT_MATCH) { $env:ENABLE_STAGE10_V2_TEXT_MATCH } else { "true" }
$env:ENABLE_STAGE10_V2_GROUP_GAP = if ($env:ENABLE_STAGE10_V2_GROUP_GAP) { $env:ENABLE_STAGE10_V2_GROUP_GAP } else { "true" }
$env:RANK_WRITE_USER_AUDIT = if ($env:RANK_WRITE_USER_AUDIT) { $env:RANK_WRITE_USER_AUDIT } else { "true" }
$env:RUN_BUCKET5_TYPED_FOCUS_EVAL = if ($env:RUN_BUCKET5_TYPED_FOCUS_EVAL) { $env:RUN_BUCKET5_TYPED_FOCUS_EVAL } else { "true" }

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

Write-Host "[INFO] Stage10 bucket5 local train+infer"
Write-Host "[INFO] repo=$RepoRoot"
Write-Host "[INFO] stage09=$Stage09RunDir"
Write-Host "[INFO] profile=$ProfileRunDir"
Write-Host "[INFO] match_channels=$MatchChannelRunDir"
Write-Host "[INFO] spark=$env:SPARK_MASTER driver=$env:SPARK_DRIVER_MEMORY executor=$env:SPARK_EXECUTOR_MEMORY"

if ($CheckOnly) {
    Write-Host "[PASS] Stage10 bucket5 local prerequisites are present."
    exit 0
}

& $Python (Join-Path $RepoRoot "scripts\10_1_rank_train.py")

$LatestTrain = Latest-RunDir $TrainOutputRoot
if (-not $LatestTrain) {
    throw "No Stage10 train output found under $TrainOutputRoot"
}
$env:RANK_MODEL_JSON = Join-Path $LatestTrain.FullName "rank_model.json"
Require-Path "Rank model" $env:RANK_MODEL_JSON "Training did not produce rank_model.json."

& $Python (Join-Path $RepoRoot "scripts\10_2_rank_infer_eval.py")

if ($env:RUN_BUCKET5_TYPED_FOCUS_EVAL -ne "true") {
    exit 0
}

$LatestInfer = Latest-RunDir $InferOutputRoot
if (-not $LatestInfer) {
    throw "No Stage10 infer output found under $InferOutputRoot"
}

$UserAuditPath = Join-Path $LatestInfer.FullName "bucket_5\user_diagnostics.parquet"
Require-Path "Stage10 user diagnostics" $UserAuditPath "Inference did not produce user_diagnostics.parquet."

$FocusEvalOutDir = Join-Path $FocusEvalOutputRoot $LatestInfer.Name
New-Item -ItemType Directory -Force -Path $FocusEvalOutDir | Out-Null

& $Python (Join-Path $RepoRoot "scripts\10_4_bucket5_focus_slice_eval.py") `
    --stage10-user-audit $UserAuditPath `
    --bucket-dir (Join-Path $Stage09RunDir "bucket_5") `
    --profile-run-dir $ProfileRunDir `
    --output-dir $FocusEvalOutDir

Write-Host "[INFO] focus_eval_output_dir=$FocusEvalOutDir"
