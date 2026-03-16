@echo off
REM DPO training launcher - low memory configuration
REM Suitable for a 4060 Laptop with 8GB VRAM

echo ========================================
echo DPO Pairwise Training - Low Memory Mode
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Select configuration
echo Select configuration:
echo [1] Standard Low Memory (recommended)
echo [2] Ultra Low Memory (maximum stability)
echo [3] Custom (use existing environment variables)
echo.
set /p config_choice="Enter choice (1-3): "

if "%config_choice%"=="1" (
    echo Loading standard low-memory config...
    set QLORA_MAX_SEQ_LEN=512
    set QLORA_LORA_R=8
    set QLORA_LORA_ALPHA=16
    set QLORA_BATCH_SIZE=1
    set QLORA_GRAD_ACC=16
    set QLORA_DPO_MAX_PAIRS=4
    set QLORA_DPO_MAX_PROMPT_LENGTH=384
    set QLORA_DPO_MAX_TARGET_LENGTH=8
) else if "%config_choice%"=="2" (
    echo Loading ultra-low-memory config...
    set QLORA_MAX_SEQ_LEN=384
    set QLORA_LORA_R=4
    set QLORA_LORA_ALPHA=8
    set QLORA_BATCH_SIZE=1
    set QLORA_GRAD_ACC=32
    set QLORA_DPO_MAX_PAIRS=2
    set QLORA_DPO_MAX_PROMPT_LENGTH=256
    set QLORA_DPO_MAX_TARGET_LENGTH=8
    set QLORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
) else if "%config_choice%"=="3" (
    echo Using existing environment variables...
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

REM Common configuration
set QLORA_USE_4BIT=true
set QLORA_USE_BF16=true
set QLORA_GRADIENT_CHECKPOINTING=true
set QLORA_DPO_BETA=0.1
set QLORA_DPO_LOSS_TYPE=sigmoid
set QLORA_DPO_PREFER_EASY_NEG=true
set QLORA_EPOCHS=1.0
set QLORA_LR=5e-5
set QLORA_BASE_MODEL=Qwen/Qwen3-4B
set QLORA_TRUST_REMOTE_CODE=true
set BUCKETS_OVERRIDE=10

echo.
echo ========================================
echo Configuration Summary
echo ========================================
echo Max Seq Length: %QLORA_MAX_SEQ_LEN%
echo LoRA Rank: %QLORA_LORA_R%
echo Batch Size: %QLORA_BATCH_SIZE%
echo Gradient Accumulation: %QLORA_GRAD_ACC%
echo DPO Max Pairs: %QLORA_DPO_MAX_PAIRS%
echo DPO Beta: %QLORA_DPO_BETA%
echo ========================================
echo.

REM Confirm launch
set /p confirm="Start training? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting DPO training...
echo Log will be saved to: dpo_train_log.txt
echo.

REM Run training
python scripts	_2_dpo_train.py 2>&1 | tee dpo_train_log.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. Check dpo_train_log.txt for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed successfully
echo ========================================
echo.
echo Next steps:
echo 1. Check output in: data\output	_qlora_modelsecho 2. Run evaluation: python scripts	_3_qlora_sidecar_eval.py
echo.
pause
