@echo off
REM DPO training launcher based on prompt-length analysis
REM Provides multiple config choices

echo ========================================
echo DPO Pairwise Training Launcher
echo Prompt-Length-Informed Configuration

echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

echo Prompt length analysis:
echo   Mean prompt length: 432 tokens
echo   P95: 477 tokens
echo   P99: 525 tokens
echo.

echo Choose a configuration:
echo.
echo [1] Conservative profile (recommended) - MAX_SEQ_LEN=512
echo     - Data coverage: 98.7%%
echo     - Truncated samples: 460 (1.3%%)
echo     - VRAM target: ~7.5GB (safe)
echo     - LoRA rank: 8
echo.
echo [2] Optimized profile (aggressive) - MAX_SEQ_LEN=640
echo     - Data coverage: 99.9%%
echo     - Truncated samples: 28 (0.1%%)
echo     - VRAM target: ~8.5-9GB (may OOM)
echo     - LoRA rank: 6
echo.
echo [3] Ultra-low-memory profile - MAX_SEQ_LEN=384
echo     - Data coverage: 5.8%%
echo     - VRAM target: ~6GB
echo     - Use only when memory is extremely tight
echo.
set /p config_choice="Enter choice (1-3): "

if "%config_choice%"=="1" (
    echo.
    echo Loading conservative config (512)...
    for /f "delims=" %%i in (config\dpo_safe_512.env) do set %%i
    echo [OK] Config loaded
) else if "%config_choice%"=="2" (
    echo.
    echo Loading optimized config (640)...
    echo [WARN] This profile may approach the 8GB VRAM limit
    for /f "delims=" %%i in (config\dpo_optimized_640.env) do set %%i
    echo [OK] Config loaded
) else if "%config_choice%"=="3" (
    echo.
    echo Loading ultra-low-memory config (384)...
    echo [WARN] This profile truncates much more data and should only be used as a fallback
    for /f "delims=" %%i in (config\dpo_ultra_low_memory.env) do set %%i
    echo [OK] Config loaded
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Configuration Summary
echo ========================================
echo MAX_SEQ_LEN: %QLORA_MAX_SEQ_LEN%
echo LoRA Rank: %QLORA_LORA_R%
echo Batch Size: %QLORA_BATCH_SIZE%
echo Gradient Accumulation: %QLORA_GRAD_ACC%
echo DPO Max Pairs: %QLORA_DPO_MAX_PAIRS%
echo DPO Beta: %QLORA_DPO_BETA%
echo ========================================
echo.

set /p confirm="Start training? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting training...
echo Log will be saved to: dpo_train_log.txt
echo.

python scripts	_2_dpo_train.py 2>&1 | tee dpo_train_log.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed
    echo.
    echo Common causes:
    echo   1. CUDA OOM - try a more conservative config or reduce MAX_SEQ_LEN
    echo   2. Windows Error 1455 - increase the page file to 40GB+
    echo   3. Pair generation failed - check the dataset
    echo.
    echo Review the log with: type dpo_train_log.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed
echo ========================================
echo.
echo Output directory: data\output	_qlora_modelsecho.
echo Next steps:
echo   1. Review the log: type dpo_train_log.txt
echo   2. Run evaluation: python scripts	_3_qlora_sidecar_eval.py
echo.
pause
