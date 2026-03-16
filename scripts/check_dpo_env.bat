@echo off
REM One-click DPO environment check
REM Runs the main environment checks and saves a report

echo ========================================
echo Full DPO Environment Check
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

echo [1/3] Checking system memory and GPU...
echo.
python tools\check_memory.py > dpo_env_check.txt 2>&1
type dpo_env_check.txt
echo.

echo ========================================
echo [2/3] Checking dataset readiness...
echo.
python tools\check_dpo_dataset.py >> dpo_env_check.txt 2>&1
type dpo_env_check.txt | findstr /C:"DPO Dataset Audit" /C:"Overall" /C:"estimated_pairs" /C:"usable for DPO training"
echo.

echo ========================================
echo [3/3] Checking Python packages...
echo.
echo Checking TRL... >> dpo_env_check.txt
python -c "import trl; print(f'TRL version: {trl.__version__}')" >> dpo_env_check.txt 2>&1
if errorlevel 1 (
    echo [WARN] TRL not installed >> dpo_env_check.txt
    echo [WARN] TRL not installed. Install with: pip install trl^>=0.9
) else (
    python -c "import trl; print(f'TRL version: {trl.__version__}')"
)

echo Checking Transformers... >> dpo_env_check.txt
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" >> dpo_env_check.txt 2>&1
if errorlevel 1 (
    echo [WARN] Transformers not installed >> dpo_env_check.txt
    echo [WARN] Transformers not installed
) else (
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
)

echo Checking PEFT... >> dpo_env_check.txt
python -c "import peft; print(f'PEFT version: {peft.__version__}')" >> dpo_env_check.txt 2>&1
if errorlevel 1 (
    echo [WARN] PEFT not installed >> dpo_env_check.txt
    echo [WARN] PEFT not installed
) else (
    python -c "import peft; print(f'PEFT version: {peft.__version__}')"
)

echo Checking BitsAndBytes... >> dpo_env_check.txt
python -c "import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')" >> dpo_env_check.txt 2>&1
if errorlevel 1 (
    echo [WARN] BitsAndBytes not installed >> dpo_env_check.txt
    echo [WARN] BitsAndBytes not installed
) else (
    python -c "import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')"
)

echo.
echo ========================================
echo Check complete
echo ========================================
echo.
echo Full report saved to: dpo_env_check.txt
echo.
echo Next steps:
echo   1. If all checks pass, run: scriptsun_dpo_low_memory.bat
echo   2. If there are warnings, install the missing dependencies first
echo   3. Review the full report with: type dpo_env_check.txt
echo.
pause
