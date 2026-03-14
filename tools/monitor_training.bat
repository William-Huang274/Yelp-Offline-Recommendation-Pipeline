@echo off
REM Monitor DPO training status on Windows.

setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "LOG_PATH=%PROJECT_ROOT%\dpo_train_log.txt"

echo ==========================================
echo DPO Training Monitor
echo ==========================================
echo.

echo [1] Python processes
tasklist | findstr python.exe
echo.

echo [2] Recent training log lines
echo ------------------------------------------
if exist "%LOG_PATH%" (
  powershell -Command "Get-Content '%LOG_PATH%' -Tail 50 | Select-String -Pattern 'Step|loss|epoch|TRAIN|DONE'"
) else (
  echo log file not found: %LOG_PATH%
)
echo.

echo [3] GPU usage
echo ------------------------------------------
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo.

echo ==========================================
echo Monitor completed
echo ==========================================
echo.
pause
