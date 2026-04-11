@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

if "%~1"=="" (
  echo usage: monitor_stage11_training.bat ^<log_path^>
  exit /b 1
)

python "%PROJECT_ROOT%\tools\monitor_stage11_training.py" "%~1"
