@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Chua co moi truong CAPTURE_APP\.venv.
    echo Hay chay setup_capture_app.bat truoc.
    pause
    exit /b 1
)
set "PYTHONPATH=%~dp0src"
"%~dp0.venv\Scripts\python.exe" -m rice_capture
if errorlevel 1 pause
endlocal
