@echo off
setlocal
title Rice Vision AI - Phone Camera Capture Server (Port 8765)

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Moi truong ao .venv chua duoc tao!
    echo Hay chay setup_ai_service.bat truoc.
    pause
    exit /b 1
)

echo [*] DANG KHOI DONG PHONE CAMERA CAPTURE SERVER...
echo     Cong may chu: 8765 (HTTPS)
echo.

".venv\Scripts\python.exe" "capture_server\start.py" 8765

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Capture Server gap su co khi chay!
    pause
)
