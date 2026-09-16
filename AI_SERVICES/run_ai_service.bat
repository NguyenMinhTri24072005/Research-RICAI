@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ===============================================================================
    echo [CANH BAO] Chua co moi truong ao AI_SERVICES\.venv
    echo Vui long double click vao file setup_ai_service.bat truoc de cai dat.
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

echo ===============================================================================
echo [*] DANG KHOI DONG RICE VISION AI INFERENCE SERVICE...
echo     Dia chi may chu  : http://localhost:8000
echo     Swagger API Docs : http://localhost:8000/docs
echo     Kiem tra Status  : http://localhost:8000/api/status
echo ===============================================================================
echo.

"%~dp0.venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo [THONG BAO] Server da dung hoac xay ra loi.
    pause
)
endlocal
