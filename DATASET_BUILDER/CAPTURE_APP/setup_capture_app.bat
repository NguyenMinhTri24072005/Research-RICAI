@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PYTHON_CMD="
where pyenv >nul 2>nul
if not errorlevel 1 (
    pyenv exec python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=pyenv exec python"
)

if not defined PYTHON_CMD (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)

if not defined PYTHON_CMD (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Khong tim thay Python 3.12 x64.
    echo Hay cai Python 3.12 tu python.org va chon Add python.exe to PATH,
    echo hoac cai Python 3.12.10 bang pyenv, sau do chay lai file nay.
    pause
    exit /b 1
)

echo [1/3] Tao moi truong ao tai CAPTURE_APP\.venv
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv ".venv"
    if errorlevel 1 (
        echo Khong the tao .venv bang Python 3.12.
        pause
        exit /b 1
    )
 ) else (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo .venv hien tai khong dung Python 3.12.
        echo Hay xoa rieng thu muc CAPTURE_APP\.venv roi chay lai file nay.
        pause
        exit /b 1
    )
)

echo [2/3] Cap nhat pip
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/3] Cai thu vien
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto :error

echo.
echo Cai dat hoan tat. Hay chay run_capture_app.bat
pause
exit /b 0

:error
echo.
echo Cai dat that bai. Kiem tra ket noi mang va thu lai.
pause
exit /b 1
