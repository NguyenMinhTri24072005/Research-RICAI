@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ===============================================================================
echo [AI_SERVICES] SETUP MOI TRUONG AO PYTHON 3.12
echo ===============================================================================

set "PYTHON_CMD="

rem 1. Kiem tra duong dan pyenv truc tiep
if exist "%USERPROFILE%\.pyenv\pyenv-win\versions\3.12.10\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\.pyenv\pyenv-win\versions\3.12.10\python.exe"
    goto :found_python
)

rem 2. Kiem tra qua pyenv lenh
where pyenv >nul 2>nul
if not errorlevel 1 (
    pyenv exec python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 12), (3, 11), (3, 10)) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=pyenv exec python"
        goto :found_python
    )
)

rem 3. Kiem tra qua py -3.12
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    goto :found_python
)

rem 4. Kiem tra python he thong
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 12), (3, 11), (3, 10)) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :found_python
)

:not_found
echo [LOI] Khong tim thay Python 3.10, 3.11 hoac 3.12.
echo Hay cai dat Python 3.12 tu pyenv hoac python.org va thu lai.
echo.
pause
exit /b 1

:found_python
echo [*] Tim thay Python hop le: %PYTHON_CMD%
echo.

echo [1/3] Tao moi truong ao tai AI_SERVICES\.venv ...
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_CMD%" -m venv ".venv"
    if errorlevel 1 (
        echo [LOI] Khong the tao .venv bang lenh nay.
        pause
        exit /b 1
    )
) else (
    echo [*] Thu muc .venv da ton tai.
)

echo [2/3] Nang cap pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/3] Dang cai dat cac thu vien tu requirements.txt...
echo (Qua trinh nay co the mat 1-3 phut tuy thuoc toc do mang...)
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 (
    echo.
    echo [CANH BAO] Co mot so goi thu vien chua the cai dat day du.
    echo Kiem tra lai ket noi mang hoac file requirements.txt.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo [THANH CONG] Da hoan tat cai dat moi truong cho AI_SERVICES!
echo Bay gio ban co the khoi dong bang cach double click: run_ai_service.bat
echo ===============================================================================
echo.
pause
exit /b 0
