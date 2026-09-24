@echo off
setlocal
cd /d "%~dp0"

echo ===============================================================================
echo [*] DANG KHOI DONG RICE ESTIMATION APPLICATION (BACKEND + FRONTEND)...
echo     Backend  : http://localhost:3000 (Node.js + Express + Nodemon)
echo     Frontend : http://localhost:5173 (React + Vite)
echo ===============================================================================
echo.

npm run dev
if errorlevel 1 (
    echo.
    echo [THONG BAO] Ung dung da dung hoac xay ra loi.
    pause
)
endlocal
