@echo off
title MediAssist AI Server Manager

echo =========================================
echo Restarting MediAssist AI Servers...
echo =========================================

echo.
echo Stopping FastAPI backend (Port 8000)...
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr :8000') do (
    if %%a neq 0 taskkill /F /PID %%a 2>nul
)

echo Stopping Streamlit frontend (Port 8501)...
for /f "tokens=5" %%a in ('netstat -a -n -o ^| findstr :8501') do (
    if %%a neq 0 taskkill /F /PID %%a 2>nul
)

echo.
echo Starting FastAPI Backend...
start "MediAssist Backend" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate && python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Streamlit Frontend...
start "MediAssist Frontend" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate && python -m streamlit run src\frontend\app.py"

echo.
echo =========================================
echo Servers successfully restarted!
echo FastAPI is running on http://localhost:8000
echo Streamlit is running on http://localhost:8501
echo =========================================
