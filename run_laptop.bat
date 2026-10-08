@echo off
setlocal enabledelayedexpansion

REM Ensure working directory is the repository root
cd /d "%~dp0"

echo ==================================================================
echo       MediAssist AI: Safety-Aware Bilingual Healthcare Triage      
echo ==================================================================

REM 1. Check Ollama Status
echo [1/3] Checking Ollama status...
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Ollama found in system PATH.
    
    REM Check if Ollama server is already running on port 11434
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [INFO] Starting Ollama server in background...
        start "" /B ollama serve
        timeout /t 3 /nobreak >nul
    )

    echo Checking for qwen2.5:7b model...
    ollama list | findstr "qwen2.5:7b" >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [INFO] Pulling qwen2.5:7b model [approx 4.7GB]...
        ollama pull qwen2.5:7b
    ) else (
        echo [OK] Model qwen2.5:7b is ready.
    )
) else (
    echo [WARNING] Ollama is not installed or not in PATH.
    echo Install from https://ollama.ai for local Qwen 2.5 7B acceleration.
)

REM 2. Check Docker vs Native Execution
echo.
echo [2/3] Checking Docker status...
docker info >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Docker daemon is running.
    echo.
    echo [3/3] Starting MediAssist via Docker Compose...
    echo       WhatsApp-style Web UI: http://localhost:8501
    echo       FastAPI Clinical API:  http://localhost:8000
    echo ==================================================================
    docker compose up --build
) else (
    echo [INFO] Docker daemon is not active. Falling back to native virtual environment.
    if exist ".venv\Scripts\activate.bat" (
        echo [OK] Found .venv virtual environment.
        echo.
        echo [3/3] Starting Streamlit natively on http://localhost:8501...
        echo ==================================================================
        call ".venv\Scripts\activate.bat"
        python -m streamlit run src/frontend/app.py --server.port 8501
    ) else (
        echo [ERROR] Neither Docker daemon nor .venv was found.
        echo Please run 'uv venv' and install requirements, or start Docker Desktop.
        pause
    )
)

pause
