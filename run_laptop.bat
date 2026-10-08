@echo off
REM ==============================================================================
REM MediAssist AI — One-Click Laptop Launcher (Windows 10/11)
REM Target: Laptop with NVIDIA RTX 4060 (8GB VRAM)
REM ==============================================================================

echo ==================================================================
echo       MediAssist AI: Safety-Aware Bilingual Healthcare Triage      
echo ==================================================================

REM 1. Check Ollama
echo [1/3] Checking Ollama status...
where ollama >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Ollama found in system PATH.
    echo Checking for qwen2.5:7b...
    ollama list | findstr "qwen2.5:7b" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo [INFO] Pulling qwen2.5:7b model (approx 4.7GB)...
        ollama pull qwen2.5:7b
    ) else (
        echo [OK] Model qwen2.5:7b is already installed.
    )
) else (
    echo [WARNING] Ollama is not installed or not in PATH.
    echo Install from https://ollama.ai for local Qwen 2.5 7B acceleration.
)

REM 2. Check Docker
echo.
echo [2/3] Checking Docker Desktop...
where docker >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Docker Desktop detected.
    echo.
    echo [3/3] Building and starting MediAssist Docker container...
    echo       WhatsApp-style Web UI: http://localhost:8501
    echo       FastAPI Clinical API:  http://localhost:8000
    echo ==================================================================
    docker compose up --build
) else (
    echo [WARNING] Docker not detected. Checking for Python virtual environment...
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
        echo [3/3] Starting Streamlit natively on http://localhost:8501...
        streamlit run src/frontend/app.py --server.port 8501
    ) else (
        echo [ERROR] Neither Docker nor .venv found. Please install Docker Desktop.
        pause
    )
)

pause
