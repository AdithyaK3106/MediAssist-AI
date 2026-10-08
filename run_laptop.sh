#!/usr/bin/env bash
# ==============================================================================
# MediAssist AI — One-Click Laptop Launcher (Linux / macOS)
# Target: Laptop with NVIDIA RTX 4060 (8GB VRAM) or equivalent
# ==============================================================================

set -e

echo "=================================================================="
echo "      MediAssist AI: Safety-Aware Bilingual Healthcare Triage      "
echo "=================================================================="

# 1. Check Ollama Status
echo "[1/3] Checking Ollama status..."
if command -v ollama &> /dev/null; then
    echo "✔ Ollama CLI found."
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✔ Ollama server is running on http://localhost:11434."
    else
        echo "⚡ Starting Ollama server in background..."
        ollama serve &> /dev/null &
        sleep 3
    fi

    echo "Checking for Qwen 2.5 7B model..."
    if ollama list | grep -q "qwen2.5:7b"; then
        echo "✔ Model qwen2.5:7b is already downloaded."
    else
        echo "📥 Pulling qwen2.5:7b (quantized, ~4.7GB)..."
        ollama pull qwen2.5:7b
    fi
else
    echo "⚠️  Ollama CLI not found in PATH."
    echo "    MediAssist will run in fallback mode with native synthesis."
    echo "    To enable local LLM on RTX 4060: install from https://ollama.ai"
fi

# 2. Check Docker
echo ""
echo "[2/3] Checking Docker & Docker Compose..."
if command -v docker &> /dev/null; then
    echo "✔ Docker is available."
    echo ""
    echo "[3/3] Building and starting MediAssist Docker container..."
    echo "      WhatsApp-style Web UI: http://localhost:8501"
    echo "      FastAPI Clinical API:  http://localhost:8000"
    echo "=================================================================="
    docker compose up --build
else
    echo "⚠️  Docker not found. Falling back to local Python virtual environment..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "[3/3] Starting Streamlit natively on http://localhost:8501..."
        streamlit run src/frontend/app.py --server.port 8501
    else
        echo "❌ Neither Docker nor .venv found. Please install Docker or run 'uv venv'."
        exit 1
    fi
fi
