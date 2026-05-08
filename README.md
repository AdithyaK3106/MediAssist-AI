# MediAssist AI: Hybrid Healthcare Assistant

MediAssist AI is a modular, research-grade healthcare assistant that combines classical machine learning, transformer-based NLP, and local Large Language Models (LLMs) to provide safety-aware medical insights.

## 🚀 Architecture Overview
- **Classical ML**: TF-IDF + Logistic Regression for fast, interpretable disease screening.
- **Transformers**: DistilBERT/BERT for deep semantic understanding of patient symptoms.
- **Local LLM**: Qwen2.5 (via Ollama) for conversational synthesis and clinical explanation.
- **Safety**: Uncertainty estimation and Top-K prediction to avoid over-confident misdiagnosis.
- **Recommendation**: Proximity-based hospital recommendation system.

## 📁 Project Structure
- `src/`: Core source code (modular components).
- `notebooks/`: Research and experimentation labs.
- `data/`: Versioned data pipeline (raw -> interim -> processed).
- `models/`: Weights, tokenizers, and experiment checkpoints.
- `api/`: FastAPI backend for model serving.
- `frontend/`: Streamlit user interface.

## 🛠️ Setup
1. **Environment**: `python -m venv venv` & `pip install -r requirements.txt`
2. **Local LLM**: Install [Ollama](https://ollama.ai/) and run `ollama pull qwen2.5`.
3. **Config**: Edit `src/config/model_config.yaml` for model hyperparameters.

## 📊 Experiment Tracking
Metrics and confusion matrices are saved in `reports/metrics/`. Training logs are located in `logs/`.
