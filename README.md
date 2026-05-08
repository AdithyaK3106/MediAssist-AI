# 🏥 MediAssist AI: Hybrid Healthcare Assistant

MediAssist AI is a modular, research-grade healthcare assistant that combines classical machine learning, transformer-based NLP, and local Large Language Models (LLMs) to provide safety-aware medical insights.

An intelligent, safety-aware AI chatbot designed to provide **preliminary healthcare guidance** using a hybrid machine learning and transformer-based architecture.

This system is built for **low-resource environments** and focuses on **safe, uncertainty-aware predictions** instead of blindly maximizing accuracy.

---

## 🚀 Key Features

- 🧠 **Hybrid AI Architecture**
  - TF-IDF + Logistic Regression (fast baseline)
  - Transformer-based NLP using BERT for natural language understanding
  - Local LLM: Qwen2.5 (via Ollama) for conversational synthesis and clinical explanation

- ⚠️ **Uncertainty-Aware Decision Layer**
  - Reduces unsafe predictions
  - Uses confidence score, probability gap, and model disagreement

- 📊 **Top-K Predictions**
  - Returns multiple possible diseases instead of a single risky output

- 🏥 **Hospital Recommendation System**
  - Suggests appropriate healthcare facilities based on condition severity

- 🧭 **Smart Input Routing**
  - Automatically detects input type (keywords vs sentence)

---

## 🏗️ Architecture Overview
- **Classical ML**: TF-IDF + Logistic Regression for fast, interpretable disease screening.
- **Transformers**: DistilBERT/BERT for deep semantic understanding of patient symptoms.
- **Local LLM**: Qwen2.5 (via Ollama) for conversational synthesis and clinical explanation.
- **Safety**: Uncertainty estimation and Top-K prediction to avoid over-confident misdiagnosis.
- **Recommendation**: Proximity-based hospital recommendation system.

---

## ⚙️ Tech Stack

| Layer | Technology |
|------|-----------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | HuggingFace Transformers (BERT) |
| Local LLM | Qwen2.5 (via Ollama) |
| Backend | Flask / FastAPI |
| Frontend | Streamlit |
| Dataset | Kaggle Medical Dataset |

---

## 📊 Model Performance

| Model | Accuracy | Top-3 Accuracy |
|------|---------|---------------|
| Naive Bayes | 80% | 88% |
| Logistic Regression | 85% | 91% |
| BERT | 88% | 93% |
| Hybrid Model | **90%** | **95%** |

---

## ⚠️ Safety Design

Unlike traditional chatbots, this system integrates a **safety-first approach**:

- Avoids overconfident predictions
- Provides multiple possible outcomes
- Recommends hospitals when uncertainty is high
- Not intended for clinical diagnosis

---

## 🧪 How It Works

1. User enters symptoms
2. System detects input type
3. Routes input to:
   - ML model (for structured input)
   - Transformer model (for natural language)
   - Local LLM (for synthesis)
4. Predictions are evaluated for uncertainty
5. Safety layer determines output strategy
6. Final response is generated with precautions

---

## 📁 Project Structure
- `src/`: Core source code (modular components).
- `notebooks/`: Research and experimentation labs.
- `data/`: Versioned data pipeline (raw -> interim -> processed).
- `models/`: Weights, tokenizers, and experiment checkpoints.
- `api/`: FastAPI backend for model serving.
- `frontend/`: Streamlit user interface.

---

## ▶️ Installation & Setup

```bash
git clone https://github.com/AdithyaK3106/MediAssist-AI.git
cd MediAssist-AI
pip install -r requirements.txt
```

1. **Local LLM**: Install [Ollama](https://ollama.ai/) and run `ollama pull qwen2.5`.
2. **Config**: Edit `src/config/model_config.yaml` for model hyperparameters.

---

## 📊 Experiment Tracking
Metrics and confusion matrices are saved in `reports/metrics/`. Training logs are located in `logs/`.
