# 🧩 Component Details

This document provides detailed information about the core components of the MediAssist AI system.

---

## 1. Classical ML Layer
- **Purpose**: Fast, interpretable baseline disease prediction.
- **Models**: Logistic Regression, Naive Bayes, Random Forest, Decision Tree, XGBoost.
- **Features**: TF-IDF vectorized symptom text.
- **Output**: Disease class probabilities.

## 2. Transformer Layer (Planned)
- **Purpose**: Deep semantic understanding of natural language symptom descriptions.
- **Model**: BERT (or DistilBERT).
- **Features**: Dense embeddings.
- **Output**: Disease class probabilities, to be ensembled with the Classical Layer.

## 3. Conversational Layer
- **Purpose**: Synthesize responses and provide empathetic, clear explanations to users.
- **Model**: Qwen2.5 (via Ollama).
- **Role**: Acts as the interface between the raw model predictions and the user, translating medical jargon into plain language.

## 4. Safety Layer
- **Purpose**: Prevent unsafe or overconfident misdiagnoses.
- **Metrics**: 
  - **Entropy**: Measures the uncertainty of the probability distribution. High entropy means the model is confused.
  - **Top-2 Gap**: The difference between the highest and second-highest probability. Small gap means the model cannot decide between two classes.
- **Actions**: Flags predictions as unreliable, triggers Top-K fallback, or routes to a recommendation for human consultation.

## 5. Hospital Recommendation Engine
- **Purpose**: Guide users to appropriate care when needed.
- **Input**: Condition severity or high uncertainty.
- **Logic**: Filters a directory of hospitals based on location, specialty, and facility level (rural clinic vs. specialty hospital).
