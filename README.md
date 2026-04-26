# 🏥 MediAssist AI — Safety-Aware Medical Chatbot

An intelligent, safety-aware AI chatbot designed to provide **preliminary healthcare guidance** using a hybrid machine learning and transformer-based architecture.

This system is built for **low-resource environments** and focuses on **safe, uncertainty-aware predictions** instead of blindly maximizing accuracy.

---

## 🚀 Key Features

- 🧠 **Hybrid AI Architecture**
  - TF-IDF + Logistic Regression (fast baseline)
  - Transformer-based NLP using BERT for natural language understanding  

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

## 🏗️ System Architecture




---

## ⚙️ Tech Stack

| Layer | Technology |
|------|-----------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | HuggingFace Transformers (BERT) |
| Backend | Flask / FastAPI |
| Dataset | Kaggle Medical Dataset |
| Visualization | Matplotlib |

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
4. Predictions are evaluated for uncertainty  
5. Safety layer determines output strategy  
6. Final response is generated with precautions  

---

## 📂 Project Structure
MediAssist-AI/
│
├── data/
├── models/
│ ├── ml_model.pkl
│ └── bert_model/
├── src/
│ ├── preprocessing.py
│ ├── model_ml.py
│ ├── model_bert.py
│ ├── uncertainty.py
│ ├── decision.py
│ └── app.py
├── requirements.txt
└── README.md


---

## ▶️ Installation & Setup

```bash
git clone https://github.com/your-username/MediAssist-AI.git
cd MediAssist-AI
pip install -r requirements.txt
