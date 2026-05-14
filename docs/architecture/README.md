# 🏛️ MediAssist AI Architecture Documentation

## 🔍 System Overview
MediAssist AI is a hybrid, safety-aware healthcare assistant designed for low-resource environments. It combines classical machine learning for speed and interpretability, transformer models for deep semantic understanding, and local Large Language Models (LLMs) for conversational synthesis.

The system prioritizes **safety** over raw accuracy by implementing an uncertainty-aware decision layer that flags unreliable predictions and suggests multiple possibilities (Top-K) instead of a single risky output.

---

## 🗺️ Hybrid Routing Pipeline

The system uses a smart routing mechanism to handle different types of user inputs (e.g., keyword lists vs. natural language descriptions).

```mermaid
graph TD
    A[User Input] --> B{Input Type Detector}
    B -- Keywords/Structured --> C[Classical ML Pipeline]
    B -- Natural Language --> D[Transformer Pipeline]
    
    C --> E[TF-IDF + Logistic Regression]
    D --> F[BERT Embeddings]
    
    E --> G[Prediction + Confidence]
    F --> G
    
    G --> H[Safety Decision Layer]
```

---

## 🛡️ Safety-Aware Decision Layer

The safety layer evaluates the model's prediction confidence and uncertainty (entropy, top-2 gap) to determine the output strategy.

```mermaid
graph TD
    A[Model Prediction] --> B{Confidence > Threshold?}
    B -- Yes --> C{Entropy < Threshold?}
    B -- No --> D[Flag as Uncertain]
    
    C -- Yes --> E[Return Top Prediction]
    C -- No --> D
    
    D --> F[Return Top-K Predictions]
    D --> G[Route to LLM for Synthesis]
    D --> H[Recommend Hospital Visit]
```

### Key Components:
1.  **Confidence Thresholding**: Predictions with low probability are flagged.
2.  **Uncertainty Estimation**: Uses entropy of the probability distribution to detect when the model is confused between multiple classes.
3.  **Top-K Predictions**: Instead of forcing a single answer, the system returns the top 3 most likely conditions.

---

## 🏥 Hospital Recommendation Engine

When uncertainty is high or specific severe conditions are detected, the system suggests appropriate healthcare facilities.

```mermaid
graph TD
    A[Severe Condition Detected OR High Uncertainty] --> B[Fetch User Location]
    B --> C[Query Hospital Directory]
    C --> D[Filter by Specialty & Distance]
    D --> E[Return Recommendations]
```

---

## 🔄 Component Interaction

```mermaid
sequenceDiagram
    participant User
    participant Frontend (Streamlit)
    participant Backend (FastAPI)
    participant ML_Model (Logistic Regression)
    participant LLM (Qwen2.5)
    
    User->>Frontend: Enters symptoms
    Frontend->>Backend: API Request
    Backend->>ML_Model: Predict Disease
    ML_Model-->>Backend: Returns Prediction + Confidence
    alt Low Confidence
        Backend->>LLM: Synthesize explanation with warning
        LLM-->>Backend: Returns safe response
    else High Confidence
        Backend->>Backend: Formulate response
    end
    Backend-->>Frontend: API Response
    Frontend-->>User: Display Results
```

---

## 📁 Folder Structure (Docs)
- `README.md`: This file.
- `component_details.md`: Detailed explanation of each module (to be created).
- `data_flow.md`: Detailed data flow diagrams (to be created).
