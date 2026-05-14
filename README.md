# 🏥 MediAssist AI: Safety-Aware Healthcare Triage Framework

![MediAssist Banner](https://img.shields.io/badge/Healthcare-Safety_First-blue) ![Python](https://img.shields.io/badge/Python-3.12-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)

**MediAssist AI** is a modular, research-grade healthcare assistant that prioritizes patient safety through a rigorous hybrid architecture. It explicitly rejects the "pure probabilistic ML" approach—which often masks life-threatening conditions behind statistical overlap—in favor of deterministic emergency overrides, classical machine learning, and medical consistency validation.

Our thesis: **In healthcare AI, safety and responsible escalation must always supersede raw prediction accuracy.**

---

## 🛑 The Problem: Why Pure ML is Insufficient
In autonomous medical triage, pure machine learning models (whether classical algorithms or Large Language Models) treat critical symptoms identically to benign ones: as statistical features. 
If a user inputs *"coughing up blood and difficulty breathing"*, an ML model trained on common datasets might predict "Asthma" with 85% confidence due to feature overlap, entirely missing the reality of a severe respiratory emergency. Relying solely on statistical likelihoods for triage leads to a high **Unsafe Prediction Rate (UPR)**.

## 🛡️ The Solution: Defense-in-Depth Architecture
MediAssist AI mitigates these risks using a multi-layered, hybrid safety architecture:

1. **Pre-ML Triage Pipeline (Emergency Rule Engine):**
   - Intercepts raw user input *before* it reaches the ML models.
   - Uses strict deterministic matching against critical medical patterns (e.g., Stroke, Hemorrhagic, Cardiac, Respiratory emergencies).
   - Immediately bypasses the ML pipeline and triggers a **🚨 Critical Medical Alert** UI if danger is detected.

2. **Uncertainty-Aware Classical ML:**
   - For non-emergencies, symptom text is parsed using TF-IDF and classified using a highly calibrated Logistic Regression model (with extensible support for Random Forest, XGBoost, etc.).
   - Returns **Top-K Predictions** rather than a single confident guess, explicitly visualizing uncertainty for the user.

3. **Medical Consistency Validation (Post-ML Safety):**
   - Extracts explicit symptom categories (e.g., "respiratory", "gastrointestinal") from the input.
   - Cross-references the ML-predicted disease category against the detected symptoms.
   - **Implausible Outputs** (e.g., predicting an eye infection when the patient reports fever and a cough) receive a harsh confidence penalty (-90% multiplier) and are suppressed in the final ranking.

---

## 🌟 Key Features

- **Dynamic Glassmorphism UI:** A premium, modern Streamlit interface designed to instill calm while clearly delineating between routine care and emergency escalation.
- **Explainability Logging:** Every emergency trigger, suppressed prediction, and consistency penalty is recorded in structured JSON logs (`logs/safety/`) for complete clinical auditability.
- **Severity-Aware Home Care:** Home remedies and rest recommendations are exclusively provided for low-severity conditions.
- **Conversational Memory:** Maintains symptom context across a session for refined follow-up predictions.

---

## 📁 Project Structure

```text
MediAssist/
├── data/                  # Disease names, categories, and home remedies JSONs
├── docs/                  # Architecture diagrams and viva/research notes
├── logs/safety/           # Audit logs for emergency overrides and consistency checks
├── models/classical/      # Saved TF-IDF vectorizers and trained classifiers
├── notebooks/             # Jupyter notebooks for EDA, error analysis, and benchmarking
├── reports/               # Auto-generated validation reports, CSV metrics, and figures
├── scripts/               # Utility scripts for training, benchmarking, and validation
├── src/
│   ├── conversational/    # Dialogue state, intent detection, and context management
│   ├── frontend/          # Streamlit app (app.py) and custom CSS styling
│   ├── models/            # ML inference pipelines and prediction logic
│   ├── safety/            # EmergencyRuleEngine and MedicalConsistencyValidator
│   └── recommendation/    # Severity-based home care engine
└── tests/                 # Automated Pytest suite enforcing critical safety axioms
```

---

## 📊 Safety Metrics & Validation
Standard ML metrics (Accuracy, F1-Score) are insufficient to measure healthcare readiness. MediAssist AI evaluates itself against a synthetic 500-case validation suite focusing on specialized safety metrics:

- **Emergency Escalation Accuracy (EEA):** % of critical emergencies successfully intercepted and escalated (Target: >95%).
- **Missed Emergency Rate (MER):** % of critical emergencies that slipped past the triage layer (Target: ~0%).
- **Unsafe Prediction Rate (UPR):** % of times the model predicts an inappropriate condition with high confidence.
- **Medical Consistency Score:** How accurately the model aligns output disease categories with input symptom profiles.

*Validation results and visualizations are automatically generated and stored in `reports/final_validation_report.md` and `reports/figures/`.*

---

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AdithyaK3106/MediAssist-AI.git
   cd MediAssist-AI
   ```

2. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run src/frontend/app.py
   ```

4. **Run the Safety Test Suite:**
   ```bash
   pytest tests/test_safety.py -v
   ```

5. **Generate Final Research Reports:**
   ```bash
   python scripts/run_final_research_validation.py
   ```

---

## ⚖️ Medical Disclaimer
**MediAssist AI is a research and triage framework, NOT a clinical diagnostic tool.** It is explicitly designed to provide preliminary insights and escalate critical symptoms. Always consult a qualified healthcare professional for medical advice, diagnoses, or treatment.
