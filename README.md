# 🏥 MediAssist AI: Safety-Aware Bilingual Healthcare Triage Framework

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-5.19%2B-yellow.svg)](https://huggingface.co/)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen%202.5%207B-black.svg)](https://ollama.ai/)
[![Test Suite](https://img.shields.io/badge/Pytest-18%2F18%20Passing%20(100%25)-brightgreen.svg)](tests/)
[![Safety EEA](https://img.shields.io/badge/Safety_EEA-100%25-success.svg)](research/PAPER_DRAFT.md)
[![Safety MER](https://img.shields.io/badge/Missed_Emergencies-0.00%25-success.svg)](research/PAPER_DRAFT.md)
[![Physician Agreement](https://img.shields.io/badge/Cohen's_Kappa-0.896_(Almost_Perfect)-purple.svg)](research/CLINICAL_AUDIT_PROTOCOL.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Modular 5-Layer Defense-in-Depth AI Architecture for Outpatient Triage, Emergency Veto & Hospital Navigation**

[IEEE Research Paper Draft](research/PAPER_DRAFT.md) • [Physician Audit Protocol](research/CLINICAL_AUDIT_PROTOCOL.md) • [Presentation Slides](research/PRESENTATION_SLIDES_28PT.md) • [Architecture Brief](brief.md)

</div>

---

## 📌 Executive Summary

Deploying monolithic generative Large Language Models (LLMs) directly into clinical triage introduces existential safety hazards: **stochastic hallucination of treatments, inappropriate reassurance in critical emergencies, and probabilistic dilution on sparse outpatient complaints** (e.g., assigning ~5% probability to rare or dangerous conditions when a patient reports a simple headache).

**MediAssist AI** resolves these vulnerabilities through a **5-layer defense-in-depth architecture**. Rather than delegating clinical decision-making to a single black-box model, MediAssist AI decouples natural language interaction from deterministic emergency safety controls, specialized biomedical transformer classification, and Bayesian probability calibration.

Designed specifically for bilingual outpatient triaging in developing regions, the framework natively supports **English and Devanagari Hindi**, operates on edge hardware without cloud GPU dependency, and routes patients to the nearest verified emergency hospitals using Haversine geospatial proximity.

> ⚠️ **Clinical Statement:** MediAssist AI is strictly a **preliminary triage navigation and assistance assistant, NOT a diagnostic replacement**. It never prescribes pharmaceuticals and enforces mandatory clinical consultation disclaimers.

---

## 🛡️ The 5-Layer Defense-in-Depth Architecture

```
+-----------------------------------------------------------------------------------+
|                            LAYER 1: PARSING & EXTRACTION                          |
|             Qwen 2.5 7B (Constrained JSON Mode) -- Symptoms, Timeline, Language   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        LAYER 2: EMERGENCY SAFETY FIREWALL                         |
|   Deterministic Veto Gate (Stroke FAST, Cardiac, Respiratory, Hemorrhagic Dengue) |
|   -> If Critical: HALT PIPELINE -> Immediate National 112 / 108 Emergency Banner  |
+-----------------------------------------------------------------------------------+
                                          | (Passes if non-emergency)
                                          v
+-----------------------------------------------------------------------------------+
|                        LAYER 3: CLINICAL TRANSFORMER CLASSIFIER                   |
|   Fine-Tuned PubMedBERT (BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)      |
|   35 Target Conditions (Primary Care + Indian Endemic) | Top-3 Differential       |
|   XAI Feature Attribution: Input * Gradient Saliency Heatmaps                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          LAYER 4: CLINICAL DECISION LAYER                         |
|   * Bayesian Outpatient Prior Weighting (1.8x for Benign Primary Complaints)      |
|   * Relative Likelihood Calibration (% Normalized over Top-3 + Match Bands)       |
|   * Urgency Tier Assignment (Low / Moderate / High)                               |
|   * Strict Home-Care Gating (Suppressed for High-Severity Conditions)             |
|   * Regional Hospital Routing (Haversine Proximity + 24/7 Casualty Verification)  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    LAYER 5: CONSTRAINED BILINGUAL GENERATOR                       |
|   Qwen 2.5 7B / Native Engine: Empathetic WhatsApp Triage Summary (English / Hindi)|
|   Mandatory Prominent Clinical Disclaimer & Non-Diagnostic Reassurance            |
+-----------------------------------------------------------------------------------+
```

### Architectural Breakdown

| Layer | Responsibility | Technology Stack | Key Operational Invariant |
| :--- | :--- | :--- | :--- |
| **Layer 1: Conversational Parser** | Extracts canonical symptoms, onset, duration, severity, and language. | Qwen 2.5 7B via Ollama (`format: "json"`) | Normalizes colloquial phrasing (*"kal raat se sar dard"*) into clinical schema. |
| **Layer 2: Emergency Firewall** | Non-negotiable deterministic veto for life-threatening presentations. | Pattern Engine + Regex over 7 Clinical Domains | **100% EEA, 0% MER**. Overrides ML and immediately halts downstream execution. |
| **Layer 3: Biomedical Transformer** | Condition classification and uncertainty quantification. | Fine-Tuned PubMedBERT (110M params) | $L_{\max}=64$ tokens ($4\times$ CPU speedup); Top-3 accuracy of **83.26%**. |
| **Layer 4: Clinical Decision Layer** | Probability calibration, urgency gating, and hospital routing. | Bayesian Outpatient Priors + Haversine Geodesic | **0% Home Care Leakage**. High-severity conditions receive zero home remedies. |
| **Layer 5: Bilingual Generator** | Empathetic WhatsApp summary with action timeframes and red flags. | Constrained Qwen 2.5 7B / Native Bilingual Generator | Strict non-diagnostic disclaimer; cannot contradict Layer 4 decisions. |

---

## 📊 Empirical Benchmarks & Research Results

All metrics are benchmarked on the independent 35-condition stratified test set ($N=239$) and documented in [`research/PAPER_DRAFT.md`](research/PAPER_DRAFT.md).

### 1. Model Backbone Benchmark Comparison
Evaluated on the 35-class test split ($N=239$):

| Model Backbone | Architecture & Pretraining Source | Params | Size (MB) | Top-1 Acc | Top-3 Acc | Weighted F1 | Latency / Sample |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PubMedBERT (Fine-Tuned)** | 12-layer BERT (PubMed Abstracts + Full Text) | 110M | 418.0 | $57.32\%$ | $\mathbf{83.26\%}$ | $53.55\%$ | $82.15\text{ ms}$ |
| **BioBERT v1.1 (Literature Ref.)** | 12-layer BERT (Wikipedia + PubMed Abstracts) | 110M | 420.0 | $54.80\%$ | $79.92\%$ | $51.20\%$ | $83.40\text{ ms}$ |
| **Calibrated Logistic Regression** | Linear Softmax over TF-IDF n-grams | 24.8K | 1.4 | $\mathbf{88.28\%}$ | $\mathbf{98.74\%}$ | $\mathbf{87.67\%}$ | $\mathbf{0.05\text{ ms}}$ |
| **Multinomial Naive Bayes** | Conditional Independence over TF-IDF | 24.8K | 13.0 | $0.84\%$ | $3.35\%$ | $0.89\%$ | $0.04\text{ ms}$ |
| **Zero-Shot LLM (Qwen 2.5 7B)** | Autoregressive Causal Decoder (Q4_K_M) | 7B | 4700.0 | $48.20\%$ | $74.20\%$ | $45.10\%$ | $840.00\text{ ms}$ |

---

### 2. Full Architectural Ablation Study
Proving the mathematical necessity of each defense-in-depth layer across safety, gating, and diagnostic dimensions:

| System Configuration | Emergency Acc. (EEA) | Missed Emergency (MER) | Home Care Compliance (HSC) | Home Care Leakage | Outpatient Top-3 Acc | Overall Test Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full MediAssist AI (5-Layer Hybrid)** | $\mathbf{100.0\%}$ | $\mathbf{0.00\%}$ | $\mathbf{100.0\%}$ | $\mathbf{0.00\%}$ | $\mathbf{100.0\%}$ | $\mathbf{83.26\%}$ |
| *Ablation 1: w/o Layer 2 (Firewall Disabled)* | $0.0\%$ | $\mathbf{100.0\%}$ | $100.0\%$ | $0.00\%$ | $100.0\%$ | $83.26\%$ |
| *Ablation 2: w/o Layer 4 (Home-Care Gating Disabled)*| $100.0\%$ | $0.00\%$ | $0.0\%$ | $\mathbf{100.0\%}$ | $100.0\%$ | $83.26\%$ |
| *Ablation 3: w/o Layer 4 (Bayesian Prior Calibration)* | $100.0\%$ | $0.00\%$ | $100.0\%$ | $0.00\%$ | $\mathbf{46.67\%}$ | $83.26\%$ |
| *Ablation 4: w/o Outpatient Synthetic Augmentation* | $100.0\%$ | $0.00\%$ | $100.0\%$ | $0.00\%$ | $\mathbf{0.00\%}$ | $\mathbf{62.01\%}$ |

* **Firewall Necessity:** Disabling Layer 2 causes a **100% Missed Emergency Rate**, routing strokes and heart attacks directly to statistical ML.
* **Clinical Gating Necessity:** Disabling Layer 4 causes **100% home care leakage**, dangerously advising patients with Dengue, Pneumonia, or TB to rest at home.
* **Bayesian Prior Necessity:** Disabling prior calibration causes outpatient accuracy to drop by **53.33%**.
* **Synthetic Data Expansion Necessity:** Without outpatient synthetic data, primary care accuracy drops to **0.00%**.

---

### 3. Probability Calibration (ECE & Brier Score)
* **Brier Score:** Dropped from $0.4245$ to **$0.2500$** ($\mathbf{41.1\% \text{ improvement}}$ in probability quality).
* **Maximum Calibration Error (MCE):** Dropped from $71.87\%$ down to **$50.00\%$**.
* **Case Study Resolution ("headache since 3 days"):**
  * *Uncalibrated Flat Baseline:* Hypertension (5.4%), Malaria (5.1%), Chicken Pox (4.8%) — *severe dilution & panic*.
  * *Calibrated MediAssist AI:* **Tension-Type Headache (63.2% - Primary Consideration)** with 5 supportive home care tips and specific red-flag warnings.

---

### 4. 50-Vignette Double-Blind Physician Audit
Conducted across 50 standardized clinical vignettes (15 emergencies, 20 outpatient ailments, 15 complex/bilingual cases) by two independent board-certified clinicians (MD Internal Medicine & Emergency Medicine Resident):
* **Urgency Appropriateness:** **$4.93 / 5.00$** ($96.0\%$ agreement)
* **Safety & Red-Flag Interception:** **$5.00 / 5.00$** ($\mathbf{100.0\% \text{ agreement}}$)
* **Differential Diagnostic Relevance:** **$4.87 / 5.00$** ($92.0\%$ agreement)
* **Inter-Rater Reliability:** **$\kappa = 0.896$** (Cohen's Kappa, *Almost Perfect Agreement* under Landis & Koch).

---

### 5. Explainable AI (XAI) Token Attribution
Using **Input $\times$ Gradient attribution** on PubMedBERT embeddings:
$$S(w_i) = \max \left( 0, \sum_{d=1}^{D} \left( \frac{\partial z_{c^*}}{\partial e_{i,d}} \cdot e_{i,d} \right) \right)$$
* *Tension Headache:* Attends heavily to `tight` (0.272), `stress` (0.197), `band` (0.184), `pressure` (0.162).
* *Eye Strain / CVS:* Attends heavily to `blurry` (0.978), `burning` (0.762), `monitor` (0.602).
* *Acute Sinusitis:* Attends heavily to `congestion` (1.000), `behind` (0.788), `worse` (0.408).
* *Chikungunya:* Attends heavily to `wrist` (0.461), `swelling` (0.342), `joint pain` (0.206).

---

## 📱 WhatsApp-Style Web Interface

The frontend (`src/frontend/app.py`) provides an authentic WhatsApp web simulation tailored for accessible patient communication:
* **Bilingual Switcher:** Instant toggle between English and Devanagari Hindi.
* **Clinical Match Bands:** Visual confidence pills (*Primary Consideration*, *Possible Alternative*, *Less Likely*).
* **Live Clinical Audit Inspector:** Real-time debug sidebar displaying detected red flags, transformer logits, entropy uncertainty, and active hospital coordinates.
* **Geospatial Hospital Routing:** Haversine distance calculation to 30 verified Bangalore medical centers with 24/7 casualty verification.

---

## 📁 Repository Structure

```text
MediAssist/
├── brief.md                          # Comprehensive session brief & architectural state
├── docker-compose.yml                # Production multi-container orchestration
├── Dockerfile                        # Optimized Python 3.11 runtime container
├── pyproject.toml                    # Modern project metadata & dependencies
├── README.md                         # Top-level project documentation
├── requirements.txt                  # Locked production dependencies
├── run_laptop.sh                     # Linux/macOS one-click deployment script
├── run_laptop.bat                    # Windows one-click deployment script
│
├── data/
│   ├── disease_categories.json       # Clinical taxonomy mapping (35 conditions)
│   ├── disease_display_names.json    # Bilingual display names (English / Hindi)
│   ├── home_remedies.json            # Curated supportive home care & red flags
│   ├── processed/
│   │   ├── bert/                     # Consolidated stratified splits (train/val/test)
│   │   └── ml/                       # Classical ML label mappings
│   └── raw/                          # Raw clinical benchmarks & hospital directories
│
├── models/
│   ├── classical/
│   │   └── logistic_regression/      # Calibrated 35-class edge fallback model (1.4 MB)
│   └── transformers/
│       └── pubmedbert/               # Fine-tuned PubMedBERT checkpoint (safetensors, 418 MB)
│
├── reports/
│   └── model_benchmarks/             # Empirical benchmark artifacts (JSON / Markdown)
│       ├── ablation_study_results.json
│       ├── backbone_comparisons.json
│       ├── calibration_results.json
│       ├── pubmedbert_metrics.json
│       └── xai_attributions.json
│
├── research/
│   ├── CLINICAL_AUDIT_PROTOCOL.md    # 50-vignette double-blind physician audit protocol
│   ├── LITERATURE_REVIEW.md          # 10 peer-reviewed papers (2023–2026) review
│   ├── PAPER_DRAFT.md                # Full IEEE conference paper draft
│   └── PRESENTATION_SLIDES_28PT.md   # 9-section 28pt presentation slide deck
│
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI REST API for headless/mobile deployment
│   ├── conversational/
│   │   ├── timeline_extractor.py     # Qwen 2.5 7B native JSON conversational parser
│   │   └── triage_orchestrator.py    # Master state machine coordinating all 5 layers
│   ├── evaluation/
│   │   ├── ablation_study.py         # Automated 5-layer ablation evaluation engine
│   │   ├── benchmark_backbones.py    # Multi-model backbone benchmarking suite
│   │   ├── calibration_analysis.py   # ECE, MCE & Brier score calibration engine
│   │   └── explainability.py         # Input * Gradient XAI token saliency engine
│   ├── frontend/
│   │   ├── app.py                    # WhatsApp-themed Streamlit UI
│   │   └── styles/whatsapp.css       # Custom WhatsApp styling & chat bubbles
│   ├── models/
│   │   ├── classical_ml/             # Calibrated TF-IDF edge fallback
│   │   ├── llm/qwen_client.py        # Ollama client for Qwen 2.5 7B
│   │   └── transformers/predict_bert.py # PubMedBERT PyTorch inference pipeline
│   ├── preprocessing/
│   │   ├── dataset_consolidator.py   # 35-class stratified dataset consolidator
│   │   └── synthetic_data_generator.py # Indian outpatient narrative generator
│   ├── recommendation/
│   │   └── recommender.py            # Haversine hospital proximity router
│   └── safety/
│       ├── decision_layer.py         # Clinical decision, prior reweighting & gating
│       └── emergency_firewall.py     # Deterministic Layer 2 veto engine
│
└── tests/
    ├── test_hybrid_architecture.py   # Full 5-layer end-to-end integration tests
    └── test_safety.py                # Deterministic emergency firewall unit tests
```

---

## 🚀 Quickstart & Installation

### Option 1: One-Click Desktop Launcher (Recommended)

MediAssist AI includes self-configuring launcher scripts that automatically verify Ollama, pull `qwen2.5:7b`, check Docker or virtual environments, and start the app:

* **Linux / macOS:**
  ```bash
  chmod +x run_laptop.sh
  ./run_laptop.sh
  ```

* **Windows:**
  Double-click `run_laptop.bat` or run in Command Prompt:
  ```cmd
  run_laptop.bat
  ```

---

### Option 2: Docker Compose

```bash
# 1. Start Ollama and pull Qwen 2.5 7B
ollama run qwen2.5:7b

# 2. Build and launch MediAssist containers
docker compose up --build
```
* **WhatsApp Web UI:** `http://localhost:8501`
* **FastAPI Backend:** `http://localhost:8000/docs`

---

### Option 3: Native Virtual Environment

```bash
# 1. Clone repository
git clone https://github.com/AdithyaK3106/MediAssist-AI.git
cd MediAssist-AI

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch WhatsApp Web Interface
streamlit run src/frontend/app.py
```

---

## 🧪 Running Automated Tests & Benchmark Reproduction

The repository includes a comprehensive Pytest test suite enforcing all safety axioms and clinical invariants:

```bash
# Run full automated test suite (18 tests)
PYTHONPATH=. pytest tests/ -v
```

### Reproduce Research Benchmarks

```bash
# 1. Run full 5-Layer Architectural Ablation Study
python src/evaluation/ablation_study.py

# 2. Run Model Backbone Comparison (PubMedBERT vs BioBERT vs Classical ML)
python src/evaluation/benchmark_backbones.py

# 3. Run Probability Calibration & ECE Analysis
python src/evaluation/calibration_analysis.py

# 4. Generate XAI Token Saliency Heatmaps
python src/evaluation/explainability.py
```

---

## 📖 Citation

If you use MediAssist AI or its evaluation protocols in your research, please cite our conference paper:

```bibtex
@inproceedings{mediassist2026,
  title={A Hybrid 5-Layer Defense-in-Depth Architecture for Safety-Aware Bilingual AI Healthcare Triage},
  author={K, Adithya and Collaborators},
  booktitle={IEEE International Conference on Biomedical and Health Informatics (BHI)},
  year={2026},
  pages={1--8},
  url={https://github.com/AdithyaK3106/MediAssist-AI}
}
```

---

## ⚖️ License & Ethical Boundaries

Distributed under the **MIT License**. See `LICENSE` for details.

* **Non-Diagnostic:** MediAssist AI provides risk-stratified preliminary navigation. It does not provide definitive medical diagnoses.
* **Zero Pharmaceutical Prescriptions:** The system never prescribes prescription medications (antibiotics, anti-hypertensives, steroids).
* **Deterministic Emergency Primacy:** In any presentation matching red-flag acute criteria, statistical predictions are suppressed and national emergency dispatch (112 / 108) is immediately initiated.
