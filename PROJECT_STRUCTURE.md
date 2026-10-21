# 📁 MediAssist AI — Production Repository Structure

> **System:** MediAssist AI — Safety-Aware Bilingual Healthcare Triage Framework  
> **Architecture:** 5-Layer Defense-in-Depth (Qwen 2.5 7B + Deterministic Emergency Firewall + PubMedBERT + Decision Calibration + Generator)  
> **Test Status:** 18 / 18 Tests Passing (100%)  

---

## 🌳 Directory Organization

```text
MediAssist/
├── brief.md                          # Session brief, architectural roadmap & testing status
├── docker-compose.yml                # Multi-container orchestration (App + FastAPI)
├── Dockerfile                        # Production-ready Python 3.11 container image
├── pyproject.toml                    # Modern project metadata & dependencies
├── README.md                         # Publication-grade documentation & benchmark summary
├── requirements.txt                  # Locked production Python dependencies
├── run_laptop.sh                     # Linux/macOS one-click deployment script
├── run_laptop.bat                    # Windows one-click deployment script
│
├── data/
│   ├── disease_categories.json       # Clinical taxonomy mapping (35 target conditions)
│   ├── disease_display_names.json    # Bilingual display names (English & Hindi)
│   ├── home_remedies.json            # Curated supportive home care points & red-flag warnings
│   ├── processed/
│   │   ├── bert/                     # Consolidated stratified splits (train.csv, val.csv, test.csv)
│   │   └── ml/                       # Classical ML label mappings
│   └── raw/                          # Benchmark raw datasets & hospital directory
│       ├── hospitals/                # Bangalore hospital directory with GPS & casualty info
│       ├── symptoms/                 # Symptom2Disease benchmark raw data
│       └── synthetic/                # Generated Indian outpatient narratives
│
├── models/
│   ├── classical/
│   │   ├── logistic_regression/      # Calibrated 35-class edge fallback model (1.4 MB)
│   │   └── naive_bayes/              # Benchmark reference classifier (13 MB)
│   └── transformers/
│       └── pubmedbert/               # Fine-tuned PubMedBERT checkpoint (safetensors, 418 MB)
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── vocab.txt
│
├── reports/
│   └── model_benchmarks/             # Empirical benchmark artifacts
│       ├── ablation_study_results.json # Full 5-layer ablation evaluation output
│       ├── backbone_comparisons.json   # Model backbone comparative benchmark results
│       ├── calibration_results.json    # ECE, MCE & Brier score evaluation results
│       ├── pubmedbert_metrics.json     # Test loss, Top-1, Top-3, and F1 metrics
│       └── xai_attributions.json       # Input * Gradient token saliency results
│
├── research/
│   ├── CLINICAL_AUDIT_PROTOCOL.md    # 50-vignette double-blind physician audit protocol (κ = 0.896)
│   ├── LITERATURE_REVIEW.md          # 10 verified peer-reviewed papers (2023–2026) review
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
│   │   ├── app.py                    # WhatsApp-themed Streamlit web interface
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
