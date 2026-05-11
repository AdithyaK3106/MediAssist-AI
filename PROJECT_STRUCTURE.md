# 📁 MediAssist AI Project Structure

This document provides an overview of the directory structure and file organization for the MediAssist AI project.

## 🌳 Directory Tree

```
MediAssist/
├── data/                       # Data storage (ignored by git except .gitkeep)
│   ├── external/               # Data from third party sources
│   ├── interim/                # Intermediate data that has been transformed
│   ├── processed/              # The final, canonical data sets for modeling
│   │   ├── bert/               # Data prepared for BERT training
│   │   ├── hospitals/          # Cleaned hospital directory data
│   │   └── ml/                 # Data prepared for classical ML
│   └── raw/                    # The original, immutable data dump
│       ├── conversations/      # Doctor-patient conversation datasets
│       ├── hospitals/          # National hospital directories
│       ├── rural_health/       # Local/rural health facility data
│       └── symptoms/           # Symptom-to-disease mapping datasets
├── docs/                       # Project documentation
├── models/                     # Trained and serialized models, model predictions, or model summaries
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_data_analysis/       # Exploratory Data Analysis notebooks
│   ├── 02_preprocessing/       # Data cleaning and transformation experiments
│   ├── 03_feature_engineering/ # Feature extraction experiments
│   └── ...                     # Other experiment folders
├── reports/                    # Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── eda/                    # EDA reports
│   ├── figures/                # Generated graphics and plots
│   └── metrics/                # Model evaluation metrics
├── src/                        # Source code for use in this project
│   ├── __init__.py             # Makes src a Python module
│   ├── api/                    # FastAPI backend for model serving
│   │   └── main.py             # API entry point
│   ├── config/                 # Configuration files (YAML, JSON)
│   ├── evaluation/             # Scripts to evaluate model performance
│   ├── explainability/         # Model explainability (SHAP, Lime)
│   ├── features/               # Scripts to turn raw data into features for modeling
│   ├── frontend/               # Streamlit user interface
│   │   └── app.py              # Streamlit application
│   ├── models/                 # Scripts to train models and then use trained models to make predictions
│   │   ├── classical_ml/       # Classical ML models (Logistic Regression, etc.)
│   │   ├── ensemble/          # Hybrid/Ensemble model logic
│   │   ├── llm/               # LLM integration (Qwen, Ollama)
│   │   └── transformers/      # Transformer models (BERT)
│   ├── preprocessing/          # Scripts to clean and process data
│   │   ├── pipeline.py         # Preprocessing pipeline orchestrator
│   │   ├── symptom_processor.py# Symptom specific processing
│   │   └── text_processor.py   # General text cleaning
│   ├── recommendation/         # Hospital recommendation engine
│   │   └── recommender.py      # Recommendation logic
│   ├── uncertainty/            # Safety and uncertainty estimation
│   │   └── estimator.py        # Uncertainty estimation logic
│   └── utils/                  # Handy functions
├── vector_store/               # FAISS index or other vector DB storage
├── .env                        # Environment variables
├── .gitignore                  # Git ignore file
├── conversations_eda.py        # Standalone EDA script for conversations
├── generate_notebooks.py       # Script to programmatically generate notebooks
├── infrastructure_eda.py       # Standalone EDA script for hospitals
├── pyproject.toml              # Project metadata and dependencies
├── README.md                   # The top-level README for developers using this project
├── run_preprocessing.py        # Script to run the full preprocessing pipeline
├── symptoms_eda.py             # Standalone EDA script for symptoms
└── requirements.txt            # The requirements file for reproducing the analysis environment
```

## 📝 Key Components

*   **`src/`**: Contains all the core logic, divided into modular components.
*   **`data/`**: Follows a pipeline approach from `raw` -> `interim` -> `processed`.
*   **`notebooks/`**: Used for research and prototyping, structured by phase.
*   **`reports/`**: Stores outputs of analysis and experiments.
