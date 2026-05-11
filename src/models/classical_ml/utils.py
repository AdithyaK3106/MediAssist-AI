import os
import json
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import config

def setup_logging(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)

logger = setup_logging(__name__)

def load_data():
    """Load train, val, test data. Split if missing."""
    train_path = os.path.join(config.DATA_DIR, "train.csv")
    val_path = os.path.join(config.DATA_DIR, "validation.csv")
    test_path = os.path.join(config.DATA_DIR, "test.csv")
    mapping_path = os.path.join(config.DATA_DIR, "label_mapping.json")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logger.info("Loading existing train/val/test splits.")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        with open(mapping_path, "r") as f:
            label_mapping = json.load(f)
            
        return train_df, val_df, test_df, label_mapping

    logger.info("Data splits not found. Creating splits from symptoms_cleaned.csv.")
    cleaned_data_path = os.path.join(config.DATA_DIR, "symptoms_cleaned.csv")
    
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_data_path}. Please run preprocessing first.")
        
    df = pd.read_csv(cleaned_data_path)
    
    # Drop rows with NaN
    df = df.dropna(subset=['symptom_text_cleaned', 'diseases_standardized'])
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['diseases_standardized'])
    
    # Save mapping
    label_mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
        
    # Split data
    train_size = config.DATA_CONFIG["train_split"]
    val_size = config.DATA_CONFIG["val_split"]
    test_size = config.DATA_CONFIG["test_split"]
    
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=config.DATA_CONFIG["random_state"],
        stratify=df['label']
    )
    
    val_relative_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_relative_size, 
        random_state=config.DATA_CONFIG["random_state"],
        stratify=temp_df['label']
    )
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Data split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df, label_mapping

def save_model(model, model_name):
    path = os.path.join(config.MODELS_DIR, model_name, "classifier.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(model_name):
    path = os.path.join(config.MODELS_DIR, model_name, "classifier.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Model {model_name} not found.")

def save_vectorizer(vectorizer, model_name):
    path = os.path.join(config.MODELS_DIR, model_name, "vectorizer.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)

def load_vectorizer(model_name):
    path = os.path.join(config.MODELS_DIR, model_name, "vectorizer.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Vectorizer for {model_name} not found.")

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def create_notebook(filename, title, cells_data):
    nb = {
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"]
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    
    for c_type, content in cells_data:
        lines = content.splitlines(keepends=True)
        if c_type == "code":
            nb["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })
        else:
            nb["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines
            })
            
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def generate_ml_notebooks():
    """Generate experiment notebooks."""
    logger.info("Generating notebooks...")
    notebook_dir = config.NOTEBOOKS_DIR
    
    models = ["logistic_regression", "naive_bayes", "svm", "random_forest", "decision_tree", "xgboost"]
    
    for model in models:
        cells = [
            ("markdown", f"## Experimentation for {model}"),
            ("code", f"from src.models.classical_ml.predict import ClassicalInferencePipeline\npipeline = ClassicalInferencePipeline(model_name='{model}')\nresults = pipeline.predict(['I have a headache and fever']) \nprint(results)"),
            ("markdown", "## Model Evaluation"),
            ("code", f"import json\nwith open('../../models/classical/{model}/metrics.json', 'r') as f:\n    print(json.load(f))")
        ]
        create_notebook(os.path.join(notebook_dir, f"{model}_experiment.ipynb"), f"{model.replace('_', ' ').title()} Experiment", cells)
        
    # Comparison notebook
    comp_cells = [
        ("markdown", "## Model Comparison"),
        ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nleaderboard = pd.read_csv('../../reports/model_benchmarks/leaderboard.csv')\nprint(leaderboard)"),
        ("code", "sns.barplot(data=leaderboard, x='model_name', y='f1_score')\nplt.title('F1 Score Comparison')\nplt.show()")
    ]
    create_notebook(os.path.join(notebook_dir, "model_comparison.ipynb"), "Model Comparison", comp_cells)
    
    # Additional Notebooks
    create_notebook(os.path.join(notebook_dir, "explainability_analysis.ipynb"), "Explainability Analysis", [
        ("markdown", "## Feature Importance\nView global feature importance."),
        ("code", "from src.explainability.feature_importance import get_feature_importance\ndf = get_feature_importance('logistic_regression')\nprint(df.head())")
    ])
    
    create_notebook(os.path.join(notebook_dir, "error_analysis.ipynb"), "Error Analysis", [
        ("markdown", "## Misclassifications\nAnalyze where the model fails."),
        ("code", "import pandas as pd\ndf = pd.read_csv('../../reports/model_benchmarks/logistic_regression_misclassifications.csv')\nprint(df.head())")
    ])
    
    create_notebook(os.path.join(notebook_dir, "uncertainty_analysis.ipynb"), "Uncertainty Analysis", [
        ("markdown", "## Uncertainty Distribution\nAnalyze confidence and entropy."),
        ("code", "from src.models.classical_ml.uncertainty import analyze_uncertainty\n# Requires model and data")
    ])
    
    logger.info(f"Notebooks generated in {notebook_dir}")
