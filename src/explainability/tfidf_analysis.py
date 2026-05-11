import os
import json
import pandas as pd
import numpy as np
from src.models.classical_ml.utils import load_model, load_vectorizer, load_json, setup_logging
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

def get_disease_indicators(model_name="logistic_regression"):
    """Extract top indicators for each disease from Linear model coefficients."""
    logger.info(f"Extracting disease indicators for {model_name}...")
    
    model = load_model(model_name)
    vectorizer = load_vectorizer(model_name)
    
    mapping_path = os.path.join(config.DATA_DIR, "label_mapping.json")
    label_mapping = load_json(mapping_path)
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    
    if not hasattr(model, "coef_"):
        logger.warning(f"Model {model_name} does not have coefficients (coef_). Skipping disease indicators.")
        return None
        
    coefs = model.coef_ # (n_classes, n_features)
    feature_names = vectorizer.get_feature_names_out()
    
    indicators = {}
    
    for i in range(coefs.shape[0]):
        disease_name = label_mapping.get(i, f"Class_{i}")
        class_coefs = coefs[i]
        
        # Get top positive coefficients
        top_indices = np.argsort(class_coefs)[-10:][::-1]
        
        indicators[disease_name] = [
            {"symptom": str(feature_names[idx]), "weight": float(class_coefs[idx])}
            for idx in top_indices if class_coefs[idx] > 0
        ]
        
    return indicators

def save_disease_indicators(indicators, model_name):
    if indicators is None:
        return
        
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORTS_DIR, f"disease_indicators_{model_name}.md")
    
    report = f"# 🩺 Disease Specific Indicators - {model_name}\n\n"
    
    for disease, ind in indicators.items():
        report += f"### {disease}\n"
        for item in ind:
            report += f"- **{item['symptom']}**: {item['weight']:.4f}\n"
        report += "\n"
        
    with open(report_path, "w") as f:
        f.write(report)
        
    # Also save as JSON
    json_path = os.path.join(config.REPORTS_DIR, f"disease_indicators_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(indicators, f, indent=4)
        
    logger.info(f"Saved disease indicators to {report_path}")

if __name__ == "__main__":
    indicators = get_disease_indicators("logistic_regression")
    save_disease_indicators(indicators, "logistic_regression")
