import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.classical_ml.utils import load_model, load_vectorizer, setup_logging
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

def get_feature_importance(model_name="logistic_regression"):
    """Extract feature importance or coefficients."""
    logger.info(f"Extracting feature importance for {model_name}...")
    
    model = load_model(model_name)
    vectorizer = load_vectorizer(model_name)
    
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        return df
        
    elif hasattr(model, "coef_"):
        # For multi-class linear models, coef_ is (n_classes, n_features)
        # We can average or take max, or return per class
        # Let's take the mean absolute coefficient across classes for global importance
        coefs = np.mean(np.abs(model.coef_), axis=0)
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": coefs
        }).sort_values(by="importance", ascending=False)
        return df
        
    else:
        logger.warning(f"Model {model_name} does not have feature_importances_ or coef_.")
        return None

def save_global_importance(df, model_name):
    if df is None:
        return
        
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(config.REPORTS_DIR, f"feature_importance_{model_name}.md")
    
    top_features = df.head(10)
    
    report = f"""# 📊 Feature Importance - {model_name}

## Top 10 Global Features
"""
    for _, row in top_features.iterrows():
        report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
        
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved feature importance report to {report_path}")

def plot_feature_importance(df, model_name):
    if df is None:
        return
        
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(20), x='importance', y='feature', palette='viridis')
    plt.title(f"Top 20 Features - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    
    figures_dir = os.path.join(config.PROJECT_ROOT, "reports/figures/explainability")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{model_name}_feature_importance.png"), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {figures_dir}")

if __name__ == "__main__":
    df = get_feature_importance("logistic_regression")
    save_global_importance(df, "logistic_regression")
    plot_feature_importance(df, "logistic_regression")
