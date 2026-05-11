import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.classical_ml.utils import load_model, load_vectorizer, setup_logging
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP package not installed. Skipping SHAP analysis.")

def run_shap_analysis(model_name="logistic_regression", X_sample=None):
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping.")
        return
        
    logger.info(f"Running SHAP analysis for {model_name}...")
    
    model = load_model(model_name)
    vectorizer = load_vectorizer(model_name)
    
    if X_sample is None:
        logger.warning("No sample data provided for SHAP. Skipping.")
        return
        
    # SHAP expects dense arrays for some explainers or specific formats
    # For linear models we can use LinearExplainer
    # For trees we can use TreeExplainer
    
    figures_dir = os.path.join(config.PROJECT_ROOT, "reports/figures/explainability")
    os.makedirs(figures_dir, exist_ok=True)
    
    try:
        if model_name in ["logistic_regression", "svm"]:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=vectorizer.get_feature_names_out(), show=False)
            plt.savefig(os.path.join(figures_dir, f"{model_name}_shap_summary.png"), bbox_inches='tight')
            plt.close()
            
        elif model_name in ["random_forest", "xgboost", "decision_tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=vectorizer.get_feature_names_out(), show=False)
            plt.savefig(os.path.join(figures_dir, f"{model_name}_shap_summary.png"), bbox_inches='tight')
            plt.close()
            
        logger.info(f"Saved SHAP plots to {figures_dir}")
        
    except Exception as e:
        logger.error(f"Failed to run SHAP for {model_name}: {e}")

if __name__ == "__main__":
    # This requires sample data to run
    pass
