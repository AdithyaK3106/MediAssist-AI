import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import config
from .utils import setup_logging

logger = setup_logging(__name__)

def calculate_entropy(probabilities):
    """Calculate prediction entropy."""
    return -np.sum(probabilities * np.log(probabilities + 1e-12))

def calculate_top2_gap(probabilities):
    """Calculate difference between top 2 probabilities."""
    sorted_probs = np.sort(probabilities)
    return sorted_probs[-1] - sorted_probs[-2]

def analyze_uncertainty(model, X_test, model_name):
    """Generate uncertainty metrics and plots."""
    logger.info(f"Analyzing uncertainty for {model_name}...")
    
    if not hasattr(model, "predict_proba"):
        logger.warning(f"Model {model_name} does not support predict_proba. Skipping uncertainty analysis.")
        return
        
    probs = model.predict_proba(X_test)
    
    entropies = np.array([calculate_entropy(p) for p in probs])
    gaps = np.array([calculate_top2_gap(p) for p in probs])
    max_confs = np.max(probs, axis=1)
    
    # Plot Confidence Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(max_confs, bins=50, kde=True, color='green')
    plt.title(f"Confidence Distribution - {model_name}")
    plt.xlabel("Max Probability")
    plt.ylabel("Frequency")
    
    os.makedirs(config.UNCERTAINTY_FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.UNCERTAINTY_FIGURES_DIR, f"{model_name}_confidence_hist.png"))
    plt.close()
    
    # Plot Entropy Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(entropies, bins=50, kde=True, color='red')
    plt.title(f"Entropy Distribution - {model_name}")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(config.UNCERTAINTY_FIGURES_DIR, f"{model_name}_entropy_hist.png"))
    plt.close()
    
    # Plot Top-2 Gap Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(gaps, bins=50, kde=True, color='blue')
    plt.title(f"Top-2 Probability Gap - {model_name}")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(config.UNCERTAINTY_FIGURES_DIR, f"{model_name}_gap_hist.png"))
    plt.close()
    
    logger.info(f"Uncertainty plots saved for {model_name}")
    
    return {
        "mean_entropy": float(np.mean(entropies)),
        "mean_gap": float(np.mean(gaps)),
        "mean_max_confidence": float(np.mean(max_confs))
    }
