import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from .config import config
from .utils import setup_logging, save_json

logger = setup_logging(__name__)

def compute_top_k_accuracy(y_true, y_probs, k=3):
    """Compute Top-K accuracy."""
    if y_probs is None:
        return 0.0
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    # Check if true label is in top k
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)

def get_model_size(model_name):
    """Estimate model size in MB."""
    path = os.path.join(config.MODELS_DIR, model_name, "classifier.pkl")
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

def evaluate_model(model, X_test, y_test, model_name, training_time=0.0):
    logger.info(f"Evaluating {model_name}...")
    
    # Inference Latency
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    latency = inference_time / X_test.shape[0]
    
    # Probabilities for Top-K and Uncertainty
    y_probs = None
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)
        
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    top_3_acc = compute_top_k_accuracy(y_test, y_probs, k=3) if y_probs is not None else 0.0
    
    model_size = get_model_size(model_name)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "top_3_accuracy": float(top_3_acc),
        "training_time_seconds": float(training_time),
        "inference_latency_seconds_per_sample": float(latency),
        "model_size_mb": float(model_size)
    }
    
    # Save metrics
    metrics_path = os.path.join(config.MODELS_DIR, model_name, "metrics.json")
    save_json(metrics, metrics_path)
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(config.MODELS_DIR, model_name, "classification_report.json")
    save_json(report, report_path)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting
    plot_confusion_matrix(cm, model_name)
    
    if y_probs is not None:
        plot_calibration_curve_custom(y_test, y_probs, model_name)
    
    return metrics

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {plot_path}")

def plot_calibration_curve_custom(y_test, y_probs, model_name):
    """Plot calibration curve for the most confident class predictions."""
    # Multi-class calibration is complex. Here we plot for the max probability (confidence) vs correctness.
    max_probs = np.max(y_probs, axis=1)
    y_pred = np.argmax(y_probs, axis=1)
    correct = (y_pred == y_test).astype(int)
    
    prob_true, prob_pred = calibration_curve(correct, max_probs, n_bins=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(f"Calibration Curve (Reliability Diagram) - {model_name}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_path = os.path.join(config.FIGURES_DIR, f"{model_name}_calibration.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved calibration plot to {plot_path}")
