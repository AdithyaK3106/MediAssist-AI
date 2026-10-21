"""
MediAssist AI — Statistical Probability Calibration & Uncertainty Quantification
Implements post-hoc Temperature Scaling (Guo et al., ICML 2017) using held-out validation data.
Evaluates Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score
on the untouched test split (N=239).
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, Any, Tuple

from src.models.transformers.predict_bert import MedicalBERTInferencePipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "bert" / "val.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "bert" / "test.csv"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "model_benchmarks" / "calibration_results.json"

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """Optimizes temperature T on validation logits to minimize Negative Log-Likelihood."""
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_t = torch.tensor(val_labels, dtype=torch.long)
        
        nll_criterion = torch.nn.CrossEntropyLoss()
        
        def nll_func(t_val):
            t = float(t_val[0])
            if t <= 0:
                return 1e5
            scaled = val_logits_t / t
            return float(nll_criterion(scaled, val_labels_t).item())
        
        res = minimize(nll_func, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        print(f"[Temperature Scaling] Learned optimal temperature T* = {self.temperature:.4f} on validation set (N={len(val_labels)})")
        return self.temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature to logits and returns calibrated softmax probabilities."""
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Top-1 Brier Score
    - Multiclass Brier Score
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    n_samples = len(labels)
    num_classes = probs.shape[1]
    
    # 1. Brier Scores
    # Multiclass Brier: mean squared error across all class probability indicators
    one_hot_labels = np.zeros_like(probs)
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1.0
    multiclass_brier = float(np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1)))
    
    # Top-1 Brier: (confidence - is_correct)^2
    top1_brier = float(np.mean((confidences - accuracies) ** 2))
    
    # 2. ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            abs_diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            
            bin_details.append({
                "bin_range": f"[{round(bin_lower, 2)}-{round(bin_upper, 2)}]",
                "sample_count": int(np.sum(in_bin)),
                "avg_confidence": round(avg_confidence_in_bin, 4),
                "empirical_accuracy": round(accuracy_in_bin, 4),
                "calibration_gap": round(abs_diff, 4)
            })
            
    return {
        "ece": round(float(ece) * 100, 2),
        "mce": round(float(mce) * 100, 2),
        "top1_brier_score": round(top1_brier, 4),
        "multiclass_brier_score": round(multiclass_brier, 4),
        "bin_details": bin_details
    }

def run_calibration_experiments():
    print("=" * 80)
    print("MEDPASSIST AI — STATISTICAL CALIBRATION EXPERIMENTS")
    print("=" * 80)
    
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    pipeline = MedicalBERTInferencePipeline()
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    device = pipeline.device
    
    model.eval()
    
    def extract_logits(texts: list) -> np.ndarray:
        all_logits = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i:i+batch_size]
                enc = tokenizer(b_texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
                out = model(**enc)
                all_logits.append(out.logits.cpu().numpy())
        return np.vstack(all_logits)
    
    print("\n[1/3] Extracting logits on Validation Set (N=238)...")
    val_logits = extract_logits(val_df["symptom_text"].tolist())
    val_labels = val_df["label"].values
    
    print("[2/3] Extracting logits on Test Set (N=239)...")
    test_logits = extract_logits(test_df["symptom_text"].tolist())
    test_labels = test_df["label"].values
    
    # 1. Fit Temperature Scaler on Validation
    scaler = TemperatureScaler()
    t_opt = scaler.fit(val_logits, val_labels)
    
    # 2. Compute Raw Uncalibrated Softmax on Test
    raw_test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    raw_test_probs = raw_test_probs / np.sum(raw_test_probs, axis=1, keepdims=True)
    raw_metrics = compute_calibration_metrics(raw_test_probs, test_labels)
    
    # 3. Compute Temperature-Scaled Calibrated Softmax on Test
    calib_test_probs = scaler.scale(test_logits)
    calib_metrics = compute_calibration_metrics(calib_test_probs, test_labels)
    
    # Compute relative changes
    ece_rel = round(((calib_metrics["ece"] - raw_metrics["ece"]) / raw_metrics["ece"]) * 100, 2)
    brier_rel = round(((calib_metrics["multiclass_brier_score"] - raw_metrics["multiclass_brier_score"]) / raw_metrics["multiclass_brier_score"]) * 100, 2)
    
    print("\n[3/3] Calibration Evaluation on Test Set:")
    print(f"  Raw Softmax        : ECE = {raw_metrics['ece']}% | MCE = {raw_metrics['mce']}% | Multiclass Brier = {raw_metrics['multiclass_brier_score']}")
    print(f"  Temperature Scaled : ECE = {calib_metrics['ece']}% | MCE = {calib_metrics['mce']}% | Multiclass Brier = {calib_metrics['multiclass_brier_score']}")
    print(f"  ECE Change         : {ece_rel}%")
    print(f"  Brier Score Change : {brier_rel}%")
    
    results = {
        "optimal_temperature_T": round(t_opt, 4),
        "val_sample_size": len(val_labels),
        "test_sample_size": len(test_labels),
        "uncalibrated_raw_softmax": raw_metrics,
        "calibrated_temperature_scaled": calib_metrics,
        "ece_relative_reduction": f"{ece_rel}%",
        "multiclass_brier_relative_reduction": f"{brier_rel}%",
        "layer4_clinical_prior_note": (
            "Layer 4 applies a 1.8x Bayesian outpatient prior multiplier for benign primary care complaints "
            "and normalizes top-3 relative likelihood for patient communication match bands (Primary >=50%, "
            "Possible Alternative 25-50%, Less Likely <25%). This is an interface decision support rule, "
            "distinct from post-hoc temperature scaling."
        )
    }
    
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved verified calibration results to {OUTPUT_REPORT}]")
    return results

if __name__ == "__main__":
    run_calibration_experiments()
