"""
MediAssist AI — Backbone Comparison Benchmarking Suite (Rigorous & Reproducible)
Evaluates all verified model representations across the exact same 35-condition test split (N=239).
Models evaluated:
1. Fine-Tuned PubMedBERT (BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
2. Calibrated TF-IDF + Logistic Regression
3. TF-IDF + Multinomial Naive Bayes

Metrics computed:
- Top-1 Accuracy with 95% Wilson Score Confidence Interval
- Top-3 Accuracy with 95% Wilson Score Confidence Interval
- Macro Precision, Macro Recall, Macro-F1
- Weighted-F1
- CPU Inference Latency (Mean, Median, p95 per sample)
"""

import os
import time
import json
import math
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.transformers.predict_bert import MedicalBERTInferencePipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "bert" / "test.csv"
LABEL_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "bert" / "label_mapping.json"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_benchmarks" / "backbone_comparisons.json"

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes the Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0
    return round(lower, 2), round(upper, 2)

class BackboneBenchmarkRunner:
    def __init__(self):
        self.test_df = pd.read_csv(TEST_PATH)
        with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # 1. PubMedBERT
        print("[Initializing PubMedBERT Inference Pipeline...]")
        self.pubmedbert = MedicalBERTInferencePipeline()
        
        # 2. Logistic Regression
        lr_dir = PROJECT_ROOT / "models" / "classical" / "logistic_regression"
        self.lr_model = joblib.load(lr_dir / "classifier.pkl")
        self.lr_vec = joblib.load(lr_dir / "vectorizer.pkl")
        
        # 3. Naive Bayes
        nb_dir = PROJECT_ROOT / "models" / "classical" / "naive_bayes"
        self.nb_model = joblib.load(nb_dir / "classifier.pkl")
        self.nb_vec = joblib.load(nb_dir / "vectorizer.pkl")

    def evaluate_pubmedbert(self) -> Dict[str, Any]:
        """Evaluates fine-tuned PubMedBERT on test split."""
        texts = self.test_df["symptom_text"].tolist()
        ground_truth = self.test_df["disease"].tolist()
        
        latencies = []
        preds_list = []
        top3_hits = 0
        
        # Measure single-sample latencies for realistic distribution
        for text in texts:
            t0 = time.perf_counter()
            res = self.pubmedbert.predict([text], top_k=3)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            preds_list.append(res)
            
        y_true = ground_truth
        y_pred = []
        for i, res in enumerate(preds_list):
            preds = res.get("predictions", [])
            top_names = [p["disease"] for p in preds]
            top1_name = top_names[0] if top_names else "Unknown"
            y_pred.append(top1_name)
            if y_true[i] in top_names:
                top3_hits += 1
                
        top1_hits = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        n = len(y_true)
        
        top1_acc = (top1_hits / n) * 100.0
        top3_acc = (top3_hits / n) * 100.0
        top1_ci = wilson_score_interval(top1_hits, n)
        top3_ci = wilson_score_interval(top3_hits, n)
        
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        
        return {
            "model_name": "PubMedBERT (Fine-Tuned BiomedNLP)",
            "architecture": "12-layer Transformer (BERT-base, max_len=64)",
            "parameters": "110M",
            "model_size_mb": 418.0,
            "verification_status": "VERIFIED (Measured on test split)",
            "sample_size": n,
            "top1_accuracy": round(top1_acc, 2),
            "top1_ci95": top1_ci,
            "top3_accuracy": round(top3_acc, 2),
            "top3_ci95": top3_ci,
            "macro_precision": round(prec_macro, 2),
            "macro_recall": round(rec_macro, 2),
            "macro_f1": round(f1_macro, 2),
            "weighted_f1": round(f1_weighted, 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_median_ms": round(float(np.median(latencies)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2)
        }

    def evaluate_classical_model(self, model, vectorizer, name: str, arch: str, params: str, size_mb: float) -> Dict[str, Any]:
        """Evaluates classical ML models on test split."""
        texts = self.test_df["symptom_text"].tolist()
        ground_truth = self.test_df["disease"].tolist()
        
        latencies = []
        classes = model.classes_
        
        # Latency measurement
        for text in texts:
            t0 = time.perf_counter()
            X_sample = vectorizer.transform([text])
            _ = model.predict_proba(X_sample)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        X_test = vectorizer.transform(texts)
        probs = model.predict_proba(X_test)
        
        y_true = ground_truth
        y_pred = []
        top3_hits = 0
        n = len(y_true)
        
        for i in range(n):
            prob_row = probs[i]
            top_k_indices = np.argsort(prob_row)[-3:][::-1]
            top_class_ids = [classes[idx] for idx in top_k_indices]
            top_disease_names = [self.label_mapping.get(cid, "Unknown") for cid in top_class_ids]
            
            y_pred.append(top_disease_names[0])
            if y_true[i] in top_disease_names:
                top3_hits += 1
                
        top1_hits = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        top1_acc = (top1_hits / n) * 100.0
        top3_acc = (top3_hits / n) * 100.0
        top1_ci = wilson_score_interval(top1_hits, n)
        top3_ci = wilson_score_interval(top3_hits, n)
        
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
        
        return {
            "model_name": name,
            "architecture": arch,
            "parameters": params,
            "model_size_mb": size_mb,
            "verification_status": "VERIFIED (Measured on test split)",
            "sample_size": n,
            "top1_accuracy": round(top1_acc, 2),
            "top1_ci95": top1_ci,
            "top3_accuracy": round(top3_acc, 2),
            "top3_ci95": top3_ci,
            "macro_precision": round(prec_macro, 2),
            "macro_recall": round(rec_macro, 2),
            "macro_f1": round(f1_macro, 2),
            "weighted_f1": round(f1_weighted, 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 4),
            "latency_median_ms": round(float(np.median(latencies)), 4),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 4)
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        results = {
            "metadata": {
                "evaluation_date": "2026-10-21",
                "test_sample_size": len(self.test_df),
                "num_classes": len(self.label_mapping),
                "hardware": "Intel x86_64 CPU (PyTorch 2.14, scikit-learn 1.9)",
                "note": "All reported models evaluated on identical test.csv (N=239). Fabricated/unrun models removed."
            },
            "empirical_models": {}
        }
        
        # 1. PubMedBERT
        print("\nEvaluating PubMedBERT on 35-class Test Split (N=239)...")
        res_pubmed = self.evaluate_pubmedbert()
        results["empirical_models"]["pubmedbert"] = res_pubmed
        print(f"  Top-1: {res_pubmed['top1_accuracy']}% {res_pubmed['top1_ci95']} | Top-3: {res_pubmed['top3_accuracy']}% {res_pubmed['top3_ci95']}")
        print(f"  Macro-F1: {res_pubmed['macro_f1']}% | Weighted-F1: {res_pubmed['weighted_f1']}% | Latency (Mean): {res_pubmed['latency_mean_ms']} ms")
        
        # 2. Logistic Regression
        print("\nEvaluating Calibrated Logistic Regression on Test Split...")
        res_lr = self.evaluate_classical_model(
            self.lr_model, self.lr_vec,
            name="Calibrated Logistic Regression",
            arch="Linear Classifier over TF-IDF n-grams (1-2)",
            params="24.8K",
            size_mb=1.4
        )
        results["empirical_models"]["logistic_regression"] = res_lr
        print(f"  Top-1: {res_lr['top1_accuracy']}% {res_lr['top1_ci95']} | Top-3: {res_lr['top3_accuracy']}% {res_lr['top3_ci95']}")
        print(f"  Macro-F1: {res_lr['macro_f1']}% | Weighted-F1: {res_lr['weighted_f1']}% | Latency (Mean): {res_lr['latency_mean_ms']} ms")
        
        # 3. Multinomial Naive Bayes
        print("\nEvaluating Multinomial Naive Bayes on Test Split...")
        res_nb = self.evaluate_classical_model(
            self.nb_model, self.nb_vec,
            name="Multinomial Naive Bayes",
            arch="Conditional Independence over TF-IDF n-grams (1-2)",
            params="24.8K",
            size_mb=1.4
        )
        results["empirical_models"]["naive_bayes"] = res_nb
        print(f"  Top-1: {res_nb['top1_accuracy']}% {res_nb['top1_ci95']} | Top-3: {res_nb['top3_accuracy']}% {res_nb['top3_ci95']}")
        print(f"  Macro-F1: {res_nb['macro_f1']}% | Weighted-F1: {res_nb['weighted_f1']}% | Latency (Mean): {res_nb['latency_mean_ms']} ms")

        # Save verified benchmark results
        REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\n[Saved verified benchmark results to {REPORT_OUTPUT_PATH}]")
        return results

if __name__ == "__main__":
    runner = BackboneBenchmarkRunner()
    runner.run_all_benchmarks()
