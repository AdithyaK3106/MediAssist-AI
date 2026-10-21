"""
MediAssist AI — Probability Calibration & Expected Calibration Error (ECE) Engine
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score
to evaluate uncertainty reliability before vs. after Bayesian prior weighting and relative normalization.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.models.transformers.predict_bert import MedicalBERTInferencePipeline
from src.safety.decision_layer import ClinicalDecisionLayer

class CalibrationEvaluator:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.pipeline = MedicalBERTInferencePipeline()
        self.decision_layer = ClinicalDecisionLayer()
        self.test_df = pd.read_csv("data/processed/bert/test.csv")
        
        with open("data/processed/bert/label_mapping.json", "r") as f:
            self.label_mapping = json.load(f)
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}

    def compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> Tuple[float, float, list]:
        """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_details = []

        n_samples = len(confidences)

        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)

                ece += abs_diff * prop_in_bin
                mce = max(mce, abs_diff)

                bin_details.append({
                    "bin_range": f"[{round(bin_lower, 2)}-{round(bin_upper, 2)}]",
                    "sample_count": int(np.sum(in_bin)),
                    "avg_confidence": round(float(avg_confidence_in_bin), 4),
                    "empirical_accuracy": round(float(accuracy_in_bin), 4),
                    "calibration_gap": round(float(abs_diff), 4)
                })

        return round(float(ece) * 100, 2), round(float(mce) * 100, 2), bin_details

    def evaluate_test_set(self) -> Dict[str, Any]:
        texts = self.test_df["symptom_text"].tolist()
        ground_truth = self.test_df["disease"].tolist()

        raw_confidences = []
        raw_accuracies = []

        calibrated_confidences = []
        calibrated_accuracies = []

        print("[Evaluating ECE on Test Set...]")
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_gt = ground_truth[i:i+batch_size]
            batch_results = self.pipeline.predict(batch_texts, top_k=3)

            for j, res in enumerate(batch_results):
                gt = batch_gt[j]
                preds = res.get("predictions", [])
                if not preds:
                    continue

                # 1. Raw Softmax Confidence & Top-1 Accuracy
                raw_top1 = preds[0]
                raw_conf = raw_top1["probability"]
                raw_is_correct = 1.0 if raw_top1["disease"].lower() == gt.lower() else 0.0

                raw_confidences.append(raw_conf)
                raw_accuracies.append(raw_is_correct)

                # 2. Calibrated Relative Likelihood Confidence & Accuracy
                bundle = self.decision_layer.evaluate(
                    bert_predictions=preds,
                    timeline={"duration_days": 3, "severity": "mild", "progression": "stable"}
                )
                calib_top1 = bundle["top_conditions"][0] if bundle["top_conditions"] else {}
                calib_conf = calib_top1.get("relative_confidence", 50.0) / 100.0
                calib_is_correct = 1.0 if calib_top1.get("disease", "").lower() == gt.lower() else 0.0

                calibrated_confidences.append(calib_conf)
                calibrated_accuracies.append(calib_is_correct)

        raw_confidences = np.array(raw_confidences)
        raw_accuracies = np.array(raw_accuracies)
        calib_confidences = np.array(calibrated_confidences)
        calib_accuracies = np.array(calibrated_accuracies)

        raw_ece, raw_mce, raw_bins = self.compute_ece(raw_confidences, raw_accuracies)
        calib_ece, calib_mce, calib_bins = self.compute_ece(calib_confidences, calib_accuracies)

        # Brier Score (lower is better)
        raw_brier = float(np.mean((raw_confidences - raw_accuracies) ** 2))
        calib_brier = float(np.mean((calib_confidences - calib_accuracies) ** 2))

        results = {
            "test_sample_size": len(texts),
            "uncalibrated_raw_softmax": {
                "expected_calibration_error_ece": raw_ece,
                "maximum_calibration_error_mce": raw_mce,
                "brier_score": round(raw_brier, 4),
                "bin_details": raw_bins
            },
            "calibrated_decision_layer": {
                "expected_calibration_error_ece": calib_ece,
                "maximum_calibration_error_mce": calib_mce,
                "brier_score": round(calib_brier, 4),
                "bin_details": calib_bins
            },
            "relative_ece_reduction": f"{round(((raw_ece - calib_ece) / raw_ece) * 100, 2)}%"
        }

        output_path = "reports/model_benchmarks/calibration_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*80)
        print("CALIBRATION & UNCERTAINTY BENCHMARK RESULTS")
        print("="*80)
        print(f"Uncalibrated Softmax ECE: {raw_ece}% | MCE: {raw_mce}% | Brier Score: {round(raw_brier, 4)}")
        print(f"Calibrated Decision ECE:  {calib_ece}% | MCE: {calib_mce}% | Brier Score: {round(calib_brier, 4)}")
        print(f"Relative Calibration Error Reduction: {results['relative_ece_reduction']}")
        print(f"[Saved Calibration Report to {output_path}]")

        return results

if __name__ == "__main__":
    evaluator = CalibrationEvaluator()
    evaluator.evaluate_test_set()
