import os
import glob
import pandas as pd
from .config import config
from .utils import load_json, save_json, setup_logging

logger = setup_logging(__name__)

class ModelSelector:
    def __init__(self):
        self.models_dir = config.MODELS_DIR
        self.reports_dir = config.REPORTS_DIR

    def select_best_model(self):
        """Rank models based on metrics and select the best one."""
        logger.info("Selecting best model...")
        
        metrics_files = glob.glob(os.path.join(self.models_dir, "*/metrics.json"))
        
        all_metrics = []
        for f in metrics_files:
            model_name = os.path.basename(os.path.dirname(f))
            metrics = load_json(f)
            metrics["model_name"] = model_name
            all_metrics.append(metrics)
            
        if not all_metrics:
            logger.warning("No metrics found. Cannot select best model.")
            return None
            
        df = pd.DataFrame(all_metrics)
        
        # Ranking criteria: Primary F1-score, Secondary Top-3 Accuracy
        df = df.sort_values(by=["f1_score", "top_3_accuracy"], ascending=False)
        
        best_model = df.iloc[0]["model_name"]
        
        # Save leaderboard
        os.makedirs(self.reports_dir, exist_ok=True)
        leaderboard_path = os.path.join(self.reports_dir, "leaderboard.csv")
        df.to_csv(leaderboard_path, index=False)
        
        # Save best model summary
        best_metrics = df.iloc[0].to_dict()
        summary_path = os.path.join(self.reports_dir, "best_model_summary.md")
        
        summary = f"""# 🏆 Best Model Summary

The best model selected based on F1-score and Top-3 Accuracy is **{best_model}**.

## Performance Metrics
- **Accuracy**: {best_metrics.get('accuracy'):.4f}
- **Precision**: {best_metrics.get('precision'):.4f}
- **Recall**: {best_metrics.get('recall'):.4f}
- **F1-Score**: {best_metrics.get('f1_score'):.4f}
- **Top-3 Accuracy**: {best_metrics.get('top_3_accuracy'):.4f}

## Resource Usage
- **Training Time**: {best_metrics.get('training_time_seconds'):.2f}s
- **Inference Latency**: {best_metrics.get('inference_latency_seconds_per_sample'):.6f}s/sample
- **Model Size**: {best_metrics.get('model_size_mb'):.2f}MB

## Conclusion
This model is recommended for deployment based on its balanced performance and efficiency.
"""
        with open(summary_path, "w") as f:
            f.write(summary)
            
        # Save to metadata registry
        os.makedirs(config.METADATA_DIR, exist_ok=True)
        save_json(best_metrics, os.path.join(config.METADATA_DIR, "best_model.json"))
        
        logger.info(f"Best model selected: {best_model}. Summary saved to {summary_path}")
        
        # Generate Safety Analysis
        self.generate_safety_analysis(df)
        
        return best_model

    def generate_safety_analysis(self, df):
        """Generate a dedicated safety analysis report."""
        safety_path = os.path.join(self.reports_dir, "safety_analysis.md")
        
        # Find models with high uncertainty or low accuracy
        risky_models = df[df['accuracy'] < 0.6]['model_name'].tolist()
        
        report = f"""# 🛡️ Safety Analysis Report

## Overview
This report analyzes the safety and reliability of the trained classical ML models for disease prediction.

## High-Confidence Incorrect Predictions
*Critical for healthcare safety.*
- We monitor cases where the model is highly confident (>80%) but incorrect.
- Recommendations: Implement a fallback to human doctors for low-confidence or highly ambiguous cases.

## Model Reliability
"""
        for _, row in df.iterrows():
            report += f"### {row['model_name']}\n"
            report += f"- **Accuracy**: {row.get('accuracy'):.4f}\n"
            report += f"- **Uncertainty Calibration**: See `reports/figures/model_analysis/{row['model_name']}_calibration.png`\n"
            
        report += """
## Risky Disease Overlaps
- Diseases with similar symptoms are hard to distinguish.
- See `error_analysis.md` for top confusion pairs.

## Recommendations
1.  **Always** display confidence scores to users.
2.  **Flag** predictions with high entropy as unreliable.
3.  **Do not** rely solely on the model for critical diagnoses.
"""
        with open(safety_path, "w") as f:
            f.write(report)
        logger.info(f"Saved safety analysis to {safety_path}")
