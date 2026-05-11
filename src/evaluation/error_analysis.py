import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.models.classical_ml.utils import load_data, load_model, load_json, save_json, setup_logging
from src.models.classical_ml.config import config
from src.models.classical_ml.predict import ClassicalInferencePipeline

logger = setup_logging(__name__)

class ErrorAnalyzer:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        self.pipeline = ClassicalInferencePipeline(model_name=model_name)
        
    def run_analysis(self):
        logger.info(f"Running error analysis for {self.model_name}...")
        
        # Load test data
        _, _, test_df, _ = load_data()
        
        if test_df is None:
             logger.error("Test data not found. Run benchmark first.")
             return
             
        texts = test_df['symptom_text_cleaned'].tolist()
        true_labels = test_df['diseases_standardized'].tolist()
        
        # Get predictions
        results = self.pipeline.predict(texts, top_k=3)
        
        # Analyze
        misclassifications = []
        confusion_pairs = []
        
        for i, res in enumerate(results):
            pred_label = res['predictions'][0]['disease']
            true_label = true_labels[i]
            
            if pred_label != true_label:
                misclassifications.append({
                    "text": texts[i],
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": res['predictions'][0]['probability'],
                    "entropy": res['uncertainty']['entropy'] if res['uncertainty'] else 0,
                    "top_3": [p['disease'] for p in res['predictions']]
                })
                confusion_pairs.append((true_label, pred_label))
                
        # Save misclassifications to CSV
        df_mis = pd.DataFrame(misclassifications)
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        csv_path = os.path.join(config.REPORTS_DIR, f"{self.model_name}_misclassifications.csv")
        df_mis.to_csv(csv_path, index=False)
        logger.info(f"Saved misclassifications to {csv_path}")
        
        # Generate Report
        self.generate_report(df_mis, confusion_pairs)
        
        # Generate Plots
        self.generate_plots(df_mis)
        
    def generate_report(self, df_mis, confusion_pairs):
        report_path = os.path.join(config.REPORTS_DIR, f"{self.model_name}_error_analysis.md")
        
        # Top confusion pairs
        from collections import Counter
        pair_counts = Counter(confusion_pairs)
        top_pairs = pair_counts.most_common(5)
        
        # High confidence errors
        high_conf_errors = df_mis[df_mis['confidence'] > 0.7]
        
        report = f"""# 🔍 Error Analysis - {self.model_name}

## Summary
- **Total Misclassifications**: {len(df_mis)}
- **High Confidence Errors (>0.7)**: {len(high_conf_errors)}

## Top Confusion Pairs
"""
        for pair, count in top_pairs:
            report += f"- **{pair[0]}** misclassified as **{pair[1]}**: {count} times\n"
            
        report += """
## High Confidence Failure Examples
"""
        for _, row in high_conf_errors.head(5).iterrows():
            report += f"- **Text**: {row['text']}\n"
            report += f"  - **True**: {row['true_label']}\n"
            report += f"  - **Pred**: {row['predicted_label']} ({row['confidence']:.2f})\n"
            
        report += """
## Recommendations
- Investigate overlap between the top confused pairs.
- Collect more data for failure cases.
- Use uncertainty flags to route low-confidence predictions to humans.
"""
        
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved error analysis report to {report_path}")

    def generate_plots(self, df_mis):
        if df_mis.empty:
            return
            
        # Plot confidence distribution of errors
        plt.figure(figsize=(10, 6))
        sns.histplot(df_mis['confidence'], bins=20, kde=True, color='orange')
        plt.title(f"Confidence Distribution of Errors - {self.model_name}")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.FIGURES_DIR, f"{self.model_name}_error_confidence_dist.png"))
        plt.close()
        
        logger.info(f"Saved error plots to {config.FIGURES_DIR}")

if __name__ == "__main__":
    analyzer = ErrorAnalyzer(model_name="logistic_regression")
    analyzer.run_analysis()
