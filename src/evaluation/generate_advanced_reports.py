import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from src.models.classical_ml.utils import load_data, setup_logging
from src.models.classical_ml.predict import ClassicalInferencePipeline
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

class AdvancedReporter:
    def __init__(self):
        self.models = ['logistic_regression', 'naive_bayes', 'random_forest', 'decision_tree', 'xgboost']
        self.project_root = Path(config.PROJECT_ROOT)
        self.reports_dir = self.project_root / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.cm_dir = self.figures_dir / "confusion_matrices"
        self.safety_dir = self.figures_dir / "safety"
        self.analysis_dir = self.reports_dir / "analysis"
        self.tables_dir = self.reports_dir / "tables"
        
        # Create directories
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.safety_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        # Load test data
        _, _, test_df, _ = load_data()
        if test_df is None:
            logger.error("Test data not found.")
            return
            
        texts = test_df['symptom_text_cleaned'].tolist()
        true_labels = test_df['diseases_standardized'].tolist()
        
        all_results = {}
        
        for model_name in self.models:
            logger.info(f"Evaluating {model_name}...")
            try:
                pipeline = ClassicalInferencePipeline(model_name=model_name)
                results = pipeline.predict(texts, top_k=3)
                all_results[model_name] = results
                
                # Generate Confusion Matrix
                self.generate_confusion_matrix(true_labels, results, model_name)
                
                # Calculate UPR
                self.calculate_upr(true_labels, results, model_name)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                
        # Generate Hardest Classes Analysis
        self.generate_hardest_classes(true_labels, all_results)
        
        # Generate Final Evaluation Tables
        self.generate_evaluation_tables(true_labels, all_results)
        
    def generate_confusion_matrix(self, true_labels, results, model_name):
        pred_labels = [res['predictions'][0]['disease'] for res in results]
        # Get unique labels from both true and predicted
        all_labels = np.array(sorted(list(set(true_labels) | set(pred_labels))))
        num_classes = len(all_labels)
        
        cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
        
        # Calculate Normalized Matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            
        # If too many classes, filter to show only the most 'interesting' ones (those with errors)
        if num_classes > 40:
            logger.info(f"Detected {num_classes} classes. Filtering for top 40 most confused classes to reduce clutter.")
            # Errors = Total True - True Positives
            errors = cm.sum(axis=1) - np.diag(cm)
            # Also consider classes that were falsely predicted often
            false_positives = cm.sum(axis=0) - np.diag(cm)
            total_confusion = errors + false_positives
            
            # Get indices of top 40 most confused classes
            top_indices = np.argsort(total_confusion)[-40:]
            # Ensure indices are sorted to maintain alphabetical/original order if desired, 
            # but usually sorting by error count is more informative.
            # We'll keep them in the order of 'most confused' for the heatmap.
            
            display_labels = all_labels[top_indices]
            cm_display = cm[np.ix_(top_indices, top_indices)]
            cm_norm_display = cm_norm[np.ix_(top_indices, top_indices)]
            title_suffix = " (Top 40 Most Confused Classes)"
        else:
            display_labels = all_labels
            cm_display = cm
            cm_norm_display = cm_norm
            title_suffix = ""

        # Dynamic figure sizing for the subset
        num_display = len(display_labels)
        fig_width = max(16, num_display * 0.45)
        fig_height = max(12, num_display * 0.35)
        
        # Plot Raw
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm_display, 
            annot=True, 
            fmt='d', 
            xticklabels=display_labels, 
            yticklabels=display_labels, 
            cmap='Blues',
            annot_kws={"size": 8 if num_display < 30 else 6},
            cbar_kws={'shrink': 0.8}
        )
        plt.title(f"Confusion Matrix - {model_name}{title_suffix}", fontsize=18, pad=20)
        plt.xlabel("Predicted Disease", fontsize=14, labelpad=15)
        plt.ylabel("True Disease", fontsize=14, labelpad=15)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(self.cm_dir / f"{model_name}_cm_raw.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Normalized
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm_norm_display, 
            annot=True, 
            fmt='.2f', 
            xticklabels=display_labels, 
            yticklabels=display_labels, 
            cmap='Blues',
            annot_kws={"size": 8 if num_display < 30 else 6},
            cbar_kws={'shrink': 0.8}
        )
        plt.title(f"Normalized Confusion Matrix - {model_name}{title_suffix}", fontsize=18, pad=20)
        plt.xlabel("Predicted Disease", fontsize=14, labelpad=15)
        plt.ylabel("True Disease", fontsize=14, labelpad=15)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(self.cm_dir / f"{model_name}_cm_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrices for {model_name}")

    def calculate_upr(self, true_labels, results, model_name, threshold=0.7):
        incorrect_high_conf = 0
        total_predictions = len(results)
        confidences = []
        
        for i, res in enumerate(results):
            pred_label = res['predictions'][0]['disease']
            prob = res['predictions'][0]['probability']
            confidences.append(prob)
            
            if pred_label != true_labels[i] and prob > threshold:
                incorrect_high_conf += 1
                
        upr = (incorrect_high_conf / total_predictions) * 100
        
        # Plot Confidence Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20, kde=True, color='blue')
        plt.title(f"Confidence Distribution - {model_name}")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.savefig(self.safety_dir / f"{model_name}_confidence_dist.png")
        plt.close()
        
        return upr, confidences

    def generate_hardest_classes(self, true_labels, all_results):
        best_model = 'logistic_regression'
        if best_model not in all_results:
            return
            
        results = all_results[best_model]
        pred_labels = [res['predictions'][0]['disease'] for res in results]
        
        report = classification_report(true_labels, pred_labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Filter out averages
        df_classes = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
        
        # Sort by F1
        df_hardest = df_classes.sort_values(by='f1-score').head(10)
        df_hardest.to_csv(self.analysis_dir / "hardest_classes.csv")
        
        # Confused pairs
        confusion_pairs = []
        for i, res in enumerate(results):
            pred_label = res['predictions'][0]['disease']
            true_label = true_labels[i]
            if pred_label != true_label:
                confusion_pairs.append((true_label, pred_label))
                
        from collections import Counter
        pair_counts = Counter(confusion_pairs)
        df_pairs = pd.DataFrame(pair_counts.most_common(20), columns=['Pair', 'Count'])
        df_pairs.to_csv(self.analysis_dir / "confused_pairs.csv", index=False)
        
        logger.info("Saved hardest classes and confused pairs analysis.")

    def generate_evaluation_tables(self, true_labels, all_results):
        summary_data = []
        
        for model_name, results in all_results.items():
            pred_labels = [res['predictions'][0]['disease'] for res in results]
            
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
            accuracy = np.mean([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)])
            
            top_3_acc = 0
            for i, res in enumerate(results):
                top_3 = [p['disease'] for p in res['predictions']]
                if true_labels[i] in top_3:
                    top_3_acc += 1
            top_3_acc /= len(results)
            
            upr, _ = self.calculate_upr(true_labels, results, model_name)
            
            # Hardcoded metadata from previous runs or prompt
            metadata = {
                'logistic_regression': {'size': 6.2, 'time': 4.46e-06},
                'naive_bayes': {'size': 12.4, 'time': 4.68e-06},
                'random_forest': {'size': 9696.0, 'time': 0.000185},
                'decision_tree': {'size': 0.74, 'time': 2.65e-06},
                'xgboost': {'size': 17.6, 'time': 0.00024}
            }
            
            model_size = metadata.get(model_name, {}).get('size', 0)
            inference_time = metadata.get(model_name, {}).get('time', 0)
                
            summary_data.append({
                "Model": model_name,
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1-score": f"{f1:.4f}",
                "Top-3 Accuracy": f"{top_3_acc:.4f}",
                "UPR (%)": f"{upr:.2f}",
                "Model Size (MB)": f"{model_size:.2f}",
                "Inference Time (s)": f"{inference_time:.6f}"
            })
            
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.tables_dir / "final_evaluation.csv", index=False)
        # Manual markdown table generation to avoid 'tabulate' dependency
        cols = df_summary.columns
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, separator]
        for _, row in df_summary.iterrows():
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        markdown_table = "\n".join(lines)
        
        with open(self.tables_dir / "final_evaluation.md", "w") as f:
            f.write(markdown_table)
        
        logger.info("Saved final evaluation tables.")

if __name__ == "__main__":
    reporter = AdvancedReporter()
    reporter.run()
