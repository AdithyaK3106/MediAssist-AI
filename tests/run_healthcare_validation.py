import json
import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.classical_ml.predict import ClassicalInferencePipeline
from src.safety.medical_consistency import MedicalConsistencyValidator
from src.recommendation.home_care import HomeCareRecommender
from src.conversational.safety_guard import SafetyGuard

test_dir = project_root / "tests" / "test_cases"
reports_dir = project_root / "reports" / "testing"
logs_dir = project_root / "logs" / "testing"

reports_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=logs_dir / "validation_runner.log", level=logging.INFO)

print("Initializing MediAssist AI Testing Framework...")
try:
    pipeline = ClassicalInferencePipeline(model_name="logistic_regression")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

consistency_layer = MedicalConsistencyValidator()
home_care_engine = HomeCareRecommender()
safety_guard = SafetyGuard()

def load_cases(filename):
    with open(test_dir / filename, "r", encoding="utf-8") as f:
        return json.load(f)

# Data Holders
metrics = {
    "total_tests": 0,
    "unsafe_predictions": 0,
    "consistency_penalties_applied": 0,
    "emergencies_detected": 0,
    "implausible_suppressed": 0,
    "failures": []
}

detailed_logs = []
confidence_data = []

def run_tests():
    files = ["common_cases.json", "ambiguous_cases.json", "emergency_cases.json", 
             "implausible_cases.json", "noisy_cases.json", "edge_cases.json"]
    
    for filename in files:
        cases = load_cases(filename)
        category = filename.replace("_cases.json", "").upper()
        print(f"Running {len(cases)} tests for category: {category}")
        
        for case in cases:
            user_input = case.get("input", "")
            metrics["total_tests"] += 1
            
            # Pre-inference safety check
            is_emergency_pre = safety_guard.detect_emergency(user_input)
            
            # Predict
            results = pipeline.predict([user_input])[0]
            raw_top_preds = results['predictions']
            
            # Consistency Validation
            consistency_res = consistency_layer.validate_and_rerank(user_input, raw_top_preds)
            reranked_preds = consistency_res['reranked_predictions']
            
            top_disease = reranked_preds[0]['disease']
            home_care = home_care_engine.get_home_care_data(top_disease)
            severity = home_care.get("severity", "medium")
            
            # Track confidence
            confidence_data.append({
                "category": category,
                "confidence": reranked_preds[0]['probability'],
                "disease": top_disease,
                "severity": severity
            })
            
            # Logging
            log_entry = {
                "input": user_input,
                "category": category,
                "top_disease": top_disease,
                "severity": severity,
                "consistency_applied": len(consistency_res['reasoning']) > 0,
                "reasoning": consistency_res['reasoning']
            }
            detailed_logs.append(log_entry)
            
            # ----------------------------------------------------
            # Evaluation Checks
            # ----------------------------------------------------
            
            # 1. Emergency Case check
            if category == "EMERGENCY":
                if not is_emergency_pre and severity != "critical":
                    metrics["failures"].append({"type": "MISSED_EMERGENCY", "input": user_input, "pred": top_disease})
                else:
                    metrics["emergencies_detected"] += 1
                    
            # 2. Implausible Check
            if category == "IMPLAUSIBLE":
                implausible_disease = case.get("implausible_disease")
                # Did the consistency layer penalize it?
                reasoning = next((r for r in consistency_res['reasoning'] if r['disease'] == implausible_disease), None)
                if reasoning and reasoning['penalty_applied'] < 1.0:
                    metrics["implausible_suppressed"] += 1
                elif top_disease == implausible_disease:
                    metrics["unsafe_predictions"] += 1
                    metrics["failures"].append({"type": "UNSAFE_IMPLAUSIBLE", "input": user_input, "pred": top_disease})
                    
            # 4. Consistency General Check
            for r in consistency_res['reasoning']:
                if r['penalty_applied'] < 1.0:
                    metrics["consistency_penalties_applied"] += 1
                    
            # 5. Home Care Safety
            if severity == "critical" and len(home_care.get("home_care", [])) > 0:
                metrics["unsafe_predictions"] += 1
                metrics["failures"].append({"type": "UNSAFE_HOME_CARE_FOR_CRITICAL", "input": user_input, "pred": top_disease})

def generate_visualizations():
    print("Generating visualizations...")
    df = pd.DataFrame(confidence_data)
    
    plt.figure(figsize=(10,6))
    sns.boxplot(x="category", y="confidence", data=df)
    plt.title("Confidence Distribution by Test Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(reports_dir / "confidence_distribution.png")
    
    plt.figure(figsize=(8,5))
    sns.countplot(x="severity", data=df, order=["low", "medium", "high", "critical"], palette="Reds")
    plt.title("Predicted Severity Distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "severity_distribution.png")

def generate_report():
    print("Generating markdown report...")
    upr = (metrics['unsafe_predictions'] / metrics['total_tests']) * 100 if metrics['total_tests'] > 0 else 0
    
    md = f"# MediAssist AI Validation Report\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "## Executive Summary\n"
    md += f"- **Total Tests Executed:** {metrics['total_tests']}\n"
    md += f"- **Unsafe Prediction Rate (UPR):** {upr:.2f}%\n"
    md += f"- **Medical Consistency Interventions:** {metrics['consistency_penalties_applied']}\n"
    md += f"- **Emergencies Detected & Escalated:** {metrics['emergencies_detected']}\n"
    md += f"- **Implausible Predictions Suppressed:** {metrics['implausible_suppressed']}\n\n"
    
    md += "## Safety Analysis\n"
    if upr == 0:
        md += "✅ **Excellent:** The system generated zero unsafe medical outputs across all test vectors.\n"
    elif upr < 5:
        md += "⚠️ **Acceptable:** The system has a low rate of unsafe predictions. Review edge cases.\n"
    else:
        md += "❌ **Critical Failure:** High rate of unsafe predictions. Do not deploy.\n"
        
    md += "\n## Failure Cases\n"
    if not metrics["failures"]:
        md += "No critical failures detected.\n"
    else:
        for f in metrics["failures"][:20]: # show top 20
            md += f"- **{f['type']}** | Input: `{f['input']}` | Prediction: `{f['pred']}`\n"
            
    md += "\n## Visualizations\n"
    md += "![Confidence Distribution](confidence_distribution.png)\n"
    md += "![Severity Distribution](severity_distribution.png)\n"

    with open(reports_dir / "final_validation_report.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    with open(logs_dir / "validation_logs.json", "w", encoding="utf-8") as f:
        json.dump(detailed_logs, f, indent=2)
        
    print(f"Validation complete. Report saved to {reports_dir / 'final_validation_report.md'}")

if __name__ == "__main__":
    run_tests()
    generate_visualizations()
    generate_report()
