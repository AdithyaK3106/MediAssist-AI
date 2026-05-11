import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import config
from .utils import load_data, setup_logging, load_json, generate_ml_notebooks
from .vectorizer import MedicalVectorizer
from .train import train_model, XGB_AVAILABLE, LGB_AVAILABLE
from .evaluate import evaluate_model
from .uncertainty import analyze_uncertainty
from .experiment_tracker import ExperimentTracker
from .model_selector import ModelSelector

# New imports for extension
from src.evaluation.error_analysis import ErrorAnalyzer
from src.explainability.feature_importance import get_feature_importance, save_global_importance, plot_feature_importance
from src.explainability.tfidf_analysis import get_disease_indicators, save_disease_indicators

logger = setup_logging(__name__)

def run_benchmark():
    logger.info("Starting Classical ML Benchmarking...")
    
    # 1. Load Data
    train_df, val_df, test_df, label_mapping = load_data()
    
    # 2. Vectorize
    vectorizer = MedicalVectorizer(model_name="benchmark_default")
    X_train = vectorizer.fit_transform(train_df['symptom_text_cleaned'])
    X_val = vectorizer.transform(val_df['symptom_text_cleaned'])
    X_test = vectorizer.transform(test_df['symptom_text_cleaned'])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # 3. Models to Train
    models_to_train = [
        "logistic_regression",
        "naive_bayes",
        "svm",
        "random_forest",
        "decision_tree"
    ]
    
    if XGB_AVAILABLE:
        models_to_train.append("xgboost")
    if LGB_AVAILABLE:
        models_to_train.append("lightgbm")
        
    tracker = ExperimentTracker()
    
    # 4. Train and Evaluate
    for model_type in models_to_train:
        try:
            # We save vectorizer per model as requested
            model_vec = MedicalVectorizer(model_name=model_type)
            X_train_m = model_vec.fit_transform(train_df['symptom_text_cleaned'])
            X_test_m = model_vec.transform(test_df['symptom_text_cleaned'])
            X_val_m = model_vec.transform(val_df['symptom_text_cleaned'])
            
            # Train
            model, train_time = train_model(model_type, X_train_m, y_train, X_val_m, y_val)
            
            # Evaluate
            metrics = evaluate_model(model, X_test_m, y_test, model_type, training_time=train_time)
            
            # Uncertainty
            uncertainty_metrics = analyze_uncertainty(model, X_test_m, model_type)
            if uncertainty_metrics:
                metrics.update(uncertainty_metrics)
                
            # Track
            tracker.log_experiment(model_type, metrics)
            
        except Exception as e:
            logger.error(f"Failed to train/evaluate {model_type}: {e}")
            
    # 5. Model Selection
    selector = ModelSelector()
    best_model = selector.select_best_model()
    
    # 6. Generate Comparison Plots
    generate_comparison_plots()
    
    # 7. Generate Notebooks
    generate_ml_notebooks()
    
    # 8. Run Error Analysis for best model (or all, but let's do best for simplicity/speed or a specific one)
    try:
        analyzer = ErrorAnalyzer(model_name=best_model)
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"Failed to run error analysis: {e}")
        
    # 9. Run Explainability for best model
    try:
        df_imp = get_feature_importance(best_model)
        save_global_importance(df_imp, best_model)
        plot_feature_importance(df_imp, best_model)
        
        indicators = get_disease_indicators(best_model)
        save_disease_indicators(indicators, best_model)
    except Exception as e:
        logger.error(f"Failed to run explainability: {e}")
    
    logger.info(f"Benchmarking complete. Best model: {best_model}")

def generate_comparison_plots():
    """Generate plots comparing all models."""
    logger.info("Generating comparison plots...")
    
    metrics_files = glob.glob(os.path.join(config.MODELS_DIR, "*/metrics.json"))
    
    all_metrics = []
    for f in metrics_files:
        model_name = os.path.basename(os.path.dirname(f))
        metrics = load_json(f)
        metrics["model_name"] = model_name
        all_metrics.append(metrics)
        
    if not all_metrics:
        return
        
    df = pd.DataFrame(all_metrics)
    
    # Plot F1-Score Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model_name', y='f1_score', palette='viridis')
    plt.title("Model F1-Score Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.FIGURES_DIR, "model_f1_comparison.png"))
    plt.close()
    
    # Plot Inference Latency
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model_name', y='inference_latency_seconds_per_sample', palette='magma')
    plt.title("Inference Latency Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "model_latency_comparison.png"))
    plt.close()
    
    logger.info(f"Comparison plots saved to {config.FIGURES_DIR}")

if __name__ == "__main__":
    run_benchmark()
