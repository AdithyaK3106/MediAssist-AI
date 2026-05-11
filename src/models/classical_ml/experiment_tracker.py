import os
import json
import csv
import time
from .config import config
from .utils import setup_logging, save_json

logger = setup_logging(__name__)

class ExperimentTracker:
    def __init__(self):
        self.tracking_dir = config.TRACKING_DIR
        os.makedirs(self.tracking_dir, exist_ok=True)
        self.csv_path = os.path.join(self.tracking_dir, "experiments.csv")
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "model_name", "accuracy", "precision", 
                    "recall", "f1_score", "top_3_accuracy", 
                    "training_time", "inference_latency", "model_size_mb"
                ])

    def log_experiment(self, model_name, metrics):
        """Log experiment metrics to CSV and JSON."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                model_name,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("top_3_accuracy"),
                metrics.get("training_time_seconds"),
                metrics.get("inference_latency_seconds_per_sample"),
                metrics.get("model_size_mb")
            ])
            
        # Save detailed run to JSON
        run_data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "metrics": metrics,
            "config": config.MODEL_CONFIGS.get(model_name, {})
        }
        
        run_id = f"{model_name}_{int(time.time())}"
        run_path = os.path.join(self.tracking_dir, f"{run_id}.json")
        save_json(run_data, run_path)
        
        logger.info(f"Logged experiment for {model_name} in {self.csv_path}")
        
    def get_leaderboard(self):
        """Read CSV and return sorted leaderboard."""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            # Sort by F1-score descending
            leaderboard = df.sort_values(by="f1_score", ascending=False)
            return leaderboard
        return None
