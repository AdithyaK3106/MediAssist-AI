import os
import yaml

class Config:
    def __init__(self):
        # Base paths
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data/processed/ml")
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "models/classical")
        self.METADATA_DIR = os.path.join(self.PROJECT_ROOT, "models/metadata")
        self.REPORTS_DIR = os.path.join(self.PROJECT_ROOT, "reports/model_benchmarks")
        self.FIGURES_DIR = os.path.join(self.PROJECT_ROOT, "reports/figures/model_analysis")
        self.UNCERTAINTY_FIGURES_DIR = os.path.join(self.PROJECT_ROOT, "reports/figures/uncertainty")
        self.TRACKING_DIR = os.path.join(self.PROJECT_ROOT, "experiments/tracking")
        self.NOTEBOOKS_DIR = os.path.join(self.PROJECT_ROOT, "notebooks/04_classical_ml")

        # Config files
        self.CONFIG_FILE = os.path.join(self.PROJECT_ROOT, "src/config/classical_ml_config.yaml")
        
        # Load yaml config
        self.yaml_config = self._load_yaml_config()
        
        # Model hyperparams
        self.MODEL_CONFIGS = self.yaml_config.get("models", {})
        
        # Vectorizer config
        self.VECTORIZER_CONFIG = self.yaml_config.get("vectorizer", {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "min_df": 2
        })
        
        # Data split config
        self.DATA_CONFIG = self.yaml_config.get("data", {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "random_state": 42
        })

    def _load_yaml_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file not found at {self.CONFIG_FILE}. Using defaults.")
            return {}

config = Config()
