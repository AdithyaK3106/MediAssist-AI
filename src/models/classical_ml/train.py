import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import config
from .utils import save_model, save_json, setup_logging
from .vectorizer import MedicalVectorizer

logger = setup_logging(__name__)

# Conditional imports for optional packages
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Skipping if selected.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not installed. Skipping if selected.")

def get_model(model_type, params):
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "naive_bayes":
        return MultinomialNB(**params)
    elif model_type == "svm":
        return SVC(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif model_type == "xgboost" and XGB_AVAILABLE:
        return xgb.XGBClassifier(**params)
    elif model_type == "lightgbm" and LGB_AVAILABLE:
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Model type {model_type} not supported or package not available.")

def train_model(model_type, X_train, y_train, X_val=None, y_val=None):
    logger.info(f"Training {model_type}...")
    
    params = config.MODEL_CONFIGS.get(model_type, {})
    model = get_model(model_type, params)
    
    start_time = time.time()
    
    # Handle lightgbm specific validation if needed, or just train normally
    if model_type == "lightgbm" and LGB_AVAILABLE and X_val is not None:
         model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
         model.fit(X_train, y_train)
         
    training_time = time.time() - start_time
    logger.info(f"Training complete in {training_time:.2f} seconds.")
    
    save_model(model, model_type)
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": params,
        "training_time_seconds": training_time
    }
    
    metadata_path = os.path.join(config.MODELS_DIR, model_type, "metadata.json")
    save_json(metadata, metadata_path)
    
    return model, training_time
