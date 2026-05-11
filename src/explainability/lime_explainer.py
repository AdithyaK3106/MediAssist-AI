import os
from src.models.classical_ml.utils import load_model, load_vectorizer, setup_logging
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

try:
    from lime import lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME package not installed. Skipping LIME analysis.")

class LimeExplainer:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        if not LIME_AVAILABLE:
            return
            
        self.model = load_model(model_name)
        self.vectorizer = load_vectorizer(model_name)
        
        self.explainer = lime_text.LimeTextExplainer(class_names=None) # Can add class names if available

    def explain(self, text):
        if not LIME_AVAILABLE:
             return {"error": "LIME not available."}
             
        def predict_proba(texts):
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)
            
        exp = self.explainer.explain_instance(text, predict_proba, num_features=6)
        return exp.as_list()

if __name__ == "__main__":
    pass
