import os
import numpy as np
from src.models.classical_ml.utils import load_model, load_vectorizer, load_json, setup_logging
from src.models.classical_ml.config import config

logger = setup_logging(__name__)

class PredictionExplainer:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        self.model = load_model(model_name)
        self.vectorizer = load_vectorizer(model_name)
        
        mapping_path = os.path.join(config.DATA_DIR, "label_mapping.json")
        self.label_mapping = load_json(mapping_path)
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}

    def explain(self, text):
        """Explain prediction for a single text input."""
        if not hasattr(self.model, "coef_"):
             return {"error": "Model does not support coefficient-based explanation."}
             
        X = self.vectorizer.transform([text])
        
        # Get prediction
        probs = self.model.predict_proba(X)[0]
        pred_class_idx = np.argmax(probs)
        pred_label = self.label_mapping.get(pred_class_idx, "Unknown")
        
        # Calculate contribution
        # contribution = TF-IDF value * Coefficient
        tfidf_vals = X.toarray()[0]
        coefs = self.model.coef_[pred_class_idx]
        
        contributions = tfidf_vals * coefs
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get positive contributions
        positive_indices = np.where(contributions > 0)[0]
        
        explanations = []
        for idx in positive_indices:
            if tfidf_vals[idx] > 0: # Only words present in input
                explanations.append({
                    "word": str(feature_names[idx]),
                    "contribution": float(contributions[idx]),
                    "tfidf": float(tfidf_vals[idx])
                })
                
        # Sort by contribution
        explanations = sorted(explanations, key=lambda x: x['contribution'], reverse=True)
        
        return {
            "text": text,
            "prediction": pred_label,
            "confidence": float(probs[pred_class_idx]),
            "explanations": explanations
        }

if __name__ == "__main__":
    explainer = PredictionExplainer("logistic_regression")
    res = explainer.explain("I have a high fever and body pain")
    print(res)
