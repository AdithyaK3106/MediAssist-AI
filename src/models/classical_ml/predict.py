import os
import numpy as np
from .config import config
from .utils import load_model, load_vectorizer, load_json, setup_logging
from .uncertainty import calculate_entropy, calculate_top2_gap

logger = setup_logging(__name__)

class ClassicalInferencePipeline:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        self.model = load_model(model_name)
        self.vectorizer = load_vectorizer(model_name)
        
        mapping_path = os.path.join(config.DATA_DIR, "label_mapping.json")
        self.label_mapping = load_json(mapping_path)
        # Convert string keys to int
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}

    def predict(self, texts, top_k=3):
        """Predict disease with confidence and uncertainty."""
        X = self.vectorizer.transform(texts)
        
        results = []
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            
            for i in range(len(texts)):
                prob = probs[i]
                
                # Uncertainty metrics
                entropy = calculate_entropy(prob)
                gap = calculate_top2_gap(prob)
                max_conf = np.max(prob)
                
                # Top K
                top_k_idx = np.argsort(prob)[-top_k:][::-1]
                
                predictions = []
                for idx in top_k_idx:
                    predictions.append({
                        "disease": self.label_mapping.get(idx, "Unknown"),
                        "probability": float(prob[idx])
                    })
                    
                results.append({
                    "text": texts[i],
                    "predictions": predictions,
                    "uncertainty": {
                        "entropy": float(entropy),
                        "top2_gap": float(gap),
                        "max_confidence": float(max_conf)
                    },
                    "flag_unreliable": bool(entropy > 0.7 or gap < 0.2) # Example threshold
                })
        else:
            # Fallback if no probabilities (e.g. some SVMs)
            preds = self.model.predict(X)
            for i in range(len(texts)):
                idx = int(preds[i])
                results.append({
                    "text": texts[i],
                    "predictions": [{
                        "disease": self.label_mapping.get(idx, "Unknown"),
                        "probability": 1.0 # Placeholder
                    }],
                    "uncertainty": None,
                    "flag_unreliable": False
                })
                
        return results
