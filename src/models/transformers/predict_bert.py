import json
import os
from pathlib import Path
import numpy as np
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "transformers" / "pubmedbert"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"

def calculate_entropy(probs):
    probs = np.array(probs)
    return float(-np.sum(probs * np.log(probs + 1e-12)))

def calculate_top2_gap(probs):
    sorted_p = np.sort(probs)
    return float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0

def get_safe_device():
    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        return "cpu"
    try:
        t = torch.tensor([1.0], device="cuda")
        _ = t + 1.0
        return "cuda"
    except Exception:
        return "cpu"

class MedicalBERTInferencePipeline:
    """
    Inference pipeline for fine-tuned Medical BERT (PubMedBERT).
    Produces Top-K condition predictions with uncertainty and entropy estimation.
    """
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR
        self.device = get_safe_device()
        
        # Load label mapping
        mapping_path = DATA_DIR / "label_mapping.json"
        if not mapping_path.exists():
            mapping_path = PROJECT_ROOT / "data" / "processed" / "ml" / "label_mapping.json"
            
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        weights_exist = (self.model_path / "model.safetensors").exists() or (self.model_path / "pytorch_model.bin").exists()
        if TORCH_AVAILABLE and self.model_path.exists() and (self.model_path / "config.json").exists() and weights_exist:
            print(f"[PubMedBERT] Loading fine-tuned model from {self.model_path} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path)).to(self.device)
            self.model.eval()
        else:
            print(f"[PubMedBERT] Checkpoint not found at {self.model_path}. Running with lightweight local calibrated fallback.")
            self.model = None

    def predict(self, texts, top_k=3):
        """
        Predict Top-K diseases and calibrated confidence.
        
        Args:
            texts (list[str]): List of symptom strings.
            top_k (int): Number of top predictions to return.
            
        Returns:
            list[dict]: Output predictions per text sample.
        """
        results = []
        
        if self.model is not None and self.tokenizer is not None:
            # Transformer model inference
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
            for i, text in enumerate(texts):
                prob = probs[i]
                entropy = calculate_entropy(prob)
                gap = calculate_top2_gap(prob)
                max_conf = float(np.max(prob))
                
                top_indices = np.argsort(prob)[-top_k:][::-1]
                top_preds = [
                    {
                        "disease": self.label_mapping.get(int(idx), "Unknown"),
                        "probability": float(prob[idx])
                    }
                    for idx in top_indices
                ]
                
                results.append({
                    "text": text,
                    "predictions": top_preds,
                    "uncertainty": {
                        "entropy": entropy,
                        "top2_gap": gap,
                        "max_confidence": max_conf
                    },
                    "flag_unreliable": bool(max_conf < 0.30 or entropy > 2.5)
                })
        else:
            # Fallback to calibrated classical ML model if transformer weights are not yet fine-tuned
            from src.models.classical_ml.predict import ClassicalInferencePipeline
            classical_pipe = ClassicalInferencePipeline(model_name="logistic_regression")
            results = classical_pipe.predict(texts, top_k=top_k)

        return results

if __name__ == "__main__":
    pipeline = MedicalBERTInferencePipeline()
    sample_text = "I have a high fever, severe headache, and joint pain for 3 days."
    res = pipeline.predict([sample_text])
    print("\nInference Output:")
    print(json.dumps(res, indent=2))
