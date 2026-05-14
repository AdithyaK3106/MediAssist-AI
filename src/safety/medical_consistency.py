import json
import logging
from pathlib import Path
import re
from datetime import datetime

# Configure logging
log_dir = Path("logs/safety")
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("medical_consistency")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir / "medical_consistency.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)

class MedicalConsistencyValidator:
    def __init__(self):
        # 1. Load disease categories
        self.disease_categories = {}
        try:
            project_root = Path(__file__).resolve().parents[2]
            json_path = project_root / "data" / "disease_categories.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                self.disease_categories = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load disease_categories.json: {e}")

        # 2. Define symptom keywords per category
        self.symptom_keywords = {
            "respiratory": ["cough", "throat", "breath", "chest pain", "runny", "sneeze", "wheez", "mucus", "phlegm"],
            "eye": ["eye", "vision", "blur", "watery", "itchy", "red eye"],
            "gastrointestinal": ["vomit", "diarrhea", "stomach", "abdomin", "nausea", "stool", "constipat", "bowel", "belly"],
            "neurological": ["dizzi", "seizure", "confus", "conscious", "headache", "migraine", "faint", "numb", "paraly"],
            "fever": ["fever", "chill", "temperature", "sweat", "hot"],
            "cardiovascular": ["chest pain", "heart", "palpitat", "pulse", "blood pressure"],
            "metabolic": ["thirst", "urinat", "weight", "sugar", "hunger"],
            "skin": ["rash", "itch", "skin", "blister", "redness", "spot", "sore"],
            "urinary": ["urinat", "urine", "bladder", "kidney", "pelvi"],
            "musculoskeletal": ["joint", "muscle", "ache", "pain", "back", "bone", "stiff"],
            "systemic": ["fatigue", "tired", "weak", "exhaust", "malaise", "swollen"]
        }

    def _extract_symptom_categories(self, user_input: str):
        detected_categories = set()
        user_input_lower = user_input.lower()
        
        for category, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\w*\b', user_input_lower):
                    detected_categories.add(category)
                    break
        
        return list(detected_categories)

    def validate_and_rerank(self, user_input: str, predictions: list) -> dict:
        """
        Validates the predictions against user input and returns reranked predictions 
        along with explainability metadata.
        """
        detected_categories = self._extract_symptom_categories(user_input)
        
        if not detected_categories:
            # If no specific categories detected, we cannot penalize
            return {
                "reranked_predictions": predictions,
                "detected_symptoms": [],
                "reasoning": []
            }

        reasoning_log = []
        adjusted_predictions = []

        print(f"\n[MEDICAL CONSISTENCY LAYER] Input Categories Detected: {detected_categories}")

        for pred in predictions:
            disease = pred['disease']
            original_prob = pred['probability']
            disease_cats = self.disease_categories.get(disease, [])
            
            # Calculate overlap
            overlap = set(detected_categories).intersection(set(disease_cats))
            
            if not disease_cats:
                penalty = 0.9
                consistency = "UNKNOWN"
            elif overlap:
                penalty = 1.0
                consistency = "HIGH"
            else:
                penalty = 0.1 # Heavily penalize implausible outputs
                consistency = "LOW"
            
            adjusted_prob = original_prob * penalty
            
            adjusted_predictions.append({
                "disease": disease,
                "probability": adjusted_prob,
                "original_probability": original_prob,
                "disease_categories": disease_cats
            })
            
            reasoning_log.append({
                "disease": disease,
                "consistency": consistency,
                "penalty_applied": penalty,
                "disease_categories": disease_cats
            })
            
            print(f"  Prediction: {disease}")
            print(f"  Disease Categories: {disease_cats}")
            print(f"  Consistency: {consistency} | Penalty: {penalty}")

        # Rerank based on adjusted probability
        adjusted_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Log the explainability details
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": user_input,
            "detected_symptoms": detected_categories,
            "reranking_changes": [],
            "suppressed_predictions": [],
            "penalties_applied": []
        }
        
        # Compare original rank to new rank
        original_ranks = {pred['disease']: i for i, pred in enumerate(predictions)}
        new_ranks = {pred['disease']: i for i, pred in enumerate(adjusted_predictions)}
        
        for r in reasoning_log:
            disease = r['disease']
            orig_rank = original_ranks[disease]
            new_rank = new_ranks[disease]
            
            if r['penalty_applied'] < 1.0:
                log_entry["penalties_applied"].append({
                    "disease": disease,
                    "penalty": r['penalty_applied'],
                    "reason": "LOW" if r['consistency'] == "LOW" else "UNKNOWN"
                })
            
            if r['consistency'] == "LOW":
                log_entry["suppressed_predictions"].append(disease)
                
            if orig_rank != new_rank:
                log_entry["reranking_changes"].append({
                    "disease": disease,
                    "from_rank": orig_rank + 1,
                    "to_rank": new_rank + 1
                })
                
        logger.info(f"CONSISTENCY CHECK: {json.dumps(log_entry)}")
        
        return {
            "reranked_predictions": adjusted_predictions,
            "detected_symptoms": detected_categories,
            "reasoning": reasoning_log
        }
