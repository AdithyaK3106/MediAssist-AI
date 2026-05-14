import json
import os
from pathlib import Path

class HomeCareRecommender:
    """Safe, non-clinical supportive wellness recommendation engine."""
    
    def __init__(self):
        self.remedies = {}
        self.generic_fallback = {
            "home_care": [
                "Stay hydrated",
                "Get adequate rest",
                "Monitor symptoms carefully",
                "Maintain proper hygiene"
            ],
            "seek_medical_attention_if": [
                "Symptoms worsen significantly",
                "Fever exceeds 102°F (38.9°C)",
                "Breathing difficulty develops",
                "You experience severe pain or confusion"
            ],
            "severity": "medium"
        }
        
        self._load_data()
        
    def _load_data(self):
        try:
            project_root = Path(__file__).resolve().parents[2]
            json_path = project_root / "data" / "home_remedies.json"
            
            with open(json_path, 'r', encoding='utf-8') as f:
                self.remedies = {k.lower(): v for k, v in json.load(f).items()}
        except Exception as e:
            # Silently fallback to empty remedies if file is missing
            self.remedies = {}

    def is_critical(self, disease: str) -> bool:
        """Check if the disease requires emergency escalation based on severity or keywords."""
        disease_lower = disease.lower()
        if disease_lower in self.remedies:
            if self.remedies[disease_lower].get("severity") == "critical":
                return True
        # Keyword fallback
        critical_keywords = ["stroke", "heart attack", "severe dengue", "heart failure", "sepsis", "pneumonia", "tuberculosis", "kidney failure"]
        return any(keyword in disease_lower for keyword in critical_keywords)

    def get_home_care_data(self, disease: str) -> dict:
        """Return safe, disease-specific remedies or a fallback."""
        data = self.remedies.get(disease.lower(), self.generic_fallback)
        
        # If the disease is critical, remove home_care and emphasize medical attention
        if self.is_critical(disease):
            return {
                "home_care": [],
                "seek_medical_attention_if": ["Immediately seek emergency medical care"],
                "severity": "critical"
            }
            
        return data
