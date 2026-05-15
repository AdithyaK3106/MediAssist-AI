
import logging
from pathlib import Path

class SafetyGuard:
    """Enhanced safety layer to filter low-confidence or medically risky predictions."""
    
    def __init__(self, confidence_threshold=0.15):
        self.confidence_threshold = confidence_threshold
        
    def check_response(self, response, top_predictions=None, uncertainty=None):
        """Adds disclaimer and applies confidence-based filtering."""
        
        # 1. Handle Low Confidence Suppression
        if top_predictions and uncertainty:
            max_prob = top_predictions[0].get('probability', 0)
            entropy = uncertainty.get('entropy', 0)
            
            # If the model is extremely guessing (very low max prob or very high entropy)
            if max_prob < self.confidence_threshold or entropy > 4.0:
                return "I am unable to provide a reliable health suggestion based on the symptoms described. The information is too vague or ambiguous. Please provide more details or consult a doctor."

        # 2. Add standard disclaimer
        disclaimer = "⚠️ **Medical Disclaimer**: This system provides statistical suggestions, NOT a clinical diagnosis. Please consult a qualified healthcare professional immediately for medical concerns."
        
        if disclaimer not in response:
            response += "\n\n" + disclaimer
            
        return response
        
    def detect_emergency(self, text):
        """Basic keyword emergency detection (backup for EmergencyRuleEngine)."""
        emergency_keywords = ["unconscious", "seizure", "heavy bleeding", "poisoning"]
        return any(k in text.lower() for k in emergency_keywords)
