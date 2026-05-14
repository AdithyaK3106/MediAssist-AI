import re

class IntentDetector:
    def detect_intent(self, text):
        text = text.lower()
        
        # Check emergency first
        if any(e in text for e in ["chest pain", "cannot breathe", "stroke"]):
            return "EMERGENCY"
            
        # Word boundaries prevent "hi" matching in "chills"
        if re.search(r'\b(hello|hi|hey)\b', text):
            return "GREETING"
            
        if re.search(r'\b(yes|no|maybe)\b', text):
            return "FOLLOWUP_ANSWER"
            
        return "SYMPTOM_INPUT"
