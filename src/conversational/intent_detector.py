
import re

class IntentDetector:
    """Robust intent detector using regex word boundaries to prevent false positives."""
    
    def detect_intent(self, text):
        text = text.lower()
        
        # 1. Check for Emergency Keywords (High priority)
        # We also have the EmergencyRuleEngine in the manager, but this is a quick filter
        emergency_pattern = r'\b(chest pain|cannot breathe|can\'t breathe|stroke|seizure|unconscious|heavy bleeding)\b'
        if re.search(emergency_pattern, text):
            return "EMERGENCY"
            
        # 2. Check for Greetings (Using word boundaries to avoid matching "hi" in "thick", "chills", etc.)
        greeting_pattern = r'\b(hello|hi|hey|greetings|good morning|good evening)\b'
        if re.search(greeting_pattern, text):
            return "GREETING"
            
        # 3. Check for Follow-up Answers
        followup_pattern = r'\b(yes|no|maybe|none|not really)\b'
        if re.search(followup_pattern, text):
            return "FOLLOWUP_ANSWER"
            
        # Default to Symptom Input
        return "SYMPTOM_INPUT"
