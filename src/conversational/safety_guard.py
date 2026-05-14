class SafetyGuard:
    def check_response(self, response):
        disclaimer = "Disclaimer: This is not a clinical diagnosis. Please consult a doctor."
        if disclaimer not in response:
            response += "\n\n" + disclaimer
        return response
        
    def detect_emergency(self, text):
        emergency_keywords = ["chest pain", "shortness of breath", "unconscious", "seizure"]
        return any(k in text.lower() for k in emergency_keywords)
