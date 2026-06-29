import json
import logging
from pathlib import Path
import re
from datetime import datetime

# Configure logging
log_dir = Path("logs/safety")
log_dir.mkdir(parents=True, exist_ok=True)
safety_logger = logging.getLogger("emergency_safety")
safety_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir / "emergency_overrides.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not safety_logger.handlers:
    safety_logger.addHandler(file_handler)

STROKE_SYMPTOMS = [
    "slurred speech",
    "facial drooping",
    "face drooping",
    "facial weakness",
    "one-sided weakness",
    "arm weakness",
    "leg weakness",
    "sudden confusion",
    "stroke symptoms",
    "difficulty speaking",
    "speech difficulty",
    "loss of speech",
    "numbness on one side",
    "sudden loss of balance",
    "sudden dizziness",
    "one sided weakness",
    "loss of balance",
    "sudden numbness",
    "confusion",
    "cannot speak"
]

RESPIRATORY_EMERGENCY = [
    "coughing up blood",
    "coughing blood",
    "difficulty breathing",
    "severe chest pain",
    "blue lips",
    "shortness of breath",
    "cant breathe",
    "can't breathe",
    "severe breathing difficulty",
    "cannot breathe"
]

SEVERE_INFECTION = [
    "stiff neck",
    "extreme confusion",
    "very high fever",
    "highest fever ever",
    "loss of consciousness"
]

HEMORRHAGIC_WARNING = [
    "bleeding gums",
    "blood in vomit",
    "severe abdominal pain",
    "uncontrolled bleeding",
    "severe bleeding"
]

CARDIAC_EMERGENCY = [
    "crushing chest pain",
    "pain radiating to arm",
    "irregular heartbeat",
    "heart attack",
    "chest pain"
]

NEUROLOGICAL_EMERGENCY = [
    "seizure",
    "unconscious",
    "loss of consciousness",
    "stroke symptoms"
]

ANAPHYLAXIS = [
    "throat swelling",
    "allergic reaction with breathing difficulty"
]

EMERGENCY_CATEGORIES = {
    "Neurological Emergency": STROKE_SYMPTOMS + NEUROLOGICAL_EMERGENCY,
    "Respiratory Emergency": RESPIRATORY_EMERGENCY,
    "Severe Infection": SEVERE_INFECTION,
    "Hemorrhagic Warning": HEMORRHAGIC_WARNING,
    "Cardiac Emergency": CARDIAC_EMERGENCY,
    "Anaphylaxis": ANAPHYLAXIS
}

class EmergencyRuleEngine:
    def __init__(self):
        pass

    def check_for_emergency(self, user_input: str) -> dict:
        user_input_lower = user_input.lower()
        
        detected_categories = []
        triggered_symptoms = []

        for category, symptoms in EMERGENCY_CATEGORIES.items():
            for symptom in symptoms:
                # Use regex for whole word or substring match
                if symptom in user_input_lower:
                    detected_categories.append(category)
                    triggered_symptoms.append(symptom)
                    break # Record category once

        if detected_categories:
            response = {
                "is_emergency": True,
                "emergency_detected": True,
                "categories": detected_categories,
                "triggered_symptoms": triggered_symptoms,
                "severity": "critical",
                "escalation_message": "⚠️ Some symptoms may indicate a serious medical condition requiring urgent professional evaluation. Please seek immediate medical attention."
            }
            self._log_emergency(user_input, response)
            return response

        return {"is_emergency": False, "emergency_detected": False}

    def _log_emergency(self, user_input: str, response: dict):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": user_input,
            "detected_categories": response["categories"],
            "triggered_symptoms": response["triggered_symptoms"],
            "escalation": response["escalation_message"]
        }
        safety_logger.info(f"EMERGENCY OVERRIDE TRIGGERED: {json.dumps(log_entry)}")
        
        # Also log in the format requested by the user
        print(f"\n[SAFETY OVERRIDE]")
        print(f"Detected Category: {', '.join(response['categories'])}")
        print(f"Triggered Override: TRUE")
        print(f"Escalation: Immediate Medical Attention")
