import json
import logging
from pathlib import Path
import re
from datetime import datetime

# Configure safety audit logging
log_dir = Path("logs/safety")
log_dir.mkdir(parents=True, exist_ok=True)
safety_logger = logging.getLogger("emergency_firewall")
safety_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir / "emergency_firewall_audit.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
if not safety_logger.handlers:
    safety_logger.addHandler(file_handler)

EMERGENCY_RULES = {
    "Neurological Emergency (Stroke / Seizure)": {
        "keywords": [
            "slurred speech", "facial drooping", "face drooping", "facial weakness",
            "one-sided weakness", "one sided weakness", "arm weakness", "leg weakness",
            "loss of speech", "cannot speak", "cant speak", "sudden numbness",
            "stroke", "seizure", "violent seizures", "loss of consciousness", "unconscious",
            "sudden confusion", "extreme confusion", "fainted suddenly"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Stroke or acute neurological symptoms detected. Immediate emergency medical intervention is required.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: स्ट्रोक या गंभीर न्यूरोलॉजिकल लक्षण पाए गए हैं। तुरंत आपातकालीन चिकित्सा सहायता लें।"
    },
    "Cardiac Emergency (Acute Coronary Syndrome)": {
        "keywords": [
            "crushing chest pain", "chest pain radiating", "pain radiating to arm",
            "pain radiating to left arm", "pain radiating to jaw", "heart attack",
            "severe crushing pain in chest", "chest tightness and sweating"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Potential acute cardiac event detected. Do not delay. Call emergency medical services immediately.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: संभावित दिल का दौरा या गंभीर हृदय समस्या के लक्षण। तुरंत 112 या 108 पर कॉल करें।"
    },
    "Respiratory Emergency (Acute Hypoxia / Hemoptysis)": {
        "keywords": [
            "coughing up blood", "coughing blood", "blood in cough", "blood in sputum",
            "cannot breathe", "cant breathe", "can't breathe", "gasping for air",
            "severe breathing difficulty", "blue lips", "bluish lips", "severe asthma attack"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Acute respiratory distress or coughing up blood detected. Seek immediate emergency room care.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: सांस लेने में अत्यधिक कठिनाई या खांसी में खून के लक्षण। तत्काल आपातकालीन विभाग (ER) पहुंचे।"
    },
    "Severe Hemorrhagic Warning (Dengue / Internal Bleeding)": {
        "keywords": [
            "bleeding gums", "blood in vomit", "vomiting blood", "black tarry stools",
            "uncontrolled bleeding", "severe bleeding", "blood in urine and fever"
        ],
        "guidance_en": "🚨 CRITICAL WARNING: Severe hemorrhagic signs detected (potential severe Dengue or internal bleeding). Requires urgent hospital admission.",
        "guidance_hi": "🚨 गंभीर चेतावनी: रक्तस्राव (Bleeding) के गंभीर लक्षण मिले हैं। तुरंत नजदीकी अस्पताल में भर्ती हों।"
    },
    "Severe Systemic Infection / Meningitis": {
        "keywords": [
            "stiff neck and extreme confusion", "stiff neck with high fever",
            "highest fever ever and stiff neck", "extreme confusion and fever"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Symptoms consistent with severe acute central nervous system infection (e.g., Meningitis). Immediate hospital care required.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: गंभीर दिमागी संक्रमण या मेनिनजाइटिस के संभावित लक्षण। तुरंत अस्पताल जाएं।"
    },
    "Anaphylaxis (Severe Allergic Reaction)": {
        "keywords": [
            "throat swelling", "swelling of tongue and difficulty breathing",
            "throat closing up", "severe allergic reaction cannot breathe"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Anaphylactic reaction detected. Administer Epinephrine (EpiPen) if available and call for emergency ambulance immediately.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: गंभीर एलर्जी प्रतिक्रिया (एनाफिलेक्सिस)। तुरंत एम्बुलेंस बुलाएं।"
    }
}

class EmergencySafetyFirewall:
    """
    Deterministic safety firewall with non-negotiable veto authority.
    Intercepts and halts all model inference if life-threatening patterns are present.
    """
    def __init__(self):
        pass

    def evaluate(self, raw_text: str, structured_profile: dict = None) -> dict:
        combined_text = raw_text.lower()
        if structured_profile:
            symptoms = structured_profile.get("symptoms", [])
            if isinstance(symptoms, list):
                combined_text += " " + " ".join([str(s).lower() for s in symptoms])
            timeline = structured_profile.get("timeline", {})
            if isinstance(timeline, dict):
                combined_text += " " + " ".join([str(v).lower() for v in timeline.values()])

        detected_categories = []
        triggered_keywords = []
        guidance_en = []
        guidance_hi = []

        for category, rules in EMERGENCY_RULES.items():
            for kw in rules["keywords"]:
                # Check for keyword in combined text
                if re.search(r'\b' + re.escape(kw) + r'\b', combined_text):
                    detected_categories.append(category)
                    triggered_keywords.append(kw)
                    guidance_en.append(rules["guidance_en"])
                    guidance_hi.append(rules["guidance_hi"])
                    break

        if detected_categories:
            response = {
                "is_emergency": True,
                "categories": detected_categories,
                "triggered_keywords": triggered_keywords,
                "guidance_en": guidance_en[0],
                "guidance_hi": guidance_hi[0],
                "contacts": {
                    "national_emergency": "112",
                    "ambulance": "108",
                    "health_helpline": "1075"
                }
            }
            self._log_audit(raw_text, response)
            return response

        return {"is_emergency": False}

    def _log_audit(self, user_input: str, response: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "categories": response["categories"],
            "triggered_keywords": response["triggered_keywords"],
            "action": "VETO_HALT_INFERENCE_TRIGGER_EMERGENCY"
        }
        safety_logger.info(f"SAFETY OVERRIDE: {json.dumps(log_entry)}")
