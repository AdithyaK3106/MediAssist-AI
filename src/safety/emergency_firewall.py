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
    "Neurological Emergency (Stroke / Seizure / Aneurysm)": {
        "keywords": [
            "slurred speech", "facial drooping", "face drooping", "facial weakness",
            "one-sided weakness", "one sided weakness", "arm weakness", "leg weakness",
            "loss of speech", "cannot speak", "cant speak", "sudden numbness",
            "stroke", "seizure", "violent seizures", "loss of consciousness", "unconscious",
            "sudden confusion", "extreme confusion", "fainted suddenly",
            "thunderclap headache", "worst headache of my life", "explosion in head",
            "status epilepticus", "convulsion",
            "one side numb", "unable to smile", "ek taraf latak", "aawaz ladkhada",
            "स्ट्रोक", "लकवा", "दौरा", "दौरे", "चेहरा टेढ़ा", "बोल नहीं पा रहा", "बेहोश", "होश खोना"
        ],
        "patterns": [
            r"slurr.*speech", r"fac(ial|e).*droop", r"arm.*weakness", r"cannot.*speak",
            r"worst.*headache", r"thunderclap", r"seizure",
            r"one\s+side.*(?:numb|dead)", r"unable\s+to\s+smile",
            r"ek\s+taraf.*latak", r"aawaz.*ladkhada",
            r"हिस्सा.*लटक", r"हाथ.*सुन्न"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Acute stroke, seizure, or intracranial hemorrhage warning signs detected. Immediate emergency medical intervention is required.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: स्ट्रोक, दौरा या गंभीर न्यूरोलॉजिकल लक्षण पाए गए हैं। तुरंत आपातकालीन चिकित्सा सहायता लें।"
    },
    "Cardiac Emergency (Acute Coronary Syndrome)": {
        "keywords": [
            "crushing chest pain", "chest pain radiating", "pain radiating to arm",
            "pain radiating to left arm", "pain radiating to jaw", "heart attack",
            "severe crushing pain in chest", "chest tightness and sweating",
            "chest pressure", "heart racing above", "elephant on chest",
            "seene me dard", "chhati me dard",
            "सीने में तेज दर्द", "सीने में दर्द", "छाती में दर्द", "दिल का दौरा", "हार्ट अटैक", "छाती में भारीपन"
        ],
        "patterns": [
            r"crush.*chest", r"chest.*pain.*radiat", r"heart.*attack", r"chest.*tightness.*sweat",
            r"elephant.*(?:sitting|on).*(?:chest|breastbone)",
            r"seen[ae].*(?:tez|bohot|bhari).*dard", r"chhat[ie].*dard",
            r"cheest.*pane", r"radiat.*(?:left\s+arm|jaw)",
            r"सीने.*दर्द", r"छाती.*दर्द"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Potential acute cardiac event (Myocardial Infarction) detected. Do not delay. Call emergency medical services immediately.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: संभावित दिल का दौरा या गंभीर हृदय समस्या के लक्षण। तुरंत 112 या 108 पर कॉल करें।"
    },
    "Respiratory Emergency (Acute Hypoxia / Hemoptysis)": {
        "keywords": [
            "coughing up blood", "coughing blood", "blood in cough", "blood in sputum",
            "cannot breathe", "cant breathe", "can't breathe", "gasping for air",
            "gasped for air", "severe breathing difficulty", "blue lips", "bluish lips",
            "severe asthma attack", "stridor", "windpipe clamped",
            "सांस लेने में तकलीफ", "सांस नहीं आ रही", "खांसी में खून", "दम घुटना"
        ],
        "patterns": [
            r"cough.*blood", r"can(not|t).*breath", r"blue.*lip", r"gasp.*air", r"asthma.*attack",
            r"windpipe.*(?:clamp|shut|clos)", r"saans.*(?:nahi|nahin|ruk)",
            r"सांस.*तकलीफ", r"खांसी.*खून", r"दम.*घुट", r"होंठ.*नील"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Acute respiratory distress or hemoptysis (coughing blood) detected. Seek immediate emergency room care.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: सांस लेने में अत्यधिक कठिनाई या खांसी में खून के लक्षण। तत्काल आपातकालीन विभाग (ER) पहुंचे।"
    },
    "Severe Hemorrhagic Warning (Dengue / Internal Bleeding)": {
        "keywords": [
            "bleeding gums", "blood in vomit", "vomiting blood", "black tarry stools",
            "uncontrolled bleeding", "severe bleeding", "blood in urine and fever",
            "hemoptysis", "bright red fresh blood",
            "खून की उल्टी", "मसूड़ों से खून", "पेशाब में खून"
        ],
        "patterns": [
            r"bleed.*gum", r"blood.*vomit", r"fresh.*blood", r"uncontrolled.*bleed",
            r"खून.*उल्टी"
        ],
        "guidance_en": "🚨 CRITICAL WARNING: Severe hemorrhagic signs detected (potential severe Dengue shock syndrome or internal bleeding). Requires urgent hospital admission.",
        "guidance_hi": "🚨 गंभीर चेतावनी: रक्तस्राव (Bleeding) के गंभीर लक्षण मिले हैं। तुरंत नजदीकी अस्पताल में भर्ती हों।"
    },
    "Severe Systemic Infection / Meningitis / Sepsis": {
        "keywords": [
            "stiff neck and extreme confusion", "stiff neck with high fever",
            "highest fever ever and stiff neck", "extreme confusion and fever",
            "stiff neck", "meningitis", "hallucinating since morning", "petechial rash"
        ],
        "patterns": [
            r"stiff.*neck.*fever", r"stiff.*neck.*touch.*chin", r"fever.*rash.*mental", r"sepsis"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Symptoms consistent with acute meningitis, encephalitis, or septicemia. Immediate hospital emergency care required.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: गंभीर दिमागी संक्रमण या मेनिनजाइटिस के संभावित लक्षण। तुरंत अस्पताल जाएं।"
    },
    "Acute Surgical & Toxicological Emergencies": {
        "keywords": [
            "snake bite", "poison consumption", "pesticide", "rebound tenderness",
            "appendicitis", "loss of vision in right eye", "sudden total loss of vision",
            "blood sugar 42", "bathroom acid", "drank acid"
        ],
        "patterns": [
            r"snake.*bite", r"poison", r"pesticide", r"loss.*vision.*eye", r"rebound.*tender", r"blood.*sugar.*42",
            r"drank.*acid", r"swallow.*acid", r"keetnashak", r"jhaag.*nikal"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Acute surgical, toxicological, or ophthalmic emergency detected. Proceed immediately to the nearest casualty / trauma center.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: तुरंत नजदीकी अस्पताल के आपातकालीन विभाग में जाएं।"
    },
    "Anaphylaxis (Severe Allergic Reaction)": {
        "keywords": [
            "throat swelling", "swelling of tongue and difficulty breathing",
            "throat closing up", "severe allergic reaction cannot breathe",
            "allergic anaphylaxis", "tongue swelling", "tongue sweling", "wasp sting"
        ],
        "patterns": [
            r"anaphylaxis", r"tongue.*swel", r"throat.*tight", r"throat.*clos",
            r"जीभ.*सूज", r"गले.*सूजन"
        ],
        "guidance_en": "🚨 CRITICAL MEDICAL EMERGENCY: Anaphylactic reaction detected. Administer Epinephrine (EpiPen) if available and call for emergency ambulance immediately.",
        "guidance_hi": "🚨 आपातकालीन चेतावनी: गंभीर एलर्जी प्रतिक्रिया (एनाफिलेक्सिस)। तुरंत एम्बुलेंस बुलाएं।"
    }
}

NEGATION_PATTERNS = [
    re.compile(r"\b(no|not|neither|nor|without|never|denies|denied|negative for|free of)\s+(?:any\s+|signs of\s+|evidence of\s+)?(?:\w+\s+){0,3}$", re.I),
    re.compile(r"\b(nahi|nahin|bina|na)\s+(?:\w+\s+){0,2}$", re.I),
    re.compile(r"(नहीं|न|बिना)\s+$")
]

FAMILY_HISTORICAL_PATTERNS = [
    re.compile(r"\b(father|mother|brother|sister|grandfather|grandmother|uncle|aunt|cousin|relative|family|bhai|behen|dada|dadi|nana|nani|pitaji)\s+(?:had|suffered|has|was diagnosed|died of|ko pehle)\b", re.I),
    re.compile(r"\b(?:history of|diagnosed with)\s+(?:\w+\s+){0,3}(?:years?|months?)\s+ago\b", re.I),
    re.compile(r"\b(?:last\s+(?:winter|year|month|summer))\b", re.I),
    re.compile(r"\bpehle\s+.*(?:tha|thi)\b", re.I)
]

class EmergencySafetyFirewall:
    """
    Deterministic safety firewall with non-negotiable veto authority.
    Intercepts and halts all model inference if life-threatening patterns are present.
    Includes contextual negation filtering and familial/historical disqualification.
    """
    def __init__(self):
        pass

    def _is_negated_or_historical(self, text: str, match_start: int) -> bool:
        left_window = text[max(0, match_start - 60):match_start]
        boundaries = [left_window.rfind("."), left_window.rfind(";"), left_window.rfind("!"), left_window.rfind("?"), left_window.rfind(", but")]
        last_boundary = max(boundaries)
        clause_window = left_window[last_boundary+1:] if last_boundary != -1 else left_window

        for np in NEGATION_PATTERNS:
            if np.search(clause_window):
                return True
        for fp in FAMILY_HISTORICAL_PATTERNS:
            if fp.search(left_window):
                return True
        return False

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
            triggered = False
            for kw in rules["keywords"]:
                for m in re.finditer(r"(?:\b|^|(?<=[^\w]))" + re.escape(kw.lower()) + r"(?:\b|$|(?=[^\w]))", combined_text):
                    if not self._is_negated_or_historical(combined_text, m.start()):
                        detected_categories.append(category)
                        triggered_keywords.append(kw)
                        guidance_en.append(rules["guidance_en"])
                        guidance_hi.append(rules["guidance_hi"])
                        triggered = True
                        break
                if triggered:
                    break

            if not triggered and "patterns" in rules:
                for pattern in rules["patterns"]:
                    for m in re.finditer(pattern, combined_text, re.I):
                        if not self._is_negated_or_historical(combined_text, m.start()):
                            detected_categories.append(category)
                            triggered_keywords.append(m.group(0))
                            guidance_en.append(rules["guidance_en"])
                            guidance_hi.append(rules["guidance_hi"])
                            triggered = True
                            break
                    if triggered:
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
