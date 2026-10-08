import json
import re

COMMON_SYMPTOM_MAP = {
    "fever": ["fever", "bukhar", "high temp", "temperature", "chills", "shivering", "thand", "बुखार", "ताप", "कपकपी"],
    "headache": ["headache", "sar dard", "head pain", "migraine", "heavy head", "सिर दर्द", "सर दर्द"],
    "cough": ["cough", "khansi", "coughing", "dry cough", "phlegm", "mucus", "खांसी", "कफ"],
    "body pain": ["body pain", "badan dard", "body ache", "muscle ache", "joint pain", "joint ache", "weakness", "बदन दर्द", "जोड़ों में दर्द", "कमजोरी"],
    "sore throat": ["sore throat", "gale me dard", "throat pain", "throat irritation", "गले में दर्द", "गले में खराश"],
    "vomiting": ["vomiting", "ulti", "nausea", "puking", "queasy", "उल्टी", "मतली"],
    "diarrhea": ["diarrhea", "loose motion", "dast", "watery stools", "stomach upset", "दस्त"],
    "stomach pain": ["stomach pain", "pet dard", "abdominal pain", "cramps", "पेट दर्द"],
    "rash": ["rash", "itching", "khujli", "red spots", "skin redness", "चकत्ते", "खुजली", "दाने"],
    "shortness of breath": ["shortness of breath", "breathless", "sans lene me takleef", "wheezing", "सांस लेने में तकलीफ"]
}

class SymptomTimelineExtractor:
    """
    Extracts structured symptoms and timeline attributes (onset, duration, severity, trend).
    Operates with LLM or a deterministic heuristic/regex engine for standalone execution.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract(self, text: str) -> dict:
        """
        Extract structured symptoms and timeline from user text.
        """
        if self.llm_client and self.llm_client.is_available():
            try:
                return self._extract_via_llm(text)
            except Exception as e:
                print(f"[Extractor] LLM extraction failed: {e}. Falling back to deterministic extractor.")
                
        return self._extract_heuristic(text)

    def _extract_via_llm(self, text: str) -> dict:
        """
        Uses Qwen 2.5 7B via Ollama's native JSON mode to extract symptoms & timeline.
        """
        system_prompt = (
            "You are a clinical entity and timeline extraction assistant for healthcare triage. "
            "Analyze patient inquiries in English, Hindi, or Hinglish. "
            "Extract all reported symptoms normalized to standard clinical English terms. "
            "Extract timeline attributes carefully. "
            "Return ONLY a valid JSON object matching the requested schema."
        )
        prompt = f"""Extract symptoms, timeline attributes, and detected language from the patient message.
Schema:
{{
  "symptoms": ["list of symptoms in standard medical English"],
  "timeline": {{
    "onset": "<e.g., '3 days ago' or 'Not specified'>",
    "duration_days": <number of days as float/int, or null if unknown>,
    "severity": "mild" | "moderate" | "severe",
    "progression": "improving" | "stable" | "worsening",
    "frequency": "constant" | "intermittent"
  }},
  "language": "en" | "hi"
}}

Patient Message: "{text}"
"""
        extracted = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt)
        
        # Validate schema fields
        symptoms = extracted.get("symptoms", [])
        if not symptoms or not isinstance(symptoms, list):
            symptoms = ["unspecified malaise"]
        
        raw_timeline = extracted.get("timeline", {})
        if not isinstance(raw_timeline, dict):
            raw_timeline = {}
            
        timeline = {
            "onset": str(raw_timeline.get("onset", "Not specified")),
            "duration_days": raw_timeline.get("duration_days"),
            "severity": raw_timeline.get("severity", "moderate") if raw_timeline.get("severity") in ["mild", "moderate", "severe"] else "moderate",
            "progression": raw_timeline.get("progression", "stable") if raw_timeline.get("progression") in ["improving", "stable", "worsening"] else "stable",
            "frequency": raw_timeline.get("frequency", "constant") if raw_timeline.get("frequency") in ["constant", "intermittent"] else "constant"
        }
        
        lang = extracted.get("language", "en")
        if lang not in ["en", "hi"]:
            lang = "en"
            
        return {
            "symptoms": list(dict.fromkeys(symptoms)),
            "timeline": timeline,
            "language": lang
        }

    def _extract_heuristic(self, text: str) -> dict:
        text_lower = text.lower()
        
        # 1. Detect language (Hindi vs English)
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        hindi_tokens = ["bukhar", "dard", "khansi", "ulti", "thand", "pet", "sar", "din", "se", "hai"]
        is_hindi = has_devanagari or any(token in text_lower.split() for token in hindi_tokens)
        language = "hi" if is_hindi else "en"

        # 2. Extract symptoms
        detected_symptoms = []
        for canonical, variants in COMMON_SYMPTOM_MAP.items():
            for v in variants:
                if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                    detected_symptoms.append(canonical)
                    break
        
        if not detected_symptoms:
            detected_symptoms = ["unspecified malaise"]

        # 3. Extract duration in days (English, Hinglish, and Devanagari)
        duration_days = None
        duration_match = re.search(r'([0-9०-९]+)\s*(day|days|din|दिन)', text_lower)
        if duration_match:
            val_str = duration_match.group(1)
            # Map devanagari numerals if present
            dev_digits = str.maketrans("०१२३४५६७८९", "0123456789")
            duration_days = int(val_str.translate(dev_digits))
        elif any(w in text_lower for w in ["week", "hafte", "हफ्ते", "सप्ताह"]):
            duration_days = 7
        elif any(w in text_lower for w in ["yesterday", "kal se", "कल से"]):
            duration_days = 1
        elif any(w in text_lower for w in ["today", "aaj", "आज"]):
            duration_days = 0.5

        # 4. Extract severity
        severity = "moderate"
        if any(w in text_lower for w in ["mild", "slight", "halka", "thoda", "हल्का", "थोड़ा", "little bit"]):
            severity = "mild"
        elif any(w in text_lower for w in ["severe", "extreme", "unbearable", "bahut", "very high", "तेज", "गंभीर", "असहनीय", "intense"]):
            severity = "severe"

        # 5. Extract progression / trend
        progression = "stable"
        if any(w in text_lower for w in ["worse", "worsening", "increasing", "badhta", "badh raha", "बढ़ता", "बढ़ रहा", "getting worse"]):
            progression = "worsening"
        elif any(w in text_lower for w in ["better", "improving", "decreasing", "kam ho raha", "कम हो रहा"]):
            progression = "improving"

        # 6. Frequency
        frequency = "constant"
        if any(w in text_lower for w in ["intermittent", "comes and goes", "sometimes", "kabhi kabhi", "periodic"]):
            frequency = "intermittent"

        timeline = {
            "onset": f"{duration_days} days ago" if duration_days else "Not specified",
            "duration_days": duration_days,
            "severity": severity,
            "progression": progression,
            "frequency": frequency
        }

        return {
            "symptoms": list(dict.fromkeys(detected_symptoms)),
            "timeline": timeline,
            "language": language
        }

    def generate_followup(self, profile: dict) -> dict:
        """
        Determines if timeline details are incomplete and formulates targeted questions.
        """
        timeline = profile.get("timeline", {})
        language = profile.get("language", "en")
        symptoms = profile.get("symptoms", [])

        needs_followup = False
        questions = []

        if timeline.get("duration_days") is None:
            needs_followup = True
            if language == "hi":
                questions.append("यह लक्षण आपको कितने दिनों से महसूस हो रहे हैं?")
            else:
                questions.append("How many days have you been experiencing these symptoms?")

        if timeline.get("progression") == "stable" and len(symptoms) <= 2:
            needs_followup = True
            if language == "hi":
                questions.append("क्या आपकी तकलीफ समय के साथ बढ़ रही है, कम हो रही है, या वैसी ही है?")
            else:
                questions.append("Is your condition worsening, improving, or remaining about the same?")

        return {
            "needs_followup": needs_followup,
            "questions": questions[:2]
        }
