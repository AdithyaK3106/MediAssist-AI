import json
from pathlib import Path
from src.recommendation.recommender import HospitalRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

class ClinicalDecisionLayer:
    """
    Evaluates BERT predictions alongside the symptom timeline.
    Determines urgency level, gates home care, and attaches hospital routing.
    """
    def __init__(self):
        self.hospital_recommender = HospitalRecommender()
        self.home_remedies = {}
        self.display_names = {}
        self._load_metadata()

    def _load_metadata(self):
        try:
            with open(DATA_DIR / "home_remedies.json", "r", encoding="utf-8") as f:
                self.home_remedies = json.load(f)
        except Exception:
            self.home_remedies = {}

        try:
            with open(DATA_DIR / "disease_display_names.json", "r", encoding="utf-8") as f:
                self.display_names = json.load(f)
        except Exception:
            self.display_names = {}

    def evaluate(self, bert_predictions: list, timeline: dict, user_lat=12.9716, user_lon=77.5946) -> dict:
        """
        Produce a safe, comprehensive Decision Bundle.
        """
        if not bert_predictions:
            return {
                "urgency_level": "Low",
                "action_timeframe": "No clinical conditions detected. Monitor well-being.",
                "allow_home_care": True,
                "home_care": ["Maintain adequate hydration and rest."],
                "red_flags": ["Visit a doctor if new or severe symptoms emerge."],
                "recommended_hospitals": []
            }

        top_pred = bert_predictions[0]
        top_disease = top_pred.get("disease", "Unknown")
        remedy_info = self.home_remedies.get(top_disease, {})
        base_severity = remedy_info.get("severity", "moderate")

        # 1. Compute Urgency Level based on condition risk and timeline
        duration_days = timeline.get("duration_days") or 1
        progression = timeline.get("progression", "stable")
        patient_severity = timeline.get("severity", "moderate")

        if base_severity == "high" or patient_severity == "severe" or (duration_days >= 7 and progression == "worsening"):
            urgency_level = "High"
            action_timeframe = "Consult a physician promptly today for clinical evaluation and diagnostics."
        elif duration_days >= 3 or patient_severity == "moderate" or progression == "worsening":
            urgency_level = "Moderate"
            action_timeframe = "Medical consultation advised within 24 to 48 hours if symptoms persist."
        else:
            urgency_level = "Low"
            action_timeframe = "Monitor symptoms closely for 48 hours; seek care if symptoms worsen."

        # 2. Home Care Safety Gating (Strictly suppressed for High Urgency)
        allow_home_care = (urgency_level != "High")
        if allow_home_care:
            home_care_points = remedy_info.get("home_care", [
                "Drink plenty of fluids (water, electrolytes, broths).",
                "Ensure sufficient bed rest.",
                "Avoid strenuous physical exertion."
            ])
        else:
            home_care_points = []  # Strictly gated

        # 3. Red-flag triggers ("When to seek immediate emergency care")
        red_flags = remedy_info.get("seek_medical_attention_if", [
            "Temperature rises above 102°F or persists >3 days.",
            "Difficulty breathing, chest tightness, or extreme dizziness develops.",
            "Inability to keep liquids down for more than 12 hours."
        ])

        # 4. Bangalore Hospital Proximity Routing
        hospitals = self.hospital_recommender.get_recommendations(
            user_lat=user_lat,
            user_lon=user_lon,
            disease_category=top_disease,
            top_k=3
        )

        # 5. Format Top-3 conditions with display names
        enriched_predictions = []
        for p in bert_predictions[:3]:
            d_name = p["disease"]
            meta = self.display_names.get(d_name, {"friendly_name": d_name, "description": ""})
            enriched_predictions.append({
                "condition": meta.get("friendly_name", d_name),
                "confidence": round(p.get("probability", 0.0) * 100, 1),
                "description": meta.get("description", "")
            })

        return {
            "top_conditions": enriched_predictions,
            "primary_condition": enriched_predictions[0]["condition"] if enriched_predictions else "Unknown",
            "urgency_level": urgency_level,
            "action_timeframe": action_timeframe,
            "allow_home_care": allow_home_care,
            "home_care": home_care_points,
            "red_flags": red_flags,
            "hospitals": hospitals
        }
