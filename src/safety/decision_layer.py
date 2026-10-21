import json
from pathlib import Path
from src.recommendation.recommender import HospitalRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

BENIGN_OUTPATIENT_CONDITIONS = {
    "Tension Headache",
    "Eye Strain / Screen Fatigue",
    "Sinusitis",
    "Acidity & Indigestion (Dyspepsia)",
    "Viral Pharyngitis (Sore Throat)",
    "Dehydration & Heat Fatigue",
    "Mechanical Low Back Strain",
    "Common Cold",
    "Allergy"
}

class ClinicalDecisionLayer:
    """
    Evaluates condition predictions alongside symptom timeline.
    Determines urgency level, gates home care, calculates calibrated relative likelihood,
    and attaches Bangalore hospital routing.
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
        Produce a safe, comprehensive Decision Bundle with calibrated relative likelihood.
        """
        if not bert_predictions:
            return {
                "top_conditions": [],
                "primary_condition": "Unknown",
                "urgency_level": "Low",
                "action_timeframe": "No clinical conditions detected. Monitor well-being.",
                "allow_home_care": True,
                "home_care": ["Maintain adequate hydration and rest."],
                "red_flags": ["Visit a doctor if new or severe symptoms emerge."],
                "hospitals": []
            }

        duration_days = timeline.get("duration_days") or 1
        progression = timeline.get("progression", "stable")
        patient_severity = timeline.get("severity", "moderate")

        # 1. Apply Clinical Epidemiological Prior Weighting for Outpatient Presentations
        # If symptoms are non-severe and lack acute red flags, common primary conditions
        # (Tension Headache, Sinusitis, Eye Strain, Acidity, etc.) receive a prevalence prior boost.
        reweighted_preds = []
        for p in bert_predictions:
            d_name = p.get("disease", "")
            raw_prob = float(p.get("probability", 0.0))
            
            prior_factor = 1.0
            if patient_severity != "severe" and duration_days <= 5 and d_name in BENIGN_OUTPATIENT_CONDITIONS:
                prior_factor = 1.8
                
            reweighted_preds.append({
                "disease": d_name,
                "weighted_score": raw_prob * prior_factor,
                "raw_probability": raw_prob
            })

        # Sort by reweighted score
        reweighted_preds.sort(key=lambda x: x["weighted_score"], reverse=True)
        top_k = reweighted_preds[:3]

        # 2. Determine Primary Condition & Base Severity
        top_disease = top_k[0]["disease"]
        remedy_info = self.home_remedies.get(top_disease, {})
        base_severity = remedy_info.get("severity", "moderate")

        # 3. Compute Urgency Level based on condition risk and timeline
        if base_severity == "high" or patient_severity == "severe" or (duration_days >= 7 and progression == "worsening"):
            urgency_level = "High"
            action_timeframe = "Consult a physician promptly today for clinical evaluation and diagnostics."
        elif duration_days >= 3 and base_severity != "low":
            urgency_level = "Moderate"
            action_timeframe = "Medical consultation advised within 24 to 48 hours if symptoms persist."
        elif duration_days >= 5 or patient_severity == "moderate" or progression == "worsening":
            urgency_level = "Moderate"
            action_timeframe = "Medical consultation advised within 24 to 48 hours if symptoms persist."
        else:
            urgency_level = "Low"
            action_timeframe = "Monitor symptoms closely for 48 hours; follow supportive home care."

        # 4. Home Care Safety Gating (Strictly suppressed for High Urgency)
        allow_home_care = (urgency_level != "High")
        if allow_home_care:
            home_care_points = remedy_info.get("home_care", [
                "Drink plenty of fluids (water, electrolytes, broths).",
                "Ensure sufficient rest in a calm environment.",
                "Avoid strenuous physical exertion."
            ])
        else:
            home_care_points = []  # Strictly gated for high severity

        # 5. Red-flag triggers ("When to seek immediate emergency care")
        red_flags = remedy_info.get("seek_medical_attention_if", [
            "Temperature rises above 102°F or persists >3 days.",
            "Difficulty breathing, chest tightness, or extreme dizziness develops.",
            "Inability to keep liquids down for more than 12 hours."
        ])

        # 6. Bangalore Hospital Proximity Routing
        hospitals = self.hospital_recommender.get_recommendations(
            user_lat=user_lat,
            user_lon=user_lon,
            disease_category=top_disease,
            top_k=3
        )

        # 7. Calculate Normalized Relative Likelihood (%) and Clinical Match Bands
        total_score = sum(item["weighted_score"] for item in top_k)
        if total_score <= 0:
            total_score = 1.0

        enriched_predictions = []
        for item in top_k:
            d_name = item["disease"]
            rel_conf = round((item["weighted_score"] / total_score) * 100, 1)

            if rel_conf >= 48.0:
                band_en = "Primary Consideration"
                band_hi = "प्राथमिक संभावना"
            elif rel_conf >= 25.0:
                band_en = "Possible Alternative"
                band_hi = "संभावित विकल्प"
            else:
                band_en = "Less Likely"
                band_hi = "कम संभावना"

            meta = self.display_names.get(d_name, {"friendly_name": d_name, "description": ""})
            enriched_predictions.append({
                "condition": meta.get("friendly_name", d_name),
                "confidence": rel_conf,
                "raw_confidence": round(item["raw_probability"] * 100, 1),
                "match_band": band_en,
                "match_band_hi": band_hi,
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
