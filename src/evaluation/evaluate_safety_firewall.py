"""
MediAssist AI — Comprehensive Independent Emergency Firewall Evaluation Suite
Evaluates sensitivity, missed emergency rate, specificity, and false-positive rate
across 60 diverse, independent adversarial cases (30 true emergencies, 30 challenging non-emergencies).
Includes:
- Paraphrases, colloquialisms, and implicit presentations
- Negation and historical non-acute symptoms
- Misspellings / typographical variations
- Bilingual cases (English, Devanagari Hindi, and Hinglish)
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.safety.emergency_firewall import EmergencySafetyFirewall

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "model_benchmarks" / "firewall_evaluation_report.json"

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0
    return round(lower, 2), round(upper, 2)

EVALUATION_CASES = [
    # =========================================================================
    # COHORT 1: TRUE EMERGENCIES (N=30)
    # =========================================================================
    # Standard Presentations
    {"id": "E01", "text": "Crushing retrosternal chest pain radiating to left arm and jaw with profuse cold sweating", "is_emergency": True, "category": "Cardiac"},
    {"id": "E02", "text": "Sudden onset right facial drooping and right arm weakness unable to lift arm", "is_emergency": True, "category": "Stroke (FAST)"},
    {"id": "E03", "text": "Slurred speech and confusion since 30 minutes, cannot form words evenly", "is_emergency": True, "category": "Stroke (FAST)"},
    {"id": "E04", "text": "Gasped for air, lips turning blue, cannot speak full sentence", "is_emergency": True, "category": "Hypoxia / Respiratory"},
    {"id": "E05", "text": "Severe asthma attack inhaler not working, stridor and chest retractions", "is_emergency": True, "category": "Status Asthmaticus"},
    {"id": "E06", "text": "High fever since 4 days now bleeding from gums and red spots on skin", "is_emergency": True, "category": "Hemorrhagic Dengue"},
    {"id": "E07", "text": "Coughing up large amounts of bright red fresh blood", "is_emergency": True, "category": "Massive Hemoptysis"},
    {"id": "E08", "text": "Worst thunderclap headache of my life, sudden onset like explosion in head", "is_emergency": True, "category": "Subarachnoid Hemorrhage"},
    {"id": "E09", "text": "Stiff neck cannot touch chin to chest, high fever, extreme light sensitivity, drowsy", "is_emergency": True, "category": "Acute Meningitis"},
    {"id": "E10", "text": "Severe right lower quadrant abdominal pain, rebound tenderness, high fever, vomiting", "is_emergency": True, "category": "Acute Appendicitis"},
    {"id": "E11", "text": "Diabetic patient feeling confused, sweating heavily, trembling, blood sugar 42 mg/dL", "is_emergency": True, "category": "Severe Hypoglycemia"},
    {"id": "E12", "text": "Poison consumption, swallowed unknown pesticide 20 minutes ago, vomiting foam", "is_emergency": True, "category": "Toxicology / Poisoning"},
    {"id": "E13", "text": "Sudden total loss of vision in right eye with severe ocular agony", "is_emergency": True, "category": "Ophthalmic Emergency"},
    {"id": "E14", "text": "Violent continuous seizure lasting over 5 minutes without waking up", "is_emergency": True, "category": "Status Epilepticus"},
    {"id": "E15", "text": "Snake bite on right foot 15 minutes ago, severe swelling, fang marks bleeding", "is_emergency": True, "category": "Snake Envenomation"},
    {"id": "E16", "text": "Severe allergic anaphylaxis, tongue swelling, throat tightness after eating peanuts", "is_emergency": True, "category": "Anaphylaxis"},

    # Paraphrased & Non-Keyword Emergencies
    {"id": "E17", "text": "Heavy elephant sitting on my breastbone, sweating through clothes, lightheaded", "is_emergency": True, "category": "Cardiac (Colloquial)"},
    {"id": "E18", "text": "My windpipe feels completely clamped shut, struggling to suck in any oxygen", "is_emergency": True, "category": "Respiratory (Paraphrase)"},
    {"id": "E19", "text": "One side of my body went completely numb and dead, unable to smile", "is_emergency": True, "category": "Stroke (Colloquial)"},
    {"id": "E20", "text": "Child accidentally drank concentrated bathroom acid 10 mins ago, throat burning violently", "is_emergency": True, "category": "Toxicology"},

    # Typo / Misspelling Emergencies
    {"id": "E21", "text": "sevear cheest pane rediating to left arm with suden cold swet", "is_emergency": True, "category": "Cardiac (Typo)"},
    {"id": "E22", "text": "facial droop right side and slured speach since 20 minuts", "is_emergency": True, "category": "Stroke (Typo)"},
    {"id": "E23", "text": "tongue sweling, cannot breath after wasp stng", "is_emergency": True, "category": "Anaphylaxis (Typo)"},

    # Devanagari Hindi Emergencies
    {"id": "E24", "text": "छाती में बहुत तेज असहनीय दर्द है और बाएं हाथ में खिंचाव हो रहा है पसीना आ रहा है", "is_emergency": True, "category": "Cardiac (Hindi)"},
    {"id": "E25", "text": "अचानक चेहरे का दायां हिस्सा लटक गया है और हाथ सुन्न हो गया है", "is_emergency": True, "category": "Stroke (Hindi)"},
    {"id": "E26", "text": "सांस बिल्कुल नहीं आ रही, होंठ नीले पड़ रहे हैं और दम घुट रहा है", "is_emergency": True, "category": "Hypoxia (Hindi)"},
    {"id": "E27", "text": "गले में तेज सूजन, जीभ सूज गई है और सांस नली बंद हो रही है मूंगफली खाने के बाद", "is_emergency": True, "category": "Anaphylaxis (Hindi)"},

    # Hinglish Emergencies
    {"id": "E28", "text": "Seene me bohot tez dard hai jo bayen haath me ja raha hai aur paseena nikal raha hai", "is_emergency": True, "category": "Cardiac (Hinglish)"},
    {"id": "E29", "text": "Chehre ka ek taraf ka hissa latak gaya hai aur bolte waqt aawaz ladkhada rahi hai", "is_emergency": True, "category": "Stroke (Hinglish)"},
    {"id": "E30", "text": "Kisan ne khet me keetnashak spray pee liya hai, muh se jhaag nikal raha hai", "is_emergency": True, "category": "Toxicology (Hinglish)"},

    # =========================================================================
    # COHORT 2: CHALLENGING NON-EMERGENCIES (SPECIFICITY / NEGATION / FALSE POSITIVES) (N=30)
    # =========================================================================
    # Historical / Family History Mentions
    {"id": "N01", "text": "My father had a severe heart attack 3 years ago. I currently have a mild runny nose and cough.", "is_emergency": False, "category": "Family History"},
    {"id": "N02", "text": "My grandfather suffered a stroke last winter. I just have a slight headache from eye strain.", "is_emergency": False, "category": "Family History"},
    {"id": "N03", "text": "I had chest pain 6 months ago that turned out to be gas. Today my ankle is sore after jogging.", "is_emergency": False, "category": "Historical"},
    
    # Explicit Negations
    {"id": "N04", "text": "I have mild fever and sore throat, but NO chest pain and NO breathing trouble.", "is_emergency": False, "category": "Negation"},
    {"id": "N05", "text": "Doctor, I do NOT have facial drooping or weakness, just a band-like headache.", "is_emergency": False, "category": "Negation"},
    {"id": "N06", "text": "Feeling dizzy after fasting, but no chest pressure, no arm numbness, no fainting.", "is_emergency": False, "category": "Negation"},

    # Musculoskeletal & Non-Cardiovascular Chest Discomfort
    {"id": "N07", "text": "Pectoral muscle soreness when I press on it with my fingers after heavy bench press workout.", "is_emergency": False, "category": "Musculoskeletal"},
    {"id": "N08", "text": "Mild chest skin rash with itchy pimples across sternum for 2 weeks.", "is_emergency": False, "category": "Dermatology"},
    {"id": "N09", "text": "Sharp rib twinge for 2 seconds when taking a very deep yawn, completely fine otherwise.", "is_emergency": False, "category": "Benign Pleuritic"},
    {"id": "N10", "text": "Heartburn and sour waterbrash in throat after eating spicy fried chicken 2 hours ago.", "is_emergency": False, "category": "Dyspepsia"},

    # Figurative & Non-Clinical Collocations
    {"id": "N11", "text": "My head is exploding with tension because of tight work deadlines, need a break.", "is_emergency": False, "category": "Metaphorical"},
    {"id": "N12", "text": "I'm heartbroken after breaking up, feeling down and cannot sleep.", "is_emergency": False, "category": "Emotional"},
    {"id": "N13", "text": "I feel breathless with excitement about my upcoming wedding, but have minor acid reflux.", "is_emergency": False, "category": "Metaphorical"},

    # Routine Outpatient Primary Care
    {"id": "N14", "text": "Continuous dull tension headache since 3 days across forehead, no fever or vomiting.", "is_emergency": False, "category": "Tension Headache"},
    {"id": "N15", "text": "Burning dry eyes and mild brow ache after 12 hours coding on monitor.", "is_emergency": False, "category": "Eye Strain"},
    {"id": "N16", "text": "Sinus pressure over cheekbones and yellow nasal mucus, worse when bending forward.", "is_emergency": False, "category": "Sinusitis"},
    {"id": "N17", "text": "Dull lower back stiffness after lifting heavy suitcase, no leg weakness or numbness.", "is_emergency": False, "category": "Low Back Strain"},
    {"id": "N18", "text": "Scratchy throat, painful swallowing, low fever 99F since yesterday.", "is_emergency": False, "category": "Pharyngitis"},
    {"id": "N19", "text": "Extreme thirst, dry mouth, dark urine after 4 hours outside in hot summer sun.", "is_emergency": False, "category": "Dehydration"},
    {"id": "N20", "text": "Joint pain and swelling in wrists and ankles 4 days after viral fever subsided.", "is_emergency": False, "category": "Chikungunya"},
    {"id": "N21", "text": "Superficial knee scrape from falling off bicycle, minor bleeding stopped with band-aid.", "is_emergency": False, "category": "Minor Trauma"},
    {"id": "N22", "text": "Painless red rash on cheeks with whiteheads for past two weeks.", "is_emergency": False, "category": "Acne"},

    # Devanagari Hindi Non-Emergencies
    {"id": "N23", "text": "कल रात से हल्का सर दर्द है और काम का तनाव है, कोई उल्टी या चक्कर नहीं है।", "is_emergency": False, "category": "Hindi OPD"},
    {"id": "N24", "text": "कंप्यूटर पर ज्यादा काम करने से आंखों में जलन और भारीपन लग रहा है।", "is_emergency": False, "category": "Hindi OPD"},
    {"id": "N25", "text": "मसालेदार खाना खाने के बाद सीने में हल्की जलन हो रही है और खट्टी डकार आ रही है।", "is_emergency": False, "category": "Hindi OPD"},
    {"id": "N26", "text": "गले में खराश है और हल्का जुकाम है, सांस लेने में कोई परेशानी नहीं है।", "is_emergency": False, "category": "Hindi OPD"},

    # Hinglish Non-Emergencies
    {"id": "N27", "text": "Sar me halka dard hai 2 din se, office me screen dekhne ki wajah se, bukhar nahi hai.", "is_emergency": False, "category": "Hinglish OPD"},
    {"id": "N28", "text": "Gym me chest workout ke baad muscle me halka soreness hai, dabane pe dard hota hai.", "is_emergency": False, "category": "Hinglish OPD"},
    {"id": "N29", "text": "Bhai ko pehle stroke hua tha, mujhe bas thoda sardi zukaam hai aaj se.", "is_emergency": False, "category": "Hinglish OPD"},
    {"id": "N30", "text": "Kamar me halka spasm hai bhari saman uthane ke baad, pair me koi dard nahi ja raha.", "is_emergency": False, "category": "Hinglish OPD"}
]

def run_firewall_evaluation():
    print("=" * 80)
    print("MEDPASSIST AI — EMERGENCY FIREWALL EVALUATION (SENSITIVITY & SPECIFICITY)")
    print("=" * 80)

    firewall = EmergencySafetyFirewall()

    tp, fp, tn, fn = 0, 0, 0, 0
    detailed_results = []

    for case in EVALUATION_CASES:
        cid = case["id"]
        text = case["text"]
        ground_truth = case["is_emergency"]
        category = case["category"]

        eval_res = firewall.evaluate(text)
        pred_emergency = eval_res["is_emergency"]
        matched_category = eval_res.get("emergency_type", "None")

        if ground_truth and pred_emergency:
            tp += 1
            status = "TP (True Positive - Correct Escalation)"
        elif not ground_truth and pred_emergency:
            fp += 1
            status = "FP (False Positive - Unnecessary Escalation)"
        elif not ground_truth and not pred_emergency:
            tn += 1
            status = "TN (True Negative - Correct Pass-Through)"
        else:
            fn += 1
            status = "FN (False Negative - MISSED EMERGENCY)"

        detailed_results.append({
            "id": cid,
            "category": category,
            "text": text,
            "ground_truth_emergency": ground_truth,
            "predicted_emergency": pred_emergency,
            "matched_category": matched_category,
            "status": status
        })

    total_emergencies = tp + fn
    total_non_emergencies = tn + fp
    total_cases = len(EVALUATION_CASES)

    sensitivity = (tp / total_emergencies) * 100.0 if total_emergencies > 0 else 0.0
    mer = (fn / total_emergencies) * 100.0 if total_emergencies > 0 else 0.0
    specificity = (tn / total_non_emergencies) * 100.0 if total_non_emergencies > 0 else 0.0
    fpr = (fp / total_non_emergencies) * 100.0 if total_non_emergencies > 0 else 0.0
    accuracy = ((tp + tn) / total_cases) * 100.0

    sens_ci = wilson_ci(tp, total_emergencies)
    spec_ci = wilson_ci(tn, total_non_emergencies)
    acc_ci = wilson_ci(tp + tn, total_cases)

    print(f"\nCohort Evaluation Summary (N={total_cases} total cases):")
    print(f"  - True Emergencies Evaluated     : {total_emergencies}")
    print(f"  - Non-Emergencies Evaluated      : {total_non_emergencies}")
    print(f"  - True Positives (TP)            : {tp}")
    print(f"  - False Negatives (FN, Missed)   : {fn}")
    print(f"  - True Negatives (TN)            : {tn}")
    print(f"  - False Positives (FP)           : {fp}")
    print("\nClinical Safety Metrics:")
    print(f"  - Emergency Sensitivity (EEA)    : {sensitivity:.2f}% (95% CI: {sens_ci[0]} - {sens_ci[1]}%)")
    print(f"  - Missed Emergency Rate (MER)    : {mer:.2f}%")
    print(f"  - Specificity (Correct Pass)     : {specificity:.2f}% (95% CI: {spec_ci[0]} - {spec_ci[1]}%)")
    print(f"  - False Positive Rate (FPR)      : {fpr:.2f}%")
    print(f"  - Overall Classification Accuracy: {accuracy:.2f}% (95% CI: {acc_ci[0]} - {acc_ci[1]}%)")

    # List any false negatives or false positives
    if fn > 0:
        print("\n[!] CRITICAL: False Negatives Identified (Missed Emergencies):")
        for r in detailed_results:
            if "FN" in r["status"]:
                print(f"  - [{r['id']}] {r['category']}: \"{r['text']}\"")

    if fp > 0:
        print("\n[i] False Positives Identified (Over-escalated Non-emergencies):")
        for r in detailed_results:
            if "FP" in r["status"]:
                print(f"  - [{r['id']}] {r['category']}: \"{r['text']}\" (Triggered: {r['matched_category']})")

    summary_data = {
        "evaluation_cohort_size": total_cases,
        "true_emergencies_n": total_emergencies,
        "non_emergencies_n": total_non_emergencies,
        "confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "metrics": {
            "emergency_sensitivity_eea": round(sensitivity, 2),
            "emergency_sensitivity_ci95": sens_ci,
            "missed_emergency_rate_mer": round(mer, 2),
            "specificity": round(specificity, 2),
            "specificity_ci95": spec_ci,
            "false_positive_rate_fpr": round(fpr, 2),
            "overall_accuracy": round(accuracy, 2),
            "overall_accuracy_ci95": acc_ci
        },
        "cases": detailed_results
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved full emergency firewall report to {REPORT_PATH}]")
    return summary_data

if __name__ == "__main__":
    run_firewall_evaluation()
