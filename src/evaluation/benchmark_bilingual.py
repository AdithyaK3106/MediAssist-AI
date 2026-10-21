"""
MediAssist AI — Comprehensive Bilingual & Cross-Lingual Evaluation Suite
Evaluates triage and differential performance across English, Devanagari Hindi, and Hinglish.
Demonstrates the architectural necessity of Layer 1 clinical entity normalization
due to PubMedBERT WordPiece vocabulary limitations ([UNK] tokenization on Indic script).
"""

import os
import json
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer

from src.models.transformers.predict_bert import MedicalBERTInferencePipeline
from src.conversational.timeline_extractor import SymptomTimelineExtractor
from src.safety.emergency_firewall import EmergencySafetyFirewall

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bilingual_benchmark")

# 10 clinical conditions represented across 3 language modalities (30 total queries)
BILINGUAL_TEST_PAIRS = [
    {
        "condition": "Tension Headache",
        "en": "Tight band-like headache around forehead and temples due to stress",
        "hi": "तनाव के कारण माथे और कनपटी पर कसने वाला तेज सिरदर्द",
        "hinglish": "Stress ki wajah se sar me band jaisa dard aur forehead pe tightness"
    },
    {
        "condition": "Acute Sinusitis",
        "en": "Frontal headache, facial heaviness behind cheeks with nasal congestion",
        "hi": "गालों और भौहों के पीछे भारीपन, चेहरे पर दबाव और बंद नाक",
        "hinglish": "Gaal aur naak ke paas bhari pan aur sar me sinus ka dard"
    },
    {
        "condition": "Eye Strain / Screen Fatigue",
        "en": "Burning eyes, blurred vision and forehead ache after working on computer",
        "hi": "कंप्यूटर पर काम करने के बाद आंखों में जलन, धुंधलापन और दर्द",
        "hinglish": "Laptop dekhne ke baad aankhon me jalan aur thakan mehsoos ho rahi hai"
    },
    {
        "condition": "Acidity & Indigestion (Dyspepsia)",
        "en": "Severe burning in chest and sour acid rising into throat after eating",
        "hi": "खाना खाने के बाद सीने में तेज जलन और गले में खट्टी डकारें",
        "hinglish": "Khane ke baad seene me jalan aur khatti dakar aa rahi hai"
    },
    {
        "condition": "Viral Pharyngitis (Sore Throat)",
        "en": "Scratchy throat, painful swallowing, low fever and mild cough",
        "hi": "गले में खराश, निगलने में दर्द, हल्का बुखार और खांसी",
        "hinglish": "Gale me kharash aur nigalne me dard ho raha hai halka bukhar hai"
    },
    {
        "condition": "Dehydration & Heat Fatigue",
        "en": "Extreme thirst, dizziness, dark urine and weakness after walking in sun",
        "hi": "धूप में चलने के बाद अत्यधिक प्यास, चक्कर, गहरा पेशाब और कमजोरी",
        "hinglish": "Dhoop me chakkar aa rahe hain, gala sookh raha hai aur weakness hai"
    },
    {
        "condition": "Mechanical Low Back Strain",
        "en": "Dull ache in lower back and muscle stiffness after lifting heavy load",
        "hi": "भारी वजन उठाने के बाद पीठ के निचले हिस्से में दर्द और जकड़न",
        "hinglish": "Bhari saman uthane ke baad kamar me dard aur jakdan ho gayi hai"
    },
    {
        "condition": "Chikungunya",
        "en": "Severe joint pain and swelling in wrists and ankles after viral fever",
        "hi": "वायरल बुखार के बाद कलाई और टखनों में असहनीय जोड़ों का दर्द और सूजन",
        "hinglish": "Bukhar theek hone ke baad jodon me tez dard aur hath pair me soojan"
    },
    {
        "condition": "Dengue",
        "en": "High sudden fever with retro-orbital eye pain and severe muscle aches",
        "hi": "आंखों के पीछे दर्द, तेज बुखार और मांसपेशियों में गंभीर दर्द",
        "hinglish": "Tez bukhar aur aankho ke peeche dard ho raha hai body me pain hai"
    },
    {
        "condition": "Typhoid",
        "en": "Step-ladder rising high fever, loss of appetite and abdominal discomfort",
        "hi": "लगातार बढ़ता तेज बुखार, भूख न लगना और पेट में बेचैनी",
        "hinglish": "Kayi din se bukhar badh raha hai, bhookh nahi lag rahi pet me dard hai"
    }
]

def run_bilingual_benchmark():
    tokenizer = AutoTokenizer.from_pretrained("models/transformers/pubmedbert")
    classifier = MedicalBERTInferencePipeline()
    extractor = SymptomTimelineExtractor()

    # 1. Measure Tokenizer [UNK] rates across languages
    token_stats = {"en": {"total": 0, "unk": 0}, "hi": {"total": 0, "unk": 0}, "hinglish": {"total": 0, "unk": 0}}
    for pair in BILINGUAL_TEST_PAIRS:
        for lang in ["en", "hi", "hinglish"]:
            toks = tokenizer.tokenize(pair[lang])
            token_stats[lang]["total"] += len(toks)
            token_stats[lang]["unk"] += toks.count("[UNK]")

    unk_rates = {
        lang: round((token_stats[lang]["unk"] / token_stats[lang]["total"]) * 100, 2)
        for lang in ["en", "hi", "hinglish"]
    }

    # 2. Evaluate Direct PubMedBERT (without Layer 1 normalizer) vs Decoupled Pipeline (with Layer 1 normalizer)
    results = {
        "direct_pubmedbert": {"en": {"top1": 0, "top3": 0}, "hi": {"top1": 0, "top3": 0}, "hinglish": {"top1": 0, "top3": 0}},
        "decoupled_pipeline": {"en": {"top1": 0, "top3": 0}, "hi": {"top1": 0, "top3": 0}, "hinglish": {"top1": 0, "top3": 0}}
    }

    n_cases = len(BILINGUAL_TEST_PAIRS)

    for pair in BILINGUAL_TEST_PAIRS:
        gt = pair["condition"]

        for lang in ["en", "hi", "hinglish"]:
            text = pair[lang]

            # A. Direct PubMedBERT
            raw_res = classifier.predict([text], top_k=3)[0]["predictions"]
            raw_preds = [p["disease"] for p in raw_res]
            if raw_preds and raw_preds[0] == gt:
                results["direct_pubmedbert"][lang]["top1"] += 1
            if gt in raw_preds:
                results["direct_pubmedbert"][lang]["top3"] += 1

            # B. Decoupled Pipeline (Layer 1 Normalization -> PubMedBERT)
            profile = extractor.extract(text)
            extracted_symptoms = " ".join(profile.get("symptoms", []))
            normalized_query = f"{text} {extracted_symptoms}".strip() if extracted_symptoms else text

            norm_res = classifier.predict([normalized_query], top_k=3)[0]["predictions"]
            norm_preds = [p["disease"] for p in norm_res]
            if norm_preds and norm_preds[0] == gt:
                results["decoupled_pipeline"][lang]["top1"] += 1
            if gt in norm_preds:
                results["decoupled_pipeline"][lang]["top3"] += 1

    summary = {
        "sample_size_per_language": n_cases,
        "tokenizer_unk_rate": unk_rates,
        "direct_pubmedbert_top3_acc": {
            lang: round((results["direct_pubmedbert"][lang]["top3"] / n_cases) * 100, 1)
            for lang in ["en", "hi", "hinglish"]
        },
        "decoupled_pipeline_top3_acc": {
            lang: round((results["decoupled_pipeline"][lang]["top3"] / n_cases) * 100, 1)
            for lang in ["en", "hi", "hinglish"]
        },
        "decoupled_pipeline_top1_acc": {
            lang: round((results["decoupled_pipeline"][lang]["top1"] / n_cases) * 100, 1)
            for lang in ["en", "hi", "hinglish"]
        }
    }

    output_path = "reports/model_benchmarks/bilingual_evaluation_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Bilingual benchmark saved to {output_path}")
    print("\n" + "="*70)
    print("BILINGUAL & CROSS-LINGUAL EVALUATION SUMMARY")
    print("="*70)
    print(f"Tokenizer [UNK] Rate: English: {unk_rates['en']}% | Devanagari Hindi: {unk_rates['hi']}% | Hinglish: {unk_rates['hinglish']}%")
    print("\nDirect PubMedBERT (No Normalizer):")
    for l in ["en", "hi", "hinglish"]:
        print(f"  {l.capitalize():10}: Top-3: {summary['direct_pubmedbert_top3_acc'][l]}%")
    print("\nDecoupled Defense-in-Depth Pipeline (Layer 1 Normalizer -> PubMedBERT):")
    for l in ["en", "hi", "hinglish"]:
        print(f"  {l.capitalize():10}: Top-1: {summary['decoupled_pipeline_top1_acc'][l]}% | Top-3: {summary['decoupled_pipeline_top3_acc'][l]}%")

    return summary

if __name__ == "__main__":
    run_bilingual_benchmark()
