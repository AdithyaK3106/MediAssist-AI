import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "symptoms" / "Symptom2Disease.csv"
BERT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"
ML_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ml"
DATA_DIR = PROJECT_ROOT / "data"

BERT_DATA_DIR.mkdir(parents=True, exist_ok=True)
ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Standard label normalization map
LABEL_CLEANING_MAP = {
    "allergy": "Allergy",
    "diabetes": "Diabetes",
    "drug reaction": "Drug Reaction",
    "gastroesophageal reflux disease": "GERD (Acid Reflux)",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    "urinary tract infection": "Urinary Tract Infection (UTI)",
    "Bronchial Asthma": "Asthma",
    "Dimorphic Hemorrhoids": "Hemorrhoids",
    "Fungal infection": "Fungal Infection",
    "Varicose Veins": "Varicose Veins",
    "Cervical spondylosis": "Cervical Spondylosis"
}

# 2. Supplementary clinical samples for regional prevalence (Flu, Tuberculosis, Gastroenteritis)
SUPPLEMENTARY_DATA = [
    # Influenza (Flu)
    {"label": "Influenza (Flu)", "text": "I have had a sudden high fever, severe body aches, shivering chills, and extreme fatigue for the past two days."},
    {"label": "Influenza (Flu)", "text": "My temperature shot up to 102 degrees suddenly. I have a dry hacky cough, severe headache behind the eyes, and exhausting muscle pain."},
    {"label": "Influenza (Flu)", "text": "Experiencing rapid onset fever, cold sweats, blocked nose, sore throat, and feeling completely drained and bedridden."},
    {"label": "Influenza (Flu)", "text": "High grade fever with intense generalized body weakness, painful coughing in the chest, and severe joint aches."},
    {"label": "Influenza (Flu)", "text": "Chills, shaking, burning sensation in the throat, headache, and sudden fever that came on within a few hours."},
    {"label": "Influenza (Flu)", "text": "I have a sore throat, runny nose, high temperature, and my whole body feels bruised and painful."},
    {"label": "Influenza (Flu)", "text": "Severe malaise, hot and cold spells, high fever, fatigue, and dry persistent cough for 48 hours."},
    {"label": "Influenza (Flu)", "text": "Woke up shivering with 101.5 fever, debilitating weakness, tender muscles, and a scratchy painful throat."},
    {"label": "Influenza (Flu)", "text": "Severe fever and body pain after traveling. I can barely get out of bed due to exhaustion, dizziness, and headache."},
    {"label": "Influenza (Flu)", "text": "Heavy fever with chills, body ache, dry cough, loss of appetite, and headache since yesterday morning."},
    
    # Tuberculosis
    {"label": "Tuberculosis", "text": "I have been coughing persistently for more than three weeks, with drenching night sweats and noticeable weight loss."},
    {"label": "Tuberculosis", "text": "Chronic cough that has lasted over a month, coughing up thick phlegm with occasional flecks of blood, and severe evening fever."},
    {"label": "Tuberculosis", "text": "Low grade fever every evening, night sweats requiring clothes change, poor appetite, and unintentional weight reduction."},
    {"label": "Tuberculosis", "text": "Chest pain when coughing, fatigue, slight blood in sputum, and a persistent dry cough for the last month."},
    {"label": "Tuberculosis", "text": "Loss of weight, pale complexion, swollen lymph nodes in the neck, and a hacking cough lasting several weeks."},
    {"label": "Tuberculosis", "text": "Persistent cough with blood tinged phlegm, shortness of breath, unexplained exhaustion, and severe night sweating."},
    {"label": "Tuberculosis", "text": "I feel chronically fatigued, lost 5 kilos in two months, with mild fever every night and continuous cough."},
    {"label": "Tuberculosis", "text": "Deep chest pain, coughing spells that do not resolve with regular cough syrups, night chills, and heavy weight drop."},
    {"label": "Tuberculosis", "text": "Sputum production with blood stains, dull chest ache, fatigue, prolonged cough over 4 weeks, and poor appetite."},
    {"label": "Tuberculosis", "text": "Chronic afternoon and night fevers with cold sweats, chronic cough, and severe loss of energy."},

    # Gastroenteritis (Food Poisoning / Stomach Infection)
    {"label": "Gastroenteritis", "text": "Severe watery diarrhea, violent nausea, abdominal cramps, and vomiting that started after eating street food."},
    {"label": "Gastroenteritis", "text": "Frequent loose watery stools, stomach churning, cramping pain, low grade fever, and dehydration."},
    {"label": "Gastroenteritis", "text": "I cannot keep any fluids down because of repeated vomiting, accompanied by watery diarrhea and severe abdominal spasms."},
    {"label": "Gastroenteritis", "text": "Sudden onset of vomiting and watery bowel movements with chills, mild fever, and dry mouth from fluid loss."},
    {"label": "Gastroenteritis", "text": "Stomach griping cramps, constant nausea, diarrhea every hour, and general weakness after a meal."},
    {"label": "Gastroenteritis", "text": "Intestinal bloating, sharp colic pain, persistent watery loose motion, and frequent vomiting episodes."},
    {"label": "Gastroenteritis", "text": "Extreme thirst, lightheadedness, frequent watery stools, and belly cramps that started yesterday night."},
    {"label": "Gastroenteritis", "text": "Vomiting yellow bile, loose motion, rumbling painful stomach, and low grade fever."},
    {"label": "Gastroenteritis", "text": "Severe food poisoning symptoms with nausea, frequent liquid bowel movements, and painful abdominal contractions."},
    {"label": "Gastroenteritis", "text": "Watery stools multiple times today, painful tummy cramps, slight fever, and complete loss of appetite."}
]

def consolidate_dataset():
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # If index column exists, drop it
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    # Standardize column names
    df = df.rename(columns={"label": "disease", "text": "symptom_text"})
    
    # Clean disease labels
    df["disease"] = df["disease"].apply(lambda x: LABEL_CLEANING_MAP.get(x, x.strip().title()))
    
    # Add supplementary samples
    supp_df = pd.DataFrame(SUPPLEMENTARY_DATA).rename(columns={"label": "disease", "text": "symptom_text"})
    # Replicate supplementary samples to ensure reasonable balance
    supp_expanded = pd.concat([supp_df] * 5, ignore_index=True)
    
    full_df = pd.concat([df, supp_expanded], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["symptom_text"]).reset_index(drop=True)
    
    print(f"Consolidated dataset: {len(full_df)} samples across {full_df['disease'].nunique()} conditions.")
    
    # Create integer label mapping
    unique_diseases = sorted(full_df["disease"].unique())
    label_to_id = {disease: idx for idx, disease in enumerate(unique_diseases)}
    id_to_label = {idx: disease for idx, disease in enumerate(unique_diseases)}
    
    full_df["label"] = full_df["disease"].map(label_to_id)
    
    # Save full dataset
    full_df.to_csv(BERT_DATA_DIR / "dataset.csv", index=False)
    
    # Train / Val / Test Splits (70 / 15 / 15)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.30, random_state=42, stratify=full_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    
    train_df.to_csv(BERT_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(BERT_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(BERT_DATA_DIR / "test.csv", index=False)
    
    # Save label mappings in both BERT and legacy ML locations
    with open(BERT_DATA_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)
    with open(ML_DATA_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)
        
    print(f"Splits generated -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 3. Generate Disease Categories JSON
    generate_clinical_knowledge(unique_diseases)

def generate_clinical_knowledge(diseases):
    # Category mapping for medical consistency
    categories = {
        "Dengue": ["fever", "systemic", "musculoskeletal", "skin"],
        "Malaria": ["fever", "systemic", "musculoskeletal", "gastrointestinal"],
        "Typhoid": ["fever", "gastrointestinal", "systemic"],
        "Influenza (Flu)": ["respiratory", "fever", "systemic", "musculoskeletal"],
        "Common Cold": ["respiratory", "fever"],
        "Pneumonia": ["respiratory", "fever", "cardiovascular", "systemic"],
        "Tuberculosis": ["respiratory", "fever", "systemic"],
        "Asthma": ["respiratory", "cardiovascular"],
        "Gastroenteritis": ["gastrointestinal", "systemic", "fever"],
        "GERD (Acid Reflux)": ["gastrointestinal", "cardiovascular"],
        "Peptic Ulcer Disease": ["gastrointestinal"],
        "Migraine": ["neurological", "eye"],
        "Hypertension": ["cardiovascular", "neurological", "systemic"],
        "Diabetes": ["metabolic", "urinary", "systemic"],
        "Urinary Tract Infection (UTI)": ["urinary", "fever", "systemic"],
        "Chicken Pox": ["skin", "fever", "systemic"],
        "Acne": ["skin"],
        "Psoriasis": ["skin", "musculoskeletal"],
        "Impetigo": ["skin"],
        "Fungal Infection": ["skin"],
        "Arthritis": ["musculoskeletal", "systemic"],
        "Cervical Spondylosis": ["musculoskeletal", "neurological"],
        "Jaundice": ["gastrointestinal", "systemic", "skin"],
        "Hemorrhoids": ["gastrointestinal"],
        "Varicose Veins": ["cardiovascular", "musculoskeletal"],
        "Allergy": ["respiratory", "skin", "eye"],
        "Drug Reaction": ["skin", "systemic", "fever"]
    }
    
    # Display names & clinical descriptions
    display_names = {
        d: {
            "friendly_name": d,
            "description": f"Preliminary presentation consistent with {d}. Clinical evaluation recommended."
        } for d in diseases
    }
    
    # Enrich key descriptions
    display_names["Dengue"] = {
        "friendly_name": "Dengue Fever",
        "description": "Mosquito-borne viral infection characterized by sudden high fever, retro-orbital eye pain, and joint aches."
    }
    display_names["Malaria"] = {
        "friendly_name": "Malaria",
        "description": "Parasitic infection transmitted by mosquitoes, typically presenting with cyclical chills, high fever, and sweating."
    }
    display_names["Pneumonia"] = {
        "friendly_name": "Pneumonia",
        "description": "Acute lower respiratory infection causing lung inflammation, productive cough, fever, and difficulty breathing."
    }
    display_names["Tuberculosis"] = {
        "friendly_name": "Tuberculosis (TB)",
        "description": "Chronic infectious bacterial disease mainly affecting the lungs, marked by persistent cough >3 weeks, weight loss, and night sweats."
    }
    display_names["Influenza (Flu)"] = {
        "friendly_name": "Influenza (Flu)",
        "description": "Contagious respiratory illness caused by influenza viruses, marked by rapid onset fever, severe body aches, and exhaustion."
    }
    display_names["Common Cold"] = {
        "friendly_name": "Common Cold",
        "description": "Mild viral upper respiratory tract infection marked by nasal congestion, sneezing, sore throat, and low-grade discomfort."
    }
    display_names["Gastroenteritis"] = {
        "friendly_name": "Gastroenteritis (Stomach Infection)",
        "description": "Inflammation of the digestive tract commonly caused by viral or bacterial infection, resulting in diarrhea, vomiting, and cramps."
    }

    # Home remedies with strict severity gating
    remedies = {
        "Common Cold": {
            "severity": "low",
            "home_care": ["Rest adequately", "Drink warm fluids like herbal tea or broth", "Steam inhalation for nasal congestion", "Warm saline gargle"],
            "seek_medical_attention_if": ["Symptoms persist beyond 10 days", "High fever above 102°F develops", "Difficulty breathing occurs"]
        },
        "Allergy": {
            "severity": "low",
            "home_care": ["Avoid known environmental triggers and dust", "Rinse eyes and nasal passages with saline", "Stay indoors during high pollen hours"],
            "seek_medical_attention_if": ["Swelling of face, lips, or throat develops", "Breathing becomes labored"]
        },
        "GERD (Acid Reflux)": {
            "severity": "low",
            "home_care": ["Avoid spicy, fried, and acidic meals", "Eat smaller and more frequent meals", "Remain upright for 2-3 hours after eating"],
            "seek_medical_attention_if": ["Difficulty or pain when swallowing", "Persistent vomiting or unexplained weight loss"]
        },
        "Influenza (Flu)": {
            "severity": "moderate",
            "home_care": ["Strict bed rest and hydration", "Electrolyte drinks and warm soups", "Isolate to prevent transmission"],
            "seek_medical_attention_if": ["Shortness of breath or chest pain", "Persistent high fever not responding to antipyretics", "Confusion or extreme dizziness"]
        },
        "Gastroenteritis": {
            "severity": "moderate",
            "home_care": ["Oral Rehydration Salts (ORS) solution regularly", "Bland diet (BRAT: bananas, rice, applesauce, toast)", "Avoid dairy and caffeine"],
            "seek_medical_attention_if": ["Signs of severe dehydration (no urination, sunken eyes)", "Blood in stool or vomit", "Inability to keep any liquids down for >12 hours"]
        },
        "Migraine": {
            "severity": "moderate",
            "home_care": ["Rest in a quiet, dark room", "Apply a cold compress to the forehead", "Stay hydrated"],
            "seek_medical_attention_if": ["Sudden severe 'thunderclap' headache", "Headache accompanied by stiff neck, fever, or vision loss"]
        },
        "Dengue": {
            "severity": "high",
            "home_care": [], # Gated: No home care for high severity
            "seek_medical_attention_if": ["Immediate consultation required for platelet monitoring", "Seek emergency care if bleeding gums, blood in stool, or persistent severe vomiting occurs"],
            "doctor_visit": "Consult a physician immediately for Dengue NS1/IgM and CBC blood tests."
        },
        "Malaria": {
            "severity": "high",
            "home_care": [], # Gated
            "seek_medical_attention_if": ["Prompt blood smear or rapid diagnostic test (RDT) required", "Immediate care if confusion, jaundice, or dark urine develops"],
            "doctor_visit": "Visit a clinic immediately for antimalarial prescription."
        },
        "Pneumonia": {
            "severity": "high",
            "home_care": [], # Gated
            "seek_medical_attention_if": ["Seek urgent emergency care if lips turn bluish or breathing is rapid and labored"],
            "doctor_visit": "Immediate clinical evaluation and chest imaging required."
        },
        "Tuberculosis": {
            "severity": "high",
            "home_care": [], # Gated
            "seek_medical_attention_if": ["Seek immediate medical attention for sputum AFB testing and chest X-ray"],
            "doctor_visit": "Strictly requires supervised clinical anti-tubercular therapy (DOTS)."
        }
    }
    
    # Save JSON files in data/
    with open(DATA_DIR / "disease_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    with open(DATA_DIR / "disease_display_names.json", "w", encoding="utf-8") as f:
        json.dump(display_names, f, indent=2)
    with open(DATA_DIR / "home_remedies.json", "w", encoding="utf-8") as f:
        json.dump(remedies, f, indent=2)
        
    print(f"Generated disease_categories.json, disease_display_names.json, home_remedies.json successfully.")

if __name__ == "__main__":
    consolidate_dataset()
