import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "symptoms" / "Symptom2Disease.csv"
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic" / "outpatient_indian_symptoms.csv"
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

# 2. Supplementary clinical samples for regional prevalence
SUPPLEMENTARY_DATA = [
    # Influenza (Flu)
    {"disease": "Influenza (Flu)", "symptom_text": "I have had a sudden high fever, severe body aches, shivering chills, and extreme fatigue for the past two days."},
    {"disease": "Influenza (Flu)", "symptom_text": "My temperature shot up to 102 degrees suddenly. I have a dry hacky cough, severe headache behind the eyes, and exhausting muscle pain."},
    {"disease": "Influenza (Flu)", "symptom_text": "Experiencing rapid onset fever, cold sweats, blocked nose, sore throat, and feeling completely drained and bedridden."},
    {"disease": "Influenza (Flu)", "symptom_text": "High grade fever with intense generalized body weakness, painful coughing in the chest, and severe joint aches."},
    {"disease": "Influenza (Flu)", "symptom_text": "Chills, shaking, burning sensation in the throat, headache, and sudden fever that came on within a few hours."},
    {"disease": "Influenza (Flu)", "symptom_text": "I have a sore throat, runny nose, high temperature, and my whole body feels bruised and painful."},
    {"disease": "Influenza (Flu)", "symptom_text": "Severe malaise, hot and cold spells, high fever, fatigue, and dry persistent cough for 48 hours."},
    {"disease": "Influenza (Flu)", "symptom_text": "Woke up shivering with 101.5 fever, debilitating weakness, tender muscles, and a scratchy painful throat."},
    {"disease": "Influenza (Flu)", "symptom_text": "Severe fever and body pain after traveling. I can barely get out of bed due to exhaustion, dizziness, and headache."},
    {"disease": "Influenza (Flu)", "symptom_text": "Heavy fever with chills, body ache, dry cough, loss of appetite, and headache since yesterday morning."},
    
    # Tuberculosis
    {"disease": "Tuberculosis", "symptom_text": "I have been coughing persistently for more than three weeks, with drenching night sweats and noticeable weight loss."},
    {"disease": "Tuberculosis", "symptom_text": "Chronic cough that has lasted over a month, coughing up thick phlegm with occasional flecks of blood, and severe evening fever."},
    {"disease": "Tuberculosis", "symptom_text": "Low grade fever every evening, night sweats requiring clothes change, poor appetite, and unintentional weight reduction."},
    {"disease": "Tuberculosis", "symptom_text": "Chest pain when coughing, fatigue, slight blood in sputum, and a persistent dry cough for the last month."},
    {"disease": "Tuberculosis", "symptom_text": "Loss of weight, pale complexion, swollen lymph nodes in the neck, and a hacking cough lasting several weeks."},
    {"disease": "Tuberculosis", "symptom_text": "Persistent cough with blood tinged phlegm, shortness of breath, unexplained exhaustion, and severe night sweating."},
    {"disease": "Tuberculosis", "symptom_text": "I feel chronically fatigued, lost 5 kilos in two months, with mild fever every night and continuous cough."},
    {"disease": "Tuberculosis", "symptom_text": "Deep chest pain, coughing spells that do not resolve with regular cough syrups, night chills, and heavy weight drop."},
    {"disease": "Tuberculosis", "symptom_text": "Sputum production with blood stains, dull chest ache, fatigue, prolonged cough over 4 weeks, and poor appetite."},
    {"disease": "Tuberculosis", "symptom_text": "Chronic afternoon and night fevers with cold sweats, chronic cough, and severe loss of energy."},

    # Gastroenteritis (Food Poisoning / Stomach Infection)
    {"disease": "Gastroenteritis", "symptom_text": "Severe watery diarrhea, violent nausea, abdominal cramps, and vomiting that started after eating street food."},
    {"disease": "Gastroenteritis", "symptom_text": "Frequent loose watery stools, stomach churning, cramping pain, low grade fever, and dehydration."},
    {"disease": "Gastroenteritis", "symptom_text": "I cannot keep any fluids down because of repeated vomiting, accompanied by watery diarrhea and severe abdominal spasms."},
    {"disease": "Gastroenteritis", "symptom_text": "Sudden onset of vomiting and watery bowel movements with chills, mild fever, and dry mouth from fluid loss."},
    {"disease": "Gastroenteritis", "symptom_text": "Stomach griping cramps, constant nausea, diarrhea every hour, and general weakness after a meal."},
    {"disease": "Gastroenteritis", "symptom_text": "Intestinal bloating, sharp colic pain, persistent watery loose motion, and frequent vomiting episodes."},
    {"disease": "Gastroenteritis", "symptom_text": "Extreme thirst, lightheadedness, frequent watery stools, and belly cramps that started yesterday night."},
    {"disease": "Gastroenteritis", "symptom_text": "Vomiting yellow bile, loose motion, rumbling painful stomach, and low grade fever."},
    {"disease": "Gastroenteritis", "symptom_text": "Severe food poisoning symptoms with nausea, frequent liquid bowel movements, and painful abdominal contractions."},
    {"disease": "Gastroenteritis", "symptom_text": "Watery stools multiple times today, painful tummy cramps, slight fever, and complete loss of appetite."}
]

def consolidate_dataset():
    print(f"Loading raw benchmark dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    df = df.rename(columns={"label": "disease", "text": "symptom_text"})
    df["disease"] = df["disease"].apply(lambda x: LABEL_CLEANING_MAP.get(x, x.strip().title()))
    
    # 2. Add supplementary clinical samples (Influenza, Tuberculosis, Gastroenteritis: 10 each = 30 samples)
    supp_df = pd.DataFrame(SUPPLEMENTARY_DATA)
    
    # 3. Add synthetic outpatient and Indian-prevalent condition samples (8 classes, 406 samples)
    if SYNTHETIC_DATA_PATH.exists():
        print(f"Loading synthetic outpatient dataset from {SYNTHETIC_DATA_PATH}...")
        synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
        full_df = pd.concat([df, supp_df, synth_df], ignore_index=True)
    else:
        print("Warning: Synthetic dataset not found at", SYNTHETIC_DATA_PATH)
        full_df = pd.concat([df, supp_df], ignore_index=True)
        
    full_df = full_df.drop_duplicates(subset=["symptom_text"]).reset_index(drop=True)
    
    unique_diseases = sorted(full_df["disease"].unique())
    num_classes = len(unique_diseases)
    print(f"\nConsolidated dataset: {len(full_df)} samples across {num_classes} conditions.")
    
    # Integer label mapping
    label_to_id = {disease: idx for idx, disease in enumerate(unique_diseases)}
    id_to_label = {idx: disease for idx, disease in enumerate(unique_diseases)}
    full_df["label"] = full_df["disease"].map(label_to_id)
    
    # Save full consolidated dataset
    full_df.to_csv(BERT_DATA_DIR / "dataset.csv", index=False)
    
    # Stratified Train / Val / Test Splits (70 / 15 / 15)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.30, random_state=42, stratify=full_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    
    train_df.to_csv(BERT_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(BERT_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(BERT_DATA_DIR / "test.csv", index=False)
    
    # Save label mappings in both BERT and classical ML locations
    with open(BERT_DATA_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)
    with open(ML_DATA_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2)
        
    print(f"Splits successfully saved -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Generate updated clinical knowledge JSONs
    generate_clinical_knowledge(unique_diseases)

def generate_clinical_knowledge(diseases):
    categories = {
        # Core Benchmark Conditions
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
        "Drug Reaction": ["skin", "systemic", "fever"],
        
        # New Frequent & Indian Outpatient Conditions
        "Tension Headache": ["neurological", "musculoskeletal"],
        "Eye Strain / Screen Fatigue": ["eye", "neurological"],
        "Sinusitis": ["respiratory", "neurological"],
        "Acidity & Indigestion (Dyspepsia)": ["gastrointestinal"],
        "Viral Pharyngitis (Sore Throat)": ["respiratory", "fever"],
        "Dehydration & Heat Fatigue": ["systemic", "neurological"],
        "Mechanical Low Back Strain": ["musculoskeletal"],
        "Chikungunya": ["fever", "musculoskeletal", "skin"]
    }
    
    display_names = {
        d: {
            "friendly_name": d,
            "description": f"Preliminary presentation consistent with {d}. Clinical evaluation recommended if symptoms persist."
        } for d in diseases
    }
    
    # Specific clinical descriptions
    descriptions = {
        "Tension Headache": ("Tension-Type Headache", "Common non-severe headache presenting with dull, band-like aching across forehead or back of head. Often triggered by stress, posture, or lack of sleep."),
        "Eye Strain / Screen Fatigue": ("Eye Strain / Screen Fatigue", "Digital eye strain (asthenopia) characterized by tired, burning eyes, frontal headache, and mild blurriness after prolonged screen use."),
        "Sinusitis": ("Sinusitis (Sinus Pressure)", "Inflammation of nasal and sinus passages causing facial fullness, pressure over cheeks and forehead, nasal congestion, and head heaviness."),
        "Acidity & Indigestion (Dyspepsia)": ("Acidity & Indigestion (Dyspepsia)", "Common upper gastrointestinal irritation causing heartburn, sour burping, bloating, and stomach discomfort after spicy or oily food."),
        "Viral Pharyngitis (Sore Throat)": ("Viral Pharyngitis (Sore Throat)", "Mild upper respiratory viral irritation marked by scratchy, painful throat, difficulty swallowing, and mild malaise."),
        "Dehydration & Heat Fatigue": ("Dehydration & Heat Fatigue", "Fluid and electrolyte deficit presenting with extreme thirst, dry mouth, dizziness when standing, and fatigue after sun or heat exposure."),
        "Mechanical Low Back Strain": ("Mechanical Low Back Strain", "Muscular stiffness and aching in lower back following lifting, prolonged sitting, or bumpy commutes, without nerve radiation."),
        "Chikungunya": ("Chikungunya", "Mosquito-borne viral infection marked by acute high fever followed by severe, debilitating symmetrical joint pain and morning stiffness."),
        "Dengue": ("Dengue Fever", "Mosquito-borne viral infection characterized by sudden high fever, retro-orbital eye pain, and joint aches."),
        "Malaria": ("Malaria", "Parasitic infection transmitted by mosquitoes, typically presenting with cyclical chills, high fever, and sweating."),
        "Pneumonia": ("Pneumonia", "Acute lower respiratory infection causing lung inflammation, productive cough, fever, and difficulty breathing."),
        "Tuberculosis": ("Tuberculosis (TB)", "Chronic infectious bacterial disease mainly affecting the lungs, marked by persistent cough >3 weeks, weight loss, and night sweats."),
        "Influenza (Flu)": ("Influenza (Flu)", "Contagious respiratory illness caused by influenza viruses, marked by rapid onset fever, severe body aches, and exhaustion."),
        "Common Cold": ("Common Cold", "Mild viral upper respiratory tract infection marked by nasal congestion, sneezing, sore throat, and low-grade discomfort."),
        "Gastroenteritis": ("Gastroenteritis (Stomach Infection)", "Inflammation of the digestive tract commonly caused by viral or bacterial infection, resulting in diarrhea, vomiting, and cramps."),
        "Migraine": ("Migraine Headache", "Recurrent, pulsating moderate-to-severe headache, often unilateral, accompanied by nausea or sensitivity to light and sound.")
    }
    
    for key, (friendly, desc) in descriptions.items():
        if key in display_names:
            display_names[key]["friendly_name"] = friendly
            display_names[key]["description"] = desc

    # Comprehensive evidence-based remedies and severity levels
    remedies = {
        "Tension Headache": {
            "severity": "low",
            "home_care": [
                "Rest in a quiet, softly-lit environment.",
                "Stay well-hydrated with water or soothing herbal teas.",
                "Apply a warm or cool compress to the forehead or back of the neck.",
                "Perform gentle neck and shoulder stretching exercises.",
                "Ensure 7 to 8 hours of restful sleep."
            ],
            "seek_medical_attention_if": [
                "Headache is sudden, explosive, and unbearable ('worst headache of life').",
                "Accompanied by high fever, stiff neck, confusion, or speech difficulty.",
                "Headache progressively worsens daily for more than 7 days."
            ]
        },
        "Eye Strain / Screen Fatigue": {
            "severity": "low",
            "home_care": [
                "Follow the 20-20-20 rule: Every 20 minutes, look at an object 20 feet away for 20 seconds.",
                "Reduce screen brightness and match ambient room lighting.",
                "Blink frequently to keep eyes moist; gently splash cool water on face.",
                "Take a 15-minute screen break every 2 hours of digital work.",
                "Ensure proper viewing distance (at least an arm's length from monitor)."
            ],
            "seek_medical_attention_if": [
                "Persistent eye pain, significant blurred vision, or double vision occurs.",
                "Severe eye redness with discharge or extreme sensitivity to normal light."
            ]
        },
        "Sinusitis": {
            "severity": "low",
            "home_care": [
                "Inhale steam with warm water 2-3 times daily to loosen trapped mucus.",
                "Apply a warm, moist towel across your cheeks, eyes, and forehead.",
                "Perform gentle saline nasal rinses or saline nasal drops.",
                "Drink plenty of warm fluids (soups, herbal teas, warm water).",
                "Sleep with head slightly elevated on an extra pillow to aid drainage."
            ],
            "seek_medical_attention_if": [
                "Symptoms persist for more than 10-12 days without improvement.",
                "High fever above 102°F or severe facial swelling/redness around eyes develops.",
                "Severe headache that does not respond to simple analgesics."
            ]
        },
        "Acidity & Indigestion (Dyspepsia)": {
            "severity": "low",
            "home_care": [
                "Sip warm water or cold milk; avoid tea, coffee, and carbonated beverages.",
                "Eat smaller, more frequent meals rather than heavy dinners.",
                "Avoid spicy, oily, deeply fried, and sour foods for 48 hours.",
                "Remain upright for at least 2 to 3 hours after eating; do not lie down immediately.",
                "Elevate the head of your bed slightly if reflux worsens at night."
            ],
            "seek_medical_attention_if": [
                "Severe burning chest pain radiating to left arm, neck, or jaw (rule out cardiac causes).",
                "Difficulty or pain while swallowing food or liquids.",
                "Black tarry stools or vomiting with dark coffee-ground appearance."
            ]
        },
        "Viral Pharyngitis (Sore Throat)": {
            "severity": "low",
            "home_care": [
                "Gargle with warm salt water (1/2 teaspoon salt in a glass of warm water) 3 to 4 times a day.",
                "Sip warm honey and ginger tea or warm broths to soothe throat lining.",
                "Rest your voice and avoid speaking loudly or straining vocal cords.",
                "Stay well hydrated with ambient temperature or warm liquids.",
                "Avoid smoking, second-hand smoke, and chilled beverages."
            ],
            "seek_medical_attention_if": [
                "Severe difficulty swallowing saliva, breathing difficulty, or inability to open mouth.",
                "Persistent high fever >101.5°F or white pus patches visible on tonsils.",
                "Sore throat lasting more than 7 days without improvement."
            ]
        },
        "Dehydration & Heat Fatigue": {
            "severity": "low",
            "home_care": [
                "Immediately sip Oral Rehydration Salts (ORS) solution, tender coconut water, or salted buttermilk (chaas).",
                "Rest in a cool, shaded, or air-conditioned room.",
                "Loosen tight clothing and sponge skin with cool water.",
                "Avoid dehydrating beverages like alcohol, excessive coffee, and highly sugary energy drinks.",
                "Gradually rehydrate over 2 to 4 hours rather than gulping massive amounts at once."
            ],
            "seek_medical_attention_if": [
                "Inability to retain liquids due to continuous vomiting.",
                "Absence of urination for more than 8 hours, sunken eyes, or extreme dry tongue.",
                "Confusion, fainting spells (loss of consciousness), or core body temperature rising above 103°F (heat stroke alert)."
            ]
        },
        "Mechanical Low Back Strain": {
            "severity": "low",
            "home_care": [
                "Rest on a firm mattress in a comfortable position, ideally with knees supported by a pillow.",
                "Apply an ice pack for the first 48 hours (15 mins at a time), then switch to warm heat packs.",
                "Avoid complete bed rest beyond 24 hours; gentle short walks prevent stiffness.",
                "Avoid forward bending, twisting, and lifting weights over 5 kg.",
                "Practice gentle lumbar and hamstring stretching once acute spasm eases."
            ],
            "seek_medical_attention_if": [
                "Pain radiates down through the leg below the knee with numbness or tingling (sciatica).",
                "Loss of bladder or bowel control, or numbness in the groin/saddle area (emergency alert).",
                "Back pain accompanied by unexplained fever or history of cancer."
            ]
        },
        "Chikungunya": {
            "severity": "moderate",
            "home_care": [
                "Strict bed rest during the acute fever and joint inflammation phase.",
                "Drink abundant fluids: water, coconut water, lemon water, and clear broths.",
                "Use paracetamol for pain/fever relief under medical guidance; avoid aspirin/NSAIDs until Dengue is ruled out.",
                "Perform gentle active range-of-motion joint exercises once acute fever subsides.",
                "Use cold ice packs wrapped in towels over swollen wrists and ankles for 10-15 minutes."
            ],
            "seek_medical_attention_if": [
                "Extreme incapacitating joint swelling preventing any mobility.",
                "Severe persistent fever lasting >5 days or emergence of dark skin discolorations.",
                "Bleeding from gums or nose, or extreme drowsiness in elderly patients."
            ]
        },
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
            "home_care": [],
            "seek_medical_attention_if": ["Immediate consultation required for platelet monitoring", "Seek emergency care if bleeding gums, blood in stool, or persistent severe vomiting occurs"],
            "doctor_visit": "Consult a physician immediately for Dengue NS1/IgM and CBC blood tests."
        },
        "Malaria": {
            "severity": "high",
            "home_care": [],
            "seek_medical_attention_if": ["Prompt blood smear or rapid diagnostic test (RDT) required", "Immediate care if confusion, jaundice, or dark urine develops"],
            "doctor_visit": "Visit a clinic immediately for antimalarial prescription."
        },
        "Pneumonia": {
            "severity": "high",
            "home_care": [],
            "seek_medical_attention_if": ["Seek urgent emergency care if lips turn bluish or breathing is rapid and labored"],
            "doctor_visit": "Immediate clinical evaluation and chest imaging required."
        },
        "Tuberculosis": {
            "severity": "high",
            "home_care": [],
            "seek_medical_attention_if": ["Seek immediate medical attention for sputum AFB testing and chest X-ray"],
            "doctor_visit": "Strictly requires supervised clinical anti-tubercular therapy (DOTS)."
        }
    }
    
    with open(DATA_DIR / "disease_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    with open(DATA_DIR / "disease_display_names.json", "w", encoding="utf-8") as f:
        json.dump(display_names, f, indent=2)
    with open(DATA_DIR / "home_remedies.json", "w", encoding="utf-8") as f:
        json.dump(remedies, f, indent=2)
        
    print(f"Generated disease_categories.json, disease_display_names.json, home_remedies.json successfully for {len(diseases)} conditions.")

if __name__ == "__main__":
    consolidate_dataset()
