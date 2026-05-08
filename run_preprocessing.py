import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.pipeline import PreprocessingPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "ml"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "bert"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "hospitals"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "evaluation"), exist_ok=True)

pipeline = PreprocessingPipeline()

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 failed for {path}, trying cp1252")
        return pd.read_csv(path, encoding='cp1252')

def process_symptoms():
    logger.info("Processing Symptoms Dataset...")
    
    # 1. Load datasets
    df_data = read_csv_safe(os.path.join(RAW_DIR, "symptoms/data.csv"))
    df_final = read_csv_safe(os.path.join(RAW_DIR, "symptoms/final_symptoms_to_disease.csv"))
    
    # 2. Use Pipeline
    df_cleaned = pipeline.process_symptoms_df(df_final, text_col='symptom_text', label_col='diseases')
    
    # 3. EDA for Symptoms
    plt.figure(figsize=(12, 6))
    df_cleaned['diseases_standardized'].value_counts()[:20].plot(kind='bar')
    plt.title("Top 20 Diseases by Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "disease_frequency.png"))
    plt.close()
    
    # 4. Save
    df_cleaned.to_csv(os.path.join(PROCESSED_DIR, "bert/symptoms_bert_ready.csv"), index=False)
    
    # Classical ML (more aggressive cleaning)
    df_ml = df_cleaned.copy()
    df_ml['symptom_text_cleaned'] = df_ml['symptom_text_cleaned'].apply(lambda x: pipeline.clean_text_field(x, preserve_sentence=False))
    df_ml.to_csv(os.path.join(PROCESSED_DIR, "ml/symptoms_cleaned.csv"), index=False)
    
    # Wide data cleaning
    df_data = pipeline.process_hospital_df(df_data) # Using hospital logic for column cleaning
    df_data.to_csv(os.path.join(PROCESSED_DIR, "ml/symptoms_wide_cleaned.csv"), index=False)
    
    logger.info("Symptoms processing complete.")

def process_conversations():
    logger.info("Processing Conversations Dataset...")
    
    # 1. Load JSON
    with open(os.path.join(RAW_DIR, "conversations/HealthCareMagic-100k.json"), 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 2. Clean
    df = df[df['input'].str.strip() != ""]
    df = df[df['output'].str.strip() != ""]
    
    df['input_cleaned'] = df['input'].apply(lambda x: pipeline.clean_text_field(x, preserve_sentence=True))
    df['output_cleaned'] = df['output'].apply(lambda x: pipeline.clean_text_field(x, preserve_sentence=True))
    
    # 3. EDA
    df['query_length'] = df['input'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['query_length'], bins=50, kde=True)
    plt.title("Query Length Distribution (Words)")
    plt.savefig(os.path.join(FIGURES_DIR, "query_length_dist.png"))
    plt.close()
    
    # 4. Save
    df.to_csv(os.path.join(PROCESSED_DIR, "bert/conversations_cleaned.csv"), index=False)
    logger.info("Conversations processing complete.")

def process_hospitals():
    logger.info("Processing Hospital Datasets...")
    
    # 1. Load
    df_hosp = read_csv_safe(os.path.join(RAW_DIR, "hospitals/hospital_directory.csv"))
    df_rural = read_csv_safe(os.path.join(RAW_DIR, "rural_health/Health_Care_Facilities_Tiruppur_0.csv"))
    
    # 2. Clean
    df_hosp_cleaned = pipeline.process_hospital_df(df_hosp)
    df_rural_cleaned = pipeline.process_hospital_df(df_rural)
    
    # 3. EDA Hospitals
    # The column is 'state' after pipeline cleaning it becomes 'state'
    if 'state' in df_hosp_cleaned.columns:
        plt.figure(figsize=(14, 7))
        df_hosp_cleaned['state'].value_counts().plot(kind='bar')
        plt.title("Hospital Distribution by State")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "hospital_state_dist.png"))
        plt.close()
    
    # Missing value heatmap for hospitals
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_hosp_cleaned.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Values Heatmap - Hospitals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hospital_missing_values.png"))
    plt.close()
        
    # 4. Save
    df_hosp_cleaned.to_csv(os.path.join(PROCESSED_DIR, "hospitals/hospitals_cleaned.csv"), index=False)
    df_rural_cleaned.to_csv(os.path.join(PROCESSED_DIR, "hospitals/rural_health_cleaned.csv"), index=False)
    
    logger.info("Hospital processing complete.")

def generate_summary_report():
    logger.info("Generating Final Summary Report...")
    # This will create a markdown file with dataset stats
    stats = {
        "symptoms": len(pd.read_csv(os.path.join(PROCESSED_DIR, "ml/symptoms_cleaned.csv"))),
        "conversations": len(pd.read_csv(os.path.join(PROCESSED_DIR, "bert/conversations_cleaned.csv"))),
        "hospitals": len(pd.read_csv(os.path.join(PROCESSED_DIR, "hospitals/hospitals_cleaned.csv"))),
        "rural_facilities": len(pd.read_csv(os.path.join(PROCESSED_DIR, "hospitals/rural_health_cleaned.csv")))
    }
    
    report = f"""# MediAssist AI Data Preprocessing Summary

## Dataset Statistics
- **Symptoms (ML Ready)**: {stats['symptoms']} rows
- **Conversations**: {stats['conversations']} rows
- **Hospitals**: {stats['hospitals']} entries
- **Rural Health Facilities**: {stats['rural_facilities']} entries

## EDA Visualizations
Visualizations have been saved to `reports/figures/`:
- `disease_frequency.png`: Top 20 diseases in the symptoms dataset.
- `query_length_dist.png`: Distribution of patient query lengths.
- `hospital_state_dist.png`: Geographical distribution of hospitals.

## Data Quality Notes
- Duplicates were removed from all datasets.
- Text was normalized (lower-cased, cleaned) while preserving sentence structure for BERT where applicable.
- Disease labels were standardized to Title Case.
- Hospital and Rural health data column names were standardized to snake_case.
"""
    with open(os.path.join(REPORTS_DIR, "data_preprocessing_summary.md"), "w") as f:
        f.write(report)
    logger.info("Summary report generated.")

if __name__ == "__main__":
    process_symptoms()
    process_conversations()
    process_hospitals()
    generate_summary_report()
