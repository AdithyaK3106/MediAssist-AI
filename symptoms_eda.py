import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = "data/raw/symptoms"
FIGURES_DIR = "reports/figures"
EDA_DIR = "reports/eda"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

def analyze_symptoms():
    logger.info("Starting Detailed Symptoms EDA...")
    
    # Load
    df = pd.read_csv(os.path.join(RAW_DIR, "final_symptoms_to_disease.csv"))
    
    # 1. Disease Distribution & Class Imbalance
    disease_counts = df['diseases'].value_counts()
    
    plt.figure(figsize=(15, 8))
    disease_counts[:30].plot(kind='bar', color='skyblue')
    plt.title("Top 30 Diseases by Frequency", fontsize=15)
    plt.xlabel("Disease", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "symptoms_disease_dist.png"), dpi=300)
    plt.close()
    
    # Rare diseases
    rare_diseases = disease_counts[disease_counts < 5]
    logger.info(f"Found {len(rare_diseases)} rare diseases (count < 5)")
    
    # 2. Symptom Analysis
    all_symptoms = []
    df['symptom_list'] = df['symptom_text'].apply(lambda x: [s.strip().lower() for s in str(x).split(',')])
    for s_list in df['symptom_list']:
        all_symptoms.extend(s_list)
    
    symptom_counts = Counter(all_symptoms)
    symptom_df = pd.DataFrame(symptom_counts.most_common(30), columns=['Symptom', 'Count'])
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=symptom_df, x='Count', y='Symptom', palette='viridis')
    plt.title("Top 30 Symptoms by Frequency", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "symptoms_top_symptoms.png"), dpi=300)
    plt.close()
    
    # 3. Symptom Co-occurrence Matrix (Top 20 symptoms)
    top_symptoms = [s for s, c in symptom_counts.most_common(20)]
    co_matrix = np.zeros((20, 20))
    for s_list in df['symptom_list']:
        for i in range(len(top_symptoms)):
            for j in range(len(top_symptoms)):
                if top_symptoms[i] in s_list and top_symptoms[j] in s_list:
                    co_matrix[i, j] += 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, xticklabels=top_symptoms, yticklabels=top_symptoms, annot=True, fmt='g', cmap='YlGnBu')
    plt.title("Symptom Co-occurrence Matrix (Top 20)", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "symptoms_cooccurrence.png"), dpi=300)
    plt.close()
    
    # 4. Disease-Symptom Overlap
    # Analyze how many symptoms are shared between diseases
    disease_symptom_map = df.groupby('diseases')['symptom_list'].apply(lambda x: set().union(*x)).to_dict()
    
    diseases = list(disease_symptom_map.keys())
    overlap_matrix = np.zeros((len(diseases[:20]), len(diseases[:20])))
    for i in range(len(diseases[:20])):
        for j in range(len(diseases[:20])):
            s1 = disease_symptom_map[diseases[i]]
            s2 = disease_symptom_map[diseases[j]]
            if len(s1.union(s2)) > 0:
                overlap_matrix[i, j] = len(s1.intersection(s2)) / len(s1.union(s2))
                
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_matrix, xticklabels=diseases[:20], yticklabels=diseases[:20], annot=True, fmt='.2f', cmap='Reds')
    plt.title("Jaccard Similarity between Diseases (Top 20)", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "symptoms_disease_similarity.png"), dpi=300)
    plt.close()
    
    # 5. Summary Report
    report = f"""# Symptoms Dataset EDA Report

## Dataset Overview
- **Total Records**: {len(df)}
- **Unique Diseases**: {len(disease_counts)}
- **Unique Symptoms Identified**: {len(symptom_counts)}

## Key Findings
- **Class Imbalance**: The dataset is {'highly' if disease_counts.max() / disease_counts.min() > 10 else 'moderately'} imbalanced.
- **Top Disease**: {disease_counts.index[0]} ({disease_counts.iloc[0]} records)
- **Top Symptom**: {symptom_df.iloc[0]['Symptom']} ({symptom_df.iloc[0]['Count']} occurrences)
- **Rare Diseases**: {len(rare_diseases)} diseases have fewer than 5 samples.
- **Average Symptoms per Disease**: {df['symptom_list'].apply(len).mean():.2f}

## Data Quality Issues
- **Ambiguity**: Several diseases show high Jaccard similarity in their symptom sets, which may affect model precision.
- **Sparsity**: Many symptoms appear in very few samples.

## Recommendations for ML
- Use **Top-K prediction** to handle disease overlap.
- Consider **oversampling** or **class weights** for rare diseases.
- Use **TF-IDF** to penalize common symptoms like '{symptom_df.iloc[0]['Symptom']}'.
"""
    with open(os.path.join(EDA_DIR, "symptoms_eda_report.md"), "w") as f:
        f.write(report)
    
    logger.info("Symptoms EDA Complete.")

if __name__ == "__main__":
    analyze_symptoms()
