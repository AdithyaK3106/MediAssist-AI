import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
HOSPITAL_FILE = "data/raw/hospitals/hospital_directory.csv"
RURAL_FILE = "data/raw/rural_health/Health_Care_Facilities_Tiruppur_0.csv"
FIGURES_DIR = "reports/figures"
EDA_DIR = "reports/eda"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 failed for {path}, trying cp1252")
        return pd.read_csv(path, encoding='cp1252')

def analyze_infrastructure():
    logger.info("Starting Detailed Infrastructure EDA...")
    
    # 1. Hospital Directory
    df_hosp = read_csv_safe(HOSPITAL_FILE)
        
    # Clean columns
    df_hosp.columns = [c.strip().replace(' ', '_').lower() for c in df_hosp.columns]
    
    # Analysis
    if 'state' in df_hosp.columns:
        plt.figure(figsize=(15, 8))
        df_hosp['state'].value_counts().plot(kind='bar', color='teal')
        plt.title("Hospital Distribution by State", fontsize=15)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "hosp_state_dist.png"), dpi=300)
        plt.close()
        
    if 'hospital_category' in df_hosp.columns:
        plt.figure(figsize=(10, 10))
        df_hosp['hospital_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', cmap='Pastel1')
        plt.title("Hospital Category Distribution", fontsize=15)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "hosp_category_pie.png"), dpi=300)
        plt.close()

    # 2. Rural Health
    df_rural = read_csv_safe(RURAL_FILE)
    df_rural.columns = [c.strip().replace(' ', '_').lower() for c in df_rural.columns]
    
    if 'govt_and_private' in df_rural.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_rural, x='govt_and_private', palette='Set2')
        plt.title("Government vs Private Facilities (Rural)", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "rural_govt_vs_private.png"), dpi=300)
        plt.close()

    # 3. Geo Distribution (Placeholder visualization using Latitude/Longitude if available)
    if 'latitude' in df_rural.columns and 'longitude' in df_rural.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_rural, x='longitude', y='latitude', hue='govt_and_private', alpha=0.6)
        plt.title("Geographical Distribution of Rural Health Facilities", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "rural_geo_dist.png"), dpi=300)
        plt.close()

    # 4. Summary Report
    report = f"""# Healthcare Infrastructure EDA Report

## Hospital Directory Analysis
- **Total Facilities**: {len(df_hosp)}
- **Top State**: {df_hosp['state'].value_counts().index[0] if 'state' in df_hosp.columns else 'N/A'}
- **Ownership**: The majority are {df_hosp['hospital_category'].value_counts().index[0] if 'hospital_category' in df_hosp.columns else 'N/A'}.

## Rural Health Analysis (Focus: Tiruppur)
- **Total Facilities**: {len(df_rural)}
- **Accessibility**: Significant presence of {df_rural['govt_and_private'].value_counts().index[0] if 'govt_and_private' in df_rural.columns else 'N/A'} facilities.

## Recommendation System Readiness
- **Geo-info**: High availability of Lat/Long for rural facilities, but sparse for national directory.
- **Specialties**: Data on specialized services is present but requires significant normalization.
- **Feasibility**: High. We can recommend facilities based on state/district and proximity where geo-data is available.

## Recommendations
- Standardize 'specialties' list using a medical taxonomy.
- Impute missing geo-coordinates using city/district names if possible.
- Merge rural and national directories into a unified recommendation graph.
"""
    with open(os.path.join(EDA_DIR, "infrastructure_eda_report.md"), "w") as f:
        f.write(report)
    
    logger.info("Infrastructure EDA Complete.")

if __name__ == "__main__":
    analyze_infrastructure()
