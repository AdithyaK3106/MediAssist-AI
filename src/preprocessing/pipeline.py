import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from .text_processor import clean_text, handle_missing_values, remove_duplicates
from .symptom_processor import normalize_symptom, standardize_disease_label

class PreprocessingPipeline:
    """Modular preprocessing for medical symptom data."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.info("Initializing Preprocessing Pipeline")

    def clean_text_field(self, text: str, preserve_sentence: bool = False) -> str:
        """Apply medical-specific text cleaning."""
        return clean_text(text, preserve_sentence=preserve_sentence)

    def normalize_symptoms_list(self, symptoms: List[str]) -> List[str]:
        """Map user-provided symptoms to standard medical terminology."""
        return [normalize_symptom(s) for s in symptoms]

    def process_symptoms_df(self, df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> pd.DataFrame:
        """Process a dataframe containing symptoms and optionally labels."""
        logger.info(f"Processing symptoms in column: {text_col}")
        
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        
        # Clean text
        df[f"{text_col}_cleaned"] = df[text_col].apply(lambda x: self.clean_text_field(x, preserve_sentence=True))
        
        if label_col and label_col in df.columns:
            df[f"{label_col}_standardized"] = df[label_col].apply(standardize_disease_label)
            
        return df

    def process_hospital_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize hospital directory columns and data."""
        logger.info("Processing hospital directory")
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
        df = remove_duplicates(df)
        
        # Capitalize names for consistency
        # Adjusted for columns: state, district, hospital_name
        for col in ['state', 'district', 'hospital_name']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.title()
        
        return df
