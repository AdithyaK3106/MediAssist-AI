import re
import string
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str, preserve_sentence: bool = False) -> str:
    """
    Standard text cleaning for medical queries.
    
    Args:
        text: The input text string.
        preserve_sentence: If True, keeps punctuation like periods for BERT context.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not preserve_sentence:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
    else:
        # Keep sentence structure but remove weird characters
        text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
        
    return text

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in a dataframe.
    """
    if strategy == 'drop':
        return df.dropna().reset_index(drop=True)
    elif strategy == 'fill_empty':
        return df.fillna("").reset_index(drop=True)
    return df

def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    logger.info(f"Removed {initial_len - len(df)} duplicates.")
    return df
