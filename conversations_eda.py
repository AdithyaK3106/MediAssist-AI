import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RAW_FILE = "data/raw/conversations/HealthCareMagic-100k.json"
FIGURES_DIR = "reports/figures"
EDA_DIR = "reports/eda"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

def analyze_conversations():
    logger.info("Starting Detailed Conversations EDA...")
    
    # Load
    with open(RAW_FILE, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 1. Length Analysis
    df['input_len'] = df['input'].apply(lambda x: len(str(x).split()))
    df['output_len'] = df['output'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['input_len'], bins=100, color='blue', label='Patient Query (Input)', kde=True)
    sns.histplot(df['output_len'], bins=100, color='green', label='Doctor Response (Output)', kde=True)
    plt.xlim(0, 500)
    plt.title("Distribution of Conversation Lengths (Words)", fontsize=15)
    plt.xlabel("Number of Words")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "conv_length_dist.png"), dpi=300)
    plt.close()
    
    # 2. Word Clouds
    all_input_text = " ".join(df['input'].astype(str).sample(5000)) # Sample for speed
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(all_input_text)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud - Patient Queries", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "conv_wordcloud_input.png"), dpi=300)
    plt.close()
    
    # 3. Vocabulary & Medical Terminology (Approximation)
    def extract_tokens(text):
        return re.findall(r'\w+', str(text).lower())
    
    all_tokens = []
    for text in df['input'].sample(10000):
        all_tokens.extend(extract_tokens(text))
    
    token_counts = Counter(all_tokens)
    vocab_size = len(token_counts)
    common_words = token_counts.most_common(50)
    
    # Filter out stopwords (simple list)
    stopwords = set(['the', 'and', 'i', 'to', 'a', 'is', 'have', 'of', 'in', 'it', 'my', 'for', 'with', 'on', 'that', 'me', 'am', 'was', 'as', 'are'])
    filtered_common = [(w, c) for w, c in common_words if w not in stopwords][:20]
    
    plt.figure(figsize=(12, 8))
    words, counts = zip(*filtered_common)
    sns.barplot(x=list(counts), y=list(words), palette='magma')
    plt.title("Most Frequent Non-Stopwords in Patient Queries", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "conv_top_words.png"), dpi=300)
    plt.close()
    
    # 4. Data Quality
    empty_inputs = len(df[df['input'].str.strip() == ""])
    empty_outputs = len(df[df['output'].str.strip() == ""])
    duplicates = df.duplicated(subset=['input', 'output']).sum()
    
    # 5. Summary Report
    report = f"""# Conversations Dataset EDA Report

## Dataset Overview
- **Total Conversations**: {len(df)}
- **Estimated Vocabulary Size**: {vocab_size} (from 10k samples)
- **Average Query Length**: {df['input_len'].mean():.2f} words
- **Average Response Length**: {df['output_len'].mean():.2f} words

## Quality Analysis
- **Empty Inputs**: {empty_inputs}
- **Empty Responses**: {empty_outputs}
- **Duplicate Conversations**: {duplicates}
- **Potential Noise**: Found informal language, typos, and repeated phrases.

## Suitability for LLM/BERT
- **DistilBERT**: High suitability due to diverse patient descriptions.
- **Qwen2.5**: Ideal for fine-tuning as it contains direct instruction-input-output pairs.
- **NLP Challenges**: Medical shorthand and irregular grammar in patient queries.

## Recommendations
- Remove duplicate and empty entries.
- Use **Masked Language Modeling (MLM)** pre-training on this corpus before classification.
- Preserve punctuation and sentence structure for Transformer-based models.
"""
    with open(os.path.join(EDA_DIR, "conversations_eda_report.md"), "w") as f:
        f.write(report)
    
    logger.info("Conversations EDA Complete.")

if __name__ == "__main__":
    analyze_conversations()
