# Conversations Dataset EDA Report

## Dataset Overview
- **Total Conversations**: 112165
- **Estimated Vocabulary Size**: 29098 (from 10k samples)
- **Average Query Length**: 84.14 words
- **Average Response Length**: 109.56 words

## Quality Analysis
- **Empty Inputs**: 1
- **Empty Responses**: 0
- **Duplicate Conversations**: 0
- **Potential Noise**: Found informal language, typos, and repeated phrases.

## Suitability for LLM/BERT
- **DistilBERT**: High suitability due to diverse patient descriptions.
- **Qwen2.5**: Ideal for fine-tuning as it contains direct instruction-input-output pairs.
- **NLP Challenges**: Medical shorthand and irregular grammar in patient queries.

## Recommendations
- Remove duplicate and empty entries.
- Use **Masked Language Modeling (MLM)** pre-training on this corpus before classification.
- Preserve punctuation and sentence structure for Transformer-based models.
