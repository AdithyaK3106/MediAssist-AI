# Symptoms Dataset EDA Report

## Dataset Overview
- **Total Records**: 192715
- **Unique Diseases**: 254
- **Unique Symptoms Identified**: 283

## Key Findings
- **Class Imbalance**: The dataset is moderately imbalanced.
- **Top Disease**: cystitis (1219 records)
- **Top Symptom**: sharp abdominal pain (25857 occurrences)
- **Rare Diseases**: 0 diseases have fewer than 5 samples.
- **Average Symptoms per Disease**: 5.61

## Data Quality Issues
- **Ambiguity**: Several diseases show high Jaccard similarity in their symptom sets, which may affect model precision.
- **Sparsity**: Many symptoms appear in very few samples.

## Recommendations for ML
- Use **Top-K prediction** to handle disease overlap.
- Consider **oversampling** or **class weights** for rare diseases.
- Use **TF-IDF** to penalize common symptoms like 'sharp abdominal pain'.
