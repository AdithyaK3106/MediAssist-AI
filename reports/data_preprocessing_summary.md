# MediAssist AI Data Preprocessing Summary

## Dataset Statistics
- **Symptoms (ML Ready)**: 163727 rows
- **Conversations**: 112164 rows
- **Hospitals**: 30273 entries
- **Rural Health Facilities**: 55 entries

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
