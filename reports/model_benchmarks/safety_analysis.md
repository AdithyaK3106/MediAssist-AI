# 🛡️ Safety Analysis Report

## Overview
This report analyzes the safety and reliability of the trained classical ML models for disease prediction.

## High-Confidence Incorrect Predictions
*Critical for healthcare safety.*
- We monitor cases where the model is highly confident (>80%) but incorrect.
- Recommendations: Implement a fallback to human doctors for low-confidence or highly ambiguous cases.

## Model Reliability
### logistic_regression
- **Accuracy**: 0.8436
- **Uncertainty Calibration**: See `reports/figures/model_analysis/logistic_regression_calibration.png`
### random_forest
- **Accuracy**: 0.8072
- **Uncertainty Calibration**: See `reports/figures/model_analysis/random_forest_calibration.png`
### naive_bayes
- **Accuracy**: 0.8051
- **Uncertainty Calibration**: See `reports/figures/model_analysis/naive_bayes_calibration.png`
### decision_tree
- **Accuracy**: 0.1527
- **Uncertainty Calibration**: See `reports/figures/model_analysis/decision_tree_calibration.png`
### xgboost
- **Accuracy**: 0.0715
- **Uncertainty Calibration**: See `reports/figures/model_analysis/xgboost_calibration.png`

## Risky Disease Overlaps
- Diseases with similar symptoms are hard to distinguish.
- See `error_analysis.md` for top confusion pairs.

## Recommendations
1.  **Always** display confidence scores to users.
2.  **Flag** predictions with high entropy as unreliable.
3.  **Do not** rely solely on the model for critical diagnoses.
