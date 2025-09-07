# Model Optimization and Feature Analysis - Summary Report

## Analysis Overview

- **Dataset Size**: 101 observations
- **Significant Features Found**: 2
- **Best Model Performance**: R² = 0.2052

## Key Findings

### Top Predictive Features

1. **Cvd Mortality**: r = -0.5149, p = 0.0000
2. **Temperature Mean**: r = -0.1661, p = 0.0986

### Model Performance Comparison

- **Linear Regression**: CV R² = 0.2015 (±0.1737)
- **Ridge Regression**: CV R² = 0.2016 (±0.1735)
- **Elastic Net**: CV R² = 0.2052 (±0.1557)
- **Random Forest**: CV R² = 0.1384 (±0.1577)
- **Gradient Boosting**: CV R² = 0.0977 (±0.1731)

### Blue Zone Scoring Results

- **High Potential Regions** (Score > 80): 32
- **Mean Blue Zone Score**: 69.7
- **Score Range**: 0.0 - 100.0

## Files Generated

- optimized_longevity_analysis.csv
- model_comparison_results.csv
- feature_importance_analysis.csv

## Methodology

- **Correlation Analysis**: Pearson correlation with significance testing
- **Model Optimization**: Grid search with cross-validation
- **Feature Importance**: Model-specific importance extraction
- **Blue Zone Scoring**: Multi-criteria scoring system

**Analysis Date**: 2025-09-06 15:27:57
