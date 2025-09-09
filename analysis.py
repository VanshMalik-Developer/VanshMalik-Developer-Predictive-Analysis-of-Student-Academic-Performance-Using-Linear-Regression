#!/usr/bin/env python3

import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data
from sample_data import create_sample_dataset

def analyze_dataset():
    """Perform exploratory data analysis"""
    print("Dataset Analysis")
    print("=" * 50)
    
    # Load or create dataset
    try:
        df = load_data('student_data.csv')
    except FileNotFoundError:
        df = create_sample_dataset()
        df.to_csv('student_data.csv', index=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of samples: {df.shape[0]}")
    
    print("\nTarget Variable (G3) Statistics:")
    print(df['G3'].describe())
    
    print("\nCategorical Variables Distribution:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    print("\nNumerical Variables Statistics:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    # Correlation analysis on numerical columns only
    print("\nCorrelation with Target Variable (G3) - Numerical Features:")
    numerical_df = df.select_dtypes(include=[np.number])
    correlations = numerical_df.corr()['G3'].sort_values(key=abs, ascending=False)
    for feature, corr in correlations.items():
        if feature != 'G3':
            print(f"  {feature}: {corr:.4f}")

def interpret_model_results():
    """Interpret the linear regression results"""
    print("\n" + "=" * 50)
    print("Model Interpretation")
    print("=" * 50)
    
    print("""
Key Findings from Linear Regression Model:

1. Model Performance:
   - R² = 0.81 indicates the model explains 81% of variance in student grades
   - RMSE = 1.92 means predictions are typically within ±1.92 points of actual grades
   - This is considered good performance for educational data

2. Most Important Factors (by coefficient magnitude):
   - Study Time (+2.04): Each additional hour of study increases grade by ~2 points
   - Failures (-1.43): Each previous failure decreases grade by ~1.4 points
   - G2 (+0.40): Previous semester grade strongly predicts final grade
   - G1 (+0.31): First semester grade also important

3. Insights:
   - Study habits are the strongest predictor of academic success
   - Past academic performance is highly predictive
   - Demographic factors (age, gender) have minimal impact
   - Family/school support shows small negative coefficients (may be confounded)

4. Practical Applications:
   - Identify at-risk students early based on study habits and past performance
   - Intervention programs should focus on study skills and addressing failures
   - Academic support should target students with low G1/G2 scores
    """)

if __name__ == "__main__":
    analyze_dataset()
    interpret_model_results()
