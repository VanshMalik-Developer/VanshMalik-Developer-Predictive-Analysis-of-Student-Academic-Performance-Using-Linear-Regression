#!/usr/bin/env python3

import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data, split_data
from model import StudentPerformanceModel
from evaluation import evaluate_model, print_metrics
from sample_data import create_sample_dataset

def main():
    print("Student Academic Performance Prediction using Linear Regression")
    print("=" * 60)
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    df.to_csv('student_data.csv', index=False)
    print("Sample dataset created and saved")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable (G3) range: {df['G3'].min():.2f} - {df['G3'].max():.2f}")
    
    # Show first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_processed)
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = StudentPerformanceModel()
    model.train(X_train, y_train)
    print("Model trained successfully")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)
    print("\n" + "="*40)
    print_metrics(metrics)
    print("="*40)
    
    # Get model insights
    coef_info = model.get_coefficients()
    print(f"\nModel Intercept: {coef_info['intercept']:.4f}")
    
    # Show feature coefficients
    print("\nFeature Coefficients:")
    if coef_info['feature_names'] is not None:
        for name, coef in zip(coef_info['feature_names'], coef_info['coefficients']):
            print(f"  {name}: {coef:.4f}")
    
    # Show sample predictions
    print("\nSample Predictions (first 10):")
    print("Actual -> Predicted")
    for i in range(min(10, len(y_test))):
        print(f"{y_test.iloc[i]:.2f} -> {y_pred[i]:.2f}")
    
    print("\nProject completed successfully!")
    print("Note: Run 'python main.py' for full version with visualizations")

if __name__ == "__main__":
    main()
