import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_correlation_matrix(df):
    """Plot correlation heatmap using matplotlib"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    
    # Create heatmap manually
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    
    # Add labels
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Grades')
    plt.ylabel('Predicted Grades')
    plt.title('Actual vs Predicted Student Grades')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(coefficients, feature_names):
    """Plot feature importance based on coefficients"""
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(coefficients))]
    
    plt.figure(figsize=(10, 6))
    
    # Sort by absolute coefficient value
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    sorted_coef = coefficients[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    plt.barh(range(len(sorted_coef)), sorted_coef)
    plt.yticks(range(len(sorted_coef)), sorted_names)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance (Linear Regression Coefficients)')
    plt.tight_layout()
    plt.show()
