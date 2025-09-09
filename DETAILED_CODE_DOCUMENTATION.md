# Detailed Code Documentation
## Student Academic Performance Prediction Project

This document provides line-by-line explanations of all files in the project.

---

## 1. **app.py** - Flask Web Application

### Purpose
Main web application that provides an interactive interface for predicting student grades.

### Line-by-Line Breakdown

```python
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import StudentPerformanceModel
from data_preprocessing import preprocess_data, split_data
from sample_data import create_sample_dataset
```
**Lines 1-6**: Import necessary libraries and custom modules
- Flask for web framework
- pandas/numpy for data handling
- Custom modules for ML functionality

```python
app = Flask(__name__)
```
**Line 8**: Create Flask application instance

```python
def load_model():
    try:
        df = pd.read_csv('student_data.csv')
    except FileNotFoundError:
        df = create_sample_dataset()
        df.to_csv('student_data.csv', index=False)
```
**Lines 11-16**: Load existing dataset or create sample data if file doesn't exist

```python
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    model = StudentPerformanceModel()
    model.train(X_train, y_train)
    return model, X_train.columns.tolist()
```
**Lines 18-23**: Process data, split into train/test sets, train model, return trained model and feature names

```python
@app.route('/')
def home():
    return render_template('index.html')
```
**Lines 27-29**: Define home route that renders the main HTML template

```python
@app.route('/predict', methods=['POST'])
def predict():
```
**Lines 31-32**: Define prediction endpoint that accepts POST requests

```python
    data = {
        'age': int(request.form['age']),
        'gender': 1 if request.form['gender'] == 'M' else 0,
        'study_time': int(request.form['study_time']),
        # ... more features
    }
```
**Lines 35-48**: Extract form data and convert to numerical format (gender: M=1, F=0, yes/no fields: yes=1, no=0)

```python
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(20, prediction))
```
**Lines 51-53**: Create DataFrame from input, make prediction, clamp result to valid grade range (0-20)

```python
    coefficients = model.get_coefficients()['coefficients']
    feature_impacts = []
```
**Lines 56-57**: Get model coefficients to calculate feature impacts

```python
    for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
        feature_value = list(data.values())[i]
        contribution = coef * feature_value
```
**Lines 61-63**: Calculate each feature's contribution to the prediction (coefficient × value)

```python
        if feature == 'gender':
            display_value = 'Male' if feature_value == 1 else 'Female'
            interpretation = f"Being {display_value}"
```
**Lines 66-68**: Convert numerical values back to meaningful text for display

```python
        feature_impacts.append({
            'feature': interpretation,
            'coefficient': round(coef, 3),
            'value': feature_value,
            'impact': round(contribution, 3),
            'abs_impact': abs(contribution)
        })
```
**Lines 90-96**: Store feature impact information for ranking

```python
    feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
    top_features = feature_impacts[:5]
```
**Lines 99-102**: Sort by absolute impact and get top 5 most influential features

```python
    return jsonify({
        'success': True,
        'prediction': round(prediction, 2),
        'grade_letter': get_letter_grade(prediction),
        'feature_impacts': top_features,
        'confidence': calculate_confidence(prediction)
    })
```
**Lines 104-110**: Return JSON response with prediction results

---

## 2. **model.py** - Linear Regression Model

### Purpose
Implements the core machine learning model using scikit-learn's LinearRegression.

### Line-by-Line Breakdown

```python
from sklearn.linear_model import LinearRegression
import numpy as np
```
**Lines 1-2**: Import LinearRegression algorithm and numpy

```python
class StudentPerformanceModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
```
**Lines 4-7**: Define model class with LinearRegression instance and feature names storage

```python
    def train(self, X_train, y_train):
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        self.model.fit(X_train, y_train)
```
**Lines 9-12**: Train method that stores feature names and fits the model to training data

```python
    def predict(self, X_test):
        return self.model.predict(X_test)
```
**Lines 14-16**: Prediction method that returns model predictions

```python
    def get_coefficients(self):
        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_,
            'feature_names': self.feature_names
        }
```
**Lines 18-23**: Return model parameters (intercept, coefficients, feature names) for interpretation

---

## 3. **data_preprocessing.py** - Data Processing Functions

### Purpose
Handles data loading, cleaning, and preparation for machine learning.

### Line-by-Line Breakdown

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```
**Lines 1-3**: Import data processing libraries

```python
def load_data(file_path):
    return pd.read_csv(file_path)
```
**Lines 5-7**: Simple function to load CSV data into pandas DataFrame

```python
def preprocess_data(df):
    df = df.dropna()
```
**Lines 9-11**: Remove rows with missing values

```python
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
```
**Lines 13-17**: Convert categorical text columns to numerical values using LabelEncoder

```python
def split_data(df, target_col='G3', test_size=0.2):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)
```
**Lines 21-26**: Split dataset into features (X) and target (y), then into train/test sets (80%/20% split)

---

## 4. **evaluation.py** - Model Evaluation Metrics

### Purpose
Calculates and displays model performance metrics.

### Line-by-Line Breakdown

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
```
**Lines 1-2**: Import evaluation metrics from scikit-learn

```python
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
```
**Lines 4-9**: Calculate four key regression metrics:
- MAE: Average absolute difference between actual and predicted
- MSE: Average squared difference
- RMSE: Square root of MSE (same units as target)
- R²: Proportion of variance explained by model

```python
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
```
**Lines 11-16**: Return metrics as dictionary

```python
def print_metrics(metrics):
    print("Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R-squared (R²): {metrics['R2']:.4f}")
```
**Lines 18-24**: Format and display metrics with 4 decimal places

---

## 5. **sample_data.py** - Sample Dataset Generator

### Purpose
Creates synthetic student data for testing when real dataset is unavailable.

### Line-by-Line Breakdown

```python
import pandas as pd
import numpy as np
```
**Lines 1-2**: Import data manipulation libraries

```python
def create_sample_dataset():
    np.random.seed(42)
    n_samples = 1000
```
**Lines 4-6**: Set random seed for reproducibility and define sample size

```python
    data = {
        'age': np.random.randint(15, 23, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'study_time': np.random.randint(1, 5, n_samples),
        'failures': np.random.randint(0, 4, n_samples),
        # ... more features
    }
```
**Lines 8-20**: Generate random values for each feature:
- age: 15-22 years
- gender: Male/Female randomly
- study_time: 1-4 scale
- failures: 0-3 previous failures
- Binary features: 0 or 1 randomly

```python
    G3 = (data['study_time'] * 2 + 
          data['G1'] * 0.3 + 
          data['G2'] * 0.4 - 
          data['failures'] * 1.5 + 
          np.random.normal(0, 2, n_samples))
```
**Lines 23-27**: Generate target variable (G3) based on realistic relationships:
- Study time has positive impact (×2)
- Previous grades (G1, G2) are predictive
- Failures have negative impact (-1.5)
- Add random noise for realism

```python
    data['G3'] = np.clip(G3, 0, 20)
    return pd.DataFrame(data)
```
**Lines 29-31**: Clamp grades to valid range (0-20) and return as DataFrame

---

## 6. **main_simple.py** - Command Line Demo

### Purpose
Provides a simple command-line interface to run the complete ML pipeline.

### Line-by-Line Breakdown

```python
#!/usr/bin/env python3
```
**Line 1**: Shebang for direct execution on Unix systems

```python
import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_data, split_data
from model import StudentPerformanceModel
from evaluation import evaluate_model, print_metrics
from sample_data import create_sample_dataset
```
**Lines 3-8**: Import all necessary modules

```python
def main():
    print("Student Academic Performance Prediction using Linear Regression")
    print("=" * 60)
```
**Lines 10-12**: Main function with header display

```python
    df = create_sample_dataset()
    df.to_csv('student_data.csv', index=False)
    print("Sample dataset created and saved")
```
**Lines 15-17**: Create and save sample dataset

```python
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable (G3) range: {df['G3'].min():.2f} - {df['G3'].max():.2f}")
```
**Lines 19-20**: Display basic dataset information

```python
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
```
**Lines 23-24**: Show sample data

```python
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
```
**Lines 27-28**: Process and split data

```python
    model = StudentPerformanceModel()
    model.train(X_train, y_train)
    print("Model trained successfully")
```
**Lines 33-35**: Create, train, and confirm model training

```python
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print_metrics(metrics)
```
**Lines 38-41**: Make predictions and evaluate performance

```python
    coef_info = model.get_coefficients()
    print(f"\nModel Intercept: {coef_info['intercept']:.4f}")
```
**Lines 44-45**: Display model intercept

```python
    if coef_info['feature_names'] is not None:
        for name, coef in zip(coef_info['feature_names'], coef_info['coefficients']):
            print(f"  {name}: {coef:.4f}")
```
**Lines 48-50**: Display feature coefficients with names

```python
    print("\nSample Predictions (first 10):")
    print("Actual -> Predicted")
    for i in range(min(10, len(y_test))):
        print(f"{y_test.iloc[i]:.2f} -> {y_pred[i]:.2f}")
```
**Lines 53-56**: Show sample predictions vs actual values

---

## 7. **visualization.py** - Data Visualization Functions

### Purpose
Creates plots and charts for data analysis and model interpretation.

### Line-by-Line Breakdown

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```
**Lines 1-3**: Import plotting libraries

```python
def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
```
**Lines 5-7**: Create correlation heatmap - calculate correlation matrix

```python
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
```
**Lines 10-11**: Display matrix as image with color scale

```python
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
```
**Lines 14-15**: Add feature names as axis labels

```python
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
```
**Lines 21-24**: Create scatter plot of actual vs predicted values with perfect prediction line

```python
def plot_feature_importance(coefficients, feature_names):
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    sorted_coef = coefficients[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
```
**Lines 30-33**: Sort features by absolute coefficient value (importance)

```python
    plt.barh(range(len(sorted_coef)), sorted_coef)
    plt.yticks(range(len(sorted_coef)), sorted_names)
```
**Lines 35-36**: Create horizontal bar chart of feature importance

---

## 8. **analysis.py** - Exploratory Data Analysis

### Purpose
Performs comprehensive data analysis and provides model interpretation.

### Line-by-Line Breakdown

```python
def analyze_dataset():
    print("Dataset Analysis")
    print("=" * 50)
```
**Lines 7-9**: Function header for dataset analysis

```python
    try:
        df = load_data('student_data.csv')
    except FileNotFoundError:
        df = create_sample_dataset()
        df.to_csv('student_data.csv', index=False)
```
**Lines 12-16**: Load existing data or create sample dataset

```python
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of samples: {df.shape[0]}")
```
**Lines 18-20**: Display basic dataset statistics

```python
    print("\nTarget Variable (G3) Statistics:")
    print(df['G3'].describe())
```
**Lines 22-23**: Show target variable distribution statistics

```python
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
```
**Lines 26-29**: Display distribution of categorical variables

```python
    numerical_df = df.select_dtypes(include=[np.number])
    correlations = numerical_df.corr()['G3'].sort_values(key=abs, ascending=False)
    for feature, corr in correlations.items():
        if feature != 'G3':
            print(f"  {feature}: {corr:.4f}")
```
**Lines 34-38**: Calculate and display correlations with target variable

---

## 9. **requirements.txt** - Dependencies

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Basic plotting
seaborn         # Statistical visualization
scikit-learn    # Machine learning algorithms
flask           # Web framework
```

---

## 10. **templates/index.html** - Web Interface

### Purpose
HTML template for the web application interface.

### Key Sections

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
```
**Lines 1-5**: Standard HTML5 document structure with responsive viewport

```html
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎓</text></svg>">
```
**Line 6**: Inline SVG favicon with graduation cap emoji

```html
<div class="form-container">
    <form id="predictionForm">
```
**Lines 14-15**: Main form container for user input

```html
<div class="form-group">
    <label>Age:</label>
    <div class="help-text">Student's current age (typically 15-25 years)</div>
    <input type="number" name="age" min="15" max="25" value="18" required>
</div>
```
**Lines 20-24**: Form field with label, help text, and validation constraints

The HTML continues with similar form fields for all student attributes, JavaScript for form handling, and CSS for styling.

---

## Project Architecture Summary

1. **Data Layer**: `sample_data.py`, `data_preprocessing.py` - Handle data creation and preparation
2. **Model Layer**: `model.py` - Implements machine learning algorithm
3. **Evaluation Layer**: `evaluation.py` - Measures model performance
4. **Visualization Layer**: `visualization.py` - Creates charts and plots
5. **Application Layer**: `app.py` - Web interface, `main_simple.py` - CLI interface
6. **Analysis Layer**: `analysis.py` - Exploratory data analysis
7. **Frontend Layer**: `templates/index.html` - User interface

Each component is modular and focused on a specific responsibility, making the codebase maintainable and extensible.
