from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import StudentPerformanceModel
from data_preprocessing import preprocess_data, split_data
from sample_data import create_sample_dataset

app = Flask(__name__)
model = pickle.load(open("model.py", "rb"))
# Load and train model on startup
def load_model():
    try:
        df = pd.read_csv('student_data.csv')
    except FileNotFoundError:
        df = create_sample_dataset()
        df.to_csv('student_data.csv', index=False)
    
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    model = StudentPerformanceModel()
    model.train(X_train, y_train)
    return model, X_train.columns.tolist()

model, feature_names = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'age': int(request.form['age']),
            'gender': 1 if request.form['gender'] == 'M' else 0,
            'study_time': int(request.form['study_time']),
            'failures': int(request.form['failures']),
            'school_support': 1 if request.form['school_support'] == 'yes' else 0,
            'family_support': 1 if request.form['family_support'] == 'yes' else 0,
            'paid_classes': 1 if request.form['paid_classes'] == 'yes' else 0,
            'activities': 1 if request.form['activities'] == 'yes' else 0,
            'higher_ed': 1 if request.form['higher_ed'] == 'yes' else 0,
            'internet': 1 if request.form['internet'] == 'yes' else 0,
            'absences': int(request.form['absences']),
            'G1': float(request.form['G1']),
            'G2': float(request.form['G2'])
        }
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = max(0, min(20, prediction))  # Clamp to valid range
        
        # Calculate feature impacts
        coefficients = model.get_coefficients()['coefficients']
        feature_impacts = []
        
        # Get baseline (intercept) for comparison
        intercept = model.get_coefficients()['intercept']
        
        for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
            feature_value = list(data.values())[i]
            
            # Calculate actual contribution to prediction
            contribution = coef * feature_value
            
            # For categorical variables, show meaningful interpretation
            if feature == 'gender':
                display_value = 'Male' if feature_value == 1 else 'Female'
                interpretation = f"Being {display_value}"
            elif feature == 'higher_ed':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"Higher Ed Goal: {display_value}"
            elif feature == 'school_support':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"School Support: {display_value}"
            elif feature == 'family_support':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"Family Support: {display_value}"
            elif feature == 'paid_classes':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"Paid Classes: {display_value}"
            elif feature == 'activities':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"Extra Activities: {display_value}"
            elif feature == 'internet':
                display_value = 'Yes' if feature_value == 1 else 'No'
                interpretation = f"Internet Access: {display_value}"
            elif feature == 'study_time':
                time_labels = {1: '<2hrs', 2: '2-5hrs', 3: '5-10hrs', 4: '>10hrs'}
                interpretation = f"Study Time: {time_labels.get(feature_value, str(feature_value))}"
            elif feature == 'failures':
                interpretation = f"Previous Failures: {feature_value}"
            else:
                interpretation = f"{get_feature_display_name(feature)}: {feature_value}"
            
            feature_impacts.append({
                'feature': interpretation,
                'coefficient': round(coef, 3),
                'value': feature_value,
                'impact': round(contribution, 3),
                'abs_impact': abs(contribution)
            })
        
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        # Get top 5 most impactful features
        top_features = feature_impacts[:5]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'grade_letter': get_letter_grade(prediction),
            'feature_impacts': top_features,
            'confidence': calculate_confidence(prediction)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def get_feature_display_name(feature):
    """Convert feature names to display names"""
    display_names = {
        'age': 'Age',
        'gender': 'Gender',
        'study_time': 'Study Time',
        'failures': 'Previous Failures',
        'school_support': 'School Support',
        'family_support': 'Family Support',
        'paid_classes': 'Paid Classes',
        'activities': 'Extra Activities',
        'higher_ed': 'Higher Education Goal',
        'internet': 'Internet Access',
        'absences': 'Absences',
        'G1': 'First Period Grade',
        'G2': 'Second Period Grade'
    }
    return display_names.get(feature, feature)

def calculate_confidence(prediction):
    """Calculate prediction confidence based on grade range"""
    if 8 <= prediction <= 16:
        return "High"
    elif 6 <= prediction <= 18:
        return "Medium"
    else:
        return "Low"

def get_letter_grade(score):
    if score >= 18: return 'A+'
    elif score >= 16: return 'A'
    elif score >= 14: return 'B+'
    elif score >= 12: return 'B'
    elif score >= 10: return 'C+'
    elif score >= 8: return 'C'
    elif score >= 6: return 'D'
    else: return 'F'

if __name__ == '__main__':
    app.run(debug=True)
p