# Predictive Analysis of Student Academic Performance Using Linear Regression

## Project Overview
This project implements a machine learning solution to predict student academic performance using Linear Regression. The system analyzes various student attributes to forecast final grades and identify key factors influencing academic success.

## Objectives
- Analyze factors influencing student academic performance
- Build and train a Linear Regression model for grade prediction
- Evaluate model performance using statistical metrics
- Provide interactive web interface for predictions
- Generate insights for educational decision-making

## Dataset Features
The dataset includes comprehensive student information:
- **Demographics**: Age, gender, address type
- **Academic History**: Previous grades, number of failures
- **Study Patterns**: Study time, extra educational support
- **Family Background**: Parent education, family support
- **Social Factors**: Free time, going out frequency
- **Target Variable**: Final grade (G3) - scale 0-20

## Installation & Setup
```bash
# Install required packages
pip install -r requirements.txt
```

## Usage

### Web Application (Interactive Interface)
```bash
python app.py
# Navigate to http://127.0.0.1:5000 in your browser
```

### Command Line Analysis
```bash
# Run complete analysis with visualizations
python analysis.py

# Run simplified demo
python main_simple.py
```

## Project Structure
```
├── app.py                    # Flask web application
├── analysis.py               # Complete data analysis script
├── main_simple.py           # Simplified command-line demo
├── data_preprocessing.py    # Data cleaning and preparation
├── model.py                 # Linear regression implementation
├── evaluation.py            # Model evaluation metrics
├── visualization.py         # Data visualization functions
├── student_data.csv         # Dataset
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Web interface template
├── static/                  # CSS and JS files
└── COLLEGE_PROJECT_REPORT.md # Detailed project report
```

## Key Features
- **Data Preprocessing**: Handles missing values and feature encoding
- **Model Training**: Implements Linear Regression with scikit-learn
- **Performance Evaluation**: R², MSE, MAE metrics
- **Interactive Predictions**: Web-based interface for real-time predictions
- **Data Visualization**: Comprehensive charts and analysis plots
- **Educational Insights**: Identifies most influential factors

## Results
The model achieves reliable performance in predicting student grades, with detailed analysis available in the project report.
