import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the data"""
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def split_data(df, target_col='G3', test_size=0.2):
    """Split data into training and testing sets"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)
