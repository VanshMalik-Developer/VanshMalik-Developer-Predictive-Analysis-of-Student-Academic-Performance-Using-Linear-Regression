import pandas as pd
import numpy as np

def create_sample_dataset():
    """Create sample student dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(15, 23, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'study_time': np.random.randint(1, 5, n_samples),
        'failures': np.random.randint(0, 4, n_samples),
        'school_support': np.random.choice([0, 1], n_samples),
        'family_support': np.random.choice([0, 1], n_samples),
        'paid_classes': np.random.choice([0, 1], n_samples),
        'activities': np.random.choice([0, 1], n_samples),
        'higher_ed': np.random.choice([0, 1], n_samples),
        'internet': np.random.choice([0, 1], n_samples),
        'absences': np.random.randint(0, 30, n_samples),
        'G1': np.random.randint(0, 21, n_samples),
        'G2': np.random.randint(0, 21, n_samples)
    }
    
    # Generate G3 based on other features with some noise
    G3 = (data['study_time'] * 2 + 
          data['G1'] * 0.3 + 
          data['G2'] * 0.4 - 
          data['failures'] * 1.5 + 
          np.random.normal(0, 2, n_samples))
    
    data['G3'] = np.clip(G3, 0, 20)
    
    return pd.DataFrame(data)
