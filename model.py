import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

class StudentPerformanceModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
    
    def train(self, X_train, y_train):
        """Train the linear regression model"""
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions on test data"""
        return self.model.predict(X_test)
    
    def get_coefficients(self):
        """Get model coefficients and intercept"""
        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_,
            'feature_names': self.feature_names
        }
