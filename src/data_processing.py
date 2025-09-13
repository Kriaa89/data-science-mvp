"""Data processing utilities."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_iris_data(self):
        """Load and prepare iris dataset."""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="species")
        
        # Add target names for better interpretation
        target_names = pd.Series(iris.target_names)
        y_labels = y.map(lambda x: target_names[x])
        
        return X, y, y_labels, iris.target_names
    
    def explore_data(self, X, y_labels):
        """Generate data exploration summary."""
        print("Dataset Shape:", X.shape)
        print("\nFirst 5 rows:")
        print(X.head())
        print(f"\nFeatures: {list(X.columns)}")
        print(f"\nTarget distribution:")
        print(y_labels.value_counts())
        print(f"\nMissing values:")
        print(X.isnull().sum())
        
        return {
            'shape': X.shape,
            'features': list(X.columns),
            'target_distribution': y_labels.value_counts().to_dict(),
            'missing_values': X.isnull().sum().to_dict()
        }
    
    def split_and_scale_data(self, X, y):
        """Split data and apply scaling."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test