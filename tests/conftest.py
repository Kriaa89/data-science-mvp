"""Test configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def iris_data():
    """Load iris dataset for testing."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y, iris.target_names

@pytest.fixture
def train_test_data(iris_data):
    """Create train/test split for testing."""
    X, y, target_names = iris_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names

@pytest.fixture
def sample_features():
    """Sample feature vector for testing."""
    return np.array([[5.1, 3.5, 1.4, 0.2]])

@pytest.fixture
def sample_prediction_request():
    """Sample API prediction request."""
    return {
        "features": [5.1, 3.5, 1.4, 0.2],
        "model_name": "test_model",
        "explain": True
    }

@pytest.fixture
def sample_batch_request():
    """Sample API batch prediction request."""
    return {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5]
        ],
        "model_name": "test_model",
        "explain": False
    }
