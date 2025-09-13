"""Tests for data processing module."""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from data_processing import DataProcessor

class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.test_size == 0.2
        assert processor.random_state == 42
        assert processor.scaler is not None
    
    def test_init_custom_params(self):
        """Test DataProcessor initialization with custom parameters."""
        processor = DataProcessor(test_size=0.3, random_state=123)
        assert processor.test_size == 0.3
        assert processor.random_state == 123
    
    def test_load_iris_data(self):
        """Test loading iris data."""
        processor = DataProcessor()
        X, y, y_labels, target_names = processor.load_iris_data()
        
        # Check data shape
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert y_labels.shape == (150,)
        
        # Check data types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(y_labels, pd.Series)
        
        # Check feature names
        expected_features = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
        assert list(X.columns) == expected_features
        
        # Check target names
        expected_targets = ['setosa', 'versicolor', 'virginica']
        assert list(target_names) == expected_targets
        
        # Check target values
        assert set(y.unique()) == {0, 1, 2}
        assert len(y_labels.unique()) == 3
    
    def test_explore_data(self, iris_data):
        """Test data exploration functionality."""
        X, y, target_names = iris_data
        processor = DataProcessor()
        
        # Convert numeric y to categorical labels for this test
        y_labels = pd.Series([target_names[i] for i in y])
        
        summary = processor.explore_data(X, y_labels)
        
        # Check summary structure
        assert 'shape' in summary
        assert 'head' in summary
        assert 'features' in summary
        assert 'target_distribution' in summary
        assert 'missing_values' in summary
        
        # Check values
        assert summary['shape'] == (150, 4)
        assert len(summary['features']) == 4
        assert len(summary['target_distribution']) == 3
    
    def test_split_and_scale_data(self, iris_data):
        """Test data splitting and scaling."""
        X, y, _ = iris_data
        processor = DataProcessor()
        
        X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
        
        # Check shapes
        assert X_train.shape[0] == int(150 * 0.8)
        assert X_test.shape[0] == int(150 * 0.2)
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4
        
        # Check that data is scaled (mean ~0, std ~1)
        assert abs(X_train.mean().mean()) < 0.1
        assert abs(X_train.std().mean() - 1.0) < 0.1
        
        # Check y shapes
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
    
    def test_split_and_scale_custom_params(self, iris_data):
        """Test data splitting with custom parameters."""
        X, y, _ = iris_data
        processor = DataProcessor(test_size=0.3, random_state=999)
        
        X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
        
        # Check shapes with custom test size
        assert X_train.shape[0] == int(150 * 0.7)
        assert X_test.shape[0] == int(150 * 0.3)
    
    def test_data_types(self, iris_data):
        """Test that data types are preserved correctly."""
        X, y, _ = iris_data
        processor = DataProcessor()
        
        X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check numeric types
        assert X_train.dtype in [np.float64, np.float32]
        assert X_test.dtype in [np.float64, np.float32]
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        processor = DataProcessor()
        
        # Test with empty data
        with pytest.raises(Exception):
            empty_X = pd.DataFrame()
            empty_y = pd.Series(dtype=float)
            processor.split_and_scale_data(empty_X, empty_y)
        
        # Test with mismatched X and y lengths
        with pytest.raises(Exception):
            X = pd.DataFrame(np.random.rand(10, 4))
            y = pd.Series(np.random.randint(0, 3, 5))
            processor.split_and_scale_data(X, y)
