"""Tests for model training module."""
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tempfile
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.model import ModelTrainer

class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        
        # Check models are initialized
        assert 'logistic_regression' in trainer.models
        assert 'random_forest' in trainer.models
        assert 'svm' in trainer.models
        
        # Check model types
        assert isinstance(trainer.models['logistic_regression'], LogisticRegression)
        assert isinstance(trainer.models['random_forest'], RandomForestClassifier)
        assert isinstance(trainer.models['svm'], SVC)
        
        # Check empty containers
        assert len(trainer.trained_models) == 0
        assert len(trainer.results) == 0
        assert len(trainer.best_params) == 0
        
        # Check hyperparameter grids exist
        assert 'logistic_regression' in trainer.hyperparameter_grids
        assert 'random_forest' in trainer.hyperparameter_grids
        assert 'svm' in trainer.hyperparameter_grids
    
    def test_train_models(self, train_test_data):
        """Test model training."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        trainer.train_models(X_train, y_train)
        
        # Check that models are trained
        assert len(trainer.trained_models) == 3
        assert 'logistic_regression' in trainer.trained_models
        assert 'random_forest' in trainer.trained_models
        assert 'svm' in trainer.trained_models
        
        # Check that models can make predictions
        for name, model in trainer.trained_models.items():
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)
            assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_evaluate_models(self, train_test_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test, target_names = train_test_data
        trainer = ModelTrainer()
        
        # Train models first
        trainer.train_models(X_train, y_train)
        trainer.evaluate_models(X_test, y_test, target_names)
        
        # Check evaluation results
        assert len(trainer.results) == 3
        
        for name, results in trainer.results.items():
            assert 'accuracy' in results
            assert 'predictions' in results
            assert 'classification_report' in results
            assert 'confusion_matrix' in results
            
            # Check accuracy is reasonable
            assert 0.0 <= results['accuracy'] <= 1.0
            
            # Check predictions shape
            assert len(results['predictions']) == len(y_test)
            
            # Check confusion matrix shape
            assert results['confusion_matrix'].shape == (3, 3)
    
    def test_cross_validate_models(self, train_test_data):
        """Test cross-validation."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        cv_results = trainer.cross_validate_models(X_train, y_train, cv=3)
        
        # Check results structure
        assert len(cv_results) == 3
        
        for name, results in cv_results.items():
            assert 'mean_score' in results
            assert 'std_score' in results
            assert 'scores' in results
            
            # Check score validity
            assert 0.0 <= results['mean_score'] <= 1.0
            assert results['std_score'] >= 0.0
            assert len(results['scores']) == 3  # 3-fold CV
    
    def test_hyperparameter_tuning_grid(self, train_test_data):
        """Test grid search hyperparameter tuning."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        # Reduce parameter grids for faster testing
        trainer.hyperparameter_grids = {
            'logistic_regression': {
                'C': [0.1, 1],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
        }
        
        # Only tune one model for testing
        trainer.models = {'logistic_regression': trainer.models['logistic_regression']}
        
        best_params = trainer.hyperparameter_tuning(
            X_train, y_train, method='grid', cv=3
        )
        
        # Check results
        assert 'logistic_regression' in best_params
        assert 'C' in best_params['logistic_regression']
        assert best_params['logistic_regression']['C'] in [0.1, 1]
    
    def test_hyperparameter_tuning_random(self, train_test_data):
        """Test random search hyperparameter tuning."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        # Reduce parameter grids for faster testing
        trainer.hyperparameter_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
        }
        
        # Only tune one model for testing
        trainer.models = {'logistic_regression': trainer.models['logistic_regression']}
        
        best_params = trainer.hyperparameter_tuning(
            X_train, y_train, method='random', cv=3, n_iter=2
        )
        
        # Check results
        assert 'logistic_regression' in best_params
        assert 'C' in best_params['logistic_regression']
    
    def test_save_load_hyperparameters(self, train_test_data):
        """Test saving and loading hyperparameters."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        # Set some test parameters
        trainer.best_params = {
            'logistic_regression': {'C': 1.0, 'penalty': 'l2'},
            'svm': {'C': 10.0, 'kernel': 'rbf'}
        }
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            trainer.save_hyperparameters(temp_path)
            
            # Test loading
            trainer2 = ModelTrainer()
            loaded_params = trainer2.load_hyperparameters(temp_path)
            
            assert loaded_params == trainer.best_params
            assert trainer2.best_params == trainer.best_params
            
        finally:
            os.unlink(temp_path)
    
    def test_save_best_model(self, train_test_data):
        """Test saving best model."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        # Train a model
        trainer.train_models(X_train, y_train)
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            trainer.save_best_model('logistic_regression', temp_path)
            
            # Check file exists
            assert os.path.exists(temp_path)
            
            # Try to load and use the model
            import joblib
            loaded_model = joblib.load(temp_path)
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == len(y_test)
            
        finally:
            os.unlink(temp_path)
    
    def test_invalid_hyperparameter_method(self, train_test_data):
        """Test invalid hyperparameter tuning method."""
        X_train, X_test, y_train, y_test, _ = train_test_data
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError):
            trainer.hyperparameter_tuning(X_train, y_train, method='invalid')
    
    def test_save_nonexistent_model(self):
        """Test saving a model that doesn't exist."""
        trainer = ModelTrainer()
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            # This should not raise an error, but should print a message
            trainer.save_best_model('nonexistent_model', temp_path)
            
            # File should not be created
            assert not os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_nonexistent_hyperparameters(self):
        """Test loading hyperparameters from nonexistent file."""
        trainer = ModelTrainer()
        
        result = trainer.load_hyperparameters('nonexistent_file.json')
        assert result is None
