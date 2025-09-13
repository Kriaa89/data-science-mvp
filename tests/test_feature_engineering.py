"""Tests for feature engineering module."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tempfile
import os

from feature_engineering import FeatureEngineer

class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        
        assert engineer.scaler is None
        assert engineer.encoder is None
        assert engineer.feature_selector is None
        assert engineer.pca is None
        assert engineer.feature_names is None
        assert engineer.selected_features is None
    
    def test_create_scaling_pipeline(self):
        """Test scaling pipeline creation."""
        engineer = FeatureEngineer()
        
        # Test standard scaling
        scaler = engineer.create_scaling_pipeline('standard')
        assert scaler is not None
        assert engineer.scaler is not None
        
        # Test minmax scaling
        scaler = engineer.create_scaling_pipeline('minmax')
        assert scaler is not None
        
        # Test robust scaling
        scaler = engineer.create_scaling_pipeline('robust')
        assert scaler is not None
        
        # Test invalid method
        with pytest.raises(ValueError):
            engineer.create_scaling_pipeline('invalid')
    
    def test_create_polynomial_features(self, iris_data):
        """Test polynomial feature creation."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_poly, feature_names, poly_transformer = engineer.create_polynomial_features(
            X, degree=2, interaction_only=False
        )
        
        # Check that features increased
        assert X_poly.shape[0] == X.shape[0]
        assert X_poly.shape[1] > X.shape[1]
        
        # Check feature names
        assert len(feature_names) == X_poly.shape[1]
        
        # Test interaction only
        X_poly_int, _, _ = engineer.create_polynomial_features(
            X, degree=2, interaction_only=True
        )
        
        # Interaction only should have fewer features than full polynomial
        assert X_poly_int.shape[1] < X_poly.shape[1]
    
    def test_create_interaction_features(self, iris_data):
        """Test interaction feature creation."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_interaction = engineer.create_interaction_features(X)
        
        # Check that features increased
        assert X_interaction.shape[0] == X.shape[0]
        assert X_interaction.shape[1] > X.shape[1]
        
        # Check that original features are preserved
        for col in X.columns:
            assert col in X_interaction.columns
        
        # Test with specific feature pairs
        feature_pairs = [(X.columns[0], X.columns[1])]
        X_interaction_specific = engineer.create_interaction_features(X, feature_pairs)
        
        # Should have original features plus one interaction
        assert X_interaction_specific.shape[1] == X.shape[1] + 1
    
    def test_feature_selection_univariate(self, iris_data):
        """Test univariate feature selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_selected = engineer.feature_selection(X, y, method='univariate', k=3)
        
        # Check that features were reduced
        assert X_selected.shape[0] == X.shape[0]
        assert X_selected.shape[1] == 3
        
        # Check that selected features are tracked
        assert len(engineer.selected_features) == 3
    
    def test_feature_selection_mutual_info(self, iris_data):
        """Test mutual information feature selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_selected = engineer.feature_selection(X, y, method='mutual_info', k=2)
        
        assert X_selected.shape[1] == 2
        assert len(engineer.selected_features) == 2
    
    def test_feature_selection_model_based(self, iris_data):
        """Test model-based feature selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_selected = engineer.feature_selection(X, y, method='model_based', k=3)
        
        assert X_selected.shape[1] == 3
        assert len(engineer.selected_features) == 3
    
    def test_feature_selection_rfe(self, iris_data):
        """Test RFE feature selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_selected = engineer.feature_selection(X, y, method='rfe', k=2)
        
        assert X_selected.shape[1] == 2
        assert len(engineer.selected_features) == 2
    
    def test_feature_selection_pca(self, iris_data):
        """Test PCA feature selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_selected = engineer.feature_selection(X, y, method='pca', k=2)
        
        assert X_selected.shape[1] == 2
        assert len(engineer.selected_features) == 2
        assert engineer.pca is not None
        
        # Check that PCA feature names are correct
        assert engineer.selected_features == ['PC1', 'PC2']
    
    def test_feature_selection_invalid_method(self, iris_data):
        """Test invalid feature selection method."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError):
            engineer.feature_selection(X, y, method='invalid', k=2)
    
    def test_create_preprocessing_pipeline(self, iris_data):
        """Test preprocessing pipeline creation."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        numeric_features = list(X.columns)
        
        preprocessor = engineer.create_preprocessing_pipeline(
            numeric_features=numeric_features,
            scaling_method='standard'
        )
        
        assert preprocessor is not None
        
        # Test with categorical features
        categorical_features = ['category']
        preprocessor_with_cat = engineer.create_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaling_method='minmax',
            encoding_method='onehot'
        )
        
        assert preprocessor_with_cat is not None
    
    def test_engineer_features_complete(self, iris_data):
        """Test complete feature engineering pipeline."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_engineered = engineer.engineer_features(
            X, y,
            include_polynomial=True,
            include_interactions=True,
            polynomial_degree=2,
            feature_selection_method='univariate',
            k_features=5
        )
        
        # Check that features were engineered and selected
        assert X_engineered.shape[0] == X.shape[0]
        assert X_engineered.shape[1] == 5  # k_features
    
    def test_engineer_features_no_selection(self, iris_data):
        """Test feature engineering without selection."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        X_engineered = engineer.engineer_features(
            X, y,
            include_polynomial=True,
            include_interactions=True,
            polynomial_degree=2
        )
        
        # Should have more features than original
        assert X_engineered.shape[1] > X.shape[1]
    
    def test_get_feature_importance(self, iris_data):
        """Test feature importance calculation."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        importance_df = engineer.get_feature_importance(X, y)
        
        # Check structure
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        
        # Check that importance values are reasonable
        assert all(importance_df['importance'] >= 0)
        assert importance_df['importance'].sum() > 0
        
        # Test with custom model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        importance_df_custom = engineer.get_feature_importance(X, y, model=model)
        
        assert isinstance(importance_df_custom, pd.DataFrame)
        assert len(importance_df_custom) == X.shape[1]
    
    def test_save_load_pipeline(self, iris_data):
        """Test saving and loading pipeline components."""
        X, y, _ = iris_data
        engineer = FeatureEngineer()
        
        # Create some components
        engineer.create_scaling_pipeline('standard')
        engineer.scaler.fit(X)
        
        engineer.feature_selection(X, y, method='univariate', k=3)
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_prefix = os.path.join(temp_dir, 'test_pipeline')
            engineer.save_pipeline(temp_prefix)
            
            # Test loading
            engineer2 = FeatureEngineer()
            loaded_components = engineer2.load_pipeline(temp_prefix)
            
            assert 'scaler' in loaded_components
            assert 'feature_selector' in loaded_components
            assert engineer2.scaler is not None
            assert engineer2.feature_selector is not None
    
    def test_save_load_empty_pipeline(self):
        """Test saving and loading empty pipeline."""
        engineer = FeatureEngineer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_prefix = os.path.join(temp_dir, 'empty_pipeline')
            
            # Should handle empty pipeline gracefully
            engineer.save_pipeline(temp_prefix)
            
            loaded_components = engineer.load_pipeline(temp_prefix)
            assert len(loaded_components) == 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        engineer = FeatureEngineer()
        
        # Test interaction features with non-DataFrame
        with pytest.raises(ValueError):
            engineer.create_interaction_features(np.array([[1, 2], [3, 4]]))
        
        # Test feature selection without y
        X = pd.DataFrame(np.random.rand(10, 4))
        with pytest.raises(TypeError):
            engineer.feature_selection(X, None, method='univariate', k=2)
