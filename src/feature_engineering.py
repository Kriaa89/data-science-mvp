"""Feature engineering pipeline utilities."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

class FeatureEngineer:
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        self.selected_features = None
        
    def create_scaling_pipeline(self, method='standard'):
        """
        Create scaling pipeline.
        
        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        print(f"📊 Created {method} scaling pipeline")
        return self.scaler
    
    def create_polynomial_features(self, X, degree=2, interaction_only=False):
        """Create polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = poly.get_feature_names_out(X.columns)
        else:
            feature_names = poly.get_feature_names_out()
        
        print(f"🔧 Created polynomial features: {X.shape[1]} -> {X_poly.shape[1]} features")
        
        return X_poly, feature_names, poly
    
    def create_interaction_features(self, X, feature_pairs=None):
        """Create interaction features between specified feature pairs."""
        if isinstance(X, pd.DataFrame):
            X_interaction = X.copy()
            
            if feature_pairs is None:
                # Create interactions between all numeric features
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                feature_pairs = [(col1, col2) for i, col1 in enumerate(numeric_cols) 
                               for col2 in numeric_cols[i+1:]]
            
            for col1, col2 in feature_pairs:
                interaction_name = f"{col1}_x_{col2}"
                X_interaction[interaction_name] = X[col1] * X[col2]
            
            print(f"🔧 Created {len(feature_pairs)} interaction features")
            return X_interaction
        else:
            raise ValueError("X must be a pandas DataFrame for interaction features")
    
    def feature_selection(self, X, y, method='univariate', k=10, model=None):
        """
        Perform feature selection.
        
        Args:
            X: Features
            y: Target
            method: 'univariate', 'model_based', 'rfe', or 'pca'
            k: Number of features to select
            model: Model for model-based selection
        """
        print(f"🎯 Performing {method} feature selection...")
        
        if method == 'univariate':
            # Use f_classif for classification
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            if hasattr(X, 'columns'):
                selected_features = X.columns[self.feature_selector.get_support()]
                self.selected_features = selected_features.tolist()
            
        elif method == 'mutual_info':
            # Use mutual information for feature selection
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            if hasattr(X, 'columns'):
                selected_features = X.columns[self.feature_selector.get_support()]
                self.selected_features = selected_features.tolist()
        
        elif method == 'model_based':
            # Use model-based selection
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            self.feature_selector = SelectFromModel(estimator=model, max_features=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            if hasattr(X, 'columns'):
                selected_features = X.columns[self.feature_selector.get_support()]
                self.selected_features = selected_features.tolist()
        
        elif method == 'rfe':
            # Recursive Feature Elimination
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            self.feature_selector = RFE(estimator=model, n_features_to_select=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            if hasattr(X, 'columns'):
                selected_features = X.columns[self.feature_selector.get_support()]
                self.selected_features = selected_features.tolist()
        
        elif method == 'pca':
            # Principal Component Analysis
            self.pca = PCA(n_components=k)
            X_selected = self.pca.fit_transform(X)
            
            # Create feature names for PCA components
            self.selected_features = [f'PC{i+1}' for i in range(k)]
        
        else:
            raise ValueError("Method must be 'univariate', 'mutual_info', 'model_based', 'rfe', or 'pca'")
        
        print(f"✅ Selected {X_selected.shape[1]} features from {X.shape[1]}")
        if hasattr(self, 'selected_features') and self.selected_features:
            print(f"Selected features: {self.selected_features}")
        
        return X_selected
    
    def create_preprocessing_pipeline(self, numeric_features, categorical_features=None, 
                                    scaling_method='standard', encoding_method='onehot'):
        """
        Create a complete preprocessing pipeline.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            scaling_method: Scaling method for numeric features
            encoding_method: Encoding method for categorical features
        """
        transformers = []
        
        # Numeric preprocessing
        if scaling_method == 'standard':
            numeric_transformer = StandardScaler()
        elif scaling_method == 'minmax':
            numeric_transformer = MinMaxScaler()
        elif scaling_method == 'robust':
            numeric_transformer = RobustScaler()
        else:
            numeric_transformer = 'passthrough'
        
        transformers.append(('num', numeric_transformer, numeric_features))
        
        # Categorical preprocessing
        if categorical_features:
            if encoding_method == 'onehot':
                categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
            elif encoding_method == 'label':
                categorical_transformer = LabelEncoder()
            else:
                categorical_transformer = 'passthrough'
            
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        print(f"🔧 Created preprocessing pipeline with {scaling_method} scaling and {encoding_method} encoding")
        
        return preprocessor
    
    def engineer_features(self, X, y=None, include_polynomial=False, include_interactions=False,
                         polynomial_degree=2, feature_selection_method=None, k_features=None):
        """
        Complete feature engineering pipeline.
        
        Args:
            X: Input features
            y: Target variable (required for feature selection)
            include_polynomial: Whether to include polynomial features
            include_interactions: Whether to include interaction features
            polynomial_degree: Degree for polynomial features
            feature_selection_method: Method for feature selection
            k_features: Number of features to select
        """
        print("🚀 Starting feature engineering pipeline...")
        
        X_engineered = X.copy() if isinstance(X, pd.DataFrame) else X
        
        # Polynomial features
        if include_polynomial:
            X_engineered, feature_names, poly_transformer = self.create_polynomial_features(
                X_engineered, degree=polynomial_degree
            )
            if isinstance(X, pd.DataFrame):
                X_engineered = pd.DataFrame(X_engineered, columns=feature_names)
        
        # Interaction features
        if include_interactions and isinstance(X_engineered, pd.DataFrame):
            X_engineered = self.create_interaction_features(X_engineered)
        
        # Feature selection
        if feature_selection_method and y is not None and k_features:
            X_engineered = self.feature_selection(
                X_engineered, y, method=feature_selection_method, k=k_features
            )
        
        print(f"✅ Feature engineering completed: {X.shape[1]} -> {X_engineered.shape[1]} features")
        
        return X_engineered
    
    def get_feature_importance(self, X, y, model=None):
        """Get feature importance scores."""
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("📊 Feature importance calculated")
            return importance_df
        else:
            print("⚠️ Model does not have feature_importances_ attribute")
            return None
    
    def save_pipeline(self, filepath_prefix='models/feature_engineering'):
        """Save feature engineering components."""
        saved_components = []
        
        if self.scaler:
            scaler_path = f"{filepath_prefix}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            saved_components.append(f"scaler -> {scaler_path}")
        
        if self.feature_selector:
            selector_path = f"{filepath_prefix}_selector.joblib"
            joblib.dump(self.feature_selector, selector_path)
            saved_components.append(f"feature_selector -> {selector_path}")
        
        if self.pca:
            pca_path = f"{filepath_prefix}_pca.joblib"
            joblib.dump(self.pca, pca_path)
            saved_components.append(f"pca -> {pca_path}")
        
        if saved_components:
            print(f"💾 Saved feature engineering components:")
            for component in saved_components:
                print(f"  - {component}")
        else:
            print("⚠️ No feature engineering components to save")
    
    def load_pipeline(self, filepath_prefix='models/feature_engineering'):
        """Load feature engineering components."""
        loaded_components = []
        
        try:
            scaler_path = f"{filepath_prefix}_scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            loaded_components.append("scaler")
        except FileNotFoundError:
            pass
        
        try:
            selector_path = f"{filepath_prefix}_selector.joblib"
            self.feature_selector = joblib.load(selector_path)
            loaded_components.append("feature_selector")
        except FileNotFoundError:
            pass
        
        try:
            pca_path = f"{filepath_prefix}_pca.joblib"
            self.pca = joblib.load(pca_path)
            loaded_components.append("pca")
        except FileNotFoundError:
            pass
        
        if loaded_components:
            print(f"📂 Loaded feature engineering components: {', '.join(loaded_components)}")
        else:
            print("⚠️ No feature engineering components found")
        
        return loaded_components
