"""Machine learning model utilities."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.trained_models = {}
        self.results = {}
        self.best_params = {}
        self.hyperparameter_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
    
    def hyperparameter_tuning(self, X_train, y_train, method='grid', cv=5, n_iter=50):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            cv: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
        """
        print(f"\n🔧 Starting hyperparameter tuning using {method} search...")
        
        tuned_models = {}
        
        for name, model in self.models.items():
            print(f"\nTuning {name}...")
            
            param_grid = self.hyperparameter_grids[name]
            
            if method == 'grid':
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
            elif method == 'random':
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                raise ValueError("Method must be 'grid' or 'random'")
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Store results
            tuned_models[name] = search.best_estimator_
            self.best_params[name] = search.best_params_
            
            print(f"Best parameters for {name}: {search.best_params_}")
            print(f"Best CV score for {name}: {search.best_score_:.4f}")
        
        # Update models with tuned versions
        self.models.update(tuned_models)
        
        return self.best_params
    
    def save_hyperparameters(self, filepath='models/best_hyperparameters.json'):
        """Save best hyperparameters to JSON file."""
        hyperparams_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'best_parameters': self.best_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(hyperparams_with_metadata, f, indent=2)
        
        print(f"💾 Best hyperparameters saved to {filepath}")
    
    def load_hyperparameters(self, filepath='models/best_hyperparameters.json'):
        """Load best hyperparameters from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.best_params = data['best_parameters']
                print(f"📂 Hyperparameters loaded from {filepath}")
                return self.best_params
        except FileNotFoundError:
            print(f"⚠️ Hyperparameters file not found: {filepath}")
            return None

    def train_models(self, X_train, y_train):
        """Train multiple models."""
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
    
    def evaluate_models(self, X_test, y_test, target_names):
        """Evaluate all trained models."""
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred, target_names=target_names),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(self.results[name]['classification_report'])
    
    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation."""
        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv)
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            print(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_best_model(self, model_name, filepath):
        """Save the best performing model."""
        if model_name in self.trained_models:
            joblib.dump(self.trained_models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")