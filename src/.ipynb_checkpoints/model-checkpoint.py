"""Machine learning model utilities."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        self.trained_models = {}
        self.results = {}
    
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