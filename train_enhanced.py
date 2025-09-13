#!/usr/bin/env python3
"""
Enhanced Training Script for Docker Training Service
Optimized for production training with hyperparameter tuning and feature engineering
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append('/app')
sys.path.append('/app/src')

from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Enhanced training pipeline for production"""
    try:
        logger.info("🚀 Starting Enhanced Training Pipeline...")
        logger.info(f"Timestamp: {datetime.now()}")
        
        # Create directories
        os.makedirs('/app/models', exist_ok=True)
        
        # Initialize components
        processor = DataProcessor()
        trainer = ModelTrainer()
        engineer = FeatureEngineer()
        
        # Step 1: Load data
        logger.info("📊 Loading data...")
        X, y, y_labels, target_names = processor.load_iris_data()
        X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 2: Feature engineering
        logger.info("⚙️ Engineering features...")
        X_train_eng = engineer.engineer_features(
            X_train, y_train,
            include_interactions=True,
            feature_selection_method='univariate',
            k_features=6
        )
        
        # Transform test set
        X_test_eng = engineer.feature_selector.transform(X_test) if engineer.feature_selector else X_test
        logger.info(f"Features: {X_train.shape[1]} → {X_train_eng.shape[1]}")
        
        # Step 3: Hyperparameter tuning
        logger.info("🔧 Tuning hyperparameters...")
        best_params = trainer.hyperparameter_tuning(
            X_train_eng, y_train,
            method='random',
            n_iter=10,
            cv=3
        )
        
        # Save hyperparameters
        trainer.save_hyperparameters()
        logger.info("Hyperparameters saved")
        
        # Step 4: Train models
        logger.info("🎯 Training models...")
        trainer.train_models(X_train_eng, y_train)
        
        # Step 5: Evaluate models
        logger.info("📈 Evaluating models...")
        trainer.evaluate_models(X_test_eng, y_test, target_names)
        
        # Step 6: Save best model
        best_model = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])
        best_model_name, best_result = best_model
        
        model_path = f'/app/models/best_model_{best_model_name}.joblib'
        trainer.save_best_model(best_model_name, model_path)
        
        # Save feature engineering pipeline
        engineer.save_pipeline()
        
        logger.info(f"🏆 Best model: {best_model_name}")
        logger.info(f"📈 Accuracy: {best_result['accuracy']:.4f}")
        logger.info(f"💾 Model saved: {model_path}")
        
        logger.info("✅ Enhanced training completed successfully!")
        
        return {
            'best_model': best_model_name,
            'accuracy': best_result['accuracy'],
            'model_path': model_path
        }
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
