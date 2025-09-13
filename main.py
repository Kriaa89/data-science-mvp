"""
Enhanced Data Science MVP Main Pipeline
Integrates feature engineering, hyperparameter tuning, and model explainability
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.visualization import DataVisualizer
from src.feature_engineering import FeatureEngineer
from src.explainability import ModelExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_enhanced_pipeline(enable_feature_engineering=True, 
                         enable_hyperparameter_tuning=True,
                         enable_explainability=True,
                         save_visualizations=True):
    """
    Run the enhanced data science pipeline with optional features
    
    Args:
        enable_feature_engineering (bool): Enable feature engineering pipeline
        enable_hyperparameter_tuning (bool): Enable hyperparameter optimization
        enable_explainability (bool): Enable model explainability analysis
        save_visualizations (bool): Save visualization plots to files
    """
    try:
        logger.info("🚀 Starting Enhanced Data Science Pipeline")
        logger.info(f"Timestamp: {datetime.now()}")
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        
        # Initialize components
        processor = DataProcessor()
        trainer = ModelTrainer()
        visualizer = DataVisualizer()
        
        # Step 1: Load and process data
        logger.info("📊 Loading and processing data...")
        X, y, y_labels, target_names = processor.load_iris_data()
        exploration_summary = processor.explore_data(X, y_labels)
        X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Step 2: Feature Engineering (optional)
        if enable_feature_engineering:
            logger.info("⚙️ Applying feature engineering...")
            engineer = FeatureEngineer()
            
            # Engineer features with interactions and selection
            X_train_engineered = engineer.engineer_features(
                X_train, y_train,
                include_interactions=True,
                feature_selection_method='univariate',
                k_features=min(10, X_train.shape[1] * 2)  # Adaptive feature count
            )
            
            # Transform test set
            X_test_engineered = engineer.feature_selector.transform(X_test) if engineer.feature_selector else X_test
            
            logger.info(f"Features engineered: {X_train.shape[1]} → {X_train_engineered.shape[1]}")
            
            # Save feature engineering pipeline
            engineer.save_pipeline()
            
            # Update data for training
            X_train, X_test = X_train_engineered, X_test_engineered
        
        # Step 3: Hyperparameter Tuning (optional)
        if enable_hyperparameter_tuning:
            logger.info("🔧 Performing hyperparameter tuning...")
            
            # Perform hyperparameter tuning
            best_params = trainer.hyperparameter_tuning(
                X_train, y_train,
                method='random',  # Use random search for faster results
                n_iter=20,
                cv=3
            )
            
            logger.info(f"Best parameters found: {len(best_params)} models optimized")
            
            # Save hyperparameters
            trainer.save_hyperparameters()
        
        # Step 4: Cross-validation
        logger.info("🔍 Performing cross-validation...")
        cv_results = trainer.cross_validate_models(X_train, y_train)
        
        # Step 5: Train models
        logger.info("🎯 Training models...")
        trainer.train_models(X_train, y_train)
        
        # Step 6: Evaluate models
        logger.info("� Evaluating models...")
        trainer.evaluate_models(X_test, y_test, target_names)
        
        # Find best model
        best_model_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = trainer.results[best_model_name]['accuracy']
        
        logger.info(f"🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Save best model
        best_model_path = f"models/best_model_{best_model_name}.joblib"
        trainer.save_best_model(best_model_name, best_model_path)
        logger.info(f"💾 Best model saved to: {best_model_path}")
        
        # Step 7: Create visualizations
        if save_visualizations:
            logger.info("📊 Creating visualizations...")
            
            # Data exploration plots
            visualizer.plot_data_distribution(X, y_labels)
            visualizer.plot_correlation_matrix(X)
            
            # Model performance plots
            visualizer.plot_confusion_matrices(trainer.results, target_names)
            visualizer.plot_model_comparison(trainer.results)
            
            logger.info("📊 Visualizations created")
        
        # Step 8: Model Explainability (optional)
        if enable_explainability:
            logger.info("🔍 Generating model explanations...")
            
            try:
                explainer = ModelExplainer()
                
                # Get best model for explanation
                best_model = trainer.models[best_model_name]
                
                # Generate SHAP explanations
                explainer.explain_with_shap(
                    model=best_model,
                    X_train=X_train,
                    X_test=X_test[:10],  # Explain first 10 test samples
                    feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
                    save_path="models/shap_explanations.png"
                )
                
                # Generate LIME explanations for a sample
                explainer.explain_with_lime(
                    model=best_model,
                    X_train=X_train,
                    instance=X_test[0],
                    feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
                    class_names=target_names,
                    save_path="models/lime_explanation.html"
                )
                
                logger.info("🔍 Model explanations saved to models/ directory")
                
            except Exception as e:
                logger.warning(f"Model explainability failed: {e}")
                logger.info("Continuing pipeline without explainability...")
        
        # Step 9: Pipeline summary
        logger.info("\n" + "="*60)
        logger.info("📋 PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Data samples: {X.shape[0]}")
        logger.info(f"Features: {X.shape[1]} original → {X_train.shape[1]} final")
        logger.info(f"Models trained: {len(trainer.models)}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        
        if enable_feature_engineering:
            logger.info("✅ Feature engineering: Enabled")
        if enable_hyperparameter_tuning:
            logger.info("✅ Hyperparameter tuning: Enabled")
        if enable_explainability:
            logger.info("✅ Model explainability: Enabled")
        
        logger.info("="*60)
        logger.info("✅ Pipeline completed successfully!")
        
        return {
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'models_trained': len(trainer.models),
            'features_engineered': X_train.shape[1],
            'results': trainer.results
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise

def main():
    """Main entry point"""
    try:
        # Run enhanced pipeline with all features enabled
        results = run_enhanced_pipeline(
            enable_feature_engineering=True,
            enable_hyperparameter_tuning=True,
            enable_explainability=True,
            save_visualizations=True
        )
        
        print(f"\n🎉 Pipeline completed! Best model: {results['best_model']} "
              f"with accuracy: {results['best_accuracy']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
