"""Main execution script for the data science MVP."""
import os
from src.data_processing import DataProcessor
from src.model import ModelTrainer
from src.visualization import DataVisualizer

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    print("🚀 Starting Data Science MVP Pipeline...")
    
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()
    
    # Load and explore data
    print("\n📊 Loading and exploring data...")
    X, y, y_labels, target_names = data_processor.load_iris_data()
    exploration_summary = data_processor.explore_data(X, y_labels)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualizer.plot_data_distribution(X, y_labels)
    visualizer.plot_correlation_matrix(X)
    
    # Prepare data for training
    print("\n🔄 Preparing data for training...")
    X_train, X_test, y_train, y_test = data_processor.split_and_scale_data(X, y)
    
    # Cross-validation
    print("\n🔍 Performing cross-validation...")
    cv_results = model_trainer.cross_validate_models(X_train, y_train)
    
    # Train models
    print("\n🤖 Training models...")
    model_trainer.train_models(X_train, y_train)
    
    # Evaluate models
    print("\n📋 Evaluating models...")
    model_trainer.evaluate_models(X_test, y_test, target_names)
    
    # Create evaluation visualizations
    print("\n📊 Creating evaluation visualizations...")
    visualizer.plot_confusion_matrices(model_trainer.results, target_names)
    visualizer.plot_model_comparison(model_trainer.results)
    
    # Save best model
    best_model = max(model_trainer.results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n💾 Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
    model_trainer.save_best_model(best_model[0], f'models/best_model_{best_model[0]}.joblib')
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
