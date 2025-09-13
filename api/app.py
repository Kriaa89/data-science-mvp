"""FastAPI application for ML model predictions."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import uvicorn
import json
import os
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Science MVP API",
    description="API for ML model predictions with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
loaded_models = {}
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: List[float] = Field(..., description="List of feature values", min_items=4, max_items=4)
    model_name: Optional[str] = Field("best_model", description="Name of the model to use")
    explain: Optional[bool] = Field(False, description="Whether to include explanations")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[List[float]] = Field(..., description="List of feature vectors")
    model_name: Optional[str] = Field("best_model", description="Name of the model to use")
    explain: Optional[bool] = Field(False, description="Whether to include explanations")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    prediction_proba: List[float]
    predicted_class: str
    confidence: float
    model_used: str
    explanation: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    type: str
    features: List[str]
    classes: List[str]
    accuracy: Optional[float] = None
    loaded: bool

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: int
    version: str

# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load available models on startup."""
    global loaded_models
    
    models_dir = Path("../models")
    
    # Try to load best model
    try:
        best_model_path = models_dir / "best_model_svm.joblib"
        if best_model_path.exists():
            loaded_models["best_model"] = joblib.load(best_model_path)
            logger.info("✅ Best model (SVM) loaded successfully")
        else:
            logger.warning("⚠️ Best model file not found")
    except Exception as e:
        logger.error(f"❌ Error loading best model: {e}")
    
    # Try to load other models
    model_files = {
        "svm": "best_model_svm.joblib",
        "random_forest": "random_forest_model.joblib",
        "logistic_regression": "logistic_regression_model.joblib"
    }
    
    for model_name, filename in model_files.items():
        try:
            model_path = models_dir / filename
            if model_path.exists():
                loaded_models[model_name] = joblib.load(model_path)
                logger.info(f"✅ {model_name} model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load {model_name}: {e}")
    
    logger.info(f"📊 Total models loaded: {len(loaded_models)}")

def get_model(model_name: str):
    """Get a specific model."""
    if model_name not in loaded_models:
        # Try to load the model if not already loaded
        models_dir = Path("../models")
        model_path = models_dir / f"{model_name}.joblib"
        
        if model_path.exists():
            try:
                loaded_models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Dynamically loaded {model_name}")
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {e}")
                raise HTTPException(status_code=404, detail=f"Model {model_name} not available")
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return loaded_models[model_name]

def validate_features(features: List[float]) -> np.ndarray:
    """Validate and preprocess features."""
    if len(features) != 4:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 4 features, got {len(features)}"
        )
    
    # Check for valid numeric values
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)) or np.isnan(feature) or np.isinf(feature):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for feature {i}: {feature}"
            )
    
    return np.array(features).reshape(1, -1)

def generate_explanation(model, features: np.ndarray, prediction: int) -> Dict[str, Any]:
    """Generate simple explanation for prediction."""
    try:
        # Basic feature importance (if available)
        explanation = {
            "feature_contributions": {},
            "explanation_method": "basic"
        }
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
            feature_values = features[0]
            
            for i, (name, value, imp) in enumerate(zip(feature_names, feature_values, importance)):
                explanation["feature_contributions"][name] = {
                    "value": float(value),
                    "importance": float(imp),
                    "contribution": float(value * imp)
                }
        
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            feature_values = features[0]
            
            for i, (name, value, coef) in enumerate(zip(feature_names, feature_values, coefficients)):
                explanation["feature_contributions"][name] = {
                    "value": float(value),
                    "coefficient": float(coef),
                    "contribution": float(value * coef)
                }
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {"error": f"Could not generate explanation: {str(e)}"}

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Data Science MVP API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(loaded_models),
        version="1.0.0"
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    models_info = []
    
    for name, model in loaded_models.items():
        model_type = type(model).__name__
        
        models_info.append(ModelInfo(
            name=name,
            type=model_type,
            features=feature_names,
            classes=target_names,
            loaded=True
        ))
    
    return models_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    # Validate features
    features_array = validate_features(request.features)
    
    # Get model
    model = get_model(request.model_name)
    
    try:
        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0].tolist()
        
        # Get predicted class and confidence
        predicted_class = target_names[prediction]
        confidence = max(prediction_proba)
        
        # Generate explanation if requested
        explanation = None
        if request.explain:
            explanation = generate_explanation(model, features_array, prediction)
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_proba=prediction_proba,
            predicted_class=predicted_class,
            confidence=confidence,
            model_used=request.model_name,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if len(request.features) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 1000 samples allowed."
        )
    
    # Get model
    model = get_model(request.model_name)
    
    predictions = []
    
    try:
        for i, features in enumerate(request.features):
            # Validate features
            features_array = validate_features(features)
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            prediction_proba = model.predict_proba(features_array)[0].tolist()
            
            # Get predicted class and confidence
            predicted_class = target_names[prediction]
            confidence = max(prediction_proba)
            
            # Generate explanation if requested
            explanation = None
            if request.explain:
                explanation = generate_explanation(model, features_array, prediction)
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                prediction_proba=prediction_proba,
                predicted_class=predicted_class,
                confidence=confidence,
                model_used=request.model_name,
                explanation=explanation
            ))
        
        # Create summary
        summary = {
            "total_predictions": len(predictions),
            "average_confidence": sum(p.confidence for p in predictions) / len(predictions),
            "class_distribution": {
                class_name: sum(1 for p in predictions if p.predicted_class == class_name)
                for class_name in target_names
            }
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining (placeholder for future implementation)."""
    background_tasks.add_task(retrain_background_task)
    return {"message": "Model retraining started", "status": "in_progress"}

async def retrain_background_task():
    """Background task for model retraining."""
    # This would implement the actual retraining logic
    logger.info("🔄 Model retraining task started (placeholder)")
    # Add your retraining logic here
    logger.info("✅ Model retraining completed (placeholder)")

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    # This would load and return saved metrics
    metrics_file = Path("../models/model_metrics.json")
    
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            raise HTTPException(status_code=500, detail="Could not load metrics")
    else:
        return {"message": "No metrics available"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
