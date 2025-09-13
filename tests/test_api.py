"""Tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
import json
import numpy as np
import tempfile
import os
import sys

# Add api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Mock the model loading for testing
class MockModel:
    """Mock model for testing."""
    
    def predict(self, X):
        """Mock prediction."""
        return np.array([0] * len(X))
    
    def predict_proba(self, X):
        """Mock prediction probabilities."""
        return np.array([[0.8, 0.1, 0.1]] * len(X))
    
    @property
    def feature_importances_(self):
        """Mock feature importances."""
        return np.array([0.4, 0.3, 0.2, 0.1])

@pytest.fixture
def mock_app():
    """Create test app with mocked models."""
    from app import app, loaded_models
    
    # Add mock model
    loaded_models["test_model"] = MockModel()
    loaded_models["best_model"] = MockModel()
    
    return app

@pytest.fixture
def client(mock_app):
    """Create test client."""
    return TestClient(mock_app)

class TestAPI:
    """Test cases for the FastAPI application."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Data Science MVP API"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "version" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # At least the mock model
        
        # Check model info structure
        model_info = data[0]
        assert "name" in model_info
        assert "type" in model_info
        assert "features" in model_info
        assert "classes" in model_info
        assert "loaded" in model_info
    
    def test_single_prediction(self, client, sample_prediction_request):
        """Test single prediction endpoint."""
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "prediction_proba" in data
        assert "predicted_class" in data
        assert "confidence" in data
        assert "model_used" in data
        
        # Check data types and ranges
        assert isinstance(data["prediction"], int)
        assert 0 <= data["prediction"] <= 2
        assert isinstance(data["prediction_proba"], list)
        assert len(data["prediction_proba"]) == 3
        assert all(0 <= prob <= 1 for prob in data["prediction_proba"])
        assert abs(sum(data["prediction_proba"]) - 1.0) < 0.01  # Should sum to 1
        assert 0 <= data["confidence"] <= 1
    
    def test_single_prediction_with_explanation(self, client):
        """Test single prediction with explanation."""
        request_data = {
            "features": [5.1, 3.5, 1.4, 0.2],
            "model_name": "test_model",
            "explain": True
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "explanation" in data
        assert data["explanation"] is not None
    
    def test_batch_prediction(self, client, sample_batch_request):
        """Test batch prediction endpoint."""
        response = client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        
        # Check predictions
        predictions = data["predictions"]
        assert len(predictions) == 3  # Same as input
        
        for pred in predictions:
            assert "prediction" in pred
            assert "prediction_proba" in pred
            assert "predicted_class" in pred
            assert "confidence" in pred
        
        # Check summary
        summary = data["summary"]
        assert "total_predictions" in summary
        assert "average_confidence" in summary
        assert "class_distribution" in summary
        assert summary["total_predictions"] == 3
    
    def test_invalid_features_count(self, client):
        """Test prediction with wrong number of features."""
        request_data = {
            "features": [5.1, 3.5, 1.4],  # Only 3 features instead of 4
            "model_name": "test_model"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Expected 4 features" in data["detail"]
    
    def test_invalid_feature_values(self, client):
        """Test prediction with invalid feature values."""
        request_data = {
            "features": [5.1, 3.5, float('nan'), 0.2],
            "model_name": "test_model"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Invalid value" in data["detail"]
    
    def test_nonexistent_model(self, client):
        """Test prediction with nonexistent model."""
        request_data = {
            "features": [5.1, 3.5, 1.4, 0.2],
            "model_name": "nonexistent_model"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_batch_prediction_too_large(self, client):
        """Test batch prediction with too many samples."""
        large_batch = {
            "features": [[5.1, 3.5, 1.4, 0.2]] * 1001,  # Over the limit
            "model_name": "test_model"
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "too large" in data["detail"].lower()
    
    def test_retrain_endpoint(self, client):
        """Test retrain endpoint."""
        response = client.post("/retrain")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "in_progress"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        # Should return 200 regardless of whether metrics exist
        assert response.status_code == 200
        
        data = response.json()
        # Either returns metrics or a message about no metrics
        assert "message" in data or isinstance(data, dict)
    
    def test_request_validation(self, client):
        """Test request validation."""
        # Test with missing required field
        invalid_request = {
            "model_name": "test_model"
            # Missing "features" field
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        assert response.status_code == 200
        
        # CORS headers should be present (added by middleware)
        # Note: TestClient might not perfectly simulate CORS headers

class TestAPIModels:
    """Test API model validation and edge cases."""
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest model validation."""
        from app import PredictionRequest
        
        # Valid request
        valid_request = PredictionRequest(
            features=[5.1, 3.5, 1.4, 0.2],
            model_name="test_model",
            explain=True
        )
        assert len(valid_request.features) == 4
        
        # Test with too few features
        with pytest.raises(ValueError):
            PredictionRequest(features=[5.1, 3.5, 1.4])
        
        # Test with too many features
        with pytest.raises(ValueError):
            PredictionRequest(features=[5.1, 3.5, 1.4, 0.2, 1.0])
    
    def test_batch_prediction_request_validation(self):
        """Test BatchPredictionRequest model validation."""
        from app import BatchPredictionRequest
        
        # Valid request
        valid_request = BatchPredictionRequest(
            features=[[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.0, 1.2]],
            model_name="test_model"
        )
        assert len(valid_request.features) == 2
    
    def test_response_models(self):
        """Test response model creation."""
        from app import PredictionResponse, BatchPredictionResponse, ModelInfo, HealthResponse
        
        # Test PredictionResponse
        pred_response = PredictionResponse(
            prediction=0,
            prediction_proba=[0.8, 0.1, 0.1],
            predicted_class="setosa",
            confidence=0.8,
            model_used="test_model"
        )
        assert pred_response.prediction == 0
        
        # Test ModelInfo
        model_info = ModelInfo(
            name="test_model",
            type="MockModel",
            features=["f1", "f2", "f3", "f4"],
            classes=["c1", "c2", "c3"],
            loaded=True
        )
        assert model_info.loaded is True
        
        # Test HealthResponse
        health_response = HealthResponse(
            status="healthy",
            timestamp="2023-01-01T00:00:00",
            models_loaded=1,
            version="1.0.0"
        )
        assert health_response.status == "healthy"
