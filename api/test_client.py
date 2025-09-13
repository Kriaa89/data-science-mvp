"""Simple client to test the API."""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_models():
    """Test models endpoint."""
    response = requests.get(f"{BASE_URL}/models")
    print("Available Models:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction():
    """Test single prediction."""
    # Sample iris data
    data = {
        "features": [5.1, 3.5, 1.4, 0.2],  # Typical setosa
        "model_name": "best_model",
        "explain": True
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Single Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_prediction():
    """Test batch prediction."""
    data = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [7.0, 3.2, 4.7, 1.4],  # versicolor
            [6.3, 3.3, 6.0, 2.5]   # virginica
        ],
        "model_name": "best_model",
        "explain": False
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print("Batch Prediction:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing Data Science MVP API")
    print("=" * 50)
    
    try:
        test_health()
        test_models()
        test_prediction()
        test_batch_prediction()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")
