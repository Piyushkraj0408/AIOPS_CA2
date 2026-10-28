from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app, FEATURE_NAMES

client = TestClient(app)


def sample_features():
    dataset = load_breast_cancer(as_frame=True)
    row = dataset.data.iloc[0]
    return {name: float(row[name]) for name in FEATURE_NAMES}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post("/predict", json={"features": sample_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"malignant", "benign"}
    assert set(body["probabilities"]) == {"malignant", "benign"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_missing_feature_rejected():
    features = sample_features()
    features.pop(next(iter(features)))
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_non_numeric_feature_rejected():
    features = sample_features()
    features[next(iter(features))] = "not-a-number"
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
