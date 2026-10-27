import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict(client):
    customer = {
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "tenure": 12,
        "phoneservice": "yes",
        "multiplelines": "no",
        "internetservice": "fiber_optic",
        "onlinesecurity": "no",
        "onlinebackup": "yes",
        "deviceprotection": "no",
        "techsupport": "no",
        "streamingtv": "yes",
        "streamingmovies": "no",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "monthlycharges": 79.85,
        "totalcharges": 950.20
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["churn"], bool)


def test_invalid_tenure(client):
    customer = {
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "tenure": -1,
        "phoneservice": "yes",
        "multiplelines": "no",
        "internetservice": "fiber_optic",
        "onlinesecurity": "no",
        "onlinebackup": "yes",
        "deviceprotection": "no",
        "techsupport": "no",
        "streamingtv": "yes",
        "streamingmovies": "no",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "monthlycharges": 79.85,
        "totalcharges": 950.20
    }

    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_predict_get_not_allowed(client):
    response = client.get("/predict")

    assert response.status_code == 405