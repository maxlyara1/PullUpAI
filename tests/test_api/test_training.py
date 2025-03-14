import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from backend.main import app

client = TestClient(app)


def test_get_training_data():
    response = client.get("/api/v1/training")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_create_training_data():
    data = {"date": datetime.now().strftime("%Y-%m-%d"), "avg_pullups": 10.0}
    response = client.post("/api/v1/training", json=data)
    assert response.status_code == 200
    assert response.json()["message"] == "Данные успешно добавлены"


def test_delete_training_data():
    response = client.delete("/api/v1/training/0")
    assert response.status_code == 200
    assert response.json()["message"] == "Данные успешно удалены"


def test_update_training_data():
    data = {"date": datetime.now().strftime("%Y-%m-%d"), "avg_pullups": 15.0}
    response = client.put("/api/v1/training/0", json=data)
    assert response.status_code == 200
    assert response.json()["message"] == "Данные успешно обновлены"


def test_reset_training_data():
    response = client.delete("/api/v1/training/reset")
    assert response.status_code == 200
    assert response.json()["message"] == "Все данные успешно сброшены"
