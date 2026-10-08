"""
Unit tests for FastAPI REST Service in api/main.py.
"""

from fastapi.testclient import TestClient
from api.main import app

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app"] == "CrisisLens"
        assert "version" in data

def test_api_version():
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["app_name"] == "CrisisLens"
        assert "HumAID" in data["dataset"]
        assert "disclaimer" in data

def test_api_analyze_single():
    with TestClient(app) as client:
        payload = {
            "text": "URGENT SOS! 5 people trapped on roof due to severe flooding in Houston! Send rescue boat immediately!",
            "event": "hurricane_harvey_2017"
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "clean_text" in data
        assert "category" in data
        assert "confidence" in data
        assert data["urgency_score"] >= 5.0
        assert data["urgency_level"] in ["HIGH", "CRITICAL"]
        assert "Rescue/Evacuation" in data["needs"]
        assert any(loc.get("name") == "Houston" for loc in data["geocoded_locations"])
        assert len(data["urgency_breakdown"]) > 0

def test_api_analyze_batch():
    with TestClient(app) as client:
        payload = {
            "tweets": [
                {"text": "Flood water entering house in Houston!", "event": "hurricane_harvey_2017"},
                {"text": "Prayers for everyone affected by the storm.", "event": "hurricane_harvey_2017"}
            ]
        }
        response = client.post("/analyze/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert "category" in data[0]
        assert "category" in data[1]

def test_api_analyze_invalid_input():
    with TestClient(app) as client:
        # Empty string violates min_length=1
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422
