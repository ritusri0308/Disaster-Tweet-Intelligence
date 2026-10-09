"""
Unit tests for FastAPI REST Service in api/main.py.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from src.config import PROCESSED_PARQUET_PATH
from src.database import load_all_feedback

pytestmark = pytest.mark.usefixtures("tiny_model")

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app"] == "CrisisLens"
        assert "version" in data
        assert data["model_loaded"] is True
        assert data["model_version"]
        assert data["model_vocab_size"] > 0

def test_api_ready_when_model_loaded():
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

def test_api_version():
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["app_name"] == "CrisisLens"
        assert "HumAID" in data["dataset"]
        assert "disclaimer" in data
        assert data["model_version"]

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
        assert {"kind": "trapped", "count": 5} in data["quantities"]
        assert data["model_version"]

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

def test_api_batch_matches_single_results():
    """Batched inference must give the same answer as analysing each tweet on its own."""
    tweets = [
        {"text": "SOS 5 people trapped near Houston, send boats", "event": "hurricane_harvey_2017"},
        {"text": "Death toll hits 83 after earthquake in Kaikoura", "event": "kaikoura_earthquake_2016"},
        {"text": "Praying for everyone affected", "event": None},
    ]
    with TestClient(app) as client:
        batch = client.post("/analyze/batch", json={"tweets": tweets}).json()
        singles = [client.post("/analyze", json=t).json() for t in tweets]
    assert batch == singles

def test_api_analyze_invalid_input():
    with TestClient(app) as client:
        # Empty string violates min_length=1
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

def test_api_batch_limit_enforced_by_schema():
    with TestClient(app) as client:
        payload = {"tweets": [{"text": "x"}] * 501}
        assert client.post("/analyze/batch", json=payload).status_code == 422

def test_api_returns_503_when_model_missing(monkeypatch):
    """A triage system must refuse to answer rather than silently label everything 'General'."""
    monkeypatch.setattr(api_main, "_MODEL_PIPELINE", None)
    client = TestClient(app)  # no lifespan: the model stays unloaded
    response = client.post("/analyze", json={"text": "SOS trapped"})
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()
    assert client.get("/health").json()["status"] == "unavailable"
    assert client.get("/ready").status_code == 503

def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("CRISISLENS_API_KEY", "s3cret")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # probes stay open
        assert client.post("/analyze", json={"text": "SOS trapped"}).status_code == 401
        assert client.post("/analyze", json={"text": "SOS trapped"}, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = client.post("/analyze", json={"text": "SOS trapped"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200

def test_api_request_id_header_propagated_and_generated():
    with TestClient(app) as client:
        echoed = client.get("/health", headers={"X-Request-ID": "trace-123"})
        assert echoed.headers["X-Request-ID"] == "trace-123"
        generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        assert generated.headers["X-Request-ID"] != "bad id with spaces"
        assert len(generated.headers["X-Request-ID"]) == 32
        assert "X-Process-Time-ms" in generated.headers

def test_api_feedback_round_trip():
    payload = {
        "tweet_id": "live-1",
        "text": "Water is rising fast, we need a boat",
        "event": "hurricane_harvey_2017",
        "original_category": "General",
        "corrected_category": "Help Request",
        "original_urgency": "LOW",
        "corrected_urgency": "high",
        "notes": "Direct plea",
        "reviewer": "op-7",
    }
    with TestClient(app) as client:
        response = client.post("/feedback", json=payload)
        assert response.status_code == 201
        assert response.json()["feedback_id"] > 0
    fb = load_all_feedback()
    row = fb[fb["tweet_id"] == "live-1"].iloc[0]
    assert row["corrected_category"] == "Help Request"
    assert row["corrected_urgency"] == "HIGH"
    assert row["tweet_text"] == "Water is rising fast, we need a boat"
    assert row["split"] == "live"
    assert row["reviewer"] == "op-7"

def test_api_feedback_rejects_unknown_category():
    bad = {
        "tweet_id": "x", "original_category": "General", "corrected_category": "Nonsense",
        "original_urgency": "LOW", "corrected_urgency": "LOW",
    }
    with TestClient(app) as client:
        assert client.post("/feedback", json=bad).status_code == 422

def test_api_surges_and_incidents_need_dataset():
    if PROCESSED_PARQUET_PATH.exists():
        PROCESSED_PARQUET_PATH.unlink()
    with TestClient(app) as client:
        assert client.get("/surges").status_code == 503
        assert client.get("/incidents", params={"event": "hurricane_harvey_2017"}).status_code == 503

def test_api_surges_and_incidents_with_dataset(rich_dataframe):
    rich_dataframe.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    api_main._DATASET_CACHE.update({"mtime": None, "df": None})
    try:
        with TestClient(app) as client:
            surges = client.get("/surges", params={"limit": 5})
            assert surges.status_code == 200
            assert isinstance(surges.json(), list)

            incidents = client.get("/incidents", params={"event": "hurricane_harvey_2017"})
            assert incidents.status_code == 200
            assert isinstance(incidents.json(), list)

            assert client.get("/incidents", params={"event": "no_such_event"}).status_code == 404
            assert client.get("/incidents").status_code == 422  # event is required
    finally:
        PROCESSED_PARQUET_PATH.unlink()
        api_main._DATASET_CACHE.update({"mtime": None, "df": None})
