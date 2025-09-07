from fastapi.testclient import TestClient
from api.main import app, FEATURES

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_roundtrip():
    payload = {"features": {f: 0.0 for f in FEATURES}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "probability" in r.json()
    assert "label" in r.json()
