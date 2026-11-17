import io
import pytest

V2_PAYLOAD = {"Name":"V2","Age":36,"Gender":"Female","Department":"Research & Development","Job_Title":"Research Scientist","Experience":5,"Satisfaction":0.6,"Promotion":"No"}
VALID = {"Name":"T","Age":30,"Gender":"Male","Department":"Engineering","Job_Title":"Engineer","Experience":5,"Promotion":"No","Satisfaction":0.7,"Hours":160}

def test_model_info(client):
    b = client.get("/api/model").get_json()
    assert b["active_version"] == "legacy" and "v2-ibm" in b["available_versions"]

def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in r.headers

def test_predict_metadata(client):
    b = client.post("/api/predict", json=VALID).get_json()
    assert b["model_version"] == "legacy" and "explanation" in b and "disclaimer" in b

def test_analytics_summary_empty(client):
    assert client.get("/api/analytics/summary").get_json()["total"] == 0

def test_rate_limiter_db(client, app):
    from app import rate_allow, RateLimitHit
    with app.app_context():
        for _ in range(5): assert rate_allow("k",5,60) is True
        assert rate_allow("k",5,60) is False

@pytest.fixture()
def v2_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path/"v2.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "v2-ibm")
    from app import create_app
    return create_app(testing=True).test_client()

def test_v2_info(v2_client):
    b = v2_client.get("/api/model").get_json()
    assert b["active_version"] == "v2-ibm" and "Hours" not in b["features"]

def test_v2_predict(v2_client):
    r = v2_client.post("/api/predict", json=V2_PAYLOAD)
    assert r.status_code == 200 and r.get_json()["model_version"] == "v2-ibm"

def test_v2_rejects_legacy(v2_client):
    r = v2_client.post("/api/predict", json={**V2_PAYLOAD, "Department":"Engineering", "Job_Title":"Engineer"})
    assert r.status_code == 422

def test_v2_hours_not_required(v2_client):
    p = dict(V2_PAYLOAD); p.pop("Hours", None)
    assert v2_client.post("/api/predict", json=p).status_code == 200

def test_unknown_version_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path/"x.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "nope")
    from app import load_all
    assert load_all() is False
