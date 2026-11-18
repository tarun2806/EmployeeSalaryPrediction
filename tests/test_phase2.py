import pytest

PAYLOAD = {"Name": "T", "Job_Title": "Software Engineer", "Experience": 4, "City": "Bengaluru"}


def test_model_info(client):
    b = client.get("/api/model").get_json()
    assert b["kind"] == "india_salary" and b["active_version"] == "india-salary"
    assert b["salary_unit"] == "inr_annual"
    assert "Job_Title" in b["categories"]
    assert "v2-ibm" not in str(b) and "ibm" not in b.get("dataset", "").lower()


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in r.headers


def test_analytics_summary_empty(client):
    assert client.get("/api/analytics/summary").get_json()["total"] == 0


def test_health(client):
    b = client.get("/api/health").get_json()
    assert b["status"] == "ok" and b["models_loaded"] is True


def test_rate_limiter_db(client, app):
    from app import rate_allow
    with app.app_context():
        for _ in range(5):
            assert rate_allow("k", 5, 60) is True
        assert rate_allow("k", 5, 60) is False


def test_unknown_version_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "nope")
    from app import load_all
    assert load_all() is False


def test_predict_india_metadata(client):
    b = client.post("/api/predict", json=PAYLOAD).get_json()
    assert b["model_version"] == "india-salary"
    assert b["salary_unit"] == "inr_annual"
    assert "canonical_title" in b and "support_level" in b
