"""Phase 2 tests: model registry, simulation, audit, analytics, security."""
import io
import json

import pytest

from conftest import ADMIN_PASS, ADMIN_USER, VALID_PAYLOAD, auth_headers, login, make_prediction


def test_model_info_endpoint(client):
    csrf = login(client)
    r = client.get("/api/model")
    assert r.status_code == 200
    body = r.get_json()
    assert body["active_version"] == "legacy"
    assert "v2-ibm" in body["available_versions"]
    assert "Hours" in body["features"]
    assert set(body["categories"]["Department"]) >= {"Engineering", "Sales"}


def test_security_headers_present(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in r.headers


def test_predict_returns_model_metadata(client):
    csrf = login(client)
    body = make_prediction(client, csrf).get_json()
    assert body["model_version"] == "legacy"
    assert "explanation" in body
    assert "disclaimer" in body


def test_simulate_does_not_persist_and_labels(client, app):
    from conftest import count_history
    csrf = login(client)
    before = count_history(app)
    r = client.post("/api/simulate", json={
        **VALID_PAYLOAD, "satisfaction_boost": 0.2, "hours_delta": -20,
    }, headers=auth_headers(csrf))
    assert r.status_code == 200
    body = r.get_json()
    assert "baseline" in body and "scenario" in body
    assert "not a causal" in body["disclaimer"] or "cannot prove" in body["disclaimer"]
    assert count_history(app) == before


def test_simulate_validation(client):
    csrf = login(client)
    r = client.post("/api/simulate", json={**VALID_PAYLOAD, "satisfaction_boost": 5.0},
                    headers=auth_headers(csrf))
    assert r.status_code == 422


def test_audit_log_admin_only_and_records_events(client, seeded_users):
    csrf = login(client)
    make_prediction(client, csrf, Name="AuditTest")
    # analyst forbidden
    client.post("/api/auth/logout", headers=auth_headers(csrf))
    csrf_an = login(client, "analyst", "userpass123")
    assert client.get("/api/audit").status_code == 403
    client.post("/api/auth/logout", headers=auth_headers(csrf_an))
    # admin sees events, no secrets stored
    csrf = login(client)
    r = client.get("/api/audit")
    assert r.status_code == 200
    items = r.get_json()["items"]
    actions = {i["action"] for i in items}
    assert "login" in actions and "predict" in actions
    for i in items:
        assert "password" not in (i["detail"] or "").lower()
        assert "csrf" not in (i["detail"] or "").lower()


def test_password_change_flow(client):
    csrf = login(client)
    # wrong current password rejected
    r = client.post("/api/auth/change-password",
                    json={"current_password": "wrong", "new_password": "newpass123"},
                    headers=auth_headers(csrf))
    assert r.status_code == 403
    # weak new password rejected
    r = client.post("/api/auth/change-password",
                    json={"current_password": ADMIN_PASS, "new_password": "short"},
                    headers=auth_headers(csrf))
    assert r.status_code == 422
    # success
    r = client.post("/api/auth/change-password",
                    json={"current_password": ADMIN_PASS, "new_password": "newpass123"},
                    headers=auth_headers(csrf))
    assert r.status_code == 200
    # old password no longer works
    client.post("/api/auth/logout", headers=auth_headers(csrf))
    assert client.post("/api/auth/login",
                       json={"username": ADMIN_USER, "password": ADMIN_PASS}).status_code == 401
    assert client.post("/api/auth/login",
                       json={"username": ADMIN_USER, "password": "newpass123"}).status_code == 200


def test_rate_limit_persistent_across_clients(client, app):
    """DB-backed limiter: hits accumulate across separate clients/sessions."""
    from app import RateLimitHit
    csrf = login(client)
    # burn through predict limit (60/min) is too many; test the mechanism directly
    from app import rate_allow
    with app.app_context():
        for _ in range(5):
            assert rate_allow("testkey", 5, 60) is True
        assert rate_allow("testkey", 5, 60) is False
        assert RateLimitHit.query.filter(RateLimitHit.key.like("testkey:%")).count() == 1


def test_analytics_upload(client):
    csrf = login(client)
    csv_data = (
        "Employee_ID,Age,Department,Satisfaction_Level,Average_Monthly_Hours,Salary,Attrition\n"
        "1,30,Sales,0.8,160,50000,0\n"
        "2,40,Sales,0.4,180,55000,1\n"
        "3,35,HR,0.9,150,52000,0\n"
    ).encode()
    r = client.post("/api/analytics/upload",
                    data={"file": (io.BytesIO(csv_data), "team.csv")},
                    content_type="multipart/form-data",
                    headers=auth_headers(csrf))
    assert r.status_code == 200
    body = r.get_json()
    assert body["stats"]["rows"] == 3
    assert "Sales" in body["stats"]["by_department"]
    assert "disclaimer" in body


def test_analytics_summary_empty(client):
    login(client)
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    assert r.get_json()["total"] == 0


# ---------- v2 model version tests ----------

V2_PAYLOAD = {
    "Name": "V2 Test", "Age": 36, "Gender": "Female",
    "Department": "Research & Development", "Job_Title": "Research Scientist",
    "Experience": 5, "Satisfaction": 0.6, "Promotion": "No",
}


@pytest.fixture()
def v2_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_DB_PATH", str(tmp_path / "v2.db"))
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN_USER)
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASS)
    monkeypatch.setenv("AURORA_MODEL_VERSION", "v2-ibm")
    application = __import__("app").create_app(testing=True)
    yield application


@pytest.fixture()
def v2_client(v2_app):
    return v2_app.test_client()


def test_v2_model_info(v2_client):
    login(v2_client)
    body = v2_client.get("/api/model").get_json()
    assert body["active_version"] == "v2-ibm"
    assert "Hours" not in body["features"]
    assert "Research & Development" in body["categories"]["Department"]
    assert body["salary_unit"] == "monthly_income_dataset_currency"


def test_v2_predict_and_explanation(v2_client):
    csrf = login(v2_client)
    r = v2_client.post("/api/predict", json=V2_PAYLOAD, headers=auth_headers(csrf))
    assert r.status_code == 200
    body = r.get_json()
    assert body["model_version"] == "v2-ibm"
    assert 0 <= body["risk"] <= 1
    assert body["salary"] > 0
    assert body.get("explanation")


def test_v2_rejects_legacy_categories(v2_client):
    csrf = login(v2_client)
    bad = dict(V2_PAYLOAD, Department="Engineering", Job_Title="Engineer")
    r = v2_client.post("/api/predict", json=bad, headers=auth_headers(csrf))
    assert r.status_code == 422


def test_v2_hours_ignored_not_required(v2_client):
    csrf = login(v2_client)
    payload = dict(V2_PAYLOAD)
    payload.pop("Hours", None)
    r = v2_client.post("/api/predict", json=payload, headers=auth_headers(csrf))
    assert r.status_code == 200


def test_unknown_model_version_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("AURORA_MODEL_VERSION", "does-not-exist")
    from app import load_all
    assert load_all() is False
