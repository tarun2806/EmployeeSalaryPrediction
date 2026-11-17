import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import User, create_app, db, login_attempts, rate_limiter  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    login_attempts.clear()
    rate_limiter._hits.clear()
    yield
    login_attempts.clear()
    rate_limiter._hits.clear()

ADMIN_USER = "admin"
ADMIN_PASS = "adminpass123"

VALID_PAYLOAD = {
    "Name": "Test Employee",
    "Age": 30,
    "Gender": "Male",
    "Department": "Engineering",
    "Job_Title": "Engineer",
    "Experience": 5,
    "Promotion": "No",
    "Satisfaction": 0.7,
    "Hours": 160,
}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN_USER)
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASS)
    application = create_app(testing=True)
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def seeded_users(app):
    with app.app_context():
        for username, role in [("hr", "HR_MANAGER"), ("analyst", "ANALYST")]:
            db.session.add(User(
                username=username,
                password_hash=generate_password_hash("userpass123"),
                role=role,
            ))
        db.session.commit()
    return app


def login(client, username=ADMIN_USER, password=ADMIN_PASS):
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()["csrf_token"]


def auth_headers(csrf):
    return {"X-CSRF-Token": csrf}


def make_prediction(client, csrf, **overrides):
    payload = dict(VALID_PAYLOAD)
    payload.update(overrides)
    return client.post("/api/predict", json=payload, headers=auth_headers(csrf))


def count_history(app):
    from app import PredictionHistory
    with app.app_context():
        return PredictionHistory.query.count()
