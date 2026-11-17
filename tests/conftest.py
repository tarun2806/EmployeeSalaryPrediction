import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db, RateLimitHit, PredictionHistory  # noqa: E402

VALID_PAYLOAD = {
    "Name": "Test Employee", "Age": 30, "Gender": "Male",
    "Department": "Engineering", "Job_Title": "Engineer",
    "Experience": 5, "Promotion": "No", "Satisfaction": 0.7, "Hours": 160,
}

@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "legacy")
    application = create_app(testing=True)
    yield application

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture(autouse=True)
def _reset_state(app):
    with app.app_context():
        RateLimitHit.query.delete()
        db.session.commit()
    yield

def login(client, username=None, password=None):
    return "noauth"  # compatibility shim; app has no auth

def auth_headers(csrf=None):
    return {}

def make_prediction(client, csrf=None, **overrides):
    payload = dict(VALID_PAYLOAD); payload.update(overrides)
    return client.post("/api/predict", json=payload)

def count_history(app):
    with app.app_context():
        return PredictionHistory.query.count()
