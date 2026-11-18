from conftest import VALID_PAYLOAD


def test_predict_success(client, app):
    before = __import__('conftest').count_history(app)
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["salary_unit"] == "inr_annual"
    assert body["salary_annual_inr"] > 0
    assert body["salary_p10_inr"] <= body["salary_annual_inr"] <= body["salary_p90_inr"]
    assert __import__('conftest').count_history(app) == before + 1


def test_predict_simulate_no_persist(client, app):
    from conftest import count_history
    before = count_history(app)
    resp = client.post("/api/predict", json={**VALID_PAYLOAD, "simulate": True})
    assert resp.status_code == 200
    assert count_history(app) == before


def test_predict_unsupported_title(client):
    r = client.post("/api/predict", json={**VALID_PAYLOAD, "Job_Title": "Astronaut"})
    assert r.status_code == 422


def test_predict_bad_body(client):
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 400


def test_predict_deterministic(client):
    a = client.post("/api/predict", json=VALID_PAYLOAD).get_json()
    b = client.post("/api/predict", json=VALID_PAYLOAD).get_json()
    assert a["salary_annual_inr"] == b["salary_annual_inr"]


def test_error_does_not_leak_internals(client):
    r = client.post("/api/predict", json={"Job_Title": 123})
    body = r.get_json()
    assert r.status_code in (400, 422)
    assert "traceback" not in str(body).lower()
    assert "File \"" not in str(body)
