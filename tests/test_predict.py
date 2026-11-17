from conftest import VALID_PAYLOAD, auth_headers, count_history, login, make_prediction


def test_predict_requires_auth(client):
    assert client.post("/api/predict", json=VALID_PAYLOAD).status_code == 401


def test_predict_success_and_persistence(client, app):
    csrf = login(client)
    before = count_history(app)
    resp = make_prediction(client, csrf)
    assert resp.status_code == 200
    body = resp.get_json()
    assert 0 <= body["risk"] <= 1
    assert body["salary"] > 0
    assert len(body["feat_imp"]) == 8
    assert count_history(app) == before + 1


def test_predict_simulate_does_not_persist(client, app):
    csrf = login(client)
    before = count_history(app)
    resp = make_prediction(client, csrf, simulate=True, Name="Sim")
    assert resp.status_code == 200
    assert count_history(app) == before


def test_predict_requires_csrf(client):
    login(client)
    assert client.post("/api/predict", json=VALID_PAYLOAD).status_code == 403


def test_predict_validation_errors(client):
    csrf = login(client)
    cases = [
        {"Age": 5},
        {"Age": "abc"},
        {"Satisfaction": 1.5},
        {"Hours": -10},
        {"Experience": 100},
        {"Gender": "Robot"},
        {"Department": "Legal"},
        {"Job_Title": "Wizard"},
        {"Promotion": "Maybe"},
        {"Name": "x" * 200},
    ]
    for override in cases:
        resp = make_prediction(client, csrf, **override)
        assert resp.status_code == 422, (override, resp.status_code, resp.get_json())
        assert "error" in resp.get_json()


def test_predict_missing_body(client):
    csrf = login(client)
    resp = client.post("/api/predict", data="not json", content_type="application/json",
                       headers=auth_headers(csrf))
    assert resp.status_code == 400
    resp2 = client.post("/api/predict", json=[1, 2, 3], headers=auth_headers(csrf))
    assert resp2.status_code == 400


def test_predict_deterministic(client):
    csrf = login(client)
    r1 = make_prediction(client, csrf).get_json()
    r2 = make_prediction(client, csrf).get_json()
    assert r1["risk"] == r2["risk"]
    assert r1["salary"] == r2["salary"]


def test_error_does_not_leak_internals(client):
    csrf = login(client)
    # trigger a failure path: oversized name passes validation? use bad type for whole body
    resp = client.post("/api/predict", json=[1, 2, 3], headers=auth_headers(csrf))
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Traceback" not in str(body)
