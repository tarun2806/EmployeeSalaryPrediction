from conftest import VALID_PAYLOAD, make_prediction


def test_predict_success(client, app):
    before = __import__('conftest').count_history(app)
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert 0 <= body["risk"] <= 1 and body["salary"] > 0
    assert len(body["feat_imp"]) == 8
    assert __import__('conftest').count_history(app) == before + 1


def test_predict_simulate_no_persist(client, app):
    from conftest import count_history
    before = count_history(app)
    resp = client.post("/api/predict", json={**VALID_PAYLOAD, "simulate": True})
    assert resp.status_code == 200
    assert count_history(app) == before


def test_predict_validation(client):
    for over in [{"Age":5},{"Satisfaction":1.5},{"Gender":"Robot"},{"Department":"Legal"}]:
        r = client.post("/api/predict", json={**VALID_PAYLOAD, **over})
        assert r.status_code == 422, (over, r.status_code)


def test_predict_bad_body(client):
    assert client.post("/api/predict", json=[1,2,3]).status_code == 400


def test_predict_deterministic(client):
    a = client.post("/api/predict", json=VALID_PAYLOAD).get_json()
    b = client.post("/api/predict", json=VALID_PAYLOAD).get_json()
    assert a["risk"] == b["risk"] and a["salary"] == b["salary"]

def test_error_does_not_leak_internals(client):
    r = client.post("/api/predict", json={"Age": "abc"})
    body = r.get_json()
    assert r.status_code == 422
    assert "traceback" not in str(body).lower()
    assert "File \"" not in str(body)
