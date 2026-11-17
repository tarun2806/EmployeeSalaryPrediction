import io


def test_history_pagination(client, app):
    for i in range(25):
        client.post("/api/predict", json={"Name": f"E{i}", "Age":30,"Gender":"Male","Department":"Engineering","Job_Title":"Engineer","Experience":5,"Promotion":"No","Satisfaction":0.7,"Hours":160})
    r = client.get("/api/history?page=1&page_size=20")
    assert r.status_code == 200
    b = r.get_json()
    assert b["total"] == 25 and len(b["items"]) == 20
    assert len(client.get("/api/history?page=2&page_size=20").get_json()["items"]) == 5


def test_history_bad_pagination(client):
    assert client.get("/api/history?page=0").status_code == 422
    assert client.get("/api/history?page_size=500").status_code == 422


def test_history_delete_and_clear(client):
    client.post("/api/predict", json={"Name":"X","Age":30,"Gender":"Male","Department":"Engineering","Job_Title":"Engineer","Experience":5,"Promotion":"No","Satisfaction":0.7,"Hours":160})
    rid = client.get("/api/history").get_json()["items"][0]["id"]
    assert client.delete(f"/api/history/{rid}").status_code == 200
    assert client.delete("/api/history/99999").status_code == 404
    assert client.delete("/api/history").status_code == 200
    assert client.get("/api/history").get_json()["total"] == 0


def test_xss_payload_inert(client):
    payload = '<script>alert(1)</script>'
    client.post("/api/predict", json={"Name":payload,"Age":30,"Gender":"Male","Department":"Engineering","Job_Title":"Engineer","Experience":5,"Promotion":"No","Satisfaction":0.7,"Hours":160})
    item = client.get("/api/history").get_json()["items"][0]
    assert item["name"] == payload


def test_frontend_safe_dom():
    src = open("static/script.js").read()
    assert "${row.name}" not in src
    for line in src.splitlines():
        if "innerHTML" in line:
            assert 'innerHTML = ""' in line, f"unsafe innerHTML: {line}"
