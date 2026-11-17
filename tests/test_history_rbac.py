from conftest import VALID_PAYLOAD, auth_headers, login, make_prediction


def _seed_history(client, csrf, app, n=3):
    for i in range(n):
        make_prediction(client, csrf, Name=f"Emp{i}", simulate=False)


def test_history_requires_auth(client):
    assert client.get("/api/history").status_code == 401


def test_history_pagination_shape(client, app):
    csrf = login(client)
    _seed_history(client, csrf, app, n=25)
    r = client.get("/api/history?page=1&page_size=20")
    assert r.status_code == 200
    body = r.get_json()
    assert body["total"] == 25
    assert len(body["items"]) == 20
    assert body["page"] == 1
    r2 = client.get("/api/history?page=2&page_size=20")
    assert len(r2.get_json()["items"]) == 5


def test_history_bad_pagination(client):
    login(client)
    assert client.get("/api/history?page=0").status_code == 422
    assert client.get("/api/history?page_size=0").status_code == 422
    assert client.get("/api/history?page_size=500").status_code == 422


def test_history_clear_admin_only(client, seeded_users):
    csrf_admin = login(client)
    _seed_history(client, csrf_admin, seeded_users, n=2)

    # analyst cannot clear
    client.post("/api/auth/logout", headers=auth_headers(csrf_admin))
    csrf_an = login(client, "analyst", "userpass123")
    r = client.delete("/api/history", headers=auth_headers(csrf_an))
    assert r.status_code == 403

    # hr cannot clear all
    client.post("/api/auth/logout", headers=auth_headers(csrf_an))
    csrf_hr = login(client, "hr", "userpass123")
    r = client.delete("/api/history", headers=auth_headers(csrf_hr))
    assert r.status_code == 403

    # admin can
    client.post("/api/auth/logout", headers=auth_headers(csrf_hr))
    csrf_admin = login(client)
    r = client.delete("/api/history", headers=auth_headers(csrf_admin))
    assert r.status_code == 200
    assert client.get("/api/history").get_json()["total"] == 0


def test_history_record_delete_roles(client, seeded_users, app):
    csrf_admin = login(client)
    make_prediction(client, csrf_admin, Name="Victim")
    items = client.get("/api/history").get_json()["items"]
    rid = items[0]["id"]

    # analyst forbidden
    client.post("/api/auth/logout", headers=auth_headers(csrf_admin))
    csrf_an = login(client, "analyst", "userpass123")
    assert client.delete(f"/api/history/{rid}", headers=auth_headers(csrf_an)).status_code == 403

    # hr allowed
    client.post("/api/auth/logout", headers=auth_headers(csrf_an))
    csrf_hr = login(client, "hr", "userpass123")
    assert client.delete(f"/api/history/{rid}", headers=auth_headers(csrf_hr)).status_code == 200
    assert client.get("/api/history").get_json()["total"] == 0


def test_history_delete_requires_csrf(client):
    csrf = login(client)
    assert client.delete("/api/history").status_code == 403
    assert client.delete("/api/history/1").status_code == 403


def test_xss_payload_stored_as_inert_string(client, app):
    csrf = login(client)
    payload = '<script>alert("xss")</script><img src=x onerror=alert(1)>'
    make_prediction(client, csrf, Name=payload)
    items = client.get("/api/history").get_json()["items"]
    assert items[0]["name"] == payload  # stored verbatim; JSON transport is safe


def test_frontend_uses_safe_dom_for_history():
    """Regression: script.js must not inject row.name via innerHTML."""
    src = open("static/script.js").read()
    assert "${row.name}" not in src
    assert "row.name}</" not in src
    # innerHTML may only be used for clearing containers
    for line in src.splitlines():
        if "innerHTML" in line:
            assert 'innerHTML = ""' in line, f"unsafe innerHTML usage: {line}"
