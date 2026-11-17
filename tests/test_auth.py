import pytest

from conftest import ADMIN_PASS, ADMIN_USER, auth_headers, login


def test_health_requires_auth(client):
    assert client.get("/api/health").status_code == 401


def test_login_success(client):
    resp = client.post("/api/auth/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["user"]["role"] == "ADMIN"
    assert "csrf_token" in body
    assert "aurora_session" in resp.headers.get("Set-Cookie", "")


def test_login_wrong_password(client):
    resp = client.post("/api/auth/login", json={"username": ADMIN_USER, "password": "wrong"})
    assert resp.status_code == 401


def test_login_missing_fields(client):
    assert client.post("/api/auth/login", json={}).status_code == 400
    assert client.post("/api/auth/login", json={"username": "admin"}).status_code == 400


def test_login_rate_limit(client):
    for _ in range(5):
        client.post("/api/auth/login", json={"username": ADMIN_USER, "password": "bad"})
    resp = client.post("/api/auth/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    assert resp.status_code == 429


def test_me_and_logout(client):
    csrf = login(client)
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.get_json()["user"]["username"] == ADMIN_USER

    out = client.post("/api/auth/logout", headers=auth_headers(csrf))
    assert out.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_logout_requires_csrf(client):
    login(client)
    assert client.post("/api/auth/logout").status_code == 403


def test_passwords_are_hashed(seeded_users):
    from app import User
    with seeded_users.app_context():
        user = User.query.filter_by(username="hr").first()
        assert user.password_hash != "userpass123"
        assert user.password_hash.startswith(("scrypt:", "pbkdf2:"))


def test_create_user_validation(client):
    csrf = login(client)
    # weak password rejected
    r = client.post("/api/users", json={"username": "newuser", "password": "short", "role": "ANALYST"},
                    headers=auth_headers(csrf))
    assert r.status_code == 422
    # bad role rejected
    r = client.post("/api/users", json={"username": "newuser", "password": "longenough1", "role": "SUPERUSER"},
                    headers=auth_headers(csrf))
    assert r.status_code == 422
    # valid creation
    r = client.post("/api/users", json={"username": "newuser", "password": "longenough1", "role": "ANALYST"},
                    headers=auth_headers(csrf))
    assert r.status_code == 201
    # duplicate rejected
    r = client.post("/api/users", json={"username": "newuser", "password": "longenough1", "role": "ANALYST"},
                    headers=auth_headers(csrf))
    assert r.status_code == 409
