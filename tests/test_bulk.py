import io

from conftest import auth_headers, login

CSV_HEADER = (
    "Employee_ID,Age,Gender,Department,Job_Title,Years_at_Company,"
    "Satisfaction_Level,Average_Monthly_Hours,Promotion_Last_5Years,Salary,Attrition"
)
GOOD_ROW = "1,30,Male,Engineering,Engineer,5,0.7,160,0,60000,0"
BAD_CAT_ROW = "2,30,Robot,Engineering,Engineer,5,0.7,160,0,60000,0"
BAD_NUM_ROW = "3,abc,Male,Engineering,Engineer,5,0.7,160,0,60000,0"


def _csv(rows):
    return (CSV_HEADER + "\n" + "\n".join(rows) + "\n").encode()


def _upload(client, csrf, data, filename="team.csv"):
    return client.post(
        "/api/bulk_predict",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        headers=auth_headers(csrf),
    )


def test_bulk_requires_auth(client):
    assert client.post("/api/bulk_predict").status_code in (400, 401, 403)


def test_bulk_success(client):
    csrf = login(client)
    r = _upload(client, csrf, _csv([GOOD_ROW, GOOD_ROW]))
    assert r.status_code == 200
    body = r.get_json()
    assert body["summary"] == {"total": 2, "succeeded": 2, "failed": 0}
    assert len(body["results"]) == 2
    assert 0 <= body["results"][0]["risk"] <= 1


def test_bulk_partial_row_errors(client):
    """One invalid row must not sink the whole batch."""
    csrf = login(client)
    r = _upload(client, csrf, _csv([GOOD_ROW, BAD_CAT_ROW, BAD_NUM_ROW, GOOD_ROW]))
    assert r.status_code == 200
    body = r.get_json()
    assert body["summary"]["succeeded"] == 2
    assert body["summary"]["failed"] == 2
    assert {e["row"] for e in body["errors"]} == {3, 4}


def test_bulk_missing_columns(client):
    csrf = login(client)
    bad = b"Age,Gender\n30,Male\n"
    r = _upload(client, csrf, bad)
    assert r.status_code == 422
    assert "Missing required columns" in r.get_json()["error"]


def test_bulk_rejects_non_csv(client):
    csrf = login(client)
    r = _upload(client, csrf, _csv([GOOD_ROW]), filename="data.txt")
    assert r.status_code == 415


def test_bulk_rejects_malformed_csv(client):
    csrf = login(client)
    r = _upload(client, csrf, b"\x00\x01\x02not,a,csv\n\xff\xfe")
    # malformed or missing columns — either way must be a clean 4xx, never 500
    assert r.status_code in (400, 415, 422)
    assert "error" in r.get_json()


def test_bulk_requires_csrf(client):
    login(client)
    r = client.post("/api/bulk_predict",
                    data={"file": (io.BytesIO(_csv([GOOD_ROW])), "t.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 403


def test_bulk_empty_file(client):
    csrf = login(client)
    r = _upload(client, csrf, b"")
    assert r.status_code in (400, 422)
