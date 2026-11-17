import io

CSV_HEADER = "Employee_ID,Age,Gender,Department,Job_Title,Years_at_Company,Satisfaction_Level,Average_Monthly_Hours,Promotion_Last_5Years,Salary,Attrition"
GOOD = "1,30,Male,Engineering,Engineer,5,0.7,160,0,60000,0"
BAD = "2,30,Robot,Engineering,Engineer,5,0.7,160,0,60000,0"

def _csv(rows): return (CSV_HEADER+"\n"+"\n".join(rows)+"\n").encode()

def _up(client, data, fn="t.csv"):
    return client.post("/api/bulk_predict", data={"file": (io.BytesIO(data), fn)}, content_type="multipart/form-data")

def test_bulk_success(client):
    r = _up(client, _csv([GOOD, GOOD]))
    assert r.status_code == 200
    b = r.get_json()
    assert b["summary"] == {"total":2,"succeeded":2,"failed":0}

def test_bulk_partial_errors(client):
    r = _up(client, _csv([GOOD, BAD, GOOD]))
    b = r.get_json()
    assert b["summary"]["succeeded"] == 2 and b["summary"]["failed"] == 1

def test_bulk_missing_cols(client):
    assert _up(client, b"Age,Gender\n30,Male\n").status_code == 422

def test_bulk_non_csv(client):
    assert _up(client, _csv([GOOD]), "x.txt").status_code == 415


def test_bulk_malformed(client):
    assert _up(client, b"\x00\x01bad").status_code in (400,415,422)
