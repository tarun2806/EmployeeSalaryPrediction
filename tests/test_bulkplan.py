import io
import pytest

HDR = ("Employee_ID,Employee_Name,Job_Title,City,Years_Experience,"
       "Current_Annual_Salary")


def row(eid="E1", name="Alice", title="Software Engineer", city="Bengaluru",
        exp=4, sal=1200000):
    return f"{eid},{name},{title},{city},{exp},{sal}"


def upload(client, content, name="sheet.csv", endpoint="validate"):
    return client.post(f"/api/bulk/{endpoint}",
                       data={"file": (io.BytesIO(content), name)},
                       content_type="multipart/form-data")


def csv(*rows):
    return (HDR + "\n" + "\n".join(rows) + "\n").encode()


def test_template_downloads(client):
    r = client.get("/api/bulk/template?format=csv")
    assert r.status_code == 200 and b"Employee_ID" in r.data
    r = client.get("/api/bulk/template?format=xlsx")
    assert r.status_code == 200 and r.data[:2] == b"PK"


def test_validate_ok(client):
    r = upload(client, csv(row(), row("E2", "Bob", "Data Scientist", "Pune", 6, 1800000)), "v.csv")
    assert r.status_code == 200
    b = r.get_json()
    assert b["valid_rows"] == 2 and b["invalid_rows"] == 0
    assert b["kind"] == "india_salary"


def test_validate_missing_and_dup(client):
    content = csv(row(),
                  "E1,Dup,Software Engineer,Bengaluru,4,1200000",
                  "E3,NoTitle,,,4,1200000")
    r = upload(client, content, "v.csv")
    b = r.get_json()
    assert b["valid_rows"] == 1 and b["invalid_rows"] == 2
    errs = " ".join(e["error"] for e in b["errors"])
    assert "duplicate" in errs


def test_validate_unsupported_title(client):
    r = upload(client, csv(row(title="Astronaut")), "v.csv")
    b = r.get_json()
    assert b["valid_rows"] == 0 and b["invalid_rows"] == 1


def test_process_creates_job(client):
    r = client.post("/api/bulk/process",
                    data={"file": (io.BytesIO(csv(row(), row("E2", "Bob", sal=1500000))), "p.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    jid = r.get_json()["job_id"]
    d = client.get(f"/api/bulk/jobs/{jid}").get_json()
    assert d["valid_rows"] == 2 and len(d["results"]) == 2
    assert d["results"][0]["Estimated_Salary"] > 0
    assert d["results"][0]["Salary_P10"] <= d["results"][0]["Estimated_Salary"] <= d["results"][0]["Salary_P90"]
    assert d["model_version"] == "india-salary"


def test_scenario_calc(client):
    r = client.post("/api/bulk/process",
                    data={"file": (io.BytesIO(csv(row(), row("E2", "Bob", sal=1500000))), "s.csv")},
                    content_type="multipart/form-data")
    jid = r.get_json()["job_id"]
    sc = client.post(f"/api/bulk/jobs/{jid}/scenario",
                     json={"increment_pct": 10, "budget": 100000000}).get_json()
    # India salaries are ANNUAL INR: current = 1200000 + 1500000
    assert sc["current_annual_payroll"] == 2700000
    assert sc["projected_annual_payroll"] == 2970000
    assert sc["total_additional_annual_cost"] == 270000
    assert sc["budget_exceeded"] is False


def test_export(client):
    r = client.post("/api/bulk/process",
                    data={"file": (io.BytesIO(csv(row())), "e.csv")},
                    content_type="multipart/form-data")
    jid = r.get_json()["job_id"]
    rx = client.get(f"/api/bulk/jobs/{jid}/export?format=xlsx")
    assert rx.status_code == 200 and rx.data[:2] == b"PK"
    rc = client.get(f"/api/bulk/jobs/{jid}/export?format=csv")
    assert rc.status_code == 200 and b"Estimated_Salary" in rc.data


def test_job_not_found(client):
    assert client.get("/api/bulk/jobs/99999").status_code == 404
    assert client.get("/api/bulk/jobs/99999/export").status_code == 404
