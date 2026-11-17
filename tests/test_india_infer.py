import io, json
import pytest
from ml import india_infer as ii

TITLES = ii.supported_titles()


def test_candidate_artifacts_load():
    r = ii.predict_salary("Software Engineer", experience_years=4, city="Bengaluru")
    assert "salary_annual_inr" in r and r["salary_p10_inr"] <= r["salary_annual_inr"] <= r["salary_p90_inr"]


def test_unsupported_title_rejected():
    r = ii.predict_salary("Astronaut", experience_years=4)
    assert "error" in r


def test_alias_and_canonical_resolution():
    canon, info, err = ii.resolve_title("software engineer")
    assert err is None and canon == "Software Engineer"


def test_prediction_deterministic():
    a = ii.predict_salary("Data Scientist", experience_years=5, city="Pune")
    b = ii.predict_salary("Data Scientist", experience_years=5, city="Pune")
    assert a["salary_annual_inr"] == b["salary_annual_inr"]


def test_range_and_metadata():
    r = ii.predict_salary("HR Executive", experience_years=3)
    assert r["salary_unit"] == "inr_annual"
    assert r["market_observations"] >= 5
    assert r["support_level"] in ("high", "medium", "low")
    assert "not a confidence interval" in r["range_note"]


def test_interval_coverage_recorded():
    v = json.loads((ii.BASE / "ml_results" / "interval_validation.json").read_text())
    assert 0.6 < v["model_coverage"] < 1.0
    assert v["model_mean_width_inr"] < v["empirical_width_inr"]


@pytest.fixture()
def india_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path / "i.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "india-salary")
    from app import create_app
    return create_app(testing=True).test_client()


def test_app_india_model_info(india_client):
    m = india_client.get("/api/model").get_json()
    assert m["kind"] == "india_salary" and m["active_version"] == "india-salary"
    assert "Job_Title" in m["categories"]


def test_app_india_predict(india_client):
    r = india_client.post("/api/predict", json={
        "Name": "T", "Job_Title": "Software Engineer", "Experience": 4, "City": "Bengaluru"})
    assert r.status_code == 200
    b = r.get_json()
    assert b["salary_unit"] == "inr_annual" and b["salary_annual_inr"] > 0
    assert b["salary_p10_inr"] <= b["salary_annual_inr"] <= b["salary_p90_inr"]


def test_app_india_unsupported_title(india_client):
    assert india_client.post("/api/predict", json={"Job_Title": "Astronaut", "Experience": 4}).status_code == 422


def test_app_india_bulk(india_client):
    csv_data = ("Employee_ID,Employee_Name,Job_Title,City,Years_Experience,Current_Monthly_Salary\n"
                "E1,Alice,Software Engineer,Bengaluru,4,50000\n").encode()
    r = india_client.post("/api/bulk/process",
                          data={"file": (io.BytesIO(csv_data), "t.csv")},
                          content_type="multipart/form-data")
    assert r.status_code == 200
    jid = r.get_json()["job_id"]
    d = india_client.get(f"/api/bulk/jobs/{jid}").get_json()
    assert d["results"][0]["Estimated_Salary"] > 0
    assert d["results"][0]["Salary_P10"] <= d["results"][0]["Estimated_Salary"] <= d["results"][0]["Salary_P90"]


def test_rollback_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPS_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ESPS_MODEL_VERSION", "legacy")
    from app import create_app
    c = create_app(testing=True).test_client()
    m = c.get("/api/model").get_json()
    assert m["kind"] == "standard" and m["active_version"] == "legacy"


def test_preprocess_persisted_not_recomputed():
    """Inference must use persisted preprocess.json, not mutable CSVs."""
    import json as _j
    pre = _j.loads((ii.MODEL_DIR / "preprocess.json").read_text())
    assert len(pre["top_cities"]) == 40 and len(pre["top_industries"]) == 15
    assert "Bengaluru" in pre["top_cities"]
    # _load exposes the persisted mapping
    assert "Bengaluru" in ii._load()["top_cities"]


def test_quantile_coverage_by_subgroup_recorded():
    v = json.loads((ii.BASE / "ml_results" / "interval_validation.json").read_text())
    assert "coverage_by_family" in v and "coverage_by_exp_band" in v
    assert all(0.5 < f["coverage"] < 1.0 for f in v["coverage_by_family"].values())
