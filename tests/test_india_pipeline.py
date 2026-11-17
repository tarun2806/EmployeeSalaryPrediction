import json
import numpy as np
import pandas as pd
import pytest
from ml import india_pipeline as ip
from ml import synthetic_generator as sg


@pytest.fixture(scope="module")
def india():
    return ip.load_india()


def test_raw_files_unchanged_hashes():
    # sha256 recorded in dataset_quality.json matches current files
    q = json.loads((ip.RESULTS / "dataset_quality.json").read_text())
    assert q["india_raw"]["sha256"] == ip.sha256(ip.INDIA_CSV)
    assert q["labour_bureau"]["sha256"] == ip.sha256(ip.LABOUR_CSV)


def test_india_loads_and_schema(india):
    assert len(india) == 12958
    assert "salary_mid_inr" in india.columns and "role_family" in india.columns


def test_cleaning_rejects_with_reasons(india):
    clean, rejected = ip.clean_india(india)
    assert len(clean) + len(rejected) == len(india)
    assert (rejected["reject_reason"] != "").all()
    assert clean["salary_mid_inr"].between(50000, 8000000).all()
    assert not clean.duplicated(subset=["company", "job_title_norm", "city_norm", "posted_month", "salary_mid_inr"]).any()


def test_cleaning_reproducible(india):
    c1, r1 = ip.clean_india(india)
    c2, r2 = ip.clean_india(india)
    assert len(c1) == len(c2) and len(r1) == len(r2)
    assert c1["salary_mid_inr"].equals(c2["salary_mid_inr"])


def test_taxonomy_support_flags():
    tax = pd.read_csv(ip.PROCESSED / "job_taxonomy.csv")
    assert tax["canonical_title"].is_unique
    assert tax["supported"].sum() < len(tax)  # long tail exists
    assert set(tax.columns) >= {"canonical_title", "job_family", "department",
                                "observations", "supported", "mapping_confidence"}


def test_features_no_leakage():
    clean = pd.read_csv(ip.PROCESSED / "india_salary_clean.csv")
    X, y, groups, _ = ip.engineer_features(clean)
    banned = {"posting_id", "company", "employee_name", "salary_mid_inr",
              "salary_min_inr", "salary_max_inr", "fliar_profile_url"}
    assert not (set(X.columns) & banned)
    assert len(X) == len(y) and y.notna().all()


def test_model_comparison_exists_and_honest():
    comp = json.loads((ip.RESULTS / "india_model_comparison.json").read_text())
    assert comp["best_model"] in comp["results"]
    for name, r in comp["results"].items():
        h = r["holdout"]
        assert 0 < h["mae"] and -1 <= h["r2"] <= 1
    # baselines present for comparison
    assert "median_global" in comp["results"] and "median_family" in comp["results"]


def test_synthetic_reproducible_and_flagged():
    df1 = sg.generate(2000, seed=7)
    df2 = sg.generate(2000, seed=7)
    assert df1.equals(df2)
    assert (df1["data_origin"] == "synthetic").all()
    assert (df1["generation_version"] == "syn-v1-seed7").all()
    assert "employee_id" not in ip.FEATURE_COLS  # synthetic ids never become features


def test_synthetic_salary_calibration():
    df = pd.read_csv(ip.BASE / "data" / "synthetic" / "workforce_synthetic_50k.csv")
    assert len(df) == 50000
    clean = pd.read_csv(ip.PROCESSED / "india_salary_clean.csv")
    # synthetic family medians should be within plausible range of real medians
    for fam in df.job_family.unique():
        real = clean[clean.role_family_norm == fam].salary_mid_inr.median()
        syn = df[df.job_family == fam].current_annual_salary_inr.median()
        assert 0.3 * real < syn < 3.0 * real, fam


def test_labour_bureau_not_treated_as_salaries():
    lb = ip.load_labour()
    assert "Absolute Wages" in lb.columns
    # values are small (avg daily wage INR), must NOT be used as annual salary
    assert lb["Absolute Wages"].max() < 5000
