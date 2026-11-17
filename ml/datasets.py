"""ESPS dataset loaders, schema validation and quality reporting.

Two dataset categories:
  A) SYNTHETIC — employee_attrition_data.csv (demonstration baseline, generated data)
  B) IBM_HR   — data/ibm_hr_attrition.csv (IBM Watson Analytics sample data;
     fictional data created by IBM data scientists for a demo, publicly mirrored.
     NOT real employee records.)
"""

import hashlib
import json

import numpy as np
import pandas as pd

BASE = __import__("pathlib").Path(__file__).resolve().parent.parent

SYNTHETIC_CSV = BASE / "data" / "employee_attrition_data.csv"
IBM_CSV = BASE / "data" / "ibm_hr_attrition.csv"

# Columns the demo app consumes (Dataset A contract)
SYNTHETIC_FEATURES = [
    "Age", "Gender", "Department", "Job_Title", "Years_at_Company",
    "Satisfaction_Level", "Average_Monthly_Hours", "Promotion_Last_5Years",
]
SYNTHETIC_TARGETS = {"attrition": "Attrition", "salary": "Salary"}

# Dataset B -> app-schema mapping (documented approximations only)
IBM_MAPPING = {
    "Age": "Age",
    "Gender": "Gender",
    "Department": "Department",
    "Job_Title": "JobRole",
    "Experience": "YearsAtCompany",
    # JobSatisfaction is 1-4 -> normalised to 0-1 to match the app contract
    "Satisfaction": ("JobSatisfaction", lambda s: (s - 1) / 3.0),
    # Proxy: "promoted within last 5 years" derived from years since last promotion.
    # Documented approximation, not a true promotion flag.
    "Promotion": ("YearsSinceLastPromotion", lambda s: (s <= 5).astype(int)),
}
IBM_ATTRITION_TARGET = "Attrition"   # 'Yes'/'No'
IBM_SALARY_TARGET = "MonthlyIncome"  # monthly income, dataset currency (USD)
# No MonthlyHours analog exists in the IBM dataset -> feature excluded for v2 models.
IBM_EXCLUDED = ["Average_Monthly_Hours"]


def file_fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_synthetic(path=SYNTHETIC_CSV):
    df = pd.read_csv(path)
    required = SYNTHETIC_FEATURES + list(SYNTHETIC_TARGETS.values()) + ["Employee_ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic dataset missing columns: {missing}")
    return df


def load_ibm(path=IBM_CSV):
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = {"Age", "Gender", "Department", "JobRole", "YearsAtCompany",
              "JobSatisfaction", "YearsSinceLastPromotion", "Attrition", "MonthlyIncome"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"IBM dataset missing columns: {missing}")
    return df


def ibm_to_app_schema(df):
    """Map IBM columns onto the app feature contract. Returns a DataFrame with
    app-style column names plus targets attrition (0/1) and salary."""
    out = pd.DataFrame()
    out["Age"] = df["Age"]
    out["Gender"] = df["Gender"]
    out["Department"] = df["Department"]
    out["Job_Title"] = df["JobRole"]
    out["Experience"] = df["YearsAtCompany"]
    out["Satisfaction"] = (df["JobSatisfaction"] - 1) / 3.0
    out["Promotion"] = (df["YearsSinceLastPromotion"] <= 5).astype(int)
    out["attrition"] = (df["Attrition"].str.strip().str.lower() == "yes").astype(int)
    out["salary"] = df["MonthlyIncome"]
    return out


def quality_report(df, name):
    """Data-quality metrics for a raw dataset frame."""
    n = len(df)
    report = {
        "dataset": name,
        "rows": int(n),
        "columns": int(df.shape[1]),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(float(df.duplicated().mean() * 100), 3),
    }
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            vc = df[col].value_counts(normalize=True)
            report.setdefault("categorical", {})[col] = {
                "n_unique": int(df[col].nunique()),
                "top_share": round(float(vc.iloc[0]), 4) if len(vc) else 0.0,
            }
    return report


def dataset_report():
    """Full quality report for both datasets (executed, not assumed)."""
    reports = {}
    synth = load_synthetic()
    reports["synthetic"] = {
        **quality_report(synth, "synthetic"),
        "fingerprint_sha256": file_fingerprint(SYNTHETIC_CSV),
        "attrition_rate": round(float(synth["Attrition"].mean()), 4),
        "provenance": "Generated synthetic demo data (employee_attrition_data.csv)",
    }
    ibm_raw = load_ibm()
    ibm = ibm_to_app_schema(ibm_raw)
    reports["ibm_hr"] = {
        **quality_report(ibm, "ibm_hr_mapped"),
        "fingerprint_sha256": file_fingerprint(IBM_CSV),
        "attrition_rate": round(float(ibm["attrition"].mean()), 4),
        "salary_target": "MonthlyIncome (USD/month, dataset currency)",
        "excluded_features": IBM_EXCLUDED,
        "mapping_notes": [
            "Job_Title <- JobRole (46 roles)",
            "Satisfaction <- (JobSatisfaction-1)/3 normalized 0-1",
            "Promotion <- YearsSinceLastPromotion <= 5 (proxy, not a true flag)",
            "Average_Monthly_Hours has no IBM analog and is excluded",
            "Source: IBM Watson Analytics sample data (fictional, created by IBM "
            "data scientists for a demo; mirrored via PacktPublishing GitHub)",
        ],
    }
    return reports


if __name__ == "__main__":
    print(json.dumps(dataset_report(), indent=2))
