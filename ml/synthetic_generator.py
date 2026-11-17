"""Reproducible synthetic workforce dataset calibrated on cleaned market data.

All records are FICTIONAL. Salary distributions are sampled per job_family
from the empirical cleaned-data distribution (lognormal fit on IQR). Never
present generated salary history as real-world salary growth.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "data" / "processed" / "india_salary_clean.csv"
OUT = BASE / "data" / "synthetic" / "workforce_synthetic_50k.csv"
META = BASE / "data" / "synthetic" / "generation_meta.json"
SEED = 42
N = 50_000
MIN_FAMILY_OBS = 30  # only generate for families with real market coverage

FIRST = ["Aarav", "Diya", "Arjun", "Ananya", "Rohan", "Priya", "Karan", "Meera",
         "Vikram", "Pooja", "Rahul", "Sneha", "Amit", "Kavya", "Suresh", "Nisha",
         "Deepak", "Riya", "Manish", "Tanvi", "Ajay", "Swati", "Nikhil", "Pallavi"]
LAST = ["Sharma", "Verma", "Patel", "Gupta", "Mehta", "Reddy", "Nair", "Iyer",
        "Singh", "Kumar", "Shah", "Das", "Bose", "Chopra", "Malhotra", "Joshi"]
CITIES = ["Bengaluru", "Hyderabad", "Mumbai", "Pune", "Gurugram", "Chennai",
          "Delhi", "Ahmedabad", "Kolkata", "Jaipur", "Lucknow", "Indore"]
STATES = {"Bengaluru": "Karnataka", "Hyderabad": "Telangana", "Mumbai": "Maharashtra",
          "Pune": "Maharashtra", "Gurugram": "Haryana", "Chennai": "Tamil Nadu",
          "Delhi": "Delhi", "Ahmedabad": "Gujarat", "Kolkata": "West Bengal",
          "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Indore": "Madhya Pradesh"}


def generate(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    clean = pd.read_csv(CLEAN)
    fam_stats = (clean.groupby("role_family_norm")["salary_mid_inr"]
                 .agg(["count", "median", "std"]).reset_index())
    fam_stats = fam_stats[fam_stats["count"] >= MIN_FAMILY_OBS]
    # sampling weights proportional to market observation count
    probs = fam_stats["count"].values / fam_stats["count"].sum()
    titles = (clean[clean.role_family_norm.isin(fam_stats.role_family_norm)]
              .groupby("role_family_norm")["job_title_norm"]
              .apply(lambda s: list(s.value_counts().head(5).index)))

    rows = []
    for i in range(n):
        fam_row = fam_stats.iloc[rng.choice(len(fam_stats), p=probs)]
        fam = fam_row["role_family_norm"]
        title_pool = titles.get(fam, [fam])
        title = title_pool[rng.integers(len(title_pool))]
        exp_band = rng.choice(["0-1 yrs", "2-3 yrs", "4-6 yrs", "7-10 yrs", "11-15 yrs"],
                              p=[0.25, 0.3, 0.25, 0.15, 0.05])
        exp_map = {"0-1 yrs": 0.5, "2-3 yrs": 2.5, "4-6 yrs": 5.0, "7-10 yrs": 8.5, "11-15 yrs": 13.0}
        exp = exp_map[exp_band] + rng.normal(0, 0.4)
        exp = float(np.clip(exp, 0, 20))
        seniority = ("entry" if exp < 2 else "mid" if exp < 5 else
                     "senior" if exp < 9 else "lead" if exp < 13 else "manager")
        city = CITIES[rng.integers(len(CITIES))]
        # salary: lognormal around family median, sigma from family std capped
        med = float(fam_row["median"]); std = float(fam_row["std"])
        sigma = min(0.35, max(0.1, std / (med * 2.2)))
        salary = float(rng.lognormal(np.log(med), sigma))
        salary = float(np.clip(salary, 150_000, 8_000_000))
        tenure = float(np.clip(exp * rng.uniform(0.4, 0.8), 0, exp))
        perf = int(np.clip(rng.normal(3.5, 0.8), 1, 5))
        name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
        rows.append({
            "employee_id": f"SIM{i+1:05d}",
            "employee_name": name,
            "department": {"software-eng": "Engineering", "devops-cloud": "Engineering",
                           "data": "Engineering", "it-support": "Engineering", "security": "Engineering",
                           "sales": "Sales", "marketing": "Marketing", "customer-service": "Sales",
                           "hr": "Human Resources", "admin": "Operations", "finance": "Finance",
                           "design": "Design", "product": "Product", "healthcare": "Healthcare",
                           "manufacturing": "Manufacturing", "construction": "Operations",
                           "logistics": "Operations", "operations": "Operations",
                           "retail-hospitality": "Operations", "education": "Education",
                           "legal": "Legal", "science-research": "Research"}.get(fam, "Other"),
            "job_title": title, "job_family": fam,
            "city": city, "state": STATES[city],
            "experience_years": round(exp, 1),
            "company_tenure_years": round(tenure, 1),
            "seniority": seniority,
            "joining_date": f"{int(2026 - tenure)}-{int(rng.integers(1,13)):02d}-{int(rng.integers(1,29)):02d}",
            "current_annual_salary_inr": round(salary, -3),
            "performance_rating": perf,
            "data_origin": "synthetic",
            "generation_version": f"syn-v1-seed{seed}",
        })
    df = pd.DataFrame(rows)
    if n == N:  # only the canonical 50k run writes artifacts
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
        META.write_text(json.dumps({
        "n": n, "seed": seed, "families": fam_stats.role_family_norm.tolist(),
        "min_family_obs": MIN_FAMILY_OBS,
            "calibration": "lognormal per job_family, median+std from india_salary_clean.csv",
            "disclaimer": "All records fictional. Salary history is generated, not real observations.",
        }, indent=2))
    return df


if __name__ == "__main__":
    df = generate()
    print(df.shape, df.current_annual_salary_inr.describe().round(0).to_dict())
    print("families:", df.job_family.nunique(), "| all synthetic:", (df.data_origin == "synthetic").all())
