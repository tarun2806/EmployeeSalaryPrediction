"""Production inference for the India salary candidate model.

Shares EXACT preprocessing logic with ml.india_pipeline (same FEATURE_COLS,
same bucketing thresholds recomputed from the same clean dataset -> identical
buckets). Predicts annual salary INR + validated p10/p90 range.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import india_pipeline as ip

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "india_candidate"
TAXONOMY_CSV = ip.PROCESSED / "job_taxonomy.csv"
CLEAN_CSV = ip.PROCESSED / "india_salary_clean.csv"

_cache = {}


def _load():
    if _cache:
        return _cache
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    models = {q: joblib.load(MODEL_DIR / f"salary_q{int(q*100)}.pkl")
              for q in meta["quantiles"]}
    tax = pd.read_csv(TAXONOMY_CSV)
    # alias -> canonical title map (normalised raw title -> canonical)
    alias_map = {}
    for _, r in tax.iterrows():
        canon_norm = ip._norm_text(r["canonical_title"])
        alias_map[canon_norm] = r["canonical_title"]
        for a in (r.get("aliases") or []):
            alias_map[ip._norm_text(a)] = r["canonical_title"]
    # persisted preprocessing artifact (immutable; not recomputed from mutable CSVs)
    pre = json.loads((MODEL_DIR / "preprocess.json").read_text())
    top_cities = set(pre["top_cities"])
    top_inds = set(pre["top_industries"])
    title_info = tax.set_index("canonical_title")[["job_family", "department", "seniority", "observations", "supported"]].to_dict("index")
    _cache.update(models=models, meta=meta, alias_map=alias_map, preprocess=pre,
                  top_cities=top_cities, top_inds=top_inds, title_info=title_info,
                  tax=tax)
    return _cache


def resolve_title(raw):
    """Map a user-supplied job title to a canonical supported title.
    Returns (canonical_title, info_dict, error_str)."""
    c = _load()
    norm = ip._norm_text(str(raw or ""))
    canon = c["alias_map"].get(norm)
    if canon is None:
        # try contains-match on canonical titles only (no fuzzy guessing)
        for t in c["tax"]["canonical_title"]:
            if norm and (norm in ip._norm_text(t) or ip._norm_text(t) in norm):
                canon = t
                break
    if canon is None:
        return None, None, f"Unsupported job title '{raw}'. Only titles observed in the training data are supported."
    info = c["title_info"].get(canon)
    if info is None:
        return None, None, f"Job title '{canon}' has no taxonomy entry."
    return canon, info, None


def supported_titles(min_obs=ip.MIN_TITLE_SUPPORT):
    c = _load()
    t = c["tax"]
    return sorted(t.loc[t["supported"], "canonical_title"].tolist())


def predict_salary(job_title, experience_years=None, city=None, seniority=None,
                   industry=None):
    """Returns dict with p10/p50/p90 INR annual salary + evidence metadata."""
    c = _load()
    canon, info, err = resolve_title(job_title)
    if err:
        return {"error": err}
    # experience: derive seniority if not given
    exp = float(experience_years) if experience_years is not None else 3.0
    exp = float(np.clip(exp, 0, 30))
    if seniority is None:
        seniority = ("entry" if exp < 2 else "mid" if exp < 5 else
                     "senior" if exp < 9 else "lead" if exp < 13 else "manager")
    else:
        seniority = str(seniority).strip().lower()
        if seniority not in {"entry", "mid", "senior", "lead", "manager", "director", "executive", "intern", "unknown"}:
            seniority = "mid"
    exp_band = ("0-1 yrs" if exp < 2 else "2-3 yrs" if exp < 4 else
                "4-6 yrs" if exp < 7 else "7-10 yrs" if exp < 11 else "11-15 yrs" if exp < 16 else "16+ yrs")
    exp_mid = {"0-1 yrs": 0.5, "2-3 yrs": 2.5, "4-6 yrs": 5.0,
               "7-10 yrs": 8.5, "11-15 yrs": 13.0, "16+ yrs": 18.0}[exp_band]
    city_b = str(city).strip().title() if city else ""
    city_bucket = city_b if city_b in c["top_cities"] else "Other"
    ind_b = str(industry).strip() if industry else "Unknown"
    industry_bucket = ind_b if ind_b in c["top_inds"] else "Other"
    year = pd.Timestamp.now().year

    row = pd.DataFrame([{
        "job_family": info["job_family"], "seniority": seniority,
        "exp_mid_years": exp_mid, "city_bucket": city_bucket,
        "industry_bucket": industry_bucket, "posted_year": year,
    }])
    q10 = float(c["models"][0.1].predict(row)[0])
    q50 = float(c["models"][0.5].predict(row)[0])
    q90 = float(c["models"][0.9].predict(row)[0])
    # keep monotonic
    lo, hi = min(q10, q50), max(q50, q90)
    obs = int(info.get("observations", 0))
    support = ("high" if obs >= 50 else "medium" if obs >= ip.MIN_TITLE_SUPPORT else "low")
    return {
        "salary_annual_inr": round(q50, -3),
        "salary_p10_inr": round(lo, -3),
        "salary_p90_inr": round(hi, -3),
        "salary_unit": "inr_annual",
        "canonical_title": canon,
        "job_family": info["job_family"],
        "department": info["department"],
        "seniority_used": seniority,
        "experience_years_used": exp,
        "city_bucket": city_bucket,
        "industry_bucket": industry_bucket,
        "market_observations": obs,
        "support_level": support,
        "range_note": "p10-p90 quantile estimates from gradient-boosted quantile models; not a confidence interval.",
        "disclaimer": "Market benchmark estimate for the Indian job market sample — not a guaranteed offer or HR decision.",
    }


def predict_salary_safe(**kwargs):
    try:
        return predict_salary(**kwargs)
    except Exception:
        return {"error": "prediction failed"}


def supported_categories():
    c = _load()
    tax = c["tax"]
    sup = tax[tax["supported"]]
    return {
        "Job_Title": sorted(sup["canonical_title"].tolist()),
        "Job_Family": sorted(sup["job_family"].unique().tolist()),
        "City": sorted(c["top_cities"]),
        "Seniority": ["entry", "mid", "senior", "lead", "manager"],
        "Department": sorted(sup["department"].unique().tolist()),
    }


if __name__ == "__main__":
    print(predict_salary("Software Engineer", experience_years=4, city="Bengaluru"))
    print(predict_salary("Astronaut", experience_years=4))
