"""India salary benchmarks pipeline: ingest -> clean -> taxonomy -> features -> evaluate.

Dataset B: data/raw/india_salary_benchmarks_2026_sample.csv
  Real market job-posting observations (Fliar sample), salary_*_inr are ANNUAL INR.
Dataset C: data/raw/labour_bureau_wages.csv
  Govt of India Labour Bureau sector wage reference (average daily wages, NOT salaries).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
RESULTS = BASE / "ml_results"
INDIA_CSV = RAW / "india_salary_benchmarks_2026_sample.csv"
LABOUR_CSV = RAW / "labour_bureau_wages.csv"

RANDOM_STATE = 42
SALARY_CAP_INR = 8_000_000      # reject postings above this (documented, reviewable)
SALARY_FLOOR_INR = 50_000       # below = almost certainly monthly or erroneous
MIN_TITLE_SUPPORT = 5           # titles with fewer obs are "unsupported"
TOP_CITIES = 40
TOP_INDUSTRIES = 15


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ ingestion

def load_india(path=INDIA_CSV):
    df = pd.read_csv(path, dtype={"posting_id": str})
    df["posted_month"] = pd.to_datetime(df["posted_month"], format="%Y-%m", errors="coerce")
    return df


def load_labour(path=LABOUR_CSV):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ------------------------------------------------------------------ cleaning

def _norm_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _title_case(s):
    if not isinstance(s, str) or not s.strip():
        return ""
    # preserve common acronyms
    out = []
    for w in _norm_text(s).split():
        out.append({"it": "IT", "ui": "UI", "ux": "UX", "hr": "HR", "ml": "ML",
                    "ai": "AI", "php": "PHP", "sql": "SQL", "dev": "Dev",
                    "sr": "Sr.", "jr": "Jr.", "ios": "iOS", "api": "API"}.get(w, w.capitalize()))
    return " ".join(out)


def clean_india(df):
    """Returns (clean_df, rejected_df). Every rejection has a reason; nothing is silently dropped."""
    d = df.copy()
    d["_src_row"] = np.arange(len(d))
    reasons = pd.Series(["" for _ in range(len(d))], index=d.index)

    def reject(mask, reason):
        nonlocal reasons
        hit = mask & (reasons == "")
        reasons.loc[hit] = reason

    # --- required target fields ---
    reject(d["salary_mid_inr"].isna(), "missing salary_mid_inr")
    reject(d["salary_min_inr"].isna() | d["salary_max_inr"].isna(), "missing salary_min/max")
    valid_target = d["salary_mid_inr"].notna() & d["salary_min_inr"].notna() & d["salary_max_inr"].notna()
    reject(valid_target & (d["salary_min_inr"] >= d["salary_max_inr"]), "salary_min >= salary_max")
    reject(valid_target & ((d["salary_mid_inr"] < d["salary_min_inr"]) | (d["salary_mid_inr"] > d["salary_max_inin" if False else "salary_max_inr"])),
           "salary_mid outside min-max range")
    reject(d["salary_mid_inr"] < SALARY_FLOOR_INR, f"salary below floor {SALARY_FLOOR_INR}")
    reject(d["salary_mid_inr"] > SALARY_CAP_INR, f"salary above cap {SALARY_CAP_INR} (implausible for annual INR)")
    reject(d["job_title"].isna() | (d["job_title"].astype(str).str.strip() == ""), "missing job_title")
    reject(d["role_family"].isna() | (d["role_family"].astype(str).str.strip() == ""), "missing role_family")
    reject(d["city"].isna() | (d["city"].astype(str).str.strip() == ""), "missing city")
    reject(d["experience_band"].isna(), "missing experience_band")

    keep = reasons == ""
    clean = d[keep].copy()
    clean["reject_reason"] = ""

    # --- normalisation (transforms, recorded) ---
    clean["job_title_norm"] = clean["job_title"].map(_title_case)
    clean["city_norm"] = clean["city"].astype(str).str.strip().str.title()
    clean["industry_norm"] = clean["industry_category"].fillna("Unknown").astype(str).str.strip()
    clean["role_family_norm"] = clean["role_family"].astype(str).str.strip().str.lower()
    clean["seniority_norm"] = clean["seniority"].fillna("unknown").astype(str).str.strip().str.lower()
    clean["exp_mid_years"] = clean["experience_band"].map({
        "0-1 yrs": 0.5, "2-3 yrs": 2.5, "4-6 yrs": 5.0,
        "7-10 yrs": 8.5, "11-15 yrs": 13.0, "16+ yrs": 18.0}).fillna(3.0)
    clean["posted_year"] = clean["posted_month"].dt.year

    # --- duplicate detection: exact dupes on natural key + near-dupes on
    # (company,title,city,month,salary_mid rounded to 50k) ---
    key_cols = ["company", "job_title_norm", "city_norm", "posted_month", "salary_mid_inr"]
    dup_exact = clean.duplicated(subset=key_cols, keep="first")
    clean["_sal_bucket"] = clean["salary_mid_inr"].round(-4)  # ~10k granularity
    dup_near = clean.duplicated(
        subset=["company", "job_title_norm", "city_norm", "posted_month", "_sal_bucket"],
        keep="first")
    clean = clean.drop(columns=["_sal_bucket"])
    rejected_dups = clean[dup_exact | dup_near].copy()
    rejected_dups["reject_reason"] = "duplicate posting (exact or near-duplicate salary bucket)"
    clean = clean[~(dup_exact | dup_near)].copy()

    # --- outlier flagging (IQR per role_family, flag only — do not delete) ---
    clean["salary_iqr_outlier"] = False
    for fam, idx in clean.groupby("role_family_norm").groups.items():
        s = clean.loc[idx, "salary_mid_inr"]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        clean.loc[idx, "salary_iqr_outlier"] = (s < lo) | (s > hi)

    rejected = d[reasons != ""].copy()
    rejected["reject_reason"] = reasons[reasons != ""]
    rejected = pd.concat([rejected, rejected_dups], ignore_index=True)
    return clean.reset_index(drop=True), rejected.reset_index(drop=True)


# ------------------------------------------------------------------ taxonomy

def build_taxonomy(clean):
    """Canonical titles = normalised job_title; support = obs count; family from role_family."""
    g = clean.groupby("job_title_norm").agg(
        aliases=("job_title", lambda s: sorted(set(s.astype(str)))),
        job_family=("role_family_norm", lambda s: s.mode().iat[0]),
        department=("role_family_norm", "first"),
        seniority=("seniority_norm", lambda s: s.mode().iat[0]),
        observations=("job_title_norm", "size"),
        cities=("city_norm", lambda s: s.nunique()),
        median_salary=("salary_mid_inr", "median"),
    ).reset_index().rename(columns={"job_title_norm": "canonical_title"})
    g["supported"] = g["observations"] >= MIN_TITLE_SUPPORT
    g["mapping_confidence"] = np.where(g["observations"] >= 50, "high",
                                       np.where(g["observations"] >= MIN_TITLE_SUPPORT, "medium", "low"))
    # department = coarse mapping from family
    fam2dept = {
        "software-eng": "Engineering", "devops-cloud": "Engineering", "data": "Engineering",
        "it-support": "Engineering", "security": "Engineering",
        "sales": "Sales", "marketing": "Marketing", "customer-service": "Sales",
        "hr": "Human Resources", "admin": "Operations", "finance": "Finance",
        "design": "Design", "product": "Product", "healthcare": "Healthcare",
        "manufacturing": "Manufacturing", "construction": "Operations",
        "logistics": "Operations", "operations": "Operations", "retail-hospitality": "Operations",
        "education": "Education", "legal": "Legal", "science-research": "Research",
        "other": "Other",
    }
    g["department"] = g["job_family"].map(fam2dept).fillna("Other")
    return g.sort_values("observations", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ features

FEATURE_COLS = ["job_family", "seniority", "exp_mid_years", "city_bucket", "industry_bucket", "posted_year"]
CATEGORICAL = ["job_family", "seniority", "city_bucket", "industry_bucket", "posted_year"]


def engineer_features(clean, taxonomy=None):
    """Leakage-safe feature frame. No IDs, names, company, or target-derived cols."""
    d = clean.copy()
    # aggregate rare categories into 'Other' using TRAIN-fit thresholds passed by caller;
    # here we compute buckets on the full frame for the prepared dataset only.
    top_cities = d["city_norm"].value_counts().head(TOP_CITIES).index
    top_inds = d["industry_norm"].value_counts().head(TOP_INDUSTRIES).index
    d["city_bucket"] = d["city_norm"].where(d["city_norm"].isin(top_cities), "Other")
    d["industry_bucket"] = d["industry_norm"].where(d["industry_norm"].isin(top_inds), "Other")
    d["job_family"] = d["role_family_norm"]
    X = d[FEATURE_COLS].copy()
    y = d["salary_mid_inr"].astype(float)
    groups = {
        "job_family": d["job_family"], "city": d["city_norm"],
        "industry": d["industry_norm"], "seniority": d["seniority_norm"],
        "exp_band": d["experience_band"], "year": d["posted_year"],
        "salary_iqr_outlier": d["salary_iqr_outlier"],
    }
    return X, y, groups, d


# ------------------------------------------------------------------ evaluation

def _metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "medae": round(float(median_absolute_error(y_true, y_pred)), 2),
        "mean_abs_pct_err": round(float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1)))), 4),
    }


def evaluate_models(X, y, groups, out_json=RESULTS / "india_model_comparison.json"):
    """Median baseline + linear + tree models. Categorical -> ordinal for linear,
    native for HistGB. 5-fold CV on train split; untouched 20% holdout."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                  HistGradientBoostingRegressor, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]

    cat_pipe = ColumnTransformer([("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), CATEGORICAL)], remainder="passthrough")

    models = {
        "median_global": ("baseline", None),
        "median_family": ("baseline", "job_family"),
        "Ridge": ("linear", Pipeline([("prep", cat_pipe), ("m", Ridge(alpha=1.0))])),
        "RandomForest": ("tree", Pipeline([("prep", cat_pipe), ("m", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1))])),
        "ExtraTrees": ("tree", Pipeline([("prep", cat_pipe), ("m", ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))])),
        "HistGradientBoosting": ("tree_native", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            categorical_features=[0, 1, 3, 4, 5], random_state=RANDOM_STATE)),
        "GradientBoosting": ("tree", Pipeline([("prep", cat_pipe), ("m", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE))])),
    }

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, (kind, obj) in models.items():
        if kind == "baseline":
            if name == "median_global":
                pred_tr = np.full(len(ytr), np.median(ytr))
                pred_te = np.full(len(yte), np.median(ytr))
            else:
                fam_tr = groups["job_family"].iloc[tr_idx].values
                fam_te = groups["job_family"].iloc[te_idx].values
                med = pd.Series(ytr.values).groupby(fam_tr).median()
                glob = np.median(ytr)
                pred_tr = np.array([med.get(f, glob) for f in fam_tr])
                pred_te = np.array([med.get(f, glob) for f in fam_te])
            results[name] = {
                "cv_mae": None, "holdout": _metrics(yte, pred_te),
                "note": "non-ML baseline",
            }
            continue
        # CV predictions on train
        if kind == "tree_native":
            cv_pred = cross_val_predict(obj, Xtr, ytr, cv=kf, n_jobs=-1)
        else:
            cv_pred = cross_val_predict(obj, Xtr, ytr, cv=kf, n_jobs=-1)
        cv_mae = float(np.mean(np.abs(ytr - cv_pred)))
        obj.fit(Xtr, ytr)
        pred_te = obj.predict(Xte)
        results[name] = {"cv_mae": round(cv_mae, 2), "holdout": _metrics(yte, pred_te)}
        # subgroup errors for the best candidates only (keep all anyway — cheap)
        results[name]["by_family"] = _subgroup_errors(yte, pred_te, groups["job_family"].iloc[te_idx], min_n=50)
        results[name]["by_city"] = _subgroup_errors(yte, pred_te, groups["city"].iloc[te_idx], min_n=100)
        results[name]["by_exp_band"] = _subgroup_errors(yte, pred_te, groups["exp_band"].iloc[te_idx], min_n=30)
        results[name]["by_seniority"] = _subgroup_errors(yte, pred_te, groups["seniority"].iloc[te_idx], min_n=50)

    best = min((k for k in results if results[k].get("cv_mae") is not None),
               key=lambda k: results[k]["cv_mae"], default=None)
    out = {
        "dataset": "india_salary_benchmarks_2026_sample.csv",
        "target": "salary_mid_inr (annual INR)",
        "split": "80/20 random, seed 42; 5-fold CV on train",
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "features": FEATURE_COLS,
        "categorical": CATEGORICAL,
        "best_model": best,
        "results": results,
    }
    out_json.write_text(json.dumps(out, indent=2))
    return out


def _subgroup_errors(y_true, y_pred, keys, min_n=30):
    df = pd.DataFrame({"yt": y_true, "yp": y_pred, "k": keys.values})
    out = {}
    for k, g in df.groupby("k"):
        if len(g) >= min_n:
            out[str(k)] = {"n": int(len(g)), "mae": round(float(np.mean(np.abs(g.yt - g.yp))), 0)}
    return out


# ------------------------------------------------------------------ range model

def train_range_models(X, y, out_dir=BASE / "models" / "india_candidate"):
    """Quantile HistGB models p10/p50/p90 for salary ranges + median model."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    out_dir.mkdir(parents=True, exist_ok=True)
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
    meta = {"features": FEATURE_COLS, "categorical": CATEGORICAL, "quantiles": [0.1, 0.5, 0.9]}
    for q in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                        max_iter=400, learning_rate=0.06,
                                        max_leaf_nodes=63, categorical_features=cat_idx,
                                        random_state=RANDOM_STATE)
        m.fit(X, y)
        import joblib
        joblib.dump(m, out_dir / f"salary_q{int(q*100)}.pkl")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def run_all():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    india_raw = load_india()
    labour = load_labour()
    clean, rejected = clean_india(india_raw)

    # --- audit JSONs ---
    audit = {
        "india_raw": {"rows": int(len(india_raw)), "cols": int(india_raw.shape[1]),
                      "sha256": sha256(INDIA_CSV)},
        "india_clean": {"rows": int(len(clean)), "rejected": int(len(rejected)),
                        "reject_reasons": rejected["reject_reason"].value_counts().to_dict()},
        "labour_bureau": {"rows": int(len(labour)), "sha256": sha256(LABOUR_CSV),
                          "note": "Sector-level avg daily wages (INR/day), Base Year 2016, 2016-2024. NOT individual salary records; kept as separate reference."},
    }
    (RESULTS / "dataset_quality.json").write_text(json.dumps(audit, indent=2))
    (RESULTS / "source_provenance.json").write_text(json.dumps({
        "india_salary_benchmarks": {
            "file": "data/raw/india_salary_benchmarks_2026_sample.csv",
            "sha256": sha256(INDIA_CSV),
            "source": "Fliar job-posting salary sample (India), columns salary_*_inr annual INR",
            "posted_range": [str(india_raw.posted_month.min()), str(india_raw.posted_month.max())],
            "nature": "Real market observations (sample), not synthetic",
        },
        "labour_bureau_wages": {
            "file": "data/raw/labour_bureau_wages.csv",
            "sha256": sha256(LABOUR_CSV),
            "source": "Govt of India Ministry of Labour — Labour Bureau wage surveys",
            "nature": "Official sector-level wage reference (avg daily wages)",
        },
    }, indent=2))

    # save cleaned + rejected (lineage preserved)
    clean.to_csv(PROCESSED / "india_salary_clean.csv", index=False)
    rejected.to_csv(PROCESSED / "india_salary_rejected.csv", index=False)

    # taxonomy
    tax = build_taxonomy(clean)
    tax.to_csv(PROCESSED / "job_taxonomy.csv", index=False)

    # features + evaluation
    X, y, groups, prepared = engineer_features(clean, tax)
    comp = evaluate_models(X, y, groups)
    train_range_models(X, y)

    print(f"clean={len(clean)} rejected={len(rejected)} titles={len(tax)} "
          f"supported={int(tax.supported.sum())}")
    print("best:", comp["best_model"], comp["results"].get(comp["best_model"], {}).get("holdout"))
    return comp


if __name__ == "__main__":
    run_all()
