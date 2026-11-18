# Employee Salary Prediction System

An HR decision-support web app that estimates **market salary benchmarks (annual INR)** for employees in the Indian job market, supports **bulk salary planning** for entire teams, and lets HR teams model **salary-increment scenarios** before committing budget.

## Problem statement

HR teams lack a fast, data-grounded way to answer: *"What should this role realistically pay in the Indian market, and what would a proposed increment cost us?"* Manual salary benchmarking is slow and inconsistent across geographies, seniority levels and industries.

## What it does

- **Individual prediction** — enter a job title, experience, city, seniority and industry; get an estimated annual salary plus a p10–p90 market range, the canonical job title, job family, number of supporting market observations and a support level.
- **Bulk salary planning** — upload a CSV/XLSX employee sheet, validate rows, run salary estimates for the whole company, review results in a paginated table, and export to Excel/CSV.
- **Increment scenarios** — apply a uniform or per-department increment percentage to current salaries, optionally capped by budget or salary ceiling, and see projected payroll impact. This is transparent arithmetic, not an ML prediction.
- **Prediction history** — individual predictions and bulk jobs are stored in SQLite and browsable in the UI.
- **Model information page** — shows the active model version, dataset, salary unit, supported categories and honest evaluation metrics.

## Technology stack

- **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-CORS
- **ML:** scikit-learn (HistGradientBoosting quantile regressors), joblib, pandas, NumPy
- **Database:** SQLite (prediction history, bulk jobs, rate-limit counters)
- **Frontend:** server-rendered Jinja template + vanilla JS, TailwindCSS (CDN), Chart.js
- **Testing:** pytest
- **Deployment:** Vercel (`vercel.json`); local dev via `python app.py`

## ML approach

The active model is **`india-salary`**: three quantile gradient-boosted regressors (p10, p50, p90) trained on a cleaned sample of real Indian job-posting salary observations (annual INR). Features: job family, seniority, experience midpoint, city bucket, industry bucket, posting year. Only job titles with ≥5 training observations are supported; unsupported titles are rejected with a clear error. The p10–p90 spread is a quantile range, **not** a confidence interval.

Older IBM-attrition and synthetic-demo model versions were removed; the registry contains only `india-salary`.

## Architecture

```
Browser (index.html + script.js)
   │  fetch /api/*
   ▼
Flask app.py ── rate limiter (DB-backed) ── security headers/CORS
   ├── /api/predict ──► ml/india_infer.py ──► models/india_candidate/*.pkl
   ├── /api/bulk/{validate,process,jobs,...} ──► BulkJob rows (SQLite)
   ├── /api/history, /api/analytics ──► PredictionHistory (SQLite)
   └── /api/model, /api/health ──► model registry + metadata
```

`ml/india_pipeline.py` is the offline training pipeline (ingest → clean → taxonomy → features → evaluate → serialize). `ml/india_infer.py` is the production inference path and shares the same preprocessing contract.

## Datasets

| File | Role |
|---|---|
| `data/raw/india_salary_benchmarks_2026_sample.csv` | Real job-posting salary observations (Fliar sample), annual INR — training source |
| `data/raw/labour_bureau_wages.csv` | Govt of India sector wage reference (daily wages) — reference only, never mixed into salary targets |
| `data/processed/` | Cleaned data, rejected rows (with reasons), job taxonomy |
| `ml_results/` | Dataset quality, provenance, model comparison and interval-coverage reports |

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # serves http://127.0.0.1:5000
```

The registry defaults to `india-salary`. To pin a version explicitly: `ESPS_MODEL_VERSION=india-salary python app.py`. Optional env vars: `ESPS_DB_PATH`, `ESPS_DATABASE_URL`, `ESPS_ALLOWED_ORIGINS`, `ESPS_TRUST_PROXY` (set to `1` only behind a real proxy such as Vercel).

## Example workflow

1. Open the app → **Individual Prediction** → Job title "Software Engineer", 4 yrs experience, Bengaluru → annual INR estimate + p10–p90 range.
2. **Bulk Salary Planning** → download the template → upload your sheet → validate → process → review → plan a 10% increment scenario → export `.xlsx`.

## Limitations

- Salary estimates reflect the training sample's market coverage; unsupported or rare titles are rejected rather than guessed.
- Predictions are benchmarks, not offers or HR decisions; the increment scenario is arithmetic only.
- SQLite history is local/demo-grade; on Vercel the DB lives at `/tmp` and is not durable across instances.
- No authentication; rate limiting is per-IP and intended for a demo deployment.
