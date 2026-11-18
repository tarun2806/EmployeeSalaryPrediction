import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("esps")

REGISTRY_PATH = MODELS_PATH / "registry.json"
MAX_CONTENT_LENGTH = 5 * 1024 * 1024
MAX_BULK_ROWS = 5000
API_RATE_LIMIT = 300
PREDICT_RATE_LIMIT = 60
BULK_RATE_LIMIT = 20
SIM_RATE_LIMIT = 120
ANALYTICS_RATE_LIMIT = 30
HISTORY_PAGE_SIZE = 20


def _default_db_path():
    if os.environ.get("ESPS_DATABASE_URL"):
        return os.environ["ESPS_DATABASE_URL"]
    if os.environ.get("ESPS_DB_PATH"):
        return os.environ["ESPS_DB_PATH"]
    if os.environ.get("VERCEL"):
        return "/tmp/history.db"
    return str(BASE_DIR / "history.db")


def _db_uri():
    target = _default_db_path()
    if target.startswith(("postgresql", "postgres", "mysql", "sqlite")):
        return target
    return f"sqlite:///{target}"


db = SQLAlchemy()
models = {}
active_version = "india-salary"
active_meta = {}


def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"active": "india-salary",
            "versions": {"india-salary": {"dir": "models/india_candidate", "kind": "india_salary"}}}


def active_models_dir():
    reg = load_registry()
    version = os.environ.get("ESPS_MODEL_VERSION") or reg.get("active", "india-salary")
    v = reg["versions"].get(version)
    if v is None:
        raise KeyError(f"Unknown model version '{version}'")
    return version, BASE_DIR / v["dir"], v






# ---------------------------------------------------------------- models

class PredictionHistory(db.Model):
    __tablename__ = "history"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    salary = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, nullable=True)
    kind = db.Column(db.String(20), nullable=False, default="prediction")


class RateLimitHit(db.Model):
    __tablename__ = "rate_limits"
    key = db.Column(db.String(200), primary_key=True)
    window = db.Column(db.Integer, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)


class BulkJob(db.Model):
    """A bulk salary-planning run: uploaded sheet + per-employee results."""
    __tablename__ = "bulk_jobs"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    model_version = db.Column(db.String(50))
    total_rows = db.Column(db.Integer, default=0)
    valid_rows = db.Column(db.Integer, default=0)
    invalid_rows = db.Column(db.Integer, default=0)
    results_json = db.Column(db.Text)
    errors_json = db.Column(db.Text)
    scenario_json = db.Column(db.Text)
    summary_json = db.Column(db.Text)


# Job-title taxonomy grouping supported titles into families (UI metadata only).


# ---------------------------------------------------------------- helpers

def load_all():
    """Lazy-load the ACTIVE model version's artifacts. Returns True on success."""
    global active_version, active_meta
    try:
        version, mdir, vconf = active_models_dir()
    except KeyError:
        log.error("Configured model version unavailable")
        return False
    if models and models.get("_version") == version:
        return True
    models.clear()
    try:
        meta_file = vconf.get("meta_file")
        if meta_file:
            active_meta = json.loads((BASE_DIR / meta_file).read_text())
        elif (mdir / "meta.json").exists():
            active_meta = json.loads((mdir / "meta.json").read_text())
        else:
            active_meta = {"version": version}
        if vconf.get("kind") != "india_salary":
            log.error("Only india_salary model kind is supported")
            return False
        from ml import india_infer
        india_infer._load()  # validates artifacts load
        models["_kind"] = "india_salary"
        active_version = version
        models["_version"] = version
        log.info("Loaded model version '%s' from %s", version, mdir)
        return True
    except Exception:
        log.exception("Failed to load models")
        return False


def _json_error(message, status=400, code=None):
    payload = {"error": message}
    if code:
        payload["code"] = code
    return jsonify(payload), status


def _client_ip():
    # Only trust X-Forwarded-For when explicitly deployed behind a proxy
    # (ESPS_TRUST_PROXY=1, e.g. Vercel). Otherwise use the direct peer address
    # so clients cannot spoof rate-limit identities.
    if os.environ.get("ESPS_TRUST_PROXY"):
        return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_allow(key, limit, window_seconds):
    """DB-backed fixed-window rate limiter."""
    window = int(time.time() // window_seconds)
    rkey = f"{key}:{window}"
    try:
        hit = db.session.get(RateLimitHit, rkey)
        if hit is None:
            hit = RateLimitHit(key=rkey, window=window, count=0)
            db.session.add(hit)
        if hit.count >= limit:
            db.session.commit()
            return False
        hit.count += 1
        db.session.commit()
        return True
    except Exception:
        db.session.rollback()
        log.exception("Rate limiter DB error; failing closed")
        return False


def _migrate_history_table():
    insp = db.inspect(db.engine)
    cols = {c["name"] for c in insp.get_columns("history")}
    with db.engine.begin() as conn:
        if "user_id" not in cols:
            conn.exec_driver_sql("ALTER TABLE history ADD COLUMN user_id INTEGER")
        if "kind" not in cols:
            conn.exec_driver_sql("ALTER TABLE history ADD COLUMN kind VARCHAR(20) NOT NULL DEFAULT 'prediction'")


# ---------------------------------------------------------------- validation

def _err(field, message):
    return _json_error(f"Invalid '{field}': {message}", 422, "VALIDATION_ERROR")


def _as_int(value, field, lo, hi):
    try:
        if isinstance(value, bool):
            raise ValueError
        v = int(value)
    except (TypeError, ValueError):
        return None, _err(field, "must be an integer")
    if not lo <= v <= hi:
        return None, _err(field, f"must be between {lo} and {hi}")
    return v, None












def _india_active():
    return bool(models) and models.get("_kind") == "india_salary"






# ================================================================ bulk salary planning

BULK_TEMPLATE_COLUMNS = [
    "Employee_ID", "Employee_Name", "Department", "Job_Title", "Age",
    "Years_Experience", "Years_At_Company", "Current_Monthly_Salary",
    "Gender", "Satisfaction_Level", "Last_Promotion_Years", "Monthly_Hours",
    "City", "Seniority", "Industry",
]
BULK_TEMPLATE_NOTES = {
    "Employee_ID": "Unique employee identifier (required, no duplicates)",
    "Employee_Name": "Display name (optional)",
    "Department": "Must match an active model category",
    "Job_Title": "Must match an active model category",
    "Age": "16-100",
    "Years_Experience": "Total years of experience (used if Years_At_Company empty)",
    "Years_At_Company": "Tenure at company -> model Experience feature",
    "Current_Monthly_Salary": "Optional; enables increment/payroll planning",
    "Gender": "Must match an active model category",
    "Satisfaction_Level": "0.0 - 1.0",
    "Last_Promotion_Years": "Years since last promotion; <=5 maps to Promotion=Yes",
    "Monthly_Hours": "Optional; used only if the active model consumes Hours",
    "City": "Optional for India model; must match a supported city or 'Other'",
    "Seniority": "Optional; entry/mid/senior/lead/manager (auto-derived from experience)",
    "Industry": "Optional; must match a supported industry or 'Other'",
    "Current_Annual_Salary": "India model: annual INR salary (preferred over monthly)",
}


def _read_upload(file):
    """Parse uploaded CSV or XLSX into a DataFrame. Returns (df, error_response)."""
    name = (file.filename or "").lower()
    raw = file.read(MAX_CONTENT_LENGTH + 1)
    if len(raw) > MAX_CONTENT_LENGTH:
        return None, _json_error("File exceeds 5 MB limit", 413, "FILE_TOO_LARGE")
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            return None, _json_error("Only .csv and .xlsx files are accepted", 415, "UNSUPPORTED_MEDIA_TYPE")
    except Exception:
        return None, _json_error("File could not be parsed", 422, "MALFORMED_FILE")
    if df is None or len(df) == 0:
        return None, _json_error("File contains no data rows", 422, "EMPTY_FILE")
    if len(df) > MAX_BULK_ROWS:
        return None, _json_error(f"Too many rows (max {MAX_BULK_ROWS})", 413, "TOO_MANY_ROWS")
    return df, None










def _scenario_calc(results, cfg):
    """Transparent HR increment scenario — arithmetic, NOT an ML prediction."""
    pct = float(cfg.get("increment_pct", 0))
    dept_pct = {str(k): float(v) for k, v in (cfg.get("dept_increments") or {}).items()}
    budget = cfg.get("budget")
    cap = cfg.get("cap")
    rows, cur_annual = [], 0.0
    dept_totals = {}
    for r in results:
        if r["Status"] != "ok" or r.get("Current_Salary") is None:
            continue
        cur = float(r["Current_Salary"])  # annual INR
        p = dept_pct.get(r["Department"], pct)
        revised = cur * (1 + p / 100)
        if cap is not None:
            revised = min(revised, float(cap) * 12)  # cap is monthly -> annual
        diff_a = revised - cur
        cur_annual += cur
        d = dept_totals.setdefault(r["Department"], {"current": 0.0, "projected": 0.0, "employees": 0})
        d["current"] += cur; d["projected"] += revised; d["employees"] += 1
        rows.append({**r, "Increment_Pct": p, "Revised_Salary": round(revised, 2),
                     "Monthly_Diff": round(diff_a / 12, 2), "Annual_Diff": round(diff_a, 2)})
    proj_annual = sum(d["projected"] for d in dept_totals.values())
    annual_cost = proj_annual - cur_annual
    return {
        "rows": rows,
        "current_monthly_payroll": round(cur_annual / 12, 2),
        "projected_monthly_payroll": round(proj_annual / 12, 2),
        "current_annual_payroll": round(cur_annual, 2),
        "projected_annual_payroll": round(proj_annual, 2),
        "total_additional_annual_cost": round(annual_cost, 2),
        "avg_increment_pct": round(pct, 2),
        "department_impact": {k: {"current_monthly": round(v["current"] / 12, 2),
                                  "projected_monthly": round(v["projected"] / 12, 2),
                                  "employees": v["employees"]}
                              for k, v in dept_totals.items()},
        "budget": budget, "budget_exceeded": bool(budget is not None and annual_cost > float(budget)),
        "cap": cap,
        "disclaimer": "HR-configured increment arithmetic only — not an ML prediction of deserved salary.",
    }


def _safe_cell(v):
    """Prevent spreadsheet formula injection in exported text cells."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v
    return v


def _export_xlsx(job, results, errors, scenario):
    """Build an .xlsx workbook: Results / Dept Summary / Scenario / Errors."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Salary Results"
    headers = ["EmpID", "Employee", "Department", "Job_Title", "Current_Salary",
               "Estimated_Salary", "Salary_Delta", "Risk", "Status"]
    ws.append([_safe_cell(h) for h in headers])
    for r in results:
        ws.append([_safe_cell(r.get(h)) for h in
                   ["EmpID", "Employee", "Department", "Job_Title", "Current_Salary",
                    "Estimated_Salary", "Salary_Delta", "Risk", "Status"]])
    ws2 = wb.create_sheet("Department Summary")
    ws2.append(["Department", "Employees", "Avg_Estimated_Salary", "Avg_Current_Salary"])
    dept = {}
    for r in results:
        if r["Status"] != "ok":
            continue
        d = dept.setdefault(r["Department"], {"n": 0, "est": [], "cur": []})
        d["n"] += 1; d["est"].append(r["Estimated_Salary"])
        if r["Current_Salary"] is not None: d["cur"].append(r["Current_Salary"])
    for k, v in dept.items():
        ws2.append([_safe_cell(k), v["n"],
                    round(sum(v["est"]) / len(v["est"]), 2) if v["est"] else None,
                    round(sum(v["cur"]) / len(v["cur"]), 2) if v["cur"] else None])
    if scenario:
        ws3 = wb.create_sheet("Increment Scenario")
        ws3.append(["Metric", "Value"])
        for k in ("current_monthly_payroll", "projected_monthly_payroll",
                  "current_annual_payroll", "projected_annual_payroll",
                  "total_additional_annual_cost", "avg_increment_pct",
                  "budget_exceeded"):
            ws3.append([k, scenario.get(k)])
        ws3.append([])
        ws3.append(["Department", "Current_Monthly", "Projected_Monthly", "Employees"])
        for k, v in (scenario.get("department_impact") or {}).items():
            ws3.append([_safe_cell(k), v["current_monthly"], v["projected_monthly"], v["employees"]])
    if errors:
        ws4 = wb.create_sheet("Validation Errors")
        ws4.append(["Row", "EmpID", "Employee", "Error"])
        for e_ in errors:
            ws4.append([e_.get("row"), _safe_cell(e_.get("EmpID")), _safe_cell(e_.get("Employee")), _safe_cell(e_.get("error"))])
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def _export_csv(results, errors, scenario):
    import csv as _csv
    out = io.StringIO()
    w = _csv.writer(out)
    w.writerow(["EmpID", "Employee", "Department", "Job_Title", "Current_Salary",
                "Estimated_Salary", "Salary_Delta", "Risk", "Status"])
    for r in results:
        w.writerow([r.get("EmpID"), r.get("Employee"), r.get("Department"),
                    r.get("Job_Title"), r.get("Current_Salary"),
                    r.get("Estimated_Salary"), r.get("Salary_Delta"),
                    r.get("Risk"), r.get("Status")])
    if scenario:
        w.writerow([])
        w.writerow(["SCENARIO", "current_monthly_payroll", scenario.get("current_monthly_payroll")])
        w.writerow(["SCENARIO", "projected_monthly_payroll", scenario.get("projected_monthly_payroll")])
        w.writerow(["SCENARIO", "total_additional_annual_cost", scenario.get("total_additional_annual_cost")])
    if errors:
        w.writerow([])
        w.writerow(["VALIDATION ERRORS"])
        w.writerow(["Row", "EmpID", "Employee", "Error"])
        for e_ in errors:
            w.writerow([e_.get("row"), e_.get("EmpID"), e_.get("Employee"), e_.get("error")])
    return out.getvalue()


def _bulk_india_rows(df):
    """Map spreadsheet rows to india-model inputs. Returns (entries, errors)."""
    from ml import india_infer
    entries, errors = [], []
    seen = set()
    for idx, row in df.iterrows():
        row_no = int(idx) + 2
        def cell(*names):
            for n in names:
                if n in row.index and pd.notna(row[n]) and str(row[n]).strip() != "":
                    return row[n]
            return None
        emp_id = cell("Employee_ID")
        emp_id = str(emp_id).strip() if emp_id is not None else None
        name = cell("Employee_Name") or emp_id or f"row-{row_no}"
        if emp_id and emp_id in seen:
            errors.append({"row": row_no, "EmpID": emp_id, "Employee": name, "error": "duplicate Employee_ID"})
            continue
        if emp_id: seen.add(emp_id)
        title = cell("Job_Title", "job_title", "Title")
        if title is None:
            errors.append({"row": row_no, "EmpID": emp_id, "Employee": name, "error": "missing Job_Title"})
            continue
        canon, info, terr = india_infer.resolve_title(title)
        if terr:
            errors.append({"row": row_no, "EmpID": emp_id, "Employee": name, "error": terr})
            continue
        exp = cell("Years_Experience", "Experience", "Years_At_Company")
        try:
            exp_f = float(exp) if exp is not None else None
        except (TypeError, ValueError):
            errors.append({"row": row_no, "EmpID": emp_id, "Employee": name, "error": "invalid experience"})
            continue
        cur = cell("Current_Annual_Salary", "Current_Monthly_Salary", "Salary")
        cur_f = None
        if cur is not None:
            try: cur_f = float(cur)
            except (TypeError, ValueError):
                errors.append({"row": row_no, "EmpID": emp_id, "Employee": name, "error": "invalid current salary"})
                continue
        entries.append({"row_no": row_no, "emp_id": emp_id, "emp_name": str(name),
                        "title": canon, "family": info["job_family"],
                        "department": info["department"],
                        "city": cell("City", "city"), "experience": exp_f,
                        "seniority": cell("Seniority", "seniority"),
                        "industry": cell("Industry", "industry_category"),
                        "current_salary": cur_f})
    return entries, errors


def _bulk_india_validate(df, filename):
    from ml import india_infer
    entries, errors = _bulk_india_rows(df)
    cats = india_infer.supported_categories()
    return jsonify({
        "filename": filename, "total_rows": int(len(df)),
        "valid_rows": len(entries), "invalid_rows": len(errors),
        "sample_valid": [{"EmpID": e["emp_id"], "Employee": e["emp_name"],
                          "Job_Title": e["title"], "Department": e["department"]} for e in entries[:10]],
        "errors": errors[:100],
        "supported_departments": cats["Department"],
        "supported_job_titles": cats["Job_Title"],
        "supported_cities": cats["City"][:40],
        "model_version": active_version, "kind": "india_salary",
    })


def _bulk_india_process(df, filename):
    from ml import india_infer
    entries, errors = _bulk_india_rows(df)
    results = []
    for e in entries:
        try:
            r = india_infer.predict_salary(e["title"], e["experience"], e["city"],
                                           e["seniority"], e["industry"])
            if r.get("error"):
                results.append({"row": e["row_no"], "EmpID": e["emp_id"], "Employee": e["emp_name"],
                                "Department": e["department"], "Job_Title": e["title"],
                                "Current_Salary": e["current_salary"], "Status": "failed",
                                "Error": r["error"]})
                continue
            cur = e["current_salary"]
            est = r["salary_annual_inr"]
            results.append({
                "row": e["row_no"], "EmpID": e["emp_id"], "Employee": e["emp_name"],
                "Department": e["department"], "Job_Title": e["title"],
                "City": r.get("city_bucket"), "Current_Salary": cur,
                "Estimated_Salary": est,
                "Salary_P10": r["salary_p10_inr"], "Salary_P90": r["salary_p90_inr"],
                "Salary_Delta": round(est - cur, 2) if cur is not None else None,
                "Support_Level": r["support_level"],
                "Market_Observations": r["market_observations"],
                "Risk": None, "Status": "ok", "Model": active_version,
            })
        except Exception:
            log.exception("India bulk row %s failed", e["row_no"])
            results.append({"row": e["row_no"], "EmpID": e["emp_id"], "Employee": e["emp_name"],
                            "Department": e["department"], "Job_Title": e["title"],
                            "Current_Salary": e["current_salary"], "Status": "failed",
                            "Error": "prediction failed"})
    ok = [r for r in results if r["Status"] == "ok"]
    with_sal = [r for r in ok if r["Current_Salary"] is not None]
    def avg(xs): return round(sum(xs)/len(xs), 2) if xs else None
    summary = {"total_rows": len(df), "processed": len(results), "succeeded": len(ok),
               "failed": len(results)-len(ok), "validation_errors": len(errors),
               "avg_estimated_salary": avg([r["Estimated_Salary"] for r in ok]),
               "avg_current_salary": avg([r["Current_Salary"] for r in with_sal]),
               "employees_with_salary": len(with_sal),
               "salary_unit": "inr_annual"}
    job = BulkJob(filename=(filename or "upload")[:255], model_version=active_version,
                  total_rows=len(df), valid_rows=len(entries), invalid_rows=len(errors),
                  results_json=json.dumps(results), errors_json=json.dumps(errors),
                  summary_json=json.dumps(summary))
    db.session.add(job); db.session.commit()
    log.info("India bulk job %s: %s", job.id, summary)
    return jsonify({"job_id": job.id, "summary": summary,
                    "sample_results": results[:10], "sample_errors": errors[:10]})


# ---------------------------------------------------------------- app factory

def create_app(testing=False):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SQLALCHEMY_DATABASE_URI=_db_uri(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        TESTING=testing,
    )
    allowed = os.environ.get(
        "ESPS_ALLOWED_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000",
    ).split(",")
    CORS(app, origins=[o.strip() for o in allowed if o.strip()], supports_credentials=True)
    db.init_app(app)

    @app.before_request
    def _global_rate_limit():
        if request.path.startswith("/api/") and not rate_allow(
            f"api:{_client_ip()}", API_RATE_LIMIT, 60
        ):
            return _json_error("Rate limit exceeded", 429, "RATE_LIMITED")

    @app.after_request
    def _security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; connect-src 'self'"
        )
        return resp

    with app.app_context():
        db.create_all()
        _migrate_history_table()

    # ---------------------------------------------------------- pages

    @app.route("/")
    def home():
        return render_template("index.html")

    # ---------------------------------------------------------- prediction

    @app.route("/api/predict", methods=["POST"])
    def predict():
        if not rate_allow(f"predict:{_client_ip()}", PREDICT_RATE_LIMIT, 60):
            return _json_error("Prediction rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Prediction models unavailable", 503, "MODELS_UNAVAILABLE")
        data = request.get_json(silent=True)
        if not _india_active():
            return _json_error("Active model does not support predictions", 503, "MODEL_UNSUPPORTED")
        return _predict_india(data)

    def _predict_india(data):
        """India salary model: annual INR benchmark + quantile range."""
        if not isinstance(data, dict):
            return _json_error("Request body must be a JSON object", 400, "BAD_REQUEST")
        from ml import india_infer
        res = india_infer.predict_salary(
            job_title=data.get("Job_Title") or data.get("job_title"),
            experience_years=data.get("Experience", data.get("experience_years")),
            city=data.get("City", data.get("city")),
            seniority=data.get("Seniority", data.get("seniority")),
            industry=data.get("Industry", data.get("industry")),
        )
        if res.get("error"):
            return _json_error(res["error"], 422, "UNSUPPORTED_TITLE")
        name = data.get("Name", "Anonymous")
        if not isinstance(name, str) or len(name) > 100:
            return _err("Name", "must be a string up to 100 chars")
        if not data.get("simulate"):
            try:
                db.session.add(PredictionHistory(
                    name=name.strip() or "Anonymous",
                    salary=res["salary_annual_inr"], risk_score=None,
                    kind="prediction"))
                db.session.commit()
            except Exception:
                db.session.rollback()
        res["model_version"] = active_version
        res["name"] = name.strip() or "Anonymous"
        return jsonify(res)

    # ---------------------------------------------------------- history

    @app.route("/api/history", methods=["GET"])
    def get_history():
        page, e = _as_int(request.args.get("page", 1), "page", 1, 10_000)
        if e: return e
        size, e = _as_int(request.args.get("page_size", HISTORY_PAGE_SIZE), "page_size", 1, 100)
        if e: return e
        q = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc(), PredictionHistory.id.desc())
        total = q.count()
        rows = q.offset((page - 1) * size).limit(size).all()
        items = [{
            "id": r.id, "name": r.name, "salary": r.salary, "risk": r.risk_score,
            "kind": r.kind,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else None,
        } for r in rows]
        return jsonify({"items": items, "page": page, "page_size": size, "total": total})

    @app.route("/api/history/<int:record_id>", methods=["DELETE"])
    def delete_history_record(record_id):
        rec = db.session.get(PredictionHistory, record_id)
        if rec is None:
            return _json_error("Record not found", 404, "NOT_FOUND")
        db.session.delete(rec)
        db.session.commit()
        log.info("History record %s deleted", record_id)
        return jsonify({"message": "Record deleted"})

    @app.route("/api/history", methods=["DELETE"])
    def clear_history():
        if not rate_allow(f"historyclear:{_client_ip()}", 5, 300):
            return _json_error("Rate limit exceeded", 429, "RATE_LIMITED")
        n = db.session.query(PredictionHistory).delete()
        db.session.commit()
        log.warning("History cleared (%d rows)", n)
        return jsonify({"message": "History cleared successfully.", "deleted": n})

    # ---------------------------------------------------------- model registry

    @app.route("/api/model", methods=["GET"])
    def model_info():
        load_all()
        reg = load_registry()
        from ml import india_infer
        cats = india_infer.supported_categories()
        title_info = india_infer._load()["title_info"]
        payload = {
            "active_version": active_version,
            "available_versions": sorted(reg.get("versions", {}).keys()),
            "kind": "india_salary",
            "salary_unit": "inr_annual",
            "dataset": active_meta.get("dataset", "india_salary_benchmarks"),
            "metrics": active_meta.get("evaluation", {}),
            "features": ["Job_Title", "Experience", "City", "Seniority", "Industry"],
            "categories": cats,
            "job_families": {t: (title_info.get(t) or {}).get("job_family", "Other")
                             for t in cats["Job_Title"]},
            "algorithm": "HistGradientBoosting quantile regressors",
            "prediction_output": "P10 · P50 · P90 annual salary (INR)",
            "caveats": [
                "India job-market sample (Fliar postings); annual INR benchmark.",
                "Only job titles with >=5 training observations are supported.",
                "Quantile range is not a statistical confidence interval.",
            ],
        }
        return jsonify(payload)

    # ---------------------------------------------------------- analytics


    @app.route("/api/analytics/summary", methods=["GET"])
    def analytics_summary():
        total = PredictionHistory.query.count()
        if total == 0:
            return jsonify({"total": 0})
        avg_risk = db.session.query(db.func.avg(PredictionHistory.risk_score)).scalar() or 0
        avg_sal = db.session.query(db.func.avg(PredictionHistory.salary)).scalar() or 0
        high = PredictionHistory.query.filter(PredictionHistory.risk_score > 0.6).count()
        return jsonify({
            "total_predictions": int(total),
            "avg_risk_score": round(float(avg_risk), 4),
            "avg_estimated_salary": round(float(avg_sal), 2),
            "high_risk_count": int(high),
            "disclaimer": "Aggregates model outputs stored in history; not verified outcomes.",
        })

    # ---------------------------------------------------------- health


    # ---------------------------------------------------------- bulk salary planning

    @app.route("/api/bulk/template", methods=["GET"])
    def bulk_template():
        fmt = request.args.get("format", "csv")
        load_all()
        if not _india_active():
            return _json_error("Model does not support bulk templates", 503, "MODEL_UNSUPPORTED")
        from ml import india_infer
        _ic = india_infer.supported_categories()
        _dept0 = (_ic.get("Department") or ["Engineering"])[0]
        _title0 = (_ic.get("Job_Title") or ["Software Engineer"])[0]
        _req_cols = {"Job_Title"}
        if fmt == "xlsx":
            from openpyxl import Workbook
            wb = Workbook(); ws = wb.active; ws.title = "Template"
            ws.append(BULK_TEMPLATE_COLUMNS)
            ws.append(["E001", "Example Employee", _dept0, _title0, 30, 5, 3, 50000,
                       "Male", 0.7, 2, 160])
            ws2 = wb.create_sheet("Column Guide")
            ws2.append(["Column", "Description", "Required"])
            for c in BULK_TEMPLATE_COLUMNS:
                ws2.append([_safe_cell(c), BULK_TEMPLATE_NOTES.get(c, ""),
                            "yes" if c in _req_cols else "no"])
            bio = io.BytesIO(); wb.save(bio)
            from flask import Response
            return Response(bio.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            headers={"Content-Disposition": "attachment; filename=salary_template.xlsx"})
        import csv as _csv
        out = io.StringIO(); w = _csv.writer(out)
        w.writerow(BULK_TEMPLATE_COLUMNS)
        w.writerow(["E001", "Example Employee", _dept0, _title0,
                    30, 5, 3, 50000, "Male", 0.7, 2, 160])
        w.writerow([])
        w.writerow(["COLUMN GUIDE"])
        w.writerow(["Column", "Description", "Required"])
        for c in BULK_TEMPLATE_COLUMNS:
            w.writerow([c, BULK_TEMPLATE_NOTES.get(c, ""),
                        "yes" if c in _req_cols else "no"])
        from flask import Response
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=salary_template.csv"})

    @app.route("/api/bulk/validate", methods=["POST"])
    def bulk_validate():
        """Parse + validate an uploaded sheet WITHOUT running predictions."""
        if not rate_allow(f"bulkval:{_client_ip()}", 60, 60):
            return _json_error("Rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Models unavailable", 503, "MODELS_UNAVAILABLE")
        if "file" not in request.files:
            return _json_error("No file part", 400, "NO_FILE")
        df, err = _read_upload(request.files["file"])
        if err: return err
        return _bulk_india_validate(df, request.files["file"].filename)

    @app.route("/api/bulk/process", methods=["POST"])
    def bulk_process():
        """Validate + predict + persist a BulkJob. Returns job id + summary."""
        if not rate_allow(f"bulkproc:{_client_ip()}", BULK_RATE_LIMIT, 60):
            return _json_error("Bulk processing rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Models unavailable", 503, "MODELS_UNAVAILABLE")
        if "file" not in request.files:
            return _json_error("No file part", 400, "NO_FILE")
        df, err = _read_upload(request.files["file"])
        if err: return err
        return _bulk_india_process(df, request.files["file"].filename)

    @app.route("/api/bulk/jobs", methods=["GET"])
    def bulk_jobs_list():
        rows = (BulkJob.query.order_by(BulkJob.created_at.desc(), BulkJob.id.desc())
                .limit(50).all())
        return jsonify({"items": [{
            "id": j.id, "filename": j.filename,
            "created_at": j.created_at.strftime("%Y-%m-%d %H:%M:%S") if j.created_at else None,
            "model_version": j.model_version, "total_rows": j.total_rows,
            "valid_rows": j.valid_rows, "invalid_rows": j.invalid_rows,
            "summary": json.loads(j.summary_json) if j.summary_json else None,
        } for j in rows]})

    @app.route("/api/bulk/jobs/<int:job_id>", methods=["GET"])
    def bulk_job_detail(job_id):
        job = db.session.get(BulkJob, job_id)
        if job is None:
            return _json_error("Job not found", 404, "NOT_FOUND")
        return jsonify({
            "id": job.id, "filename": job.filename,
            "created_at": job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else None,
            "model_version": job.model_version, "total_rows": job.total_rows,
            "valid_rows": job.valid_rows, "invalid_rows": job.invalid_rows,
            "summary": json.loads(job.summary_json) if job.summary_json else None,
            "results": json.loads(job.results_json) if job.results_json else [],
            "errors": json.loads(job.errors_json) if job.errors_json else [],
            "scenario": json.loads(job.scenario_json) if job.scenario_json else None,
        })

    @app.route("/api/bulk/jobs/<int:job_id>/scenario", methods=["POST"])
    def bulk_job_scenario(job_id):
        """Compute an HR increment scenario for a stored job (arithmetic only)."""
        job = db.session.get(BulkJob, job_id)
        if job is None:
            return _json_error("Job not found", 404, "NOT_FOUND")
        data = request.get_json(silent=True) or {}
        results = json.loads(job.results_json) if job.results_json else []
        try:
            scenario = _scenario_calc(results, data)
        except (TypeError, ValueError):
            return _err("increment_pct", "must be a number")
        job.scenario_json = json.dumps(scenario)
        db.session.commit()
        return jsonify(scenario)

    @app.route("/api/bulk/jobs/<int:job_id>/export", methods=["GET"])
    def bulk_job_export(job_id):
        job = db.session.get(BulkJob, job_id)
        if job is None:
            return _json_error("Job not found", 404, "NOT_FOUND")
        fmt = request.args.get("format", "xlsx")
        results = json.loads(job.results_json) if job.results_json else []
        errors = json.loads(job.errors_json) if job.errors_json else []
        scenario = json.loads(job.scenario_json) if job.scenario_json else None
        from flask import Response
        if fmt == "csv":
            return Response(_export_csv(results, errors, scenario), mimetype="text/csv",
                            headers={"Content-Disposition": f"attachment; filename=salary_results_job{job_id}.csv"})
        return Response(_export_xlsx(job, results, errors, scenario),
                        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f"attachment; filename=salary_results_job{job_id}.xlsx"})

    @app.route("/api/health", methods=["GET"])
    def health():
        try:
            load_ok = load_all()
            return jsonify({
                "status": "ok" if load_ok else "degraded",
                "models_loaded": bool(models),
                "db_ok": db.session.execute(db.text("SELECT 1")).scalar() == 1,
            })
        except Exception:
            log.exception("Health check failed")
            return jsonify({"status": "error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
