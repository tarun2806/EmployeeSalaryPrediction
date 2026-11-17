import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("esps")

REGISTRY_PATH = MODELS_PATH / "registry.json"
ALL_APP_FIELDS = ["Age", "Gender", "Department", "Job_Title", "Experience",
                  "Satisfaction", "Hours", "Promotion"]
FEATURE_LABELS = ["Age", "Gender", "Dept", "Title", "Tenure", "Sat", "Hours", "Promo"]
BULK_REQUIRED_COLUMNS = [
    "Age", "Gender", "Department", "Job_Title", "Years_at_Company",
    "Satisfaction_Level", "Average_Monthly_Hours", "Promotion_Last_5Years",
]
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
active_version = "legacy"
active_meta = {}


def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"active": "legacy", "versions": {"legacy": {"dir": "models"}}}


def active_models_dir():
    reg = load_registry()
    version = os.environ.get("ESPS_MODEL_VERSION") or reg.get("active", "legacy")
    v = reg["versions"].get(version)
    if v is None:
        raise KeyError(f"Unknown model version '{version}'")
    return version, BASE_DIR / v["dir"], v


def active_feature_fields():
    feats = active_meta.get("features")
    if feats:
        return [f for f in ALL_APP_FIELDS if f in feats]
    return ALL_APP_FIELDS


def active_categories():
    return {
        "Gender": sorted(models["le_gender"].classes_.tolist()),
        "Department": sorted(models["le_dept"].classes_.tolist()),
        "Job_Title": sorted(models["le_title"].classes_.tolist()),
    }


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
        else:
            active_meta = {"version": version, "features": ALL_APP_FIELDS}
        for key, fname in [
            ("attr_model", "attr_model.pkl"), ("sal_model", "sal_model.pkl"),
            ("attr_scaler", "attr_scaler.pkl"), ("sal_scaler", "sal_scaler.pkl"),
            ("le_gender", "le_gender.pkl"), ("le_dept", "le_dept.pkl"),
            ("le_title", "le_title.pkl"),
        ]:
            models[key] = joblib.load(mdir / fname)
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
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


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


def _as_float(value, field, lo, hi):
    try:
        if isinstance(value, bool):
            raise ValueError
        v = float(value)
    except (TypeError, ValueError):
        return None, _err(field, "must be a number")
    if not lo <= v <= hi:
        return None, _err(field, f"must be between {lo} and {hi}")
    return v, None


def _as_choice(value, field, choices):
    if not isinstance(value, str) or value not in choices:
        return None, _err(field, f"must be one of {sorted(choices)}")
    return value, None


def _validate_predict_payload(data):
    if not isinstance(data, dict):
        return None, _json_error("Request body must be a JSON object", 400, "BAD_REQUEST")

    fields = set(active_feature_fields())
    clean = {}
    name = data.get("Name", "Anonymous")
    if not isinstance(name, str) or len(name) > 100:
        return None, _err("Name", "must be a string up to 100 chars")
    clean["Name"] = name.strip() or "Anonymous"

    age, e = _as_int(data.get("Age"), "Age", 16, 100)
    if e: return None, e
    clean["Age"] = age
    exp, e = _as_int(data.get("Experience"), "Experience", 0, 60)
    if e: return None, e
    clean["Experience"] = exp
    if "Hours" in fields:
        hours, e = _as_int(data.get("Hours"), "Hours", 0, 500)
        if e: return None, e
        clean["Hours"] = hours
    else:
        clean["Hours"] = data.get("Hours", 160)
    sat, e = _as_float(data.get("Satisfaction"), "Satisfaction", 0.0, 1.0)
    if e: return None, e
    clean["Satisfaction"] = sat
    promo, e = _as_choice(data.get("Promotion"), "Promotion", {"Yes", "No"})
    if e: return None, e
    clean["Promotion"] = promo
    cats = active_categories()
    for key in ("Gender", "Department", "Job_Title"):
        val, e = _as_choice(data.get(key), key, set(cats[key]))
        if e: return None, e
        clean[key] = val
    clean["simulate"] = bool(data.get("simulate", False))
    return clean, None


def _encode_row(clean):
    g_enc = models["le_gender"].transform([clean["Gender"]])[0]
    d_enc = models["le_dept"].transform([clean["Department"]])[0]
    t_enc = models["le_title"].transform([clean["Job_Title"]])[0]
    p = 1 if clean["Promotion"] == "Yes" else 0
    vals = {
        "Age": clean["Age"], "Gender": g_enc, "Department": d_enc,
        "Job_Title": t_enc, "Experience": clean["Experience"],
        "Satisfaction": clean["Satisfaction"], "Hours": clean.get("Hours", 0), "Promotion": p,
    }
    return np.array([vals[f] for f in active_feature_fields()], dtype=float).reshape(1, -1)


def _predict_row(X_arr):
    risk = float(models["attr_model"].predict_proba(models["attr_scaler"].transform(X_arr))[0][1])
    sal = float(models["sal_model"].predict(models["sal_scaler"].transform(X_arr))[0])
    return risk, sal


def _shap_explanation(X_arr):
    try:
        import shap
        explainer = shap.TreeExplainer(models["attr_model"])
        sv = explainer.shap_values(models["attr_scaler"].transform(X_arr))
        arr = sv[1] if isinstance(sv, list) else sv
        vals = np.asarray(arr).reshape(-1)
        labels = active_feature_fields()
        order = np.argsort(-np.abs(vals))[:5]
        return [{"feature": labels[i], "impact": round(float(vals[i]), 4)} for i in order]
    except Exception:
        log.exception("SHAP explanation failed")
        return None


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
        clean, err = _validate_predict_payload(data)
        if err:
            return err
        try:
            X_arr = _encode_row(clean)
            risk, sal = _predict_row(X_arr)
        except Exception:
            log.exception("Prediction failed")
            return _json_error("Prediction failed", 500, "PREDICTION_ERROR")

        explanation = _shap_explanation(X_arr)

        if not clean["simulate"]:
            try:
                entry = PredictionHistory(
                    name=clean["Name"], salary=sal, risk_score=risk, kind="prediction",
                )
                db.session.add(entry)
                db.session.commit()
            except Exception:
                db.session.rollback()
                log.exception("Failed to persist prediction history")

        field_labels = {
            "Age": "Age", "Gender": "Gender", "Department": "Dept",
            "Job_Title": "Title", "Experience": "Tenure",
            "Satisfaction": "Sat", "Hours": "Hours", "Promotion": "Promo",
        }
        return jsonify({
            "risk": round(risk, 4),
            "salary": round(sal, 2),
            "salary_unit": active_meta.get("salary_unit", "inr_monthly"),
            "model_version": active_version,
            "feat_imp": models["attr_model"].feature_importances_.tolist(),
            "features": [field_labels[f] for f in active_feature_fields()],
            "explanation": explanation,
            "disclaimer": "Model estimate only — not a causal prediction or employment decision.",
        })

    # ---------------------------------------------------------- bulk predict

    @app.route("/api/bulk_predict", methods=["POST"])
    def bulk_predict():
        if not rate_allow(f"bulk:{_client_ip()}", BULK_RATE_LIMIT, 60):
            return _json_error("Bulk analysis rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Prediction models unavailable", 503, "MODELS_UNAVAILABLE")

        if "file" not in request.files:
            return _json_error("No file part", 400, "NO_FILE")
        file = request.files["file"]
        if not file or not file.filename:
            return _json_error("No selected file", 400, "NO_FILE")
        if not file.filename.lower().endswith(".csv"):
            return _json_error("Only .csv files are accepted", 415, "UNSUPPORTED_MEDIA_TYPE")

        raw = file.read(MAX_CONTENT_LENGTH + 1)
        if len(raw) > MAX_CONTENT_LENGTH:
            return _json_error("File exceeds 5 MB limit", 413, "FILE_TOO_LARGE")

        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            log.warning("Malformed CSV upload")
            return _json_error("File is not a valid CSV", 422, "MALFORMED_CSV")

        required = [c for c in BULK_REQUIRED_COLUMNS
                    if c != "Average_Monthly_Hours" or "Hours" in active_feature_fields()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return _json_error(f"Missing required columns: {', '.join(missing)}", 422, "MISSING_COLUMNS")
        if len(df) > MAX_BULK_ROWS:
            return _json_error(f"Too many rows (max {MAX_BULK_ROWS})", 413, "TOO_MANY_ROWS")
        if len(df) == 0:
            return _json_error("CSV contains no data rows", 422, "EMPTY_FILE")

        results, errors = [], []
        for idx, row in df.iterrows():
            row_no = int(idx) + 2
            emp_id = row.get("Employee_ID", row_no - 1)
            try:
                payload = {
                    "Name": str(emp_id),
                    "Age": row["Age"], "Gender": row["Gender"],
                    "Department": row["Department"], "Job_Title": row["Job_Title"],
                    "Experience": row["Years_at_Company"],
                    "Satisfaction": row["Satisfaction_Level"],
                    "Hours": row.get("Average_Monthly_Hours", 160),
                    "Promotion": "Yes" if str(row["Promotion_Last_5Years"]).strip() in ("1", "1.0", "Yes", "yes", "True") else "No",
                }
                clean, verr = _validate_predict_payload(payload)
                if verr:
                    errors.append({"row": row_no, "EmpID": emp_id, "error": "validation failed"})
                    continue
                X_arr = _encode_row(clean)
                risk, sal = _predict_row(X_arr)
                results.append({"EmpID": emp_id, "risk": risk, "salary": sal, "dept": clean["Department"]})
            except Exception:
                log.exception("Bulk row %s failed", row_no)
                errors.append({"row": row_no, "EmpID": emp_id, "error": "prediction failed"})

        log.info("Bulk predict: %d ok, %d errors", len(results), len(errors))
        return jsonify({
            "results": results, "errors": errors,
            "summary": {"total": len(df), "succeeded": len(results), "failed": len(errors)},
        })

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
        return jsonify({
            "active_version": active_version,
            "available_versions": sorted(reg.get("versions", {}).keys()),
            "features": active_feature_fields(),
            "categories": active_categories() if models else {},
            "salary_unit": active_meta.get("salary_unit", "inr_monthly"),
            "dataset": active_meta.get("dataset", "synthetic_demo"),
            "metrics": active_meta.get("evaluation", {}),
            "caveats": active_meta.get("caveats", []),
        })

    # ---------------------------------------------------------- simulation

    @app.route("/api/simulate", methods=["POST"])
    def simulate():
        if not rate_allow(f"sim:{_client_ip()}", SIM_RATE_LIMIT, 60):
            return _json_error("Simulation rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Prediction models unavailable", 503, "MODELS_UNAVAILABLE")
        data = request.get_json(silent=True)
        clean, err = _validate_predict_payload(data)
        if err:
            return err
        sat_boost, e = _as_float(data.get("satisfaction_boost", 0.0), "satisfaction_boost", -1.0, 1.0)
        if e: return e
        hours_delta, e = _as_int(data.get("hours_delta", 0), "hours_delta", -200, 200)
        if e: return e

        scenario = dict(clean)
        scenario["Satisfaction"] = min(1.0, max(0.0, clean["Satisfaction"] + sat_boost))
        if "Hours" in active_feature_fields():
            scenario["Hours"] = max(0, min(500, clean.get("Hours", 160) + hours_delta))

        try:
            risk_b, sal_b = _predict_row(_encode_row(clean))
            risk_s, sal_s = _predict_row(_encode_row(scenario))
        except Exception:
            log.exception("Simulation failed")
            return _json_error("Simulation failed", 500, "SIMULATION_ERROR")

        return jsonify({
            "baseline": {"risk": round(risk_b, 4), "salary": round(sal_b, 2)},
            "scenario": {"risk": round(risk_s, 4), "salary": round(sal_s, 2)},
            "delta_risk": round(risk_s - risk_b, 4),
            "delta_salary": round(sal_s - sal_b, 2),
            "salary_unit": active_meta.get("salary_unit", "inr_monthly"),
            "model_version": active_version,
            "disclaimer": ("Model-based estimate under changed inputs only. A predictive "
                           "model cannot prove that a policy change causes attrition "
                           "or salary outcomes."),
        })

    # ---------------------------------------------------------- analytics

    @app.route("/api/analytics/upload", methods=["POST"])
    def analytics_upload():
        if not rate_allow(f"analytics:{_client_ip()}", ANALYTICS_RATE_LIMIT, 60):
            return _json_error("Analytics rate limit exceeded", 429, "RATE_LIMITED")
        if "file" not in request.files:
            return _json_error("No file part", 400, "NO_FILE")
        file = request.files["file"]
        if not file or not file.filename or not file.filename.lower().endswith(".csv"):
            return _json_error("Only .csv files are accepted", 415, "UNSUPPORTED_MEDIA_TYPE")
        raw = file.read(MAX_CONTENT_LENGTH + 1)
        if len(raw) > MAX_CONTENT_LENGTH:
            return _json_error("File exceeds 5 MB limit", 413, "FILE_TOO_LARGE")
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            return _json_error("File is not a valid CSV", 422, "MALFORMED_CSV")
        if len(df) == 0 or len(df) > MAX_BULK_ROWS:
            return _json_error("Invalid row count", 422, "BAD_ROWS")

        stats = {"rows": int(len(df)), "columns": int(df.shape[1])}
        group_col = "Department" if "Department" in df.columns else None
        if group_col:
            dept = {}
            for name, grp in df.groupby(group_col):
                entry = {"count": int(len(grp))}
                for col, key in [("Satisfaction_Level", "avg_satisfaction"),
                                 ("Average_Monthly_Hours", "avg_hours"),
                                 ("Salary", "avg_salary"), ("Attrition", "attrition_rate")]:
                    if col in grp.columns:
                        vals = pd.to_numeric(grp[col], errors="coerce")
                        entry[key] = round(float(vals.mean()), 4) if vals.notna().any() else None
                dept[str(name)] = entry
            stats["by_department"] = dept
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        stats["numeric_summary"] = {
            c: {"mean": round(float(df[c].mean()), 2), "min": float(df[c].min()),
                "max": float(df[c].max())}
            for c in numeric_cols[:10]
        }

        insights = []
        if "Attrition" in df.columns and group_col:
            rates = {k: v.get("attrition_rate") for k, v in stats.get("by_department", {}).items()
                     if v.get("attrition_rate") is not None}
            if rates:
                worst = max(rates, key=rates.get)
                insights.append({
                    "type": "attrition_concentration",
                    "message": f"Department '{worst}' shows the highest attrition rate "
                               f"({rates[worst]:.1%}) in this file. Review as a hypothesis, "
                               "not a confirmed finding.",
                })
        log.info("Analytics upload: %d rows", stats["rows"])
        return jsonify({"stats": stats, "insights": insights,
                        "disclaimer": "Descriptive statistics on uploaded data only."})

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
    app.run(port=5000, debug=False)
