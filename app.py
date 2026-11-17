import io
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("aurora")

ROLES = ("ADMIN", "HR_MANAGER", "ANALYST")
ROLE_RANK = {"ANALYST": 1, "HR_MANAGER": 2, "ADMIN": 3}

# Feature contract shared with train.py
FEATURES = [
    "Age", "Gender", "Department", "Job_Title", "Years_at_Company",
    "Satisfaction_Level", "Average_Monthly_Hours", "Promotion_Last_5Years",
]
FEATURE_LABELS = ["Age", "Gender", "Dept", "Title", "Tenure", "Sat", "Hours", "Promo"]
BULK_REQUIRED_COLUMNS = [
    "Age", "Gender", "Department", "Job_Title", "Years_at_Company",
    "Satisfaction_Level", "Average_Monthly_Hours", "Promotion_Last_5Years",
]
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload cap
MAX_BULK_ROWS = 5000
SESSION_TTL_HOURS = int(os.environ.get("AURORA_SESSION_HOURS", "12"))
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60
API_RATE_LIMIT = 300          # requests / minute / IP
BULK_RATE_LIMIT = 20          # requests / minute / user
HISTORY_PAGE_SIZE = 20


REGISTRY_PATH = MODELS_PATH / "registry.json"
# App-schema fields in canonical model-vector order
ALL_APP_FIELDS = ["Age", "Gender", "Department", "Job_Title", "Experience",
                  "Satisfaction", "Hours", "Promotion"]


def _default_db_path():
    if os.environ.get("AURORA_DATABASE_URL"):
        return os.environ["AURORA_DATABASE_URL"]  # e.g. postgresql://...
    if os.environ.get("AURORA_DB_PATH"):
        return os.environ["AURORA_DB_PATH"]
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
    """Model registry: maps version names to artifact dirs + metadata."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"active": "legacy", "versions": {"legacy": {"dir": "models"}}}


def active_models_dir():
    reg = load_registry()
    version = os.environ.get("AURORA_MODEL_VERSION") or reg.get("active", "legacy")
    v = reg["versions"].get(version)
    if v is None:
        raise KeyError(f"Unknown model version '{version}'")
    return version, BASE_DIR / v["dir"], v


def active_feature_fields():
    """Ordered app-schema fields the ACTIVE model consumes."""
    feats = active_meta.get("features")
    if feats:
        return [f for f in ALL_APP_FIELDS if f in feats]
    return ALL_APP_FIELDS  # legacy contract


def active_categories():
    """Allowed categorical values for the active model (from fitted encoders)."""
    return {
        "Gender": sorted(models["le_gender"].classes_.tolist()),
        "Department": sorted(models["le_dept"].classes_.tolist()),
        "Job_Title": sorted(models["le_title"].classes_.tolist()),
    }


# ---------------------------------------------------------------- models

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="ANALYST")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Session(db.Model):
    __tablename__ = "sessions"
    token = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    csrf_token = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)


class PredictionHistory(db.Model):
    __tablename__ = "history"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    salary = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    kind = db.Column(db.String(20), nullable=False, default="prediction")


class RateLimitHit(db.Model):
    """Persistent fixed-window rate limit counters (shared across instances)."""
    __tablename__ = "rate_limits"
    key = db.Column(db.String(200), primary_key=True)
    window = db.Column(db.Integer, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)


class AuditEvent(db.Model):
    """Append-only audit trail. Never store secrets or raw employee PII."""
    __tablename__ = "audit_events"
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    user_id = db.Column(db.Integer, nullable=True)
    username = db.Column(db.String(80))
    action = db.Column(db.String(60), nullable=False, index=True)
    detail = db.Column(db.String(500))
    ip = db.Column(db.String(64))


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
    """DB-backed fixed-window rate limiter. Returns True if request allowed."""
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


def audit(action, detail=None):
    """Write an audit event. Never include passwords/tokens/PII in detail."""
    try:
        user = getattr(g, "user", None)
        db.session.add(AuditEvent(
            user_id=user.id if user else None,
            username=user.username if user else None,
            action=action,
            detail=json.dumps(detail)[:500] if detail else None,
            ip=_client_ip(),
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()
        log.exception("Audit write failed for action=%s", action)


login_attempts = {}  # kept for API compatibility; authoritative lockout is DB-backed below


def rate_peek(key, window_seconds):
    """Current count in the active window without consuming a slot."""
    window = int(time.time() // window_seconds)
    hit = db.session.get(RateLimitHit, f"{key}:{window}")
    return hit.count if hit else 0


def rate_reset_prefix(prefix):
    RateLimitHit.query.filter(RateLimitHit.key.like(f"{prefix}:%")).delete()
    db.session.commit()


# ---------------------------------------------------------------- auth

def _get_session():
    token = request.cookies.get("aurora_session")
    if not token:
        return None
    sess = db.session.get(Session, token)
    if sess is None:
        return None
    if sess.expires_at <= datetime.utcnow():
        db.session.delete(sess)
        db.session.commit()
        return None
    user = db.session.get(User, sess.user_id)
    if user is None or not user.is_active:
        return None
    return sess, user


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapper(*args, **kwargs):
        current = _get_session()
        if current is None:
            return _json_error("Authentication required", 401, "UNAUTHENTICATED")
        g.session, g.user = current
        return view(*args, **kwargs)

    return wrapper


def role_required(*roles):
    from functools import wraps

    def decorator(view):
        @wraps(view)
        @login_required
        def wrapper(*args, **kwargs):
            if g.user.role not in roles:
                log.warning("RBAC deny: user=%s role=%s path=%s", g.user.username, g.user.role, request.path)
                return _json_error("Insufficient permissions", 403, "FORBIDDEN")
            return view(*args, **kwargs)

        return wrapper

    return decorator


def require_csrf():
    """Validate CSRF token for state-changing requests on authenticated routes."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    token = request.headers.get("X-CSRF-Token", "")
    if not token or not g.session or token != g.session.csrf_token:
        return _json_error("Invalid or missing CSRF token", 403, "CSRF_FAILED")
    return None


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
    """Returns (clean_dict, error_response). Version-aware via active schema."""
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
        clean["Hours"] = data.get("Hours", 160)  # accepted but unused by this model
    sat, e = _as_float(data.get("Satisfaction"), "Satisfaction", 0.0, 1.0)
    if e: return None, e
    clean["Satisfaction"] = sat
    promo, e = _as_choice(data.get("Promotion"), "Promotion", {"Yes", "No"})
    if e: return None, e
    clean["Promotion"] = promo
    cats = active_categories()
    for key, model_key in [("Gender", "Gender"), ("Department", "Department"), ("Job_Title", "Job_Title")]:
        val, e = _as_choice(data.get(key), key, set(cats[model_key]))
        if e: return None, e
        clean[key] = val
    clean["simulate"] = bool(data.get("simulate", False))
    return clean, None


def _encode_row(clean):
    """Build the feature vector in the ACTIVE model's expected field order."""
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


def _shap_explanation(X_arr):
    """Top feature contributions for the attrition model (SHAP TreeExplainer)."""
    try:
        import shap
        explainer = shap.TreeExplainer(models["attr_model"])
        sv = explainer.shap_values(models["attr_scaler"].transform(X_arr))
        # sklearn GBM binary -> list per class or single array; take positive class
        arr = sv[1] if isinstance(sv, list) else sv
        vals = np.asarray(arr).reshape(-1)
        labels = active_feature_fields()
        order = np.argsort(-np.abs(vals))[:5]
        return [{"feature": labels[i], "impact": round(float(vals[i]), 4)} for i in order]
    except Exception:
        log.exception("SHAP explanation failed")
        return None


def _predict_row(X_arr):
    risk = float(models["attr_model"].predict_proba(models["attr_scaler"].transform(X_arr))[0][1])
    sal = float(models["sal_model"].predict(models["sal_scaler"].transform(X_arr))[0])
    return risk, sal


def _migrate_history_table():
    """Add Phase-1 columns to the legacy history table if missing."""
    insp = db.inspect(db.engine)
    cols = {c["name"] for c in insp.get_columns("history")}
    with db.engine.begin() as conn:
        if "user_id" not in cols:
            conn.exec_driver_sql("ALTER TABLE history ADD COLUMN user_id INTEGER REFERENCES users(id)")
        if "kind" not in cols:
            conn.exec_driver_sql("ALTER TABLE history ADD COLUMN kind VARCHAR(20) NOT NULL DEFAULT 'prediction'")


def _seed_admin():
    """Create the initial admin from env vars. Never hardcodes credentials."""
    if User.query.count() > 0:
        return
    username = os.environ.get("ADMIN_USERNAME", "admin").strip()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not password:
        password = secrets.token_urlsafe(12)
        log.warning(
            "No ADMIN_PASSWORD set; generated one-time admin password for '%s': %s "
            "(set ADMIN_PASSWORD env var to choose your own; change it after first login)",
            username, password,
        )
    db.session.add(User(username=username, password_hash=generate_password_hash(password), role="ADMIN"))
    db.session.commit()
    log.info("Seeded initial admin user '%s'", username)


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
        "AURORA_ALLOWED_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000",
    ).split(",")
    CORS(app, origins=[o.strip() for o in allowed if o.strip()], supports_credentials=True)
    db.init_app(app)

    @app.before_request
    def _global_rate_limit():
        if not rate_allow(f"api:{_client_ip()}", API_RATE_LIMIT, 60):
            return _json_error("Rate limit exceeded", 429, "RATE_LIMITED")

    @app.before_request
    def _load_session():
        g.session = None
        g.user = None
        if request.path.startswith("/api/") and request.path != "/api/auth/login":
            current = _get_session()
            if current is not None:
                g.session, g.user = current

    @app.context_processor
    def _inject_user():
        return {"current_user": getattr(g, "user", None)}

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
        _seed_admin()

    # ---------------------------------------------------------- pages

    @app.route("/")
    def home():
        return render_template("index.html")

    # ---------------------------------------------------------- auth

    @app.route("/api/auth/login", methods=["POST"])
    def login():
        ip = _client_ip()
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        password = data.get("password")
        if not isinstance(username, str) or not isinstance(password, str) or not username or not password:
            return _json_error("Username and password are required", 400, "BAD_REQUEST")
        username = username.strip()[:80]
        lk = f"loginfail:{ip}:{username.lower()}"
        if rate_peek(lk, LOGIN_LOCKOUT_SECONDS) >= LOGIN_MAX_ATTEMPTS:
            log.warning("Login locked out: ip=%s user=%s", ip, username)
            audit("login_locked", {"username": username})
            return _json_error("Too many failed attempts. Try again later.", 429, "LOCKED_OUT")
        user = User.query.filter_by(username=username).first()
        if user is None or not user.is_active or not check_password_hash(user.password_hash, password):
            rate_allow(lk, LOGIN_MAX_ATTEMPTS, LOGIN_LOCKOUT_SECONDS)
            log.info("Failed login: ip=%s user=%s", ip, username)
            audit("login_failed", {"username": username})
            return _json_error("Invalid credentials", 401, "INVALID_CREDENTIALS")
        rate_reset_prefix(lk)
        sess = Session(
            token=secrets.token_urlsafe(32),
            user_id=user.id,
            csrf_token=secrets.token_urlsafe(32),
            expires_at=datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
        )
        db.session.add(sess)
        db.session.commit()
        resp = jsonify({
            "user": {"username": user.username, "role": user.role},
            "csrf_token": sess.csrf_token,
        })
        resp.set_cookie(
            "aurora_session", sess.token,
            httponly=True, samesite="Lax",
            secure=request.is_secure, max_age=SESSION_TTL_HOURS * 3600,
        )
        log.info("Login success: user=%s role=%s ip=%s", user.username, user.role, ip)
        audit("login", {"role": user.role})
        return resp

    @app.route("/api/auth/logout", methods=["POST"])
    def logout():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        if g.session is not None:
            db.session.delete(g.session)
            db.session.commit()
            log.info("Logout: user=%s", g.user.username if g.user else "?")
            audit("logout")
        resp = jsonify({"message": "Logged out"})
        resp.delete_cookie("aurora_session")
        return resp

    @app.route("/api/auth/me", methods=["GET"])
    def me():
        if g.user is None:
            return _json_error("Authentication required", 401, "UNAUTHENTICATED")
        return jsonify({
            "user": {"username": g.user.username, "role": g.user.role},
            "csrf_token": g.session.csrf_token,
        })

    # ---------------------------------------------------------- prediction

    @app.route("/api/predict", methods=["POST"])
    @login_required
    def predict():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        if not rate_allow(f"predict:{g.user.id}", 60, 60):
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
                    name=clean["Name"], salary=sal, risk_score=risk,
                    user_id=g.user.id, kind="prediction",
                )
                db.session.add(entry)
                db.session.commit()
                audit("predict", {"persisted": True})
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
    @login_required
    def bulk_predict():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        if not rate_allow(f"bulk:{g.user.id}", BULK_RATE_LIMIT, 60):
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
            log.warning("Malformed CSV upload by user %s", g.user.username)
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

        known = {
            "Gender": set(models["le_gender"].classes_),
            "Department": set(models["le_dept"].classes_),
            "Job_Title": set(models["le_title"].classes_),
        }
        results, errors = [], []
        for idx, row in df.iterrows():
            row_no = int(idx) + 2  # 1-based + header
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
                results.append({
                    "EmpID": emp_id, "risk": risk, "salary": sal,
                    "dept": clean["Department"],
                })
            except Exception:
                log.exception("Bulk row %s failed", row_no)
                errors.append({"row": row_no, "EmpID": emp_id, "error": "prediction failed"})

        log.info("Bulk predict by %s: %d ok, %d errors", g.user.username, len(results), len(errors))
        audit("bulk_predict", {"total": len(df), "succeeded": len(results), "failed": len(errors)})
        return jsonify({
            "results": results,
            "errors": errors,
            "summary": {"total": len(df), "succeeded": len(results), "failed": len(errors)},
        })

    # ---------------------------------------------------------- history

    @app.route("/api/history", methods=["GET"])
    @login_required
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
    @role_required("HR_MANAGER", "ADMIN")
    def delete_history_record(record_id):
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        rec = db.session.get(PredictionHistory, record_id)
        if rec is None:
            return _json_error("Record not found", 404, "NOT_FOUND")
        db.session.delete(rec)
        db.session.commit()
        log.info("History record %s deleted by %s", record_id, g.user.username)
        audit("history_delete", {"record_id": record_id})
        return jsonify({"message": "Record deleted"})

    @app.route("/api/history", methods=["DELETE"])
    @role_required("ADMIN")
    def clear_history():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        n = db.session.query(PredictionHistory).delete()
        db.session.commit()
        log.warning("History cleared (%d rows) by admin %s", n, g.user.username)
        audit("history_clear", {"deleted": n})
        return jsonify({"message": "History cleared successfully.", "deleted": n})

    # ---------------------------------------------------------- user admin

    @app.route("/api/users", methods=["GET"])
    @role_required("ADMIN")
    def list_users():
        users = User.query.order_by(User.id).all()
        return jsonify([{
            "id": u.id, "username": u.username, "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else None,
        } for u in users])

    @app.route("/api/users", methods=["POST"])
    @role_required("ADMIN")
    def create_user():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        password = data.get("password")
        role = data.get("role", "ANALYST")
        if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{3,80}", username):
            return _err("username", "3-80 chars, letters/digits/_.-")
        if not isinstance(password, str) or len(password) < 8:
            return _err("password", "minimum 8 characters")
        role, e = _as_choice(role, "role", set(ROLES))
        if e: return e
        if User.query.filter_by(username=username).first():
            return _json_error("Username already exists", 409, "CONFLICT")
        user = User(username=username, password_hash=generate_password_hash(password), role=role)
        db.session.add(user)
        db.session.commit()
        log.info("User created: %s role=%s by %s", username, role, g.user.username)
        audit("user_create", {"username": username, "role": role})
        return jsonify({"id": user.id, "username": user.username, "role": user.role}), 201

    # ---------------------------------------------------------- model registry

    @app.route("/api/model", methods=["GET"])
    @login_required
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

    # ---------------------------------------------------------- audit

    @app.route("/api/audit", methods=["GET"])
    @role_required("ADMIN")
    def get_audit():
        page, e = _as_int(request.args.get("page", 1), "page", 1, 10_000)
        if e: return e
        size, e = _as_int(request.args.get("page_size", 50), "page_size", 1, 200)
        if e: return e
        q = AuditEvent.query.order_by(AuditEvent.timestamp.desc(), AuditEvent.id.desc())
        total = q.count()
        rows = q.offset((page - 1) * size).limit(size).all()
        items = [{
            "id": r.id, "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else None,
            "username": r.username, "action": r.action, "detail": r.detail, "ip": r.ip,
        } for r in rows]
        return jsonify({"items": items, "page": page, "page_size": size, "total": total})

    # ---------------------------------------------------------- password change

    @app.route("/api/auth/change-password", methods=["POST"])
    @login_required
    def change_password():
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        data = request.get_json(silent=True) or {}
        current = data.get("current_password")
        new = data.get("new_password")
        if not isinstance(current, str) or not isinstance(new, str) or len(new) < 8:
            return _err("new_password", "minimum 8 characters")
        user = db.session.get(User, g.user.id)
        if not check_password_hash(user.password_hash, current):
            audit("password_change_failed")
            return _json_error("Current password is incorrect", 403, "BAD_PASSWORD")
        user.password_hash = generate_password_hash(new)
        # revoke all OTHER sessions for this user
        Session.query.filter(Session.user_id == user.id, Session.token != g.session.token).delete()
        db.session.commit()
        audit("password_change")
        log.info("Password changed by %s", user.username)
        return jsonify({"message": "Password updated; other sessions revoked"})

    # ---------------------------------------------------------- scenario simulation

    @app.route("/api/simulate", methods=["POST"])
    @login_required
    def simulate():
        """Isolated what-if scenario. Never persists records. Returns model
        estimates explicitly labelled as non-causal."""
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        if not rate_allow(f"sim:{g.user.id}", 120, 60):
            return _json_error("Simulation rate limit exceeded", 429, "RATE_LIMITED")
        if not load_all():
            return _json_error("Prediction models unavailable", 503, "MODELS_UNAVAILABLE")
        data = request.get_json(silent=True)
        clean, err = _validate_predict_payload(data)
        if err:
            return err
        # scenario deltas: satisfaction_boost, hours_delta, promotion_to
        sat_boost = _as_float(data.get("satisfaction_boost", 0.0), "satisfaction_boost", -1.0, 1.0)
        if sat_boost[1]: return sat_boost[1]
        boost = sat_boost[0]
        hours_delta, e = _as_int(data.get("hours_delta", 0), "hours_delta", -200, 200)
        if e: return e

        scenario = dict(clean)
        scenario["Satisfaction"] = min(1.0, max(0.0, clean["Satisfaction"] + boost))
        if "Hours" in active_feature_fields():
            scenario["Hours"] = max(0, min(500, clean.get("Hours", 160) + hours_delta))

        try:
            X_base = _encode_row(clean)
            X_scen = _encode_row(scenario)
            risk_b, sal_b = _predict_row(X_base)
            risk_s, sal_s = _predict_row(X_scen)
        except Exception:
            log.exception("Simulation failed")
            return _json_error("Simulation failed", 500, "SIMULATION_ERROR")

        audit("simulate", {"satisfaction_boost": boost, "hours_delta": hours_delta})
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

    # ---------------------------------------------------------- workforce analytics

    @app.route("/api/analytics/upload", methods=["POST"])
    @login_required
    def analytics_upload():
        """Workforce statistics from an uploaded CSV. Descriptive only."""
        csrf_err = require_csrf()
        if csrf_err:
            return csrf_err
        if not rate_allow(f"analytics:{g.user.id}", 30, 60):
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
        audit("analytics_upload", {"rows": stats["rows"]})
        return jsonify({"stats": stats, "insights": insights,
                        "disclaimer": "Descriptive statistics on uploaded data only."})

    @app.route("/api/analytics/summary", methods=["GET"])
    @login_required
    def analytics_summary():
        """Aggregates over stored prediction history (not employee records)."""
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

    @app.route("/api/health", methods=["GET"])
    @login_required
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
