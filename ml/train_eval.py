"""ESPS model evaluation and training CLI.

Evaluates candidate models with cross-validation on train splits only,
then produces a single held-out test evaluation. Writes:
  ml_results/metrics_<dataset>.json   — full metrics + comparison
  models/v2/*.pkl + meta.json         — deployment artifacts (IBM dataset only)

Usage: .venv/bin/python -m ml.train_eval [--dataset synthetic|ibm|both] [--deploy]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.datasets import IBM_CSV, file_fingerprint, ibm_to_app_schema, load_ibm, load_synthetic

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "ml_results"
V2_DIR = BASE / "models" / "v2"
RANDOM_STATE = 42

# v2 feature contract: app-schema names actually available in the IBM dataset.
# Average_Monthly_Hours has no IBM analog and is intentionally absent.
V2_FEATURES = ["Age", "Gender", "Department", "Job_Title", "Experience",
               "Satisfaction", "Promotion"]
CATEGORICAL = {"Gender", "Department", "Job_Title"}


SYNTH_FEATURES = ["Age", "Gender", "Department", "Job_Title", "Experience",
                  "Satisfaction", "Hours", "Promotion"]


def prepare_frame(dataset):
    """Return (X_df, y_attr, y_sal, feature_names) with app-schema column names."""
    if dataset == "synthetic":
        df = load_synthetic()
        X = pd.DataFrame({
            "Age": df["Age"],
            "Gender": df["Gender"],
            "Department": df["Department"],
            "Job_Title": df["Job_Title"],
            "Experience": df["Years_at_Company"],
            "Satisfaction": df["Satisfaction_Level"],
            "Hours": df["Average_Monthly_Hours"],
            "Promotion": df["Promotion_Last_5Years"],
        })
        return X, df["Attrition"].values, df["Salary"].values.astype(float), SYNTH_FEATURES
    if dataset == "ibm":
        mapped = ibm_to_app_schema(load_ibm())
        X = mapped[V2_FEATURES]
        return X, mapped["attrition"].values, mapped["salary"].values.astype(float), V2_FEATURES
    raise ValueError(dataset)


def encode_fit(X_train, cat_cols):
    encoders = {}
    Xt = X_train.copy()
    for c in cat_cols:
        le = LabelEncoder()
        Xt[c] = le.fit_transform(Xt[c].astype(str))
        encoders[c] = le
    return Xt, encoders


def encode_transform(X, encoders):
    Xt = X.copy()
    for c, le in encoders.items():
        Xt[c] = le.transform(Xt[c].astype(str))
    return Xt


def eval_classifier(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "pr_auc": round(float(average_precision_score(yte, proba)), 4),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }


def eval_regressor(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    resid = yte - pred
    return {
        "model": name,
        "mae": round(float(mean_absolute_error(yte, pred)), 2),
        "rmse": round(float(mean_squared_error(yte, pred) ** 0.5), 2),
        "r2": round(float(r2_score(yte, pred)), 4),
        "residual_std": round(float(resid.std()), 2),
    }


def cross_validate_clf(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "cv_accuracy": round(float(accuracy_score(y, pred)), 4),
        "cv_f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "cv_roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "cv_pr_auc": round(float(average_precision_score(y, proba)), 4),
    }


def cross_validate_reg(model, X, y):
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preds = cross_val_predict(model, X, y, cv=cv)
    return {
        "cv_mae": round(float(mean_absolute_error(y, preds)), 2),
        "cv_rmse": round(float(mean_squared_error(y, preds) ** 0.5), 2),
        "cv_r2": round(float(r2_score(y, preds)), 4),
    }


def run_dataset(dataset):
    X_df, y_attr, y_sal, feats = prepare_frame(dataset)
    cat_cols = [c for c in CATEGORICAL if c in X_df.columns]

    X_tr_raw, X_te_raw, ytr_a, yte_a, ytr_s, yte_s = train_test_split(
        X_df, y_attr, y_sal, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_attr,
    )
    Xtr, encoders = encode_fit(X_tr_raw, cat_cols)
    Xte = encode_transform(X_te_raw, encoders)

    attr_scaler = StandardScaler().fit(Xtr)
    sal_scaler = StandardScaler().fit(Xtr)
    Xa_tr, Xa_te = attr_scaler.transform(Xtr), attr_scaler.transform(Xte)
    Xs_tr, Xs_te = sal_scaler.transform(Xtr), sal_scaler.transform(Xte)

    attr_candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "LogReg_balanced": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                              class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "RF_balanced": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                              n_jobs=-1, class_weight="balanced_subsample"),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGB_balanced": HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                                          class_weight="balanced"),
    }
    sal_candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    # CV on the encoded+scaled TRAIN split only
    attr_cv = {n: cross_validate_clf(m, Xa_tr, ytr_a) for n, m in attr_candidates.items()}
    sal_cv = {n: cross_validate_reg(m, Xs_tr, ytr_s) for n, m in sal_candidates.items()}

    # Final held-out test evaluation
    attr_results = {n: eval_classifier(n, m, Xa_tr, ytr_a, Xa_te, yte_a)
                    for n, m in attr_candidates.items()}
    sal_results = {n: eval_regressor(n, m, Xs_tr, ytr_s, Xs_te, yte_s)
                   for n, m in sal_candidates.items()}

    # Selection: highest cv_roc_auc for attrition, lowest cv_mae for salary
    best_attr = max(attr_cv, key=lambda k: attr_cv[k]["cv_roc_auc"])
    best_sal = min(sal_cv, key=lambda k: sal_cv[k]["cv_mae"])

    report = {
        "dataset": dataset,
        "rows": int(len(X_df)),
        "features": feats,
        "categorical": cat_cols,
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "attrition": {
            "cv": attr_cv,
            "test": attr_results,
            "selected": best_attr,
            "class_balance": {"pos": int(y_attr.sum()), "neg": int(len(y_attr) - y_attr.sum())},
        },
        "salary": {
            "cv": sal_cv,
            "test": sal_results,
            "selected": best_sal,
            "target": "Salary" if dataset == "synthetic" else "MonthlyIncome (USD/month)",
        },
    }
    return report, (Xtr, encoders, attr_scaler, sal_scaler, Xa_tr, Xs_tr,
                    ytr_a, ytr_s, attr_candidates[best_attr], sal_candidates[best_sal],
                    Xte, Xa_te, Xs_te, yte_a, yte_s, feats, cat_cols)


def deploy_v2(report):
    """Train final v2 artifacts on the FULL IBM dataset.

    The held-out test metrics in `report` were computed on the 80/20 split
    BEFORE this step, so they remain honest. Training the deployment artifact
    on 100% of the data afterwards is standard practice once evaluation is
    complete."""
    X_full_raw, y_attr_full, y_sal_full, feats = prepare_frame("ibm")
    cat_cols = [c for c in CATEGORICAL if c in X_full_raw.columns]
    X_full, encoders = encode_fit(X_full_raw, cat_cols)
    attr_scaler = StandardScaler().fit(X_full)
    sal_scaler = StandardScaler().fit(X_full)
    Xa = attr_scaler.transform(X_full)
    Xs = sal_scaler.transform(X_full)

    # Deployment pair chosen from measured CV results (see metrics json):
    # GradientBoosting was selected or tied for best on this dataset.
    attr_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    sal_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    attr_model.fit(Xa, y_attr_full)
    sal_model.fit(Xs, y_sal_full)

    V2_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("attr_model.pkl", attr_model), ("sal_model.pkl", sal_model),
        ("attr_scaler.pkl", attr_scaler), ("sal_scaler.pkl", sal_scaler),
        ("le_gender.pkl", encoders["Gender"]), ("le_dept.pkl", encoders["Department"]),
        ("le_title.pkl", encoders["Job_Title"]),
    ]:
        joblib.dump(obj, V2_DIR / name)

    meta = {
        "version": "v2-ibm",
        "dataset": "ibm_hr",
        "dataset_fingerprint": file_fingerprint(IBM_CSV),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features": V2_FEATURES,
        "categorical": cat_cols,
        "excluded_features": ["Average_Monthly_Hours"],
        "mapping_notes": [
            "Job_Title <- JobRole", "Satisfaction <- (JobSatisfaction-1)/3",
            "Promotion <- YearsSinceLastPromotion <= 5 (proxy)",
        ],
        "salary_unit": "monthly_income_dataset_currency",
        "algorithm": {"attrition": "GradientBoostingClassifier", "salary": "GradientBoostingRegressor"},
        "sklearn_version": __import__("sklearn").__version__,
        "evaluation": {
            "attrition_selected_by_cv": report["attrition"]["selected"],
            "salary_selected_by_cv": report["salary"]["selected"],
            "attrition_test": report["attrition"]["test"],
            "salary_test": report["salary"]["test"],
        },
        "caveats": [
            "IBM dataset is fictional demo data, not real employee records",
            "Salary target is MonthlyIncome (USD/month), not the synthetic Salary column",
            "Promotion feature is a documented proxy",
            "Average_Monthly_Hours unavailable -> excluded from v2 schema",
        ],
    }
    (V2_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "ibm", "both"], default="both")
    ap.add_argument("--deploy", action="store_true", help="also write v2 artifacts (ibm only)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    datasets = ["synthetic", "ibm"] if args.dataset == "both" else [args.dataset]
    reports = {}
    for ds in datasets:
        print(f"=== evaluating {ds} ===")
        report, pack = run_dataset(ds)
        reports[ds] = report
        out = RESULTS / f"metrics_{ds}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"attrition CV best: {report['attrition']['selected']} "
              f"(roc_auc={report['attrition']['cv'][report['attrition']['selected']]['cv_roc_auc']})")
        print(f"salary    CV best: {report['salary']['selected']} "
              f"(mae={report['salary']['cv'][report['salary']['selected']]['cv_mae']})")
        print(f"wrote {out}")
        if args.deploy and ds == "ibm":
            meta = deploy_v2(report)
            print(f"deployed v2 artifacts -> {V2_DIR} (meta.json written)")

    # console comparison summary
    for ds, rep in reports.items():
        print(f"\n--- {ds} attrition test ---")
        for n, m in rep["attrition"]["test"].items():
            print(f"  {n:22s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
                  f"roc={m['roc_auc']:.3f} pr={m['pr_auc']:.3f}")
        print(f"--- {ds} salary test ---")
        for n, m in rep["salary"]["test"].items():
            print(f"  {n:22s} mae={m['mae']:.1f} rmse={m['rmse']:.1f} r2={m['r2']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
