import joblib
import numpy as np
import pytest

from app import MODELS_PATH, FEATURE_LABELS, _encode_row, _predict_row, load_all
from conftest import VALID_PAYLOAD


def test_all_artifacts_load():
    assert load_all() is True
    for key in ["attr_model", "sal_model", "attr_scaler", "sal_scaler",
                "le_gender", "le_dept", "le_title"]:
        assert key in __import__("app").models


def test_model_files_exist():
    for f in ["attr_model.pkl", "sal_model.pkl", "attr_scaler.pkl", "sal_scaler.pkl",
              "le_gender.pkl", "le_dept.pkl", "le_title.pkl"]:
        assert (MODELS_PATH / f).exists(), f


def test_feature_contract():
    assert FEATURE_LABELS == ["Age", "Gender", "Dept", "Title", "Tenure", "Sat", "Hours", "Promo"]


def test_prediction_bounds_and_determinism():
    load_all()
    clean = {
        "Age": 30, "Gender": "Male", "Department": "Engineering",
        "Job_Title": "Engineer", "Experience": 5, "Satisfaction": 0.7,
        "Hours": 160, "Promotion": "No",
    }
    X = _encode_row(clean)
    assert X.shape == (1, 8)
    r1, s1 = _predict_row(X)
    r2, s2 = _predict_row(X)
    assert r1 == r2 and s1 == s2
    assert 0.0 <= r1 <= 1.0
    assert s1 > 0


def test_label_encoders_cover_ui_categories():
    load_all()
    from app import models
    assert set(models["le_gender"].classes_) == {"Female", "Male"}
    assert set(models["le_dept"].classes_) == {"Engineering", "Finance", "HR", "Marketing", "Sales"}
    assert set(models["le_title"].classes_) == {"Accountant", "Analyst", "Engineer", "HR Specialist", "Manager"}
