import joblib

from app import MODELS_PATH, load_all


def test_india_artifacts_exist():
    for f in ["salary_q10.pkl", "salary_q50.pkl", "salary_q90.pkl",
              "meta.json", "preprocess.json"]:
        assert (MODELS_PATH / "india_candidate" / f).exists(), f


def test_registry_points_to_india():
    import json
    reg = json.loads((MODELS_PATH / "registry.json").read_text())
    assert reg["active"] == "india-salary"
    assert reg["versions"]["india-salary"]["kind"] == "india_salary"


def test_models_load_india():
    assert load_all() is True
    import app
    assert app.models.get("_kind") == "india_salary"
    assert app.active_version == "india-salary"


def test_no_ibm_or_legacy_artifacts():
    assert not (MODELS_PATH / "v2").exists()
    for f in ["attr_model.pkl", "sal_model.pkl", "attr_scaler.pkl",
              "sal_scaler.pkl", "le_gender.pkl", "le_dept.pkl", "le_title.pkl"]:
        assert not (MODELS_PATH / f).exists(), f
