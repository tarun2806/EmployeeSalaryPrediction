import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Load artifacts - bundled inside the api folder
MODELS_PATH = Path(__file__).resolve().parent

# Global model pointers (initially None)
models = {}

def load_all():
    if not models:
        try:
            models["attr_model"] = joblib.load(MODELS_PATH / "attr_model.pkl")
            models["sal_model"] = joblib.load(MODELS_PATH / "sal_model.pkl")
            models["attr_scaler"] = joblib.load(MODELS_PATH / "attr_scaler.pkl")
            models["sal_scaler"] = joblib.load(MODELS_PATH / "sal_scaler.pkl")
            models["le_gender"] = joblib.load(MODELS_PATH / "le_gender.pkl")
            models["le_dept"] = joblib.load(MODELS_PATH / "le_dept.pkl")
            models["le_title"] = joblib.load(MODELS_PATH / "le_title.pkl")
        except Exception as e:
            raise Exception(f"Failed to load models at {MODELS_PATH}: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        load_all() # Ensure models are loaded
        
        # Transform categoricals
        g = models["le_gender"].transform([data['Gender']])[0]
        d = models["le_dept"].transform([data['Department']])[0]
        t = models["le_title"].transform([data['Job_Title']])[0]
        p = 1 if data['Promotion'] == "Yes" else 0
        
        # Columns matching training
        X = [int(data['Age']), g, d, t, int(data['Experience']), float(data['Satisfaction']), int(data['Hours']), p]
        X_arr = np.array(X).reshape(1, -1)
        
        # Attrition Prediction
        attr_s = models["attr_scaler"].transform(X_arr)
        risk = models["attr_model"].predict_proba(attr_s)[0][1]
        
        # Salary Prediction
        sal_s = models["sal_scaler"].transform(X_arr)
        sal = models["sal_model"].predict(sal_s)[0]
        
        # Feature Importance approximation
        feat_imp = models["attr_model"].feature_importances_.tolist()
        
        return jsonify({
            'risk': round(float(risk), 4),
            'salary': round(float(sal), 2),
            'feat_imp': feat_imp,
            'features': ['Age', 'Gender', 'Dept', 'Title', 'Tenure', 'Sat', 'Hours', 'Promo']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    try:
        load_all()
        return jsonify({'status': 'ok', 'message': 'Models reached and loaded', 'path': str(MODELS_PATH)})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Deleted home route to allow static file serving at root

if __name__ == "__main__":
    app.run(port=5000)
