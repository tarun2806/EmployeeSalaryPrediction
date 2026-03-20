import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Use the 'api' directory as the root for Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
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
            # We don't raise here so the server stays up; predict will error
            print(f"FAILED TO LOAD MODELS: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/script.js')
def serve_script():
    return send_from_directory('static', 'script.js')

@app.route('/salary_banner.jpg')
def serve_banner():
    return send_from_directory('static', 'salary_banner.jpg')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        load_all() # Ensure models are loaded
        if not models:
            return jsonify({'error': 'Models not loaded on server'}), 500
            
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
        return jsonify({
            'status': 'ok', 
            'models_loaded': len(models) > 0,
            'path': str(MODELS_PATH)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Handler for Vercel
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(port=5000)
