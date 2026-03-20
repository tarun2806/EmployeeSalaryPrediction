import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Absolute paths for Vercel stability
BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global model pointers
models = {}

def load_all():
    if not models:
        try:
            # Re-verify the models directory on every request if needed
            if not MODELS_PATH.exists():
                print(f"ERROR: Models path {MODELS_PATH} not found!")
                return
                
            models["attr_model"] = joblib.load(MODELS_PATH / "attr_model.pkl")
            models["sal_model"] = joblib.load(MODELS_PATH / "sal_model.pkl")
            models["attr_scaler"] = joblib.load(MODELS_PATH / "attr_scaler.pkl")
            models["sal_scaler"] = joblib.load(MODELS_PATH / "sal_scaler.pkl")
            models["le_gender"] = joblib.load(MODELS_PATH / "le_gender.pkl")
            models["le_dept"] = joblib.load(MODELS_PATH / "le_dept.pkl")
            models["le_title"] = joblib.load(MODELS_PATH / "le_title.pkl")
            print("Models successfully loaded from", MODELS_PATH)
        except Exception as e:
             print(f"FAILED TO LOAD MODELS: {e}")

@app.route('/')
def home():
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
        load_all()
        if not models:
            return jsonify({'error': 'Models not found on server at: ' + str(MODELS_PATH)}), 500
            
        g = models["le_gender"].transform([data['Gender']])[0]
        d = models["le_dept"].transform([data['Department']])[0]
        t = models["le_title"].transform([data['Job_Title']])[0]
        p = 1 if data['Promotion'] == "Yes" else 0
        
        X = [int(data['Age']), g, d, t, int(data['Experience']), float(data['Satisfaction']), int(data['Hours']), p]
        X_arr = np.array(X).reshape(1, -1)
        
        attr_s = models["attr_scaler"].transform(X_arr)
        risk = models["attr_model"].predict_proba(attr_s)[0][1]
        
        sal_s = models["sal_scaler"].transform(X_arr)
        sal = models["sal_model"].predict(sal_s)[0]
        
        feat_imp = models["attr_model"].feature_importances_.tolist()
        
        return jsonify({
            'risk': round(float(risk), 4),
            'salary': round(float(sal), 2),
            'feat_imp': feat_imp,
            'features': ['Age', 'Gender', 'Dept', 'Title', 'Tenure', 'Sat', 'Hours', 'Promo']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    try:
        load_all()
        return jsonify({
            'status': 'ok',
            'models': list(models.keys()),
            'path': str(BASE_DIR),
            'templates_exist': (BASE_DIR / 'templates' / 'index.html').exists()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Function name must be app for Vercel
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(port=5000)
