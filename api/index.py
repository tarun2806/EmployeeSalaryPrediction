import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load artifacts
# We use relative paths for Vercel
# Load artifacts - bundled inside the api folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = BASE_DIR

attr_model = joblib.load(os.path.join(MODELS_PATH, "attr_model.pkl"))
sal_model = joblib.load(os.path.join(MODELS_PATH, "sal_model.pkl"))
attr_scaler = joblib.load(os.path.join(MODELS_PATH, "attr_scaler.pkl"))
sal_scaler = joblib.load(os.path.join(MODELS_PATH, "sal_scaler.pkl"))
le_gender = joblib.load(os.path.join(MODELS_PATH, "le_gender.pkl"))
le_dept = joblib.load(os.path.join(MODELS_PATH, "le_dept.pkl"))
le_title = joblib.load(os.path.join(MODELS_PATH, "le_title.pkl"))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Transform categoricals
        g = le_gender.transform([data['Gender']])[0]
        d = le_dept.transform([data['Department']])[0]
        t = le_title.transform([data['Job_Title']])[0]
        p = 1 if data['Promotion'] == "Yes" else 0
        
        # Columns: Age, Gender, Department, Job_Title, Years_at_Company, Satisfaction_Level, Average_Monthly_Hours, Promotion_Last_5Years
        X = [int(data['Age']), g, d, t, int(data['Experience']), float(data['Satisfaction']), int(data['Hours']), p]
        # In sklearn v1.2+, using a DataFrame with the same feature names as training preserves the scaler's context
        # But for GradientBoosting, numpy array is usually fine
        X_arr = np.array(X).reshape(1, -1)
        
        # Attrition Prediction
        attr_s = attr_scaler.transform(X_arr)
        risk = attr_model.predict_proba(attr_s)[0][1]
        
        # Salary Prediction
        sal_s = sal_scaler.transform(X_arr)
        sal = sal_model.predict(sal_s)[0]
        
        # Feature Importance approximation
        feat_imp = attr_model.feature_importances_.tolist()
        
        return jsonify({
            'risk': round(float(risk), 4),
            'salary': round(float(sal), 2),
            'feat_imp': feat_imp,
            'features': ['Age', 'Gender', 'Dept', 'Title', 'Tenure', 'Sat', 'Hours', 'Promo']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

# Handler for Vercel
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(port=5000)
