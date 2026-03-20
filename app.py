import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"
# Use /tmp on Vercel since the root is read-only
if os.environ.get('VERCEL'):
    DB_PATH = "/tmp/history.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "history.db")

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model for Prediction History
class PredictionHistory(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    salary = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database
with app.app_context():
    db.create_all()

# Global model pointers
models = {}

def load_all():
    if not models:
        try:
            if not MODELS_PATH.exists():
                 print(f"ERROR: Models path {MODELS_PATH} not found!")
                 return False
                
            models["attr_model"] = joblib.load(MODELS_PATH / "attr_model.pkl")
            models["sal_model"] = joblib.load(MODELS_PATH / "sal_model.pkl")
            models["attr_scaler"] = joblib.load(MODELS_PATH / "attr_scaler.pkl")
            models["sal_scaler"] = joblib.load(MODELS_PATH / "sal_scaler.pkl")
            models["le_gender"] = joblib.load(MODELS_PATH / "le_gender.pkl")
            models["le_dept"] = joblib.load(MODELS_PATH / "le_dept.pkl")
            models["le_title"] = joblib.load(MODELS_PATH / "le_title.pkl")
            print("Successfully loaded all models from", MODELS_PATH)
            return True
        except Exception as e:
            print(f"CRITICAL: Failed to load models: {str(e)}")
            return False
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        if not load_all():
            return jsonify({'error': 'Models configuration missing on server.'}), 500
            
        g = models["le_gender"].transform([data['Gender']])[0]
        d = models["le_dept"].transform([data['Department']])[0]
        t = models["le_title"].transform([data['Job_Title']])[0]
        p = 1 if data['Promotion'] == "Yes" else 0
        
        X = [int(data['Age']), g, d, t, int(data['Experience']), float(data['Satisfaction']), int(data['Hours']), p]
        X_arr = np.array(X).reshape(1, -1)
        
        attr_s = models["attr_scaler"].transform(X_arr)
        risk = float(models["attr_model"].predict_proba(attr_s)[0][1])
        
        sal_s = models["sal_scaler"].transform(X_arr)
        sal = float(models["sal_model"].predict(sal_s)[0])
        
        # Get feature importance
        feat_imp = models["attr_model"].feature_importances_.tolist()
        
        # Save to history if name is provided
        name = data.get('Name', 'Anonymous')
        entry = PredictionHistory(name=name, salary=sal, risk_score=risk)
        db.session.add(entry)
        db.session.commit()
        
        return jsonify({
            'risk': round(risk, 4),
            'salary': round(sal, 2),
            'feat_imp': feat_imp,
            'features': ['Age', 'Gender', 'Dept', 'Title', 'Tenure', 'Sat', 'Hours', 'Promo']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET', 'DELETE'])
def handle_history():
    if request.method == 'DELETE':
        try:
            db.session.query(PredictionHistory).delete()
            db.session.commit()
            return jsonify({'message': 'History cleared successfully.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
        history = [{
            'id': r.id,
            'name': r.name,
            'salary': r.salary,
            'risk': r.risk_score,
            'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for r in records]
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_predict', methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if not load_all():
            return jsonify({'error': 'Models configuration missing on server.'}), 500
            
        df = pd.read_csv(file)
        # Required columns mapping from CSV to prediction
        # Expected CSV columns: Age, Gender, Department, Job_Title, Years_at_Company, Satisfaction_Level, Average_Monthly_Hours, Promotion_Last_5Years
        
        results = []
        for _, row in df.iterrows():
            g = models["le_gender"].transform([row['Gender']])[0]
            d = models["le_dept"].transform([row['Department']])[0]
            t = models["le_title"].transform([row['Job_Title']])[0]
            p = 1 if row['Promotion_Last_5Years'] == 1 else 0
            
            X = [row['Age'], g, d, t, row['Years_at_Company'], row['Satisfaction_Level'], row['Average_Monthly_Hours'], p]
            X_arr = np.array(X).reshape(1, -1)
            
            risk = float(models["attr_model"].predict_proba(models["attr_scaler"].transform(X_arr))[0][1])
            sal = float(models["sal_model"].predict(models["sal_scaler"].transform(X_arr))[0])
            
            results.append({
                'EmpID': row.get('Employee_ID', _),
                'risk': risk,
                'salary': sal,
                'dept': row['Department']
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    try:
        load_ok = load_all()
        return jsonify({
            'status': 'ok' if load_ok else 'partially_loaded',
            'models_in_memory': list(models.keys()),
            'db_exists': os.path.exists(DB_PATH)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)

