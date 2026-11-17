# Employee Salary Prediction System

**Employee Salary Prediction System**

This application is a cutting-edge HR tool designed to help organizations predict **Employee Attrition Risk** and determine **Optimal Market Salaries**. Combining machine learning with a premium user interface, it provides actionable insights for talent retention and resource management.

---

## 🚀 Key Features

- **👤 Employee Prediction**: Real-time analysis of individual employees to determine their risk of leaving and their fair market salary based on experience and performance.
- **📊 Team Analysis**: Bulk assessment mode where you can upload a company-wide CSV to generate risk heatmaps and identifying systemic stability issues.
- **🧪 Prediction Simulator**: A "What-If" lab that allows HR managers to simulate how specific changes (e.g., salary boosts or satisfaction improvements) directly impact retention risk.
- **📜 Past Predictions**: Persistent audit logs backed by a local database to track and review all previous strategic assessments.

---

## 🛠️ Technology Stack

- **Core**: Python 3.14
- **Backend / API**: Flask & Flask-SQLAlchemy
- **Frontend**: Vanilla Javascript, HTML, TailwindCSS (Advanced Glassmorphism UI)
- **Machine Learning**: 
  - `Scikit-learn` (Gradient Boosting Classifier & Regressor)
  - `Joblib` (Model Serialization)
- **Data Engineering**: 
  - `Pandas` & `NumPy` (Preprocessing & Feature Scaling)
- **Visualizations**: 
  - `Chart.js` (Interactive Dashboard & Importance Charts)
- **Database**: 
  - `SQLite` with `flask-sqlalchemy` (Persistent History Logging)
- **Deployment**: 
  - `Vercel` Ready

---

## 🔍 The Machine Learning Approach

The application uses two specialized models trained on historical workforce datasets:

1.  **Attrition Model**: A **Gradient Boosting Classifier** that analyzes factors like satisfaction score, work hours, and tenure to predict the probability of a resignation.
2.  **Salary Model**: A **Gradient Boosting Regressor** that calculates a competitive market salary amount based on the employee's role, experience, and department.
3.  **Interpretability**: The tool provides real-time "Decision Drivers" charts, showing exactly why the AI arrived at a specific result.

---

## 📦 Project Structure

- `app.py`: Main Flask application and API endpoints.
- `train.py`: The background script for model training and pipeline building.
- `models/`: Directory containing the serialized AI brains of the project (`attr_model.pkl`, `sal_model.pkl`, encoders, scalers).
- `templates/` & `static/`: Frontend HTML, CSS, and JS files.
- `employee_attrition_data.csv`: Source dataset for analysis.
- `vercel.json`: Configuration for cloud deployment.

---

## 💻 Local Setup & Running

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/tarun2806/EmployeeSalaryPrediction.git
    cd EmployeeSalaryPrediction
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Predictor**:
    ```bash
    python app.py
    ```

---

*Developed with ❤️ as part of the Strategic HR Intelligence Initative.*
