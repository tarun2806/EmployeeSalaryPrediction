import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Load data
df = pd.read_csv('employee_attrition_data.csv')

# Preprocessing
le_gender = LabelEncoder()
le_dept = LabelEncoder()
le_title = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Department'] = le_dept.fit_transform(df['Department'])
df['Job_Title'] = le_title.fit_transform(df['Job_Title'])

# Features
features = ['Age', 'Gender', 'Department', 'Job_Title', 'Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours', 'Promotion_Last_5Years']
X = df[features]
y_attr = df['Attrition']
y_sal = df['Salary']

# Split
# For Attrition
X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(X, y_attr, test_size=0.2, random_state=42, stratify=y_attr)
# For Salary
X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(X, y_sal, test_size=0.2, random_state=42)

# Scale
attr_scaler = StandardScaler()
X_train_attr_scaled = attr_scaler.fit_transform(X_train_attr)
X_test_attr_scaled = attr_scaler.transform(X_test_attr)

sal_scaler = StandardScaler()
X_train_sal_scaled = sal_scaler.fit_transform(X_train_sal)
X_test_sal_scaled = sal_scaler.transform(X_test_sal)

# Train Attrition (Gradient Boosting)
attr_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
attr_model.fit(X_train_attr_scaled, y_train_attr)
attr_acc = accuracy_score(y_test_attr, attr_model.predict(X_test_attr_scaled))
print(f"Attr Accuracy: {attr_acc:.4f}")

# Train Salary (Gradient Boosting Regressor)
sal_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
sal_model.fit(X_train_sal_scaled, y_train_sal)
sal_preds = sal_model.predict(X_test_sal_scaled)
sal_mae = mean_absolute_error(y_test_sal, sal_preds)
sal_r2 = r2_score(y_test_sal, sal_preds)
print(f"Sal MAE: {sal_mae:.2f}, R2: {sal_r2:.4f}")

# Save
import os
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(attr_model, 'models/attr_model.pkl')
joblib.dump(sal_model, 'models/sal_model.pkl')
joblib.dump(attr_scaler, 'models/attr_scaler.pkl')
joblib.dump(sal_scaler, 'models/sal_scaler.pkl')
joblib.dump(le_gender, 'models/le_gender.pkl')
joblib.dump(le_dept, 'models/le_dept.pkl')
joblib.dump(le_title, 'models/le_title.pkl')

print("Retrained successfully with Gradient Boosting Models. Saved to models/")
