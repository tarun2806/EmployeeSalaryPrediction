import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# --- Configuration & Paths ---
st.set_page_config(page_title="AURORA AI | Strategic HR Intelligence", page_icon="📈", layout="wide")

BANNER_PATH = "api/static/salary_banner.jpg"
MODELS_DIR = "api/"

# --- Database for History ---
Base = declarative_base()
class PredictionHistory(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    salary = Column(Float)
    risk_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine('sqlite:///history.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- Load ML Assets ---
@st.cache_resource
def load_all():
    models = {
        "attr_model": joblib.load(os.path.join(MODELS_DIR, "attr_model.pkl")),
        "sal_model": joblib.load(os.path.join(MODELS_DIR, "sal_model.pkl")),
        "attr_scaler": joblib.load(os.path.join(MODELS_DIR, "attr_scaler.pkl")),
        "sal_scaler": joblib.load(os.path.join(MODELS_DIR, "sal_scaler.pkl")),
        "le_gender": joblib.load(os.path.join(MODELS_DIR, "le_gender.pkl")),
        "le_dept": joblib.load(os.path.join(MODELS_DIR, "le_dept.pkl")),
        "le_title": joblib.load(os.path.join(MODELS_DIR, "le_title.pkl"))
    }
    return models

models = load_all()

def process_data(row):
    # Encodings
    g = models["le_gender"].transform([row['Gender']])[0]
    d = models["le_dept"].transform([row['Department']])[0]
    t = models["le_title"].transform([row['Job_Title']])[0]
    p = 1 if row['Promotion'] == "Yes" else 0
    
    # Columns: Age, Gender, Department, Job_Title, Years_at_Company, Satisfaction_Level, Average_Monthly_Hours, Promotion_Last_5Years
    X = [row['Age'], g, d, t, row['Experience'], row['Satisfaction'], row['Hours'], p]
    X_arr = np.array(X).reshape(1, -1)
    
    # Scale & Predict
    attr_s = models["attr_scaler"].transform(X_arr)
    risk = models["attr_model"].predict_proba(attr_s)[0][1]
    
    sal_s = models["sal_scaler"].transform(X_arr)
    sal = models["sal_model"].predict(sal_s)[0]
    
    return risk, sal, X_arr

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h1 style='color: #6366f1; font-weight: 800;'>HR SMART ASSISTANT</h1>", unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio("Main Menu", ["👤 Employee Prediction", "📊 Team Analysis", "📜 Past Predictions", "🧪 Prediction Simulator"])
    st.markdown("---")
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, use_container_width=True)
    st.caption("AI-Powered HR Strategic Tool v4.0")

# --- Main Logic ---
if choice == "👤 Employee Prediction":
    st.markdown('<div class="hero-banner"><h1>EMPLOYEE PREDICTION</h1><p>Check the risk of an employee leaving and their market salary</p></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Employee Profile Details</p>', unsafe_allow_html=True)
    with st.container(border=True):
        name = st.text_input("Employee Name", "John Doe")
        col_main1, col_main2, col_main3 = st.columns(3)
        with col_main1:
            age = st.number_input("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col_main2:
            dept = st.selectbox("Department", ["Engineering", "HR", "Sales", "Finance", "Marketing"])
            title = st.selectbox("Job Title", ["Manager", "Engineer", "Accountant", "Analyst", "HR Specialist"])
        with col_main3:
            exp = st.number_input("Experience (Years)", 0, 40, 5)
            promo = st.radio("Promoted recently?", ["No", "Yes"], horizontal=True)
        
        col_slider1, col_slider2 = st.columns(2)
        with col_slider1:
            sat = st.slider("Satisfaction Score (0.0 to 1.0)", 0.0, 1.0, 0.7)
        with col_slider2:
            hrs = st.slider("Average Monthly Work Hours", 80, 400, 160)
        
        btn_analyze = st.button("🚀 EXECUTE AI DEEP ANALYSIS", type="primary", use_container_width=True)

    if btn_analyze:
        input_row = {'Age': age, 'Gender': gender, 'Department': dept, 'Job_Title': title, 
                    'Experience': exp, 'Satisfaction': sat, 'Hours': hrs, 'Promotion': promo}
        
        with st.spinner("Calculating results..."):
            risk, sal, X_arr = process_data(input_row)
            time.sleep(1)
        
        # Persist
        db_sess = Session()
        db_sess.add(PredictionHistory(name=name, salary=sal, risk_score=risk))
        db_sess.commit()
        db_sess.close()
        
        # Displays
        st.markdown('<p class="section-header">AI Predictions Dashboard</p>', unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns([1.5, 1.5, 2])
        with r1:
            st.markdown(f'''<div class="glass-card">
                <p style="color: #94a3b8; font-weight: 600;">RISK OF LEAVING</p>
                <p class="metric-value">{risk*100:.1f}%</p>
                <span class="status-badge" style="background: {'#ef4444' if risk > 0.5 else '#22c55e'}; color: white;">
                    {'🔴 High Risk' if risk > 0.5 else '🟢 Secure'}
                </span>
            </div>''', unsafe_allow_html=True)
        
        with r2:
            st.markdown(f'''<div class="glass-card">
                <p style="color: #94a3b8; font-weight: 600;">EXPECTED SALARY (AMOUNT)</p>
                <p class="metric-value">₹ {sal:,.0f}</p>
                <p style="color: #64748b; font-size: 0.8rem;">Relative To Experience Band: <b>COMPETITIVE</b></p>
            </div>''', unsafe_allow_html=True)
        
        with r3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p style="color: #94a3b8; font-weight: 600;">TALENT RETENTION INSIGHT</p>', unsafe_allow_html=True)
            if risk > 0.5:
                st.error(f"Potential resignation risk detected for {name}. Focus on boosting satisfaction levels.")
            else:
                st.success(f"Profile for {name} is stable. Ensure continued engagement and growth opportunities.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<p class="section-header">Decision Factors (Why this result?)</p>', unsafe_allow_html=True)
        with st.container(border=True):
            feat_imp = models["attr_model"].feature_importances_
            fe_cols = ['Age', 'Gender', 'Dept', 'Title', 'Exp', 'Sat', 'Hrs', 'Promo']
            fig_imp = px.bar(x=fe_cols, y=feat_imp, title="Key Drivers for this Result", color=feat_imp, color_continuous_scale='Viridis')
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=350)
            st.plotly_chart(fig_imp, use_container_width=True)

elif choice == "📊 Team Analysis":
    st.markdown('<div class="hero-banner"><h1>TEAM ANALYSIS</h1><p>Check the stability and salary of your entire team</p></div>', unsafe_allow_html=True)
    
    file_up = st.file_uploader("Upload Team CSV Data", type="csv")
    if file_up:
        bulk_df = pd.read_csv(file_up)
        # Check required columns
        req = ['Age', 'Gender', 'Department', 'Job_Title', 'Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours', 'Promotion_Last_5Years']
        if set(req).issubset(set(bulk_df.columns)):
            with st.spinner("Processing team data..."):
                results = []
                for idx, row in bulk_df.iterrows():
                    mapped = {'Age': row['Age'], 'Gender': row['Gender'], 'Department': row['Department'], 'Job_Title': row['Job_Title'], 'Experience': row['Years_at_Company'], 'Satisfaction': row['Satisfaction_Level'], 'Hours': row['Average_Monthly_Hours'], 'Promotion': "Yes" if row['Promotion_Last_5Years'] == 1 else "No"}
                    risk, sal, _ = process_data(mapped)
                    results.append({'EmpID': row.get('Employee_ID', idx), 'Risk': risk, 'Salary': sal, 'Dept': row['Department']})
                
                res_df = pd.DataFrame(results)
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Overall Team Stability", f"{100 - (res_df['Risk'].mean()*100):.1f}%")
                k2.metric("Total Salary Amount", f"₹ {res_df['Salary'].sum():,.0f}")
                k3.metric("High-Risk Employees", len(res_df[res_df['Risk'] > 0.6]))
                
                st.markdown("---")
                c_heat, c_dist = st.columns([1.5, 1])
                with c_heat:
                    fig_heat = px.scatter(res_df, x="Salary", y="Risk", color="Dept", hover_name="EmpID", size="Risk", title="Salary vs Risk")
                    fig_heat.update_layout(height=450)
                    st.plotly_chart(fig_heat, use_container_width=True)
                with c_dist:
                    fig_dept = px.box(res_df, x="Dept", y="Risk", color="Dept", title="Risk per Department")
                    fig_dept.update_layout(height=450)
                    st.plotly_chart(fig_dept, use_container_width=True)
        else:
            st.error("Invalid CSV columns. Required: " + ", ".join(req))

elif choice == "📜 Past Predictions":
    st.markdown('<div class="hero-banner"><h1>PAST PREDICTIONS</h1><p>Review the history of calculations</p></div>', unsafe_allow_html=True)
    db_sess = Session()
    records = db_sess.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).all()
    db_sess.close()
    
    if records:
        df_h = pd.DataFrame([{ "Name": r.name, "Risk of Leaving": f"{r.risk_score*100:.1f}%", "Salary Amount": f"₹{r.salary:,.0f}", "Date / Time": r.timestamp.strftime('%Y-%m-%d %H:%M:%S') } for r in records])
        st.dataframe(df_h, use_container_width=True)
        if st.button("CLEAR HISTORY"):
            db_sess = Session()
            db_sess.query(PredictionHistory).delete()
            db_sess.commit()
            db_sess.close()
            st.success("History cleared.")
            st.rerun()
    else:
        st.info("No prediction history found.")

elif choice == "🧪 Prediction Simulator":
    st.markdown('<div class="hero-banner"><h1>PREDICTION SIMULATOR</h1><p>Simulate how changes affect the risk of leaving</p></div>', unsafe_allow_html=True)
    st.write("Change the parameters below to see how results change.")
    
    wi_c1, wi_c2 = st.columns([1, 2])
    with wi_c1:
        st.subheader("Base Factors")
        wi_age = st.slider("Age", 18, 65, 30, key="wi_1")
        wi_exp = st.slider("Experience (Yrs)", 0, 40, 5, key="wi_2")
        wi_hrs = st.slider("Monthly Work Hours", 80, 400, 160, key="wi_3")
        wi_sat_orig = st.slider("Original Satisfaction", 0.0, 1.0, 0.5)
        
    with wi_c2:
        st.subheader("Simulate Change")
        boost_sat = st.slider("Improve Satisfaction by", 0.0, 0.5, 0.2)
        
        row_base = {'Age': wi_age, 'Gender': 'Male', 'Department': 'Engineering', 'Job_Title': 'Engineer', 'Experience': wi_exp, 'Satisfaction': wi_sat_orig, 'Hours': wi_hrs, 'Promotion': 'No'}
        row_opt = {'Age': wi_age, 'Gender': 'Male', 'Department': 'Engineering', 'Job_Title': 'Engineer', 'Experience': wi_exp, 'Satisfaction': min(1.0, wi_sat_orig + boost_sat), 'Hours': wi_hrs, 'Promotion': 'No'}
        
        r_base, _, _ = process_data(row_base)
        r_opt, _, _ = process_data(row_opt)
        
        st.markdown(f'''<div class="glass-card" style="text-align: center;">
            <h3>Simulation Result</h3>
            <p>Risk Reduction: <b style="color: #22c55e;">-{(r_base-r_opt)*100:.1f}%</b></p>
            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                <div><p>Original Risk</p><h2 style="color: #ef4444;">{r_base*100:.1f}%</h2></div>
                <div><p>New Risk</p><h2 style="color: #22c55e;">{r_opt*100:.1f}%</h2></div>
            </div>
        </div>''', unsafe_allow_html=True)
        st.success(f"Result: Better satisfaction reduces risk by {(r_base-r_opt)*100:.1f}%.")

# Corporate Footer
st.divider()
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>AURORA Intelligence Layer • Protocol 2.0 • Phase 5 Operational</p>", unsafe_allow_html=True)

# Custom Styling (Shared with the Dashboard)
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        color: #94a3b8 !important;
        font-size: 1.2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6366f1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        height: 100%;
        transition: transform 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-5px); }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)
