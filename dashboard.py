import streamlit as st
import sqlite3
import pandas as pd
import requests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    layout="wide"
)

st.title("Diabetes Prediction Dashboard")

# =========================
# INPUT FORM
# =========================
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        AGE = st.number_input("Age", 1, 120, 50)
        BMI = st.number_input("BMI", 10.0, 60.0, 24.0)
        gender_label = st.selectbox("Gender", ["Female", "Male"])
        Gender = 0 if gender_label == "Female" else 1

    with col2:
        HbA1c = st.number_input("HbA1c", 3.0, 15.0, 5.0)
        Chol = st.number_input("Cholesterol", 1.0, 10.0, 4.2)
        TG = st.number_input("Triglycerides", 0.1, 5.0, 0.9)

    with col3:
        HDL = st.number_input("HDL", 0.1, 5.0, 2.4)
        LDL = st.number_input("LDL", 0.1, 5.0, 1.4)
        VLDL = st.number_input("VLDL", 0.1, 5.0, 0.5)

    Urea = st.number_input("Urea", 1.0, 20.0, 4.7)
    Cr = st.number_input("Creatinine", 10.0, 200.0, 46.0)

    submitted = st.form_submit_button("Predict")

# =========================
# PREDICTION CALL
# =========================
if submitted:
    payload = {
        "Gender": Gender,
        "AGE": AGE,
        "Urea": Urea,
        "Cr": Cr,
        "HbA1c": HbA1c,
        "Chol": Chol,
        "TG": TG,
        "HDL": HDL,
        "LDL": LDL,
        "VLDL": VLDL,
        "BMI": BMI
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
            st.info(f"Confidence: **{result['confidence']:.2f}**")
        else:
            st.error(" API error occurred")

    except requests.exceptions.ConnectionError:
        st.error("Flask API is not running. Start the API first.")
    except requests.exceptions.Timeout:
        st.error("API request timed out.")

# =========================
# DASHBOARD SECTION
# =========================
st.divider()
st.subheader("📈 Prediction History")

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df["prediction"].value_counts())

        with col2:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.line_chart(df.groupby(df["timestamp"].dt.date).size())

    else:
        st.info("No predictions yet.")

except Exception:
    st.warning("No database found yet. Make a prediction first.")
