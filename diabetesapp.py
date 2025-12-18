from flask import Flask, request, jsonify
import numpy as np
import sqlite3
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# ======================
# LOAD MODEL & SCALER
# ======================
model = load_model(os.path.join(BASE_DIR, "model", "diabetes_model.keras"), compile=False)
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

CLASSES = ['N', 'P', 'Y']

# ======================
# DATABASE INIT
# ======================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            bmi REAL,
            hba1c REAL,
            prediction TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ======================
# LOG PREDICTION
# ======================
def log_prediction(age, bmi, hba1c, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (age, bmi, hba1c, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
    """, (age, bmi, hba1c, prediction, probability))
    conn.commit()
    conn.close()

# ======================
# PREDICTION API
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = [
            'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol',
            'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[
            float(data['AGE']),
            float(data['Urea']),
            float(data['Cr']),
            float(data['HbA1c']),
            float(data['Chol']),
            float(data['TG']),
            float(data['HDL']),
            float(data['LDL']),
            float(data['VLDL']),
            float(data['BMI']),
            float(data['Gender'])
        ]])

        features_scaled = scaler.transform(features)
        probs = model.predict(features_scaled, verbose=0)[0]

        idx = int(np.argmax(probs))
        prediction = CLASSES[idx]
        confidence = float(probs[idx])

        log_prediction(
            data['AGE'],
            data['BMI'],
            data['HbA1c'],
            prediction,
            confidence
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# HEALTH CHECK
# ======================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"})

# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
