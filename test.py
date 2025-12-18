import requests

payload = {
    "Gender": 1, "AGE": 50, "Urea": 4.7, "Cr": 46,
    "HbA1c": 5.0, "Chol": 4.2, "TG": 0.9,
    "HDL": 2.4, "LDL": 1.4, "VLDL": 0.5, "BMI": 24.0
}

r = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(r.json())