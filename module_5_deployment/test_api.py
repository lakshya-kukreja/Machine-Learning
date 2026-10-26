import requests

url = "http://127.0.0.1:8000/predict"

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 5,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 79.85,
    "totalcharges": 950.20
}

response = requests.post(url, json=customer, timeout=10)

print("Status Code : ", response.status_code)
print("Prediction : ", response.json())