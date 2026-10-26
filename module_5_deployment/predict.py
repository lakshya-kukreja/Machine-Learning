from pathlib import Path
import pickle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.bin"

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# New customer data
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 12,
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

probability = pipeline.predict_proba([customer])[0, 1]

churn = probability >= 0.5

print(f"Churn Probability : {probability : 0.4f}")
print(f"Will Churn : {churn}")