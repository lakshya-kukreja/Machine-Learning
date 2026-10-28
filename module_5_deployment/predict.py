import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


from pathlib import Path
import pickle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.bin"

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)
    logger.info("Prediction Logging Successful")

# New customer data
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 15,
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

try:
    probability = pipeline.predict_proba([customer])[0, 1]

    churn = probability >= 0.5

    logger.info("Prediction completed successfully")

    logger.info(f"Churn Probability : {probability : 0.4f}")
    logger.info(f"Will Churn : {churn}")

except Exception:
    logger.exception("Prediction Failed")
