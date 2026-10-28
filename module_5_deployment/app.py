import logging


from pathlib import Path
from contextlib import asynccontextmanager
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import time
from fastapi import Request


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)



metrics = {
    "prediction_count" : 0,
    "error_count" : 0,
    "total_latency_seconds" : 0.0
}



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.bin"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        models["pipeline"] = pickle.load(f)

    logger.info("Model Loaded Successfully")
    yield

    models.clear()

app = FastAPI(
    title = "Customer Churn Prediction API",
    lifespan=lifespan
)

class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool

@app.get("/health")
def health():
    return {"status":"ok"}


@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    start_time = time.perf_counter()
    try:
        customer_dict = customer.model_dump()

        probability = float(
            models["pipeline"].predict_proba([customer_dict])[0, 1]
        )
        metrics["prediction_count"] += 1

        return{
            "churn_probability" : probability,
            "churn" : probability >= 0.5
        }
    except Exception:
        logger.exception("Prediction Failed")
        metrics["error_count"] += 1
        raise HTTPException(
            status_code = 500,
            details = "Prediction failed due to an internal error"
        )
        
    finally:
        end_time = time.perf_counter()
        latency = end_time - start_time
        metrics["total_latency_seconds"] += latency

        logger.info("Prediction request latency: %.4f seconds", latency)