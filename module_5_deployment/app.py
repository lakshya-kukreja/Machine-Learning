from pathlib import Path
from contextlib import asynccontextmanager
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline.bin"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        models["pipeline"] = pickle.load(f)

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

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    customer_dict = customer.model_dump()

    probability = float(
        models["pipeline"].predict_proba([customer_dict])[0, 1]
    )

    return{
        "churn_probability" : probability,
        "churn" : probability >= 0.5
    }