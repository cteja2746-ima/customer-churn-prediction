from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("models/random_forest_model.pkl")
features = joblib.load("models/model_features.pkl")

class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict_churn")
def predict_churn(data: CustomerInput):
    input_df = pd.DataFrame([data.dict()])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    return {
        "churn_prediction": int(prediction),
        "result": "Customer may churn" if prediction == 1 else "Customer may stay"
    }