from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model + scaler
model = joblib.load("svr_happiness_model.pkl")
scaler = joblib.load("scaler.pkl")

# Required feature order
FEATURES = [
    "device_hours_per_day",
    "physical_activity_days",
    "sleep_quality",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "focus_score",
    "productivity_score",
    "digital_dependence_score",
    "notifications_per_day"
]

@app.get("/")
def home():
    return {"message": "Happiness Prediction API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df[FEATURES])
    prediction = model.predict(df_scaled)[0]
    return {"predicted_happiness_score": float(prediction)}
