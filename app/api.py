import os, hopsworks, joblib, pandas as pd
from fastapi import FastAPI

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = "your_project"

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
fs = project.get_feature_store()
mr = project.get_model_registry()

model = mr.get_model("aqi_rf_model").download()
model = joblib.load("best_model.pkl")

fv = fs.get_feature_view("aqi_features_fv", version=1)

app = FastAPI()

@app.get("/predict")
def predict():
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"])
    preds = []

    for _ in range(3):
        y = float(model.predict(X)[0])
        preds.append(round(y, 2))
        X["aqi_lag_3"] = X["aqi_lag_1"]
        X["aqi_lag_1"] = y

    return {"forecast": preds}
