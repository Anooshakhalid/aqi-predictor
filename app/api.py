import os
import joblib
import hopsworks
import pandas as pd
from fastapi import FastAPI
from tensorflow.keras.models import load_model

# -------------------------
# Config
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FV_VERSION = 1
MODEL_NAME = "your_model_name"  # <-- replace with your model name

# -------------------------
# Login
# -------------------------
project = hopsworks.login(
    project=PROJECT,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()
mr = project.get_model_registry()

# -------------------------
# Fetch latest model
# -------------------------
latest_model = mr.get_model(MODEL_NAME)  # latest version automatically
print(f"Latest model: {latest_model.name}, version: {latest_model.version}")

r2 = latest_model.metrics.get("R2", None) if latest_model.metrics else None
print(f"R2: {r2}")

# -------------------------
# Download + load model
# -------------------------
model_dir = latest_model.download()

if latest_model.name == "nn_aqi_model":
    model = load_model(os.path.join(model_dir, "best_model.keras"))
else:
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))

# -------------------------
# Load Feature View
# -------------------------
fv = fs.get_feature_view(FEATURE_VIEW, version=FV_VERSION)

# -------------------------
# FastAPI
# -------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"status": "AQI Predictor is running"}

@app.get("/model_info")
def model_info():
    return {
        "name": latest_model.name,
        "version": latest_model.version,
        "metrics": latest_model.metrics
    }

@app.get("/predict")
def predict():
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"])

    preds = []
    for _ in range(3):
        y = float(model.predict(X)[0])
        preds.append(round(y, 2))

        # update lags
        X["aqi_lag_3"] = X["aqi_lag_1"]
        X["aqi_lag_1"] = y

    return {"forecast": preds}
