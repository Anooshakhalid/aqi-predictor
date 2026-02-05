import os
import joblib
import hopsworks
import pandas as pd
from fastapi import FastAPI
from tensorflow import keras
import numpy as np
from datetime import datetime, timedelta

# -------------------------
# Config
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FV_VERSION = 1
FORECAST_FG = "aqi_forecast_fg"

# Exact model names from the registry
CANDIDATE_MODELS = [
    "ridge_aqi_model",
    "nn_aqi_model",
    "aqi_rf_model",   # v5 → newest
    "rf_aqi_model"
]

# -------------------------
# Login to Hopsworks
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
latest_model = None
latest_version = -1
for name in CANDIDATE_MODELS:
    try:
        models = mr.get_models(name)
        for m in models:
            if m.version > latest_version:
                latest_version = m.version
                latest_model = m
    except Exception as e:
        print(f"⚠️ Could not fetch {name}: {e}")

if latest_model is None:
    raise RuntimeError("❌ No models found")

print(f"✅ Using model: {latest_model.name} v{latest_model.version}")

# -------------------------
# Download & load model
# -------------------------
model_dir = latest_model.download()
print("Files:", os.listdir(model_dir))

def load_model(model_dir):
    for f in os.listdir(model_dir):
        path = os.path.join(model_dir, f)
        if f.endswith(".pkl"):
            return joblib.load(path)
        elif f.endswith(".keras") or f.endswith(".h5"):
            return keras.models.load_model(path)
    raise RuntimeError("No loadable model file")

model = load_model(model_dir)

# -------------------------
# Feature View
# -------------------------
fv = fs.get_feature_view(FEATURE_VIEW, version=FV_VERSION)

# -------------------------
# Forecast Feature Group
# -------------------------
try:
    forecast_fg = fs.get_feature_group(name=FORECAST_FG, version=1)
except:
    forecast_fg = fs.create_feature_group(
        name=FORECAST_FG,
        version=1,
        description="3-day AQI forecast for Karachi",
        primary_key=["date"],
        online_enabled=True
    )

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="AQI Predictor API")

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/model_info")
def model_info():
    return {
        "name": latest_model.name,
        "version": latest_model.version,
        "metrics": latest_model.metrics
    }

@app.get("/predict")
def predict():
    # Fetch latest feature row
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"], errors="ignore")

    preds = []
    X_copy = X.copy()

    # Compute 3-day forecast
    for _ in range(3):
        if isinstance(model, keras.Model):
            y = float(model.predict(X_copy).flatten()[0])
        else:
            y = float(model.predict(X_copy)[0])
        preds.append(round(y, 2))

        # Update lag features
        if "aqi_lag_1" in X_copy.columns and "aqi_lag_3" in X_copy.columns:
            X_copy["aqi_lag_3"] = X_copy["aqi_lag_1"]
            X_copy["aqi_lag_1"] = y

    # Prepare DataFrame for Feature Store
    forecast_df = pd.DataFrame({
        "date": [pd.Timestamp.today() + pd.Timedelta(days=i) for i in range(1, 4)],
        "pred_aqi": preds
    })

    # Insert into forecast feature group
    forecast_fg.insert(forecast_df, write_options={"wait_for_job": True})

    return {"forecast_3_days": preds}
