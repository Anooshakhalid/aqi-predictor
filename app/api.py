import os
import joblib
import hopsworks
import pandas as pd
from fastapi import FastAPI
from tensorflow import keras


# -------------------------
# Config
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FV_VERSION = 1

# Exact model names from the registry
CANDIDATE_MODELS = [
    "ridge_aqi_model",
    "nn_aqi_model",
    "aqi_rf_model",   # v5 → newest
    "rf_aqi_model"
]

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
# Find highest version model
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
# Download
# -------------------------
model_dir = latest_model.download()
print("Files:", os.listdir(model_dir))

# -------------------------
# Load model
# -------------------------
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
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"])

    preds = []
    for _ in range(3):
        y = float(model.predict(X)[0])
        preds.append(round(y, 2))
        X["aqi_lag_3"] = X["aqi_lag_1"]
        X["aqi_lag_1"] = y

    return {"forecast_3_days": preds}
