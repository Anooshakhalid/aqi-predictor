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
# Get newest model (SDK compatible)
# -------------------------
model_names = mr.get_model_names()

if not model_names:
    raise RuntimeError("❌ No models found in registry")

latest_model = None
latest_version = -1

for name in model_names:
    versions = mr.get_models(name)
    for m in versions:
        if m.version > latest_version:
            latest_version = m.version
            latest_model = m

if latest_model is None:
    raise RuntimeError("❌ No models discovered")

print(f"✅ Using model: {latest_model.name} v{latest_model.version}")

# -------------------------
# Download model
# -------------------------
model_dir = latest_model.download()
print("Downloaded files:", os.listdir(model_dir))

# -------------------------
# Load model dynamically
# -------------------------
def load_latest_model(model_dir):
    for f in os.listdir(model_dir):
        path = os.path.join(model_dir, f)
        if f.endswith(".pkl"):
            return joblib.load(path)
        elif f.endswith(".keras") or f.endswith(".h5"):
            return keras.models.load_model(path)
    raise RuntimeError("❌ No loadable model file found")

model = load_latest_model(model_dir)

# -------------------------
# Load Feature View
# -------------------------
fv = fs.get_feature_view(FEATURE_VIEW, version=FV_VERSION)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="AQI Predictor API")

@app.get("/")
def root():
    return {"status": "AQI Predictor is running"}

@app.get("/model_info")
def model_info():
    return {
        "name": latest_model.name,
        "version": latest_model.version,
        "created": str(latest_model.creation_time),
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
