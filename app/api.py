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
# Pick best model automatically
# -------------------------
def get_best_model(mr):
    models = []
    for name in ["aqi_rf_model", "nn_aqi_model", "ridge_aqi_model"]:
        try:
            models.append(mr.get_model(name))  # gets latest
        except:
            pass

    if not models:
        raise Exception("No models found in registry")

    best = max(models, key=lambda m: m.metrics.get("R2", 0))
    return best

best_model_meta = get_best_model(mr)
print(f"Loaded model: {best_model_meta.name} v{best_model_meta.version}")

# -------------------------
# Download + load model
# -------------------------
model_dir = best_model_meta.download()

if best_model_meta.name == "nn_aqi_model":
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
        "name": best_model_meta.name,
        "version": best_model_meta.version,
        "metrics": best_model_meta.metrics
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
