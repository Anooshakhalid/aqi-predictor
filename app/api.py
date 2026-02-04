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
# Login to Hopsworks & Feature Store
# -------------------------
project = hopsworks.login(
    project=PROJECT,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()
mr = project.get_model_registry()

# -------------------------
# Candidate model names
# -------------------------
candidate_model_names = ["rf_aqi_model", "ridge_aqi_model", "nn_aqi_model"]  # your 3 models

all_models = []

for name in candidate_model_names:
    try:
        models_of_name = mr.get_models(name=name)
        all_models.extend(models_of_name)
    except Exception as e:
        print(f"Warning: Could not fetch models for {name}: {e}")

if not all_models:
    raise RuntimeError("No models found in Model Registry")

# Pick the latest by creation date
latest_model = max(all_models, key=lambda m: m.created)

print(
    f"Latest model selected: {latest_model.name} "
    f"(v{latest_model.version}) | created: {latest_model.created}"
)


# -------------------------
# Pick the latest model by creation timestamp
# -------------------------
latest_model = max(all_models, key=lambda m: m.created)

print(
    f"Latest model selected: {latest_model.name} "
    f"(v{latest_model.version}) | created: {latest_model.created}"
)

# -------------------------
# Download and load model
# -------------------------
model_dir = latest_model.download()

if "nn" in latest_model.name.lower():
    # TensorFlow / Keras model
    model = load_model(os.path.join(model_dir, "best_model.keras"))
else:
    # scikit-learn / joblib model
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))

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
        "metrics": latest_model.metrics
    }

@app.get("/predict")
def predict():
    # Get latest feature row
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"])

    preds = []
    for _ in range(3):
        y = float(model.predict(X)[0])
        preds.append(round(y, 2))

        # Update lags for next prediction
        X["aqi_lag_3"] = X["aqi_lag_1"]
        X["aqi_lag_1"] = y

    return {"forecast_3_days": preds}
