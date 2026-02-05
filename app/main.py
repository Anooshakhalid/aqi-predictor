import streamlit as st
import hopsworks
import pandas as pd
import joblib
from tensorflow import keras
import os
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="🌫 Karachi AQI Predictor", layout="centered")
st.title("🌫 Karachi AQI Predictor")

# -------------------------
# Config
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY") # Use Streamlit secrets
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FV_VERSION = 1
FORECAST_FG = "aqi_forecast_fg"

CANDIDATE_MODELS = [
    "ridge_aqi_model",
    "nn_aqi_model",
    "aqi_rf_model",
    "rf_aqi_model"
]

# -------------------------
# Login to Hopsworks
# -------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
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
        st.warning(f"Could not fetch {name}: {e}")

if latest_model is None:
    st.error("❌ No models found")
    st.stop()

st.write(f"✅ Using model: {latest_model.name} v{latest_model.version}")

# -------------------------
# Download & load model
# -------------------------
model_dir = latest_model.download()

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
        online_enabled=True,
        offline_enabled=True
    )

# -------------------------
# Predict button
# -------------------------
if st.button("Compute 3-Day Forecast"):

    # Fetch latest features
    df = fv.get_batch_data().sort_values("date").tail(1)
    X = df.drop(columns=["date", "aqi"], errors="ignore")

    preds = []
    X_copy = X.copy()

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

    # Insert into forecast feature group
    forecast_df = pd.DataFrame({
        "date": [pd.Timestamp.today() + pd.Timedelta(days=i) for i in range(1, 4)],
        "pred_aqi": preds
    })

    forecast_fg.insert(forecast_df, write_options={"wait_for_job": True})

    st.success("✅ Predictions computed and saved to Feature Store!")

# -------------------------
# Display latest predictions
# -------------------------
try:
    df_forecast = forecast_fg.read().sort_values("date")
    if not df_forecast.empty:
        st.subheader("Latest 3-Day AQI Forecast")
        for i, row in df_forecast.iterrows():
            st.metric(f"Day {i+1} ({row['date'].date()})", f"{row['pred_aqi']}")
    else:
        st.warning("No predictions available yet. Click the button above to compute.")
except Exception as e:
    st.error(f"Error reading forecast: {e}")
