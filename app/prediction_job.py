import os
import joblib
import hopsworks
import pandas as pd
from tensorflow import keras
from datetime import datetime, timedelta

# -------------------------
# Configs
# -------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"        # FV from karachi_aqishine_fg
FV_VERSION = 1
FORECAST_FG = "aqi_forecast_fg"

# Candidate models from registry (newest will be picked)
CANDIDATE_MODELS = [
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
        print(f"⚠️ Could not fetch {name}: {e}")

if latest_model is None:
    raise RuntimeError("❌ No models found")

print(f"✅ Using model: {latest_model.name} v{latest_model.version}")

# -------------------------
# Download & load model
# -------------------------
model_dir = latest_model.download()
print("Files in model dir:", os.listdir(model_dir))

def load_model(model_dir):
    for f in os.listdir(model_dir):
        path = os.path.join(model_dir, f)
        if f.endswith(".pkl"):
            return joblib.load(path)
        elif f.endswith(".keras") or f.endswith(".h5"):
            return keras.models.load_model(path)
    raise RuntimeError("No loadable model file")

model = load_model(model_dir)

# Print expected features
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    print("MODEL EXPECTS:", expected)
else:
    raise RuntimeError("Model does not have feature_names_in_")

# -------------------------
# Feature View (for input features)
# -------------------------
fv = fs.get_feature_view(FEATURE_VIEW, version=FV_VERSION)

# -------------------------
# Forecast Feature Group (for predictions)
# -------------------------
forecast_fg = fs.get_feature_group(name=FORECAST_FG, version=1)
if forecast_fg is None:
    print(f"Feature group {FORECAST_FG} not found, creating it...")
    forecast_fg = fs.create_feature_group(
    name="aqi_forecast_fg",
    version=1,
    description="3-day AQI forecast for Karachi",
    primary_key=["date"],   # date as string PK
    online_enabled=True     # keeps it online if needed

)



# -------------------------
# Fetch latest feature row
# -------------------------
df = fv.get_batch_data().sort_values("date").tail(1)
X = df[expected]      # enforce exact feature order
X_copy = X.copy()

preds = []

# ------------------------
# 3-Day Forecast
# ------------------------
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

# -------------------------
# Prepare DataFrame for Feature Store
# -------------------------
forecast_df = pd.DataFrame({
    "date": [pd.Timestamp.today() + pd.Timedelta(days=i) for i in range(1, 4)],
    "pred_aqi": preds
})

# -------------------------
# Insert into Feature Store
# -------------------------
forecast_df = pd.DataFrame({
    "date": [(pd.Timestamp.today() + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)],
    "pred_aqi": preds
})
forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
print("Forecast inserted into Feature Store:", FORECAST_FG)
print("3-Day AQI Forecast:", preds)
