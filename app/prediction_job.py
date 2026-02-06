import hopsworks
import pandas as pd
import joblib
from tensorflow import keras
import os

# -------------------------
# Config
# -------------------------
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FV_VERSION = 1
FORECAST_FG = "aqi_forecast_fg"

CANDIDATE_MODELS = [
    "aqi_ridge_model",
    "aqi_rf_model",
    "aqi_nn_model"
]

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# -------------------------
# Login
# -------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
fs = project.get_feature_store()
mr = project.get_model_registry()

# -------------------------
# Load best model
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
    except:
        pass

if latest_model is None:
    raise RuntimeError("No models found in registry")

print(f"Using {latest_model.name} v{latest_model.version}")

model_dir = latest_model.download()

def load_model(model_dir):
    for f in os.listdir(model_dir):
        p = os.path.join(model_dir, f)
        if f.endswith(".pkl"):
            return joblib.load(p)
        elif f.endswith(".keras") or f.endswith(".h5"):
            return keras.models.load_model(p)
    raise RuntimeError("No model file found")

model = load_model(model_dir)

# -------------------------
# Read from Feature View
# -------------------------
fv = fs.get_feature_view(FEATURE_VIEW, version=FV_VERSION)

df = fv.get_batch_data().sort_values("date").tail(1)
X = df.drop(columns=["date", "aqi"], errors="ignore")

if hasattr(model, "feature_names_in_"):
    print("MODEL EXPECTS:", list(model.feature_names_in_))


# -------------------------
# 3-day forecast
# -------------------------
preds = []
X_copy = X.copy()

for _ in range(3):
    if isinstance(model, keras.Model):
        y = float(model.predict(X_copy).flatten()[0])
    else:
        y = float(model.predict(X_copy)[0])

    preds.append(round(y, 2))

    if "aqi_lag_1" in X_copy.columns and "aqi_lag_3" in X_copy.columns:
        X_copy["aqi_lag_3"] = X_copy["aqi_lag_1"]
        X_copy["aqi_lag_1"] = y

# -------------------------
# Save to Feature Group
# -------------------------
try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
except:
    forecast_fg = fs.create_feature_group(
        name=FORECAST_FG,
        version=1,
        primary_key=["date"],
        description="AQI 3-day forecast",
        online_enabled=True,
        offline_enabled=True
    )

forecast_df = pd.DataFrame({
    "date": pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=3),
    "pred_aqi": preds
})

forecast_fg.insert(forecast_df, write_options={"wait_for_job": True})

print("Forecast saved to aqi_forecast_fg")
