import streamlit as st
import hopsworks
import pandas as pd
import os

st.set_page_config(page_title="🌫 Karachi AQI Forecast", layout="centered")
st.title("🌫 Karachi AQI 3-Day AQI Forecast")

# -------------------------
# Config
# -------------------------
PROJECT = "AQIPred"
FORECAST_FG = "aqi_forecast_fg"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# -------------------------
# Login
# -------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
fs = project.get_feature_store()

forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)

# -------------------------
# Read predictions
# -------------------------
try:
    df = forecast_fg.read().sort_values("date")

    if df.empty:
        st.warning("No forecast available yet.")
    else:
        st.subheader("Latest AQI Forecast")
        for i, row in df.tail(3).iterrows():
            st.metric(
                label=row["date"].strftime("%A"),
                value=f"{row['pred_aqi']}"
            )
except Exception as e:
    st.error(f"Could not read forecast: {e}")
