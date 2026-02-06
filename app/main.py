import streamlit as st
import hopsworks
import pandas as pd
import os
from datetime import datetime

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Karachi AQI Forecast", layout="wide")
st.title("Karachi AQI Dashboard")

# -------------------------
# Config
# -------------------------
PROJECT = "AQIPred"
FEATURE_VIEW = "aqi_features_fv"
FORECAST_FG = "aqi_forecast_fg"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# -------------------------
# Login to Hopsworks
# -------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
fs = project.get_feature_store()

# -------------------------
# Fetch today's AQI from feature view
# -------------------------
try:
    fv = fs.get_feature_view(FEATURE_VIEW, version=1)
    today_data = fv.get_batch_data().sort_values("date").tail(1)
    today_aqi = today_data["aqi"].values[0] if not today_data.empty else None
except Exception as e:
    st.error(f"Error fetching today's AQI: {e}")
    today_aqi = None

# -------------------------
# Display today's AQI
# -------------------------
st.subheader("Current AQI in Karachi")
if today_aqi is not None:
    st.metric(label="Today", value=f"{today_aqi}")
else:
    st.warning("No data for today available.")

# -------------------------
# Read 3-day forecast
# -------------------------
try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df = forecast_fg.read().sort_values("date")

    if df.empty:
        st.warning("No forecast available yet.")
    else:
        st.subheader("3-Day Forecast")
        cols = st.columns(3)
        for idx, row in df.tail(3).iterrows():
            day_name = datetime.strptime(row["date"], "%Y-%m-%d").strftime("%A")
            with cols[idx % 3]:
                st.markdown(f"### {day_name}")
                st.markdown(f"**Predicted AQI:** {row['pred_aqi']}")
except Exception as e:
    st.error(f"Could not read forecast: {e}")
