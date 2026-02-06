import streamlit as st
import hopsworks
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Karachi AQI Dashboard", layout="centered")
st.title("Karachi AQI Dashboard")

# -------------------------
# Config
# -------------------------
PROJECT = "AQIPred"
AQI_FG = "karachi_aqishine_fg"       # Original AQI data
FORECAST_FG = "aqi_forecast_fg"      # 3-day forecast
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# -------------------------
# Login to Hopsworks
# -------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT)
fs = project.get_feature_store()

# -------------------------
# Today's AQI
# -------------------------
try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("date", ascending=False)
    if not df_aqi.empty:
        latest_aqi_row = df_aqi.iloc[0]
        today_date = latest_aqi_row["date"]
        today_aqi = latest_aqi_row["aqi"]
        st.subheader("Today's AQI")
        st.write(f"**Date:** {today_date}")
        st.metric(label="AQI", value=f"{today_aqi}")
    else:
        st.warning("No AQI data available yet.")
except Exception as e:
    st.error(f"Could not read today's AQI: {e}")

# -------------------------
# 3-Day Forecast
# -------------------------
try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read().sort_values("date")
    if not df_forecast.empty:
        st.subheader("3-Day AQI Forecast")
        for i, row in df_forecast.tail(3).iterrows():
            # Convert string to datetime if needed
            if isinstance(row["date"], str):
                day_name = datetime.strptime(row["date"], "%Y-%m-%d").strftime("%A")
            else:
                day_name = row["date"].strftime("%A")
            st.metric(label=day_name, value=f"{row['pred_aqi']}")
    else:
        st.warning("No forecast available yet.")
except Exception as e:
    st.error(f"Could not read forecast: {e}")
