import streamlit as st
import hopsworks
import pandas as pd
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    layout="wide"
)

st.title("🌫 Karachi AQI Dashboard")

# -------------------------
# Config
# -------------------------
PROJECT = "AQIPred"
AQI_FG = "karachi_aqishine_fg"
FORECAST_FG = "aqi_forecast_fg"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# -------------------------
# Login to Hopsworks
# -------------------------
project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=PROJECT
)
fs = project.get_feature_store()

# =========================
# CURRENT AQI (TOP)
# =========================
try:
    aqi_fg = fs.get_feature_group(AQI_FG, version=1)
    df_aqi = aqi_fg.read().sort_values("timestamp", ascending=False)

    if not df_aqi.empty:
        latest_aqi = int(df_aqi.iloc[0]["aqi"])

        st.subheader("Current AQI")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.metric(
                label="Karachi Air Quality Index",
                value=latest_aqi
            )
    else:
        st.warning("No AQI data available.")

except Exception as e:
    st.error(f"Error reading AQI data: {e}")

# =========================
# 3-DAY FORECAST (BELOW, HORIZONTAL)
# =========================
try:
    forecast_fg = fs.get_feature_group(FORECAST_FG, version=1)
    df_forecast = forecast_fg.read().sort_values("timestamp")

    if not df_forecast.empty:
        st.subheader("3-Day AQI Forecast")

        cols = st.columns(3)

        for col, (_, row) in zip(cols, df_forecast.tail(3).iterrows()):
            with col:
                st.metric(
                    label=row["day"],   # e.g. Monday, Tuesday
                    value=int(row["pred_aqi"])
                )

    else:
        st.warning("No forecast data available.")

except Exception as e:
    st.error(f"Error reading forecast data: {e}")
